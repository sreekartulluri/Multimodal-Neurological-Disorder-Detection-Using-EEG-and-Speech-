"""
Streamlit Web Application.
Provides a premium clinical dashboard UI for uploading EEG and Speech files,
visualizing waveforms and features, running predictions, and displaying SHAP explanations.
"""

import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import librosa

from predict import load_prediction_pipeline, predict_disorder
from explainability import explain_prediction

# Page config
st.set_page_config(
    page_title="AURA-Net: Multimodal Neurological Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism/dark mode medical styling
st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header styling */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #58a6ff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(88, 166, 255, 0.2);
    }
    
    h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #f0f6fc !important;
        font-weight: 600 !important;
    }
    
    /* Custom Card container */
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .status-healthy {
        border-left: 5px solid #2ea44f;
    }
    .status-parkinsons {
        border-left: 5px solid #d29922;
    }
    .status-alzheimers {
        border-left: 5px solid #db6d28;
    }
    .status-epilepsy {
        border-left: 5px solid #f85149;
    }
    
    /* Custom button style */
    div.stButton > button {
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        border-color: #58a6ff !important;
        color: #58a6ff !important;
        box-shadow: 0 0 8px rgba(88, 166, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Helper to plot raw speech waveform
def plot_speech_waveform(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    t = np.linspace(0, len(y)/sr, len(y))
    
    fig, ax = plt.subplots(figsize=(8, 2.5), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    ax.plot(t, y, color='#58a6ff', linewidth=0.5, alpha=0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.set_xlabel('Time (seconds)', color='#8b949e', fontsize=9)
    ax.set_ylabel('Amplitude', color='#8b949e', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.2, color='#30363d')
    plt.tight_layout()
    return fig

# Helper to plot raw EEG signals
def plot_eeg_signals(edf_path, channels_to_plot=['Fp1', 'Cz', 'O1']):
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    fs = int(raw.info['sfreq'])
    duration = 10
    n_samples = int(duration * fs)
    
    fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(8, 4), sharex=True, facecolor='#161b22')
    if len(channels_to_plot) == 1:
        axes = [axes]
        
    t = np.linspace(0, duration, n_samples)
    
    from eeg_features import find_channel_mapping
    ch_mapping = find_channel_mapping(raw.ch_names)
    
    colors = ['#ff7b72', '#7ee787', '#d29922']
    
    for idx, ch in enumerate(channels_to_plot):
        ax = axes[idx]
        ax.set_facecolor('#161b22')
        
        if ch in ch_mapping:
            edf_ch = ch_mapping[ch]
            data, _ = raw[edf_ch]
            data = data.squeeze()[:n_samples]
            # Convert V to uV
            data = data * 1e6 if np.max(np.abs(data)) < 1e-2 else data
        else:
            # Fallback noise
            data = np.random.normal(0, 10.0, n_samples)
            
        ax.plot(t, data, color=colors[idx % len(colors)], linewidth=0.7, alpha=0.9)
        ax.set_ylabel(ch, color='#f0f6fc', fontsize=10, rotation=0, labelpad=15)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#30363d')
        ax.spines['bottom'].set_color('#30363d')
        
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.grid(True, linestyle=':', alpha=0.2, color='#30363d')
        
    axes[-1].set_xlabel('Time (seconds)', color='#8b949e', fontsize=9)
    plt.suptitle('Raw EEG Signals (Key Channels)', color='#f0f6fc', fontsize=11)
    plt.tight_layout()
    return fig

# Helper to plot probability distribution
def plot_probability_distribution(probabilities):
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='#161b22')
    ax.set_facecolor('#161b22')
    
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Shorten names for the x-axis
    clean_classes = [c.replace(" Disease", "").replace(" Subject", "") for c in classes]
    
    # Highlight the max probability bar in electric blue, others in charcoal gray
    max_idx = np.argmax(probs)
    colors = ['#58a6ff' if i == max_idx else '#21262d' for i in range(len(probs))]
    
    bars = ax.bar(clean_classes, probs, color=colors, width=0.5, edgecolor='#30363d', linewidth=0.8)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.set_ylabel('Probability', color='#8b949e', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', linestyle=':', alpha=0.15, color='#30363d')
    
    # Add percentage label on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=8, fontweight='bold',
            color='#f0f6fc'
        )
        
    plt.tight_layout()
    return fig

# Load Model Pipeline once
@st.cache_resource
def get_cached_pipeline():
    try:
        return load_prediction_pipeline('multimodal_checkpoint.pth')
    except Exception as e:
        st.error(f"Failed to load pipeline checkpoint: {e}")
        return None

# App Layout
st.title("🧠 AURA-Net: Multimodal Neurological Diagnostics")
st.markdown("### Attention-Based Speech & EEG Deep Learning Classifier")
st.write("---")

pipeline = get_cached_pipeline()

# Checkpoint verification warning
if pipeline is None:
    st.warning("⚠️ No trained checkpoint found. Please click the button below to train the model first on mock data.")
    if st.button("Train Model Now"):
        with st.spinner("Generating mock dataset and training model..."):
            import subprocess
            subprocess.run(["python", "train.py"])
            st.cache_resource.clear()
            st.rerun()
    st.stop()

# Sidebar controls
st.sidebar.markdown("## 📊 Configuration Panel")
st.sidebar.write("Choose input method:")

input_mode = st.sidebar.radio("Input Source", ["Clinical Demo Samples (Recommended)", "Upload Patient Files"])

# Pre-packaged demo selection
demo_class = None
if input_mode == "Clinical Demo Samples (Recommended)":
    demo_selection = st.sidebar.selectbox(
        "Select Demo Patient Profile",
        ["Healthy Subject", "Parkinson's Disease Patient", "Alzheimer's Disease Patient", "Epileptic Patient"]
    )
    # Map selection to sample filenames
    demo_map = {
        "Healthy Subject": "sample_healthy",
        "Parkinson's Disease Patient": "sample_parkinsons_disease",
        "Alzheimer's Disease Patient": "sample_alzheimers_disease",
        "Epileptic Patient": "sample_epilepsy"
    }
    demo_class = demo_map[demo_selection]
    
# Architecture Info
st.sidebar.write("---")
st.sidebar.markdown("### 🧬 Architecture Overview")
st.sidebar.markdown("""
- **Speech Branch**: 1D Dense Encoder (66 inputs)
- **EEG Branch**: 1D Dense Encoder (494 inputs)
- **Fusion**: Bilateral Multi-Head Cross-Attention (MHA)
- **Classification**: Deep Fused MLP (4 Classes)
""")

# Model Performance Tab
st.sidebar.write("---")
show_metrics = st.sidebar.checkbox("Show Model Training Metrics")

# MAIN APP LOGIC
wav_path = None
edf_path = None

if input_mode == "Upload Patient Files":
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_wav = st.file_uploader("Upload Speech File (.wav)", type=["wav"])
        if uploaded_wav:
            # Save uploaded file temporarily
            wav_path = os.path.join("temp_upload.wav")
            with open(wav_path, "wb") as f:
                f.write(uploaded_wav.read())
    with col_up2:
        uploaded_edf = st.file_uploader("Upload EEG File (.edf)", type=["edf"])
        if uploaded_edf:
            # Save uploaded file temporarily
            edf_path = os.path.join("temp_upload.edf")
            with open(edf_path, "wb") as f:
                f.write(uploaded_edf.read())
else:
    # Use demo files
    wav_path = f"samples/{demo_class}.wav"
    edf_path = f"samples/{demo_class}.edf"
    
    st.info(f"Loaded Clinical Demo Files for **{demo_selection}**:")
    col_d1, col_d2 = st.columns(2)
    col_d1.code(f"Speech Audio: {wav_path}")
    col_d2.code(f"EEG Signal:  {edf_path}")

st.write("---")

if show_metrics:
    st.subheader("📈 Model Training Performance & Diagnostics")
    col_m1, col_m2 = st.columns([1, 1])
    
    with col_m1:
        st.markdown("#### Confusion Matrix")
        if os.path.exists('confusion_matrix.png'):
            st.image('confusion_matrix.png', use_container_width=True)
        else:
            st.write("Confusion matrix plot not found. Run training again to generate.")
            
    with col_m2:
        st.markdown("#### Training History")
        # Load training metrics from checkpoint
        checkpoint = torch.load('multimodal_checkpoint.pth', map_location='cpu', weights_only=False)
        history = checkpoint['history']
        
        hist_df = pd.DataFrame({
            'Train Loss': history['train_loss'],
            'Val Loss': history['val_loss'],
            'Val Accuracy': history['val_acc']
        })
        
        st.line_chart(hist_df[['Train Loss', 'Val Loss']])
        st.line_chart(hist_df['Val Accuracy'])
        
    st.write("---")

# Prediction and Diagnostic Run
if wav_path and edf_path:
    # Diagnostic Button
    if st.button("🔴 RUN MULTIMODAL DIAGNOSTIC SCAN", use_container_width=True):
        with st.spinner("Extracting speech biomarkers and EEG spectrum..."):
            # Predict
            results = predict_disorder(wav_path, edf_path, pipeline)
            
        with st.spinner("Calculating SHAP feature importances..."):
            # SHAP Explanations
            importances, shap_plot_path = explain_prediction(results, pipeline)
            
        # Display Results
        st.subheader("🩺 Diagnostic Assessment")
        
        pred_class = results['predicted_class']
        confidence = results['confidence']
        
        # Color coding class
        card_class = "metric-card"
        if pred_class == "Healthy":
            card_class += " status-healthy"
            badge_color = "green"
        elif pred_class == "Parkinson's Disease":
            card_class += " status-parkinsons"
            badge_color = "orange"
        elif pred_class == "Alzheimer's Disease":
            card_class += " status-alzheimers"
            badge_color = "red"
        else:
            card_class += " status-epilepsy"
            badge_color = "red"
            
        # UI Presentation of Prediction
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin-top:0;">Prediction Result</h3>
            <p style="font-size: 24px; font-weight: bold; margin-bottom:5px;">
                {pred_class} <span style="font-size:16px; color:#8b949e; font-weight:normal;">(Confidence: {confidence*100:.2f}%)</span>
            </p>
            <div style="background-color:#30363d; border-radius:5px; height:10px; width:100%;">
                <div style="background-color: #58a6ff; width: {confidence*100}%; height: 10px; border-radius:5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown("#### Probability Distribution")
            fig_prob = plot_probability_distribution(results['probabilities'])
            st.pyplot(fig_prob)
            
            # Display Speech Waveform
            st.markdown("#### Audio Waveform Visualizer")
            fig_speech = plot_speech_waveform(wav_path)
            st.pyplot(fig_speech)
            
        with col_res2:
            st.markdown("#### EEG Multi-Channel Activity")
            fig_eeg = plot_eeg_signals(edf_path)
            st.pyplot(fig_eeg)
            
        st.write("---")
        
        # Explainability Block
        st.subheader("🔍 Explainable AI (SHAP Clinical Insights)")
        
        col_exp1, col_exp2 = st.columns([6, 4])
        
        with col_exp1:
            if os.path.exists(shap_plot_path):
                st.image(shap_plot_path, caption="SHAP Explanations (Red pushes prediction towards disease class, Blue reduces it)", use_container_width=True)
            else:
                st.error("SHAP explanation plot failed to generate.")
                
        with col_exp2:
            st.markdown("#### Clinical Interpretation")
            
            # Dynamic diagnostic text generation based on top features
            top_features_list = importances[:5]
            
            st.write(f"The model's classification of **{pred_class}** is primarily influenced by the following physiological and acoustic features:")
            
            for rank, (feat_name, val) in enumerate(top_features_list, 1):
                sign = "increases" if val >= 0 else "decreases"
                impact_color = "red" if val >= 0 else "blue"
                
                # Format feature name for explanation
                clean_name = feat_name.replace('connectivity_', 'EEG Connectivity: ').replace('_', ' ').title()
                
                st.markdown(f"**{rank}. {clean_name}**")
                st.markdown(f"- Direction: *{sign.capitalize()}* likelihood (SHAP = `{val:+.4f}`)")
                
                # Clinical description lookup
                desc = "Indicates significant alterations in acoustic pattern signatures."
                if 'mfcc' in feat_name:
                    desc = "Reflects changes in vocal tract shape, indicative of speech rigidity or motor coordination symptoms (dysarthria)."
                elif 'jitter' in feat_name or 'shimmer' in feat_name:
                    desc = "Reflects micro-instability and tremor in the laryngeal muscles, highly typical in motor neurological conditions."
                elif 'hnr' in feat_name:
                    desc = "Harmonic-to-Noise Ratio. Lower values indicate vocal breathiness, hoarseness, or air leak in vocal folds."
                elif 'theta' in feat_name:
                    desc = "Reflects increased slow-wave theta oscillations (4-8 Hz), which is a characteristic biomarker of cortical slowing (Alzheimer's)."
                elif 'beta' in feat_name:
                    desc = "Beta band oscillations are closely linked to motor cortex inhibition (Parkinson's)."
                elif 'hjorth' in feat_name:
                    desc = "Hjorth complexity/mobility. Reflects signal irregularity and structural alterations in brain electrical rhythms."
                elif 'connectivity' in feat_name:
                    desc = "Measures functional connectivity between brain regions. Abnormal synchrony is associated with network disintegration."
                    
                st.caption(desc)
                st.write("")
                
            st.markdown("""
            > [!NOTE]
            > **AURA-Net Warning**: These results are for research purposes only and must be verified by a board-certified neurologist.
            """)
            
        # Clean up temporary uploads if they exist
        if input_mode == "Upload Patient Files":
            if os.path.exists("temp_upload.wav"):
                os.remove("temp_upload.wav")
            if os.path.exists("temp_upload.edf"):
                os.remove("temp_upload.edf")
else:
    st.info("💡 Please upload patient WAV and EDF files or use the sidebar clinical demo selections, then click 'Run Multimodal Diagnostic Scan'.")
