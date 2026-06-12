"""
Explainability module using SHAP.
Explains model predictions by calculating SHAP values for Speech and EEG features
and visualizing the top contributing biomarkers.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings

# Suppress SHAP warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class MultimodalModelWrapper:
    """
    Wraps the Multimodal PyTorch model to accept a single concatenated 
    feature vector (Speech + EEG) of shape (B, 560).
    This wrapper outputs probabilities, matching the interface needed for SHAP KernelExplainer.
    """
    def __init__(self, model, speech_dim, eeg_dim, device):
        self.model = model
        self.speech_dim = speech_dim
        self.eeg_dim = eeg_dim
        self.device = device
        
    def predict_probs(self, concatenated_features):
        """
        Parameters:
            concatenated_features (np.ndarray): Shape (B, 560)
            
        Returns:
            np.ndarray: Probabilities of shape (B, 4)
        """
        # Ensure it is a 2D numpy array
        concatenated_features = np.atleast_2d(concatenated_features)
        
        # Split into speech and eeg vectors
        speech_feats = concatenated_features[:, :self.speech_dim]
        eeg_feats = concatenated_features[:, self.speech_dim:]
        
        # Convert to PyTorch tensors
        speech_tensor = torch.tensor(speech_feats, dtype=torch.float32).to(self.device)
        eeg_tensor = torch.tensor(eeg_feats, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(speech_tensor, eeg_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
        return probs

def get_feature_names(speech_dict, eeg_dict):
    """Combines speech and EEG feature keys in sorted order."""
    speech_keys = sorted(list(speech_dict.keys()))
    eeg_keys = sorted(list(eeg_dict.keys()))
    return speech_keys + eeg_keys

def explain_prediction(results, pipeline, dataset_path='features_dataset.pt', num_background_samples=20):
    """
    Calculates SHAP values for a single prediction and generates an explanation plot.
    
    Parameters:
        results (dict): Output dict from predict_disorder().
        pipeline (dict): Pre-loaded model pipeline from load_prediction_pipeline().
        dataset_path (str): Path to features dataset for background distribution.
        num_background_samples (int): Number of background samples to use for explainer.
        
    Returns:
        list: Sorted list of tuples (feature_name, shap_value) representing feature importances.
        str: Path to the generated SHAP explanation plot.
    """
    model = pipeline['model']
    speech_scaler = pipeline['speech_scaler']
    eeg_scaler = pipeline['eeg_scaler']
    classes = pipeline['classes']
    device = pipeline['device']
    
    # 1. Prepare Background Data for KernelExplainer
    # We load training features to establish a reference baseline
    data_dict = torch.load(dataset_path, weights_only=False)
    speech_train = data_dict['speech_features'].numpy()
    eeg_train = data_dict['eeg_features'].numpy()
    
    # Scale background data
    speech_train_scaled = speech_scaler.transform(speech_train)
    eeg_train_scaled = eeg_scaler.transform(eeg_train)
    
    # Concatenate scaled speech and eeg features
    bg_data = np.concatenate([speech_train_scaled, eeg_train_scaled], axis=1)
    
    # Subsample background data to speed up KernelExplainer calculations
    np.random.seed(42)
    bg_indices = np.random.choice(len(bg_data), min(num_background_samples, len(bg_data)), replace=False)
    bg_data_subset = bg_data[bg_indices]
    
    # 2. Prepare Current Sample
    sample_speech_scaled = results['speech_vector_scaled'] # (1, 66)
    sample_eeg_scaled = results['eeg_vector_scaled']     # (1, 494)
    sample_concatenated = np.concatenate([sample_speech_scaled, sample_eeg_scaled], axis=1) # (1, 560)
    
    # 3. Setup Model Wrapper and Explainer
    wrapper = MultimodalModelWrapper(
        model=model, 
        speech_dim=sample_speech_scaled.shape[1], 
        eeg_dim=sample_eeg_scaled.shape[1], 
        device=device
    )
    
    # Instantiate KernelExplainer
    explainer = shap.KernelExplainer(wrapper.predict_probs, bg_data_subset)
    
    # Calculate SHAP values for the sample
    shap_values = explainer.shap_values(sample_concatenated)
    
    # 4. Map SHAP values to Feature Names for the predicted class
    pred_class = results['predicted_class']
    pred_idx = classes.index(pred_class)
    
    # Extract values if it is an Explanation object
    if hasattr(shap_values, 'values'):
        shap_values_arr = shap_values.values
    else:
        shap_values_arr = shap_values
        
    # Dynamically extract SHAP values for the target class based on shape/type
    if isinstance(shap_values_arr, list):
        # List of arrays, one per class: list length = classes, each array shape = (samples, features)
        class_shap = shap_values_arr[pred_idx][0]
    elif isinstance(shap_values_arr, np.ndarray):
        if shap_values_arr.ndim == 3:
            # Array shape: (samples, features, classes)
            class_shap = shap_values_arr[0, :, pred_idx]
        elif shap_values_arr.ndim == 2:
            # Array shape: (samples, features)
            # If classes dimension is squeezed or binary, extract the first sample
            class_shap = shap_values_arr[0]
        else:
            class_shap = shap_values_arr.flatten()
    else:
        try:
            class_shap = np.array(shap_values_arr)[0]
        except Exception:
            class_shap = np.zeros(len(feature_names))
            
    feature_names = get_feature_names(results['speech_features_dict'], results['eeg_features_dict'])
    
    # Zip names and SHAP values
    feature_importances = list(zip(feature_names, class_shap))
    # Sort by absolute SHAP value (magnitude of impact)
    feature_importances = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)
    
    # 5. Generate Explanation Plot
    top_n = 15
    top_features = feature_importances[:top_n]
    
    # Reverse to plot largest contribution at the top
    top_features.reverse()
    
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    
    # Styling features: shorten long connectivity names
    clean_names = []
    for n in names:
        if n.startswith('connectivity_'):
            clean_names.append(n.replace('connectivity_', 'EEG Corr: '))
        else:
            # Capitalize speech features and format
            clean_names.append(n.replace('_', ' ').title())
            
    # Draw plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars: Red for positive impact (increased likelihood), Blue for negative
    colors = ['#FF4D4D' if v >= 0 else '#4D94FF' for v in values]
    
    bars = ax.barh(clean_names, values, color=colors, height=0.6)
    
    # Grid and spines styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#e0e0e0')
    ax.set_axisbelow(True)
    
    # Titles and labels
    ax.set_title(f"SHAP Explanations for Predicting '{pred_class}'\n(Top {top_n} Contributing Features)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('SHAP Value (Impact on Model Confidence)', fontsize=11, labelpad=10)
    
    # Add vertical reference line at 0
    ax.axvline(x=0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.7)
    
    # Add labels on the bars
    for bar in bars:
        width = bar.get_width()
        label_x = width + (0.005 if width >= 0 else -0.005)
        align = 'left' if width >= 0 else 'right'
        ax.annotate(
            f"{width:+.4f}",
            xy=(label_x, bar.get_y() + bar.get_height() / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha=align, va='center',
            fontsize=9, fontweight='bold',
            color='#333333'
        )
        
    plt.tight_layout()
    output_plot_path = 'shap_explanation.png'
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    
    return feature_importances, output_plot_path

if __name__ == "__main__":
    from predict import load_prediction_pipeline, predict_disorder
    
    print("Testing SHAP explanation pipeline...")
    wav_file = "samples/sample_parkinsons_disease.wav"
    edf_file = "samples/sample_parkinsons_disease.edf"
    
    if os.path.exists(wav_file) and os.path.exists(edf_file):
        pipeline = load_prediction_pipeline()
        results = predict_disorder(wav_file, edf_file, pipeline)
        
        print("Calculating SHAP values (this may take a few seconds)...")
        import time
        t0 = time.time()
        importances, plot_path = explain_prediction(results, pipeline)
        print(f"SHAP explanation completed in {time.time() - t0:.2f} seconds.")
        print(f"Explanation plot saved at: {plot_path}")
        print("\nTop 5 contributing features:")
        for name, val in importances[:5]:
            print(f"  {name}: {val:+.6f}")
    else:
        print("Sample files not found. Run generate_mock_data.py first.")
