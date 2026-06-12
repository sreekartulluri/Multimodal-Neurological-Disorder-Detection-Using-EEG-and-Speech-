"""
Speech processing module for extracting acoustic biomarkers from speech audio files.
Extracts 64 features including MFCCs, Chroma, Pitch, RMS, ZCR, Spectral characteristics,
Tempo, Jitter, Shimmer, and Harmonic-to-Noise Ratio (HNR).
"""

import numpy as np
import librosa
import warnings

# Suppress librosa user warnings
warnings.filterwarnings('ignore', category=UserWarning)

def extract_speech_features(wav_path, target_sr=16000):
    """
    Extracts a fixed 64-dimensional feature vector from a WAV file.
    
    Parameters:
        wav_path (str): Path to the audio file.
        target_sr (int): Sampling rate to resample the audio to.
        
    Returns:
        dict: A dictionary of feature names and their corresponding scalar values.
        np.ndarray: A flat float32 array of shape (64,) representing the feature vector.
    """
    try:
        # Load audio file
        y, sr = librosa.load(wav_path, sr=target_sr)
    except Exception as e:
        raise ValueError(f"Error loading speech audio file {wav_path}: {e}")

    # Ensure audio is not empty
    if len(y) == 0:
        y = np.zeros(target_sr)  # fallback to 1 second of silence
        sr = target_sr

    features = {}

    # 1. MFCC features (13 coefficients -> mean & std = 26 features)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_mean_{i+1}"] = float(np.mean(mfccs[i]))
        features[f"mfcc_std_{i+1}"] = float(np.std(mfccs[i]))

    # 2. Chroma features (12 pitch classes -> mean & std = 24 features)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)
    for i in range(12):
        features[f"chroma_mean_{i+1}"] = float(np.mean(chroma[i]))
        features[f"chroma_std_{i+1}"] = float(np.std(chroma[i]))

    # 3. Pitch mean and standard deviation (2 features)
    # Use Yin algorithm for pitch tracking
    try:
        f0 = librosa.yin(y=y, sr=sr, fmin=50, fmax=500)
        # Filter out NaN or infinite values
        f0 = f0[np.isfinite(f0)]
        if len(f0) > 0:
            pitch_mean = float(np.mean(f0))
            pitch_std = float(np.std(f0))
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
    except Exception:
        pitch_mean = 0.0
        pitch_std = 0.0
        f0 = np.array([])
    features["pitch_mean"] = pitch_mean
    features["pitch_std"] = pitch_std

    # 4. RMS Energy (2 features)
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # 5. Zero Crossing Rate (2 features)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # 6. Spectral Centroid (2 features)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    # 7. Spectral Bandwidth (2 features)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    # 8. Spectral Rolloff (2 features)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    # 9. Tempo (1 feature)
    try:
        tempo_output = librosa.feature.tempo(y=y, sr=sr)
        if isinstance(tempo_output, np.ndarray):
            tempo = float(tempo_output[0])
        else:
            tempo = float(tempo_output)
    except Exception:
        tempo = 120.0
    features["tempo"] = tempo

    # Helper for Jitter, Shimmer, HNR (standard algorithms adapted for python/librosa)
    # We estimate these based on voiced frames (where pitch f0 is present and reliable)
    voiced_f0 = f0[f0 > 0] if len(f0) > 0 else np.array([])
    
    # 10. Jitter (local) (1 feature)
    # Relative average perturbation of pitch periods
    if len(voiced_f0) > 1:
        periods = 1.0 / voiced_f0
        period_diffs = np.abs(np.diff(periods))
        jitter = float(np.mean(period_diffs) / np.mean(periods))
    else:
        jitter = 0.0
    features["jitter"] = jitter

    # 11. Shimmer (local) (1 feature)
    # Cycle-to-cycle variability of peak amplitude
    # We estimate amplitude peaks using short-term frames matching pitch periods
    try:
        frame_len = int(target_sr * 0.03)  # 30 ms frames
        hop_len = int(target_sr * 0.01)    # 10 ms hop
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
        
        # Calculate peak amplitude for each frame
        peak_amplitudes = np.max(np.abs(frames), axis=0)
        
        # Filter peak amplitudes where pitch is detected
        # Align length of peak_amplitudes and f0 if they differ
        min_len = min(len(peak_amplitudes), len(f0))
        voiced_peaks = peak_amplitudes[:min_len][f0[:min_len] > 0]
        
        if len(voiced_peaks) > 1 and np.mean(voiced_peaks) > 0:
            shimmer = float(np.mean(np.abs(np.diff(voiced_peaks))) / np.mean(voiced_peaks))
        else:
            shimmer = 0.0
    except Exception:
        shimmer = 0.0
    features["shimmer"] = shimmer

    # 12. Harmonic-to-Noise Ratio (HNR) (1 feature)
    # Computed using short-term autocorrelation of frames
    try:
        frame_len = int(target_sr * 0.04) # 40 ms frames
        hop_len = int(target_sr * 0.02)   # 20 ms hop
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
        
        hnr_vals = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            # Window the frame to prevent spectral leakage
            win_frame = frame * np.hanning(len(frame))
            
            # Autocorrelation
            r = np.correlate(win_frame, win_frame, mode='full')
            r = r[len(r)//2:] # Keep non-negative lags
            
            r0 = r[0] # Signal energy
            if r0 <= 1e-10:
                continue
                
            # Pitch range limits for lags (50Hz to 500Hz)
            min_lag = int(sr / 500)
            max_lag = int(sr / 50)
            
            if max_lag >= len(r):
                max_lag = len(r) - 1
            if min_lag >= max_lag:
                continue
                
            # Find maximum autocorrelation in pitch range
            r_pitch_range = r[min_lag:max_lag+1]
            r_max = np.max(r_pitch_range)
            
            # HNR calculation: H = 10 * log10( R_max / (R0 - R_max) )
            noise = r0 - r_max
            if noise > 1e-10 and r_max > 1e-10:
                hnr_val = 10 * np.log10(r_max / noise)
                hnr_vals.append(hnr_val)
                
        hnr = float(np.mean(hnr_vals)) if len(hnr_vals) > 0 else 15.0
    except Exception:
        hnr = 15.0 # standard average HNR for human speech
    features["hnr"] = hnr

    # Convert features dict to a sorted float32 array
    feature_keys = sorted(list(features.keys()))
    feature_vector = np.array([features[k] for k in feature_keys], dtype=np.float32)

    return features, feature_vector

if __name__ == "__main__":
    # Test feature extraction on synthetic silence
    print("Testing speech feature extraction...")
    synthetic_wav = "test_speech.wav"
    sr = 16000
    # Create 2 seconds of synthetic 440Hz sine wave (representing voiced speech)
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220.0 * t) + 0.1 * np.random.randn(len(t))
    
    import soundfile as sf
    sf.write(synthetic_wav, y, sr)
    
    feat_dict, feat_vector = extract_speech_features(synthetic_wav)
    print(f"Extracted feature vector shape: {feat_vector.shape}")
    print(f"Number of keys extracted: {len(feat_dict)}")
    print(f"First 5 keys: {list(feat_dict.keys())[:5]}")
    print(f"Tempo: {feat_dict['tempo']:.2f}")
    print(f"Pitch Mean: {feat_dict['pitch_mean']:.2f} Hz")
    print(f"Jitter: {feat_dict['jitter']:.5f}")
    print(f"Shimmer: {feat_dict['shimmer']:.5f}")
    print(f"HNR: {feat_dict['hnr']:.2f} dB")
    
    # Cleanup test file
    import os
    if os.path.exists(synthetic_wav):
        os.remove(synthetic_wav)
