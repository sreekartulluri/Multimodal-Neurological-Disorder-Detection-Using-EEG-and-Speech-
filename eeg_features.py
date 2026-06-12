"""
EEG processing module for extracting physiological biomarkers from EDF signal files.
Extracts 494 features including channel-specific band powers, relative powers, band ratios,
Shannon entropy, Hjorth parameters, and functional connectivity (correlation matrix).
"""

import numpy as np
import mne
import scipy.signal
import scipy.stats
import warnings

# Suppress MNE info logging to keep console clean unless there is an error
mne.set_log_level('WARNING')

# Standard 19-channel 10-20 EEG system channels
STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]

def find_channel_mapping(edf_ch_names):
    """
    Maps EDF channel names to standard 10-20 channel names.
    
    Parameters:
        edf_ch_names (list): List of channel names in the EDF file.
        
    Returns:
        dict: Mapping from standard channel name to EDF channel name.
    """
    mapping = {}
    for std in STANDARD_CHANNELS:
        # Standardize std name
        std_clean = std.lower().replace(' ', '').replace('eeg', '')
        
        # Try to find a match in the EDF channels
        for edf_ch in edf_ch_names:
            edf_clean = edf_ch.lower().replace(' ', '').replace('eeg', '').replace('-ref', '').replace('-le', '').replace('-gnd', '')
            # Handle T3/T4 vs T7/T8 or T5/T6 vs P7/P8 (standard modern naming)
            if std_clean == 't3' and edf_clean == 't7':
                mapping[std] = edf_ch
                break
            elif std_clean == 't4' and edf_clean == 't8':
                mapping[std] = edf_ch
                break
            elif std_clean == 't5' and edf_clean == 'p7':
                mapping[std] = edf_ch
                break
            elif std_clean == 't6' and edf_clean == 'p8':
                mapping[std] = edf_ch
                break
            elif std_clean == edf_clean:
                mapping[std] = edf_ch
                break
            # Substring match (e.g. "fp1-a1" contains "fp1")
            elif std_clean in edf_clean and len(std_clean) >= 2:
                # Avoid matching 'F3' with 'F34' or something similar
                # Simple check: make sure numeric suffix matches if present
                import re
                std_num = re.findall(r'\d+', std_clean)
                edf_num = re.findall(r'\d+', edf_clean)
                if std_num == edf_num:
                    mapping[std] = edf_ch
                    break
        
        # If no match is found, standard channel remains unmapped (will be padded with zeros)
    return mapping

def extract_eeg_features(edf_path):
    """
    Extracts a fixed 494-dimensional feature vector from an EDF file.
    
    Parameters:
        edf_path (str): Path to the EDF signal file.
        
    Returns:
        dict: A dictionary of feature names and their corresponding scalar values.
        np.ndarray: A flat float32 array of shape (494,) representing the feature vector.
    """
    try:
        # Load EDF file using MNE
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        raise ValueError(f"Error reading EDF file {edf_path}: {e}")

    # Standardize sampling rate to 250 Hz (common EEG sampling rate)
    target_fs = 250
    if raw.info['sfreq'] != target_fs:
        try:
            raw.resample(target_fs, verbose=False)
        except Exception:
            pass  # keep original sfreq if resample fails
    
    fs = int(raw.info['sfreq'])

    # Apply bandpass filter (0.5 to 45 Hz)
    try:
        raw.filter(0.5, 45.0, fir_design='firwin', verbose=False)
    except Exception as e:
        # If MNE filtering fails, we will filter manually per channel using scipy
        pass

    # Map EDF channels to standard 10-20 channels
    ch_mapping = find_channel_mapping(raw.ch_names)
    
    # Extract data for the standard channels (pad with zeros if channel is missing)
    duration_sec = 10  # process a fixed 10 seconds of data to ensure consistency
    n_samples = int(duration_sec * fs)
    
    # Get raw data matrix
    eeg_data = np.zeros((len(STANDARD_CHANNELS), n_samples), dtype=np.float32)
    
    for i, std_ch in enumerate(STANDARD_CHANNELS):
        if std_ch in ch_mapping:
            edf_ch = ch_mapping[std_ch]
            # Extract data
            ch_data, _ = raw[edf_ch]
            ch_data = ch_data.squeeze()
            
            # Apply manual bandpass filter if MNE filtering didn't run or to be safe
            try:
                nyq = 0.5 * fs
                b, a = scipy.signal.butter(4, [0.5 / nyq, 45.0 / nyq], btype='band')
                ch_data = scipy.signal.filtfilt(b, a, ch_data)
            except Exception:
                pass
                
            # Truncate or zero-pad to n_samples
            if len(ch_data) >= n_samples:
                eeg_data[i, :] = ch_data[:n_samples]
            else:
                eeg_data[i, :len(ch_data)] = ch_data
        else:
            # Missing channel: pad with low-level random noise to keep signal math valid
            eeg_data[i, :] = np.random.normal(0, 1e-6, n_samples)

    features = {}
    
    # Frequency Bands definitions (Hz)
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta': (12.0, 30.0),
        'gamma': (30.0, 45.0)
    }

    # 1. Compute PSD per channel and extract spectral features
    for i, std_ch in enumerate(STANDARD_CHANNELS):
        channel_signal = eeg_data[i, :]
        
        # Compute PSD using Welch's method
        nperseg = min(len(channel_signal), 2 * fs) # 2-second windows
        freqs, psd = scipy.signal.welch(channel_signal, fs=fs, nperseg=nperseg)
        
        # Extract band powers
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            # Find frequency indices
            idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            if len(idx) > 0:
                # Power is the integral of PSD (trapezoidal integration)
                power = np.trapz(psd[idx], freqs[idx])
            else:
                power = 0.0
            band_powers[band_name] = max(power, 1e-12) # avoid zero power
            features[f"{std_ch}_{band_name}_power"] = float(band_powers[band_name])
            
        # Total Power (0.5 to 45 Hz)
        total_idx = np.where((freqs >= 0.5) & (freqs <= 45.0))[0]
        total_power = np.trapz(psd[total_idx], freqs[total_idx]) if len(total_idx) > 0 else 1.0
        total_power = max(total_power, 1e-10)
        
        # Relative Band Powers
        for band_name in bands.keys():
            rel_power = band_powers[band_name] / total_power
            features[f"{std_ch}_{band_name}_relative"] = float(rel_power)
            
        # Band Ratios
        features[f"{std_ch}_theta_alpha_ratio"] = float(band_powers['theta'] / band_powers['alpha'])
        features[f"{std_ch}_alpha_beta_ratio"] = float(band_powers['alpha'] / band_powers['beta'])
        features[f"{std_ch}_theta_beta_ratio"] = float(band_powers['theta'] / band_powers['beta'])
        
        # Shannon Entropy on normalized PSD (probability distribution)
        psd_05_45 = psd[total_idx]
        if np.sum(psd_05_45) > 0:
            psd_norm = psd_05_45 / np.sum(psd_05_45)
            # Shannon entropy: -sum(p * log2(p))
            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        else:
            entropy = 0.0
        features[f"{std_ch}_shannon_entropy"] = float(entropy)
        
        # Hjorth Parameters (Activity, Mobility, Complexity)
        # First-order derivative
        d1 = np.diff(channel_signal) * fs
        # Second-order derivative
        d2 = np.diff(d1) * fs
        
        var_x = np.var(channel_signal)
        var_d1 = np.var(d1)
        var_d2 = np.var(d2)
        
        hj_activity = var_x
        
        if var_x > 1e-12:
            hj_mobility = np.sqrt(var_d1 / var_x)
        else:
            hj_mobility = 0.0
            
        if var_d1 > 1e-12:
            hj_complexity = np.sqrt(var_d2 / var_d1) / (hj_mobility + 1e-12)
        else:
            hj_complexity = 0.0
            
        features[f"{std_ch}_hjorth_activity"] = float(hj_activity)
        features[f"{std_ch}_hjorth_mobility"] = float(hj_mobility)
        features[f"{std_ch}_hjorth_complexity"] = float(hj_complexity)

    # 2. Compute EEG Connectivity using Pearson correlation matrix (171 features)
    corr_matrix = np.corrcoef(eeg_data)
    # Fill NaNs with 0 in case of zero-signal channels
    corr_matrix = np.nan_to_num(corr_matrix)
    
    # Extract the upper triangle of the correlation matrix (excluding diagonal)
    triu_indices = np.triu_indices(len(STANDARD_CHANNELS), k=1)
    for index, (row, col) in enumerate(zip(triu_indices[0], triu_indices[1])):
        feat_name = f"connectivity_{STANDARD_CHANNELS[row]}_{STANDARD_CHANNELS[col]}"
        features[feat_name] = float(corr_matrix[row, col])

    # Convert features dict to a sorted float32 array
    feature_keys = sorted(list(features.keys()))
    feature_vector = np.array([features[k] for k in feature_keys], dtype=np.float32)

    return features, feature_vector

if __name__ == "__main__":
    # Test feature extraction on synthetic data
    print("Testing EEG feature extraction...")
    synthetic_edf = "test_eeg.edf"
    
    # MNE does not directly support writing raw to EDF out-of-the-box easily without export,
    # let's test by creating a mock RawArray and checking if we can save/load it.
    import os
    info = mne.create_info(ch_names=STANDARD_CHANNELS, sfreq=250, ch_types='eeg')
    data = np.random.normal(0, 1e-5, (19, 2500)) # 10 seconds of noise
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # To export to EDF, we can use raw.export(..., fmt='edf') or similar, but let's test if raw.export is supported
    # In MNE, raw.export requires edfio or pyedfio. Let's try raw.export or fallback to saving as FIFA (.fif)
    try:
        raw.export(synthetic_edf, fmt='edf', overwrite=True, verbose=False)
        print("MNE exported EDF successfully!")
        
        # Test loading it
        feat_dict, feat_vector = extract_eeg_features(synthetic_edf)
        print(f"Extracted feature vector shape: {feat_vector.shape}")
        print(f"Number of keys extracted: {len(feat_dict)}")
        print(f"First 5 keys: {list(feat_dict.keys())[:5]}")
        
        # Cleanup
        if os.path.exists(synthetic_edf):
            os.remove(synthetic_edf)
    except Exception as e:
        print(f"EDF export failed (expected if external writer edfio is not installed): {e}")
        # Note: In our dataset/app module, we will implement a direct binary EDF writer to avoid external writer dependency!
