"""
Helper script to generate mock WAV and EDF files and a pre-extracted training dataset.
Implements a custom binary EDF writer to avoid external library dependencies.
"""

import os
import numpy as np
import soundfile as sf
import torch
import pandas as pd
from speech_features import extract_speech_features
from eeg_features import extract_eeg_features, STANDARD_CHANNELS

def write_custom_edf(filename, data, ch_names, fs=250):
    """
    Writes a 2D numpy array of EEG signals to a standard-compliant EDF file.
    
    Parameters:
        filename (str): Output EDF path.
        data (np.ndarray): Shape (channels, samples).
        ch_names (list): List of channel names.
        fs (int): Sampling frequency.
    """
    n_channels = len(ch_names)
    n_samples = data.shape[1]
    
    # We will write 1-second records
    record_duration = 1
    samples_per_record = fs
    n_records = n_samples // fs
    
    # Format header fields
    def fmt(val, length):
        return f"{val:<{length}}"[:length].encode('ascii')
    
    header = []
    header.append(fmt("0", 8)) # Version
    header.append(fmt("Mock Patient", 80)) # Patient ID
    header.append(fmt("Mock Recording", 80)) # Recording ID
    header.append(fmt("01.01.26", 8)) # Start date
    header.append(fmt("12.00.00", 8)) # Start time
    header_bytes = 256 + 256 * n_channels
    header.append(fmt(str(header_bytes), 8)) # Header size
    header.append(fmt("EDF+C", 44)) # Reserved / format
    header.append(fmt(str(n_records), 8)) # Num records
    header.append(fmt(str(record_duration), 8)) # Record duration
    header.append(fmt(str(n_channels), 4)) # Num signals
    
    # Signal headers
    for name in ch_names:
        header.append(fmt(f"EEG {name}", 16)) # Label
    for _ in ch_names:
        header.append(fmt("Ag-AgCl electrode", 80)) # Transducer type
    for _ in ch_names:
        header.append(fmt("uV", 8)) # Physical dimension
    for _ in ch_names:
        header.append(fmt("-500", 8)) # Physical min
    for _ in ch_names:
        header.append(fmt("500", 8)) # Physical max
    for _ in ch_names:
        header.append(fmt("-32768", 8)) # Digital min
    for _ in ch_names:
        header.append(fmt("32767", 8)) # Digital max
    for _ in ch_names:
        header.append(fmt("HP:0.5Hz LP:45Hz", 80)) # Prefiltering
    for _ in ch_names:
        header.append(fmt(str(samples_per_record), 8)) # Samples per record
    for _ in ch_names:
        header.append(fmt("Reserved", 32))
        
    header_data = b"".join(header)
    
    # Scale physical data to digital range [-32768, 32767]
    # Physical range is [-500, 500] uV
    scaled_data = np.clip(data, -500.0, 500.0)
    scaled_data = (scaled_data / 500.0) * 32767.0
    digital_data = np.round(scaled_data).astype(np.int16)
    
    # Write to file
    with open(filename, 'wb') as f:
        f.write(header_data)
        # Multiplex data: for each record, write samples_per_record for each channel
        for r in range(n_records):
            start_idx = r * samples_per_record
            end_idx = start_idx + samples_per_record
            for c in range(n_channels):
                chunk = digital_data[c, start_idx:end_idx]
                f.write(chunk.tobytes())

def generate_mock_speech(filename, label, sr=16000, duration=5.0):
    """Generates a mock speech WAV file with acoustic signatures of a given disease."""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    
    # Base pitch frequency (fundamental frequency)
    if label == "Parkinson's Disease":
        # Parkinson's: Monotone voice, lower HNR, higher jitter/shimmer
        f0 = 130.0 + 2.0 * np.sin(2 * np.pi * 1.5 * t)  # rigid pitch
        speech = 0.5 * np.sin(2 * np.pi * f0 * t)
        # Add tremor amplitude modulation (shimmer)
        tremor = 1.0 + 0.15 * np.sin(2 * np.pi * 6.0 * t)
        speech = speech * tremor
        # Lower HNR (more noise)
        speech = speech + 0.25 * np.random.randn(len(t))
    elif label == "Alzheimer's Disease":
        # Alzheimer's: Slur/pauses, breathy voice, lower volume
        f0 = 180.0 + 15.0 * np.sin(2 * np.pi * 0.5 * t)
        speech = 0.3 * np.sin(2 * np.pi * f0 * t)
        # Silences/pauses
        pause_mask = (np.sin(2 * np.pi * 0.3 * t) > -0.3).astype(float)
        speech = speech * pause_mask
        speech = speech + 0.15 * np.random.randn(len(t))
    elif label == "Epilepsy":
        # Epilepsy: Standard speech, slight deviations
        f0 = 200.0 + 10.0 * np.sin(2 * np.pi * 1.0 * t)
        speech = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.08 * np.random.randn(len(t))
    else:  # Healthy
        # Healthy: Rich modulation, high HNR, low jitter
        f0 = 210.0 + 30.0 * np.sin(2 * np.pi * 2.0 * t)
        speech = 0.6 * np.sin(2 * np.pi * f0 * t)
        speech = speech + 0.02 * np.random.randn(len(t))
        
    # Normalize
    if np.max(np.abs(speech)) > 0:
        speech = speech / np.max(np.abs(speech)) * 0.9
        
    sf.write(filename, speech, sr)

def generate_mock_eeg(filename, label, fs=250, duration=10.0):
    """Generates a mock EEG EDF file with spectral signatures of a given disease."""
    n_channels = len(STANDARD_CHANNELS)
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    eeg_data = np.zeros((n_channels, n_samples))
    
    for c in range(n_channels):
        # Base background noise
        noise = np.random.normal(0, 10.0, n_samples) # 10 uV noise
        
        # Add spectral band oscillations based on class
        if label == "Alzheimer's Disease":
            # Alzheimer's: Pronounced Theta slowing (4-8 Hz) and Delta (0.5-4 Hz)
            theta = 35.0 * np.sin(2 * np.pi * 6.0 * t + np.random.rand() * 2 * np.pi)
            delta = 25.0 * np.sin(2 * np.pi * 2.0 * t + np.random.rand() * 2 * np.pi)
            alpha = 5.0 * np.sin(2 * np.pi * 10.0 * t + np.random.rand() * 2 * np.pi) # reduced alpha
            beta = 3.0 * np.sin(2 * np.pi * 20.0 * t + np.random.rand() * 2 * np.pi)
            signal = noise + theta + delta + alpha + beta
        elif label == "Parkinson's Disease":
            # Parkinson's: Increased Beta (12-30 Hz) and Theta
            beta = 25.0 * np.sin(2 * np.pi * 18.0 * t + np.random.rand() * 2 * np.pi)
            theta = 15.0 * np.sin(2 * np.pi * 5.0 * t + np.random.rand() * 2 * np.pi)
            alpha = 10.0 * np.sin(2 * np.pi * 9.5 * t + np.random.rand() * 2 * np.pi)
            signal = noise + beta + theta + alpha
        elif label == "Epilepsy":
            # Epilepsy: Intermittent spike-wave discharges (3 Hz spike-and-wave)
            # We add large transient spikes
            spikes = np.zeros(n_samples)
            spike_locs = np.arange(fs, n_samples - fs, int(1.5 * fs)) # every 1.5s
            for loc in spike_locs:
                # Add a sharp biphasic spike
                spikes[loc-5:loc] = np.linspace(0, 150.0, 5)
                spikes[loc:loc+10] = np.linspace(150.0, -100.0, 10)
                spikes[loc+10:loc+25] = np.linspace(-100.0, 0.0, 15)
            # Also standard alpha
            alpha = 15.0 * np.sin(2 * np.pi * 10.0 * t)
            signal = noise + spikes + alpha
        else:  # Healthy
            # Healthy: Prominent Alpha rhythm (8-12 Hz) in posterior channels, low delta/theta
            if STANDARD_CHANNELS[c] in ['O1', 'O2', 'P3', 'P4', 'Pz']:
                alpha = 40.0 * np.sin(2 * np.pi * 10.0 * t + np.random.rand() * 2 * np.pi)
            else:
                alpha = 15.0 * np.sin(2 * np.pi * 10.0 * t + np.random.rand() * 2 * np.pi)
            beta = 10.0 * np.sin(2 * np.pi * 20.0 * t + np.random.rand() * 2 * np.pi)
            theta = 5.0 * np.sin(2 * np.pi * 6.0 * t + np.random.rand() * 2 * np.pi)
            signal = noise + alpha + beta + theta
            
        eeg_data[c, :] = signal
        
    write_custom_edf(filename, eeg_data, STANDARD_CHANNELS, fs)

def generate_and_save_dataset():
    """Generates 200 samples (50 per class) of feature data and saves it for training."""
    print("Generating synthetic feature dataset (200 samples)...")
    
    classes = ["Healthy", "Parkinson's Disease", "Alzheimer's Disease", "Epilepsy"]
    class_map = {name: idx for idx, name in enumerate(classes)}
    
    samples_per_class = 50
    
    speech_data = []
    eeg_data = []
    labels = []
    
    # We will generate mock files, extract features, and then delete them to build the dataset.
    temp_wav = "temp_speech.wav"
    temp_edf = "temp_eeg.edf"
    
    for label_idx, cls_name in enumerate(classes):
        print(f"  Processing class: {cls_name}")
        for s in range(samples_per_class):
            # Generate mock files
            generate_mock_speech(temp_wav, cls_name)
            generate_mock_eeg(temp_edf, cls_name)
            
            # Extract features
            _, s_vector = extract_speech_features(temp_wav)
            _, e_vector = extract_eeg_features(temp_edf)
            
            speech_data.append(s_vector)
            eeg_data.append(e_vector)
            labels.append(label_idx)
            
    # Clean up temp files
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    if os.path.exists(temp_edf):
        os.remove(temp_edf)
        
    speech_tensor = torch.tensor(np.array(speech_data), dtype=torch.float32)
    eeg_tensor = torch.tensor(np.array(eeg_data), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Save as dictionary
    dataset_dict = {
        'speech_features': speech_tensor,
        'eeg_features': eeg_tensor,
        'labels': labels_tensor,
        'classes': classes
    }
    
    torch.save(dataset_dict, 'features_dataset.pt')
    print("Dataset successfully saved as 'features_dataset.pt'!")

def generate_sample_demo_files():
    """Generates persistent sample files for Streamlit demonstrations."""
    os.makedirs('samples', exist_ok=True)
    classes = ["Healthy", "Parkinson's Disease", "Alzheimer's Disease", "Epilepsy"]
    
    print("Generating demo files in 'samples/' directory...")
    for cls in classes:
        name_clean = cls.lower().replace("'", "").replace(" ", "_")
        wav_path = f"samples/sample_{name_clean}.wav"
        edf_path = f"samples/sample_{name_clean}.edf"
        
        generate_mock_speech(wav_path, cls)
        generate_mock_eeg(edf_path, cls)
        print(f"  Created: {wav_path} & {edf_path}")
    print("Demo files successfully generated!")

if __name__ == "__main__":
    # Generate the dataset
    generate_and_save_dataset()
    # Generate sample files
    generate_sample_demo_files()
