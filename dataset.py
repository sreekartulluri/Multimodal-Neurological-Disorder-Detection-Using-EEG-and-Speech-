"""
Dataset module for loading and preprocessing EEG and Speech features.
Defines the PyTorch Dataset class and helper function to create data loaders.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np

class MultimodalDataset(Dataset):
    """
    PyTorch Dataset for Multimodal Neurological Disorder Detection.
    Loads Speech features, EEG features, and target labels.
    """
    def __init__(self, speech_feats, eeg_feats, labels):
        """
        Parameters:
            speech_feats (torch.Tensor or np.ndarray): Speech feature matrix.
            eeg_feats (torch.Tensor or np.ndarray): EEG feature matrix.
            labels (torch.Tensor or np.ndarray): Labels.
        """
        # Convert to float32 PyTorch Tensors
        self.speech_feats = torch.tensor(speech_feats, dtype=torch.float32) if not isinstance(speech_feats, torch.Tensor) else speech_feats.float()
        self.eeg_feats = torch.tensor(eeg_feats, dtype=torch.float32) if not isinstance(eeg_feats, torch.Tensor) else eeg_feats.float()
        self.labels = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.speech_feats[idx], self.eeg_feats[idx], self.labels[idx]

def get_dataloaders(dataset_path='features_dataset.pt', batch_size=16, val_split=0.2, random_seed=42):
    """
    Loads features from features_dataset.pt, splits them into train and validation sets,
    normalizes features using StandardScaler, and returns PyTorch DataLoaders.
    
    Parameters:
        dataset_path (str): Path to the saved dataset file.
        batch_size (int): Size of batches.
        val_split (float): Fraction of dataset to use for validation.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        DataLoader: Training data loader.
        DataLoader: Validation data loader.
        StandardScaler: Fitted speech features scaler.
        StandardScaler: Fitted EEG features scaler.
    """
    # Load dataset
    try:
        data = torch.load(dataset_path, weights_only=False)
    except Exception as e:
        raise ValueError(f"Could not load dataset from {dataset_path}: {e}")
        
    speech_all = data['speech_features'].numpy()
    eeg_all = data['eeg_features'].numpy()
    labels_all = data['labels'].numpy()
    
    n_samples = len(labels_all)
    indices = np.arange(n_samples)
    
    # Shuffle indices
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    val_size = int(n_samples * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    # Split features
    speech_train, speech_val = speech_all[train_idx], speech_all[val_idx]
    eeg_train, eeg_val = eeg_all[train_idx], eeg_all[val_idx]
    labels_train, labels_val = labels_all[train_idx], labels_all[val_idx]
    
    # Normalize features using StandardScaler
    speech_scaler = StandardScaler()
    eeg_scaler = StandardScaler()
    
    speech_train_scaled = speech_scaler.fit_transform(speech_train)
    speech_val_scaled = speech_scaler.transform(speech_val)
    
    eeg_train_scaled = eeg_scaler.fit_transform(eeg_train)
    eeg_val_scaled = eeg_scaler.transform(eeg_val)
    
    # Create datasets
    train_dataset = MultimodalDataset(speech_train_scaled, eeg_train_scaled, labels_train)
    val_dataset = MultimodalDataset(speech_val_scaled, eeg_val_scaled, labels_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, speech_scaler, eeg_scaler

if __name__ == "__main__":
    # Test dataset loading and scaling
    print("Testing Dataset & DataLoader functions...")
    if torch.os.path.exists('features_dataset.pt'):
        train_loader, val_loader, s_scaler, e_scaler = get_dataloaders('features_dataset.pt')
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Inspect first batch
        for speech_b, eeg_b, labels_b in train_loader:
            print(f"Speech batch shape: {speech_b.shape}")
            print(f"EEG batch shape: {eeg_b.shape}")
            print(f"Labels batch shape: {labels_b.shape}")
            break
    else:
        print("features_dataset.pt not found. Run generate_mock_data.py first.")
