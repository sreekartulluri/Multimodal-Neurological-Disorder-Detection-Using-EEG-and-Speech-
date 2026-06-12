"""
Training module for the Multimodal Deep Learning Model.
Handles model training, evaluation, early stopping, LR scheduling,
metrics computation (Accuracy, Precision, Recall, F1), and model saving.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_dataloaders
from model import MultimodalAttentionClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_model(dataset_path='features_dataset.pt', epochs=50, batch_size=16, lr=0.001, patience=7):
    """
    Trains the multimodal attention model and saves the best checkpoint.
    """
    print("Initializing Data Loaders...")
    train_loader, val_loader, speech_scaler, eeg_scaler = get_dataloaders(
        dataset_path=dataset_path, 
        batch_size=batch_size, 
        val_split=0.2
    )
    
    # Get dimensions
    speech_dim = train_loader.dataset.speech_feats.shape[1]
    eeg_dim = train_loader.dataset.eeg_feats.shape[1]
    print(f"Features: Speech Dim = {speech_dim}, EEG Dim = {eeg_dim}")
    
    # Initialize Model, Loss, Optimizer, Scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = MultimodalAttentionClassifier(
        speech_dim=speech_dim, 
        eeg_dim=eeg_dim, 
        num_classes=4,
        dropout=0.3
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Early stopping trackers
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    # History logs
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    classes = ["Healthy", "Parkinson's Disease", "Alzheimer's Disease", "Epilepsy"]
    
    print("\nStarting Training Loop...")
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        for speech_b, eeg_b, labels_b in train_loader:
            speech_b = speech_b.to(device)
            eeg_b = eeg_b.to(device)
            labels_b = labels_b.to(device)
            
            optimizer.zero_grad()
            logits = model(speech_b, eeg_b)
            loss = criterion(logits, labels_b)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * speech_b.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for speech_b, eeg_b, labels_b in val_loader:
                speech_b = speech_b.to(device)
                eeg_b = eeg_b.to(device)
                labels_b = labels_b.to(device)
                
                logits = model(speech_b, eeg_b)
                loss = criterion(logits, labels_b)
                val_loss += loss.item() * speech_b.size(0)
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_b.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # Calculate validation metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Update histories
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(acc)
        history['val_f1'].append(f1)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        
        # Check Early Stopping & Save Best Weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = pickle.dumps(model.state_dict())
            print(f"  --> Validation loss improved. Saving checkpoint.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs of no improvement.")
                break
                
    # Restore best weights for final evaluation and packaging
    if best_model_state is not None:
        model.load_state_dict(pickle.loads(best_model_state))
        
    # Final validation evaluation to print report and compute confusion matrix
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for speech_b, eeg_b, labels_b in val_loader:
            speech_b = speech_b.to(device)
            eeg_b = eeg_b.to(device)
            logits = model(speech_b, eeg_b)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_b.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    final_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS:")
    print(f"Accuracy:  {final_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*40)
    
    # Save the consolidated checkpoint containing:
    # 1. Model State Dict
    # 2. Fitted Scalers
    # 3. Features metadata
    # 4. History
    # 5. Class Mapping
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'speech_scaler': speech_scaler,
        'eeg_scaler': eeg_scaler,
        'classes': classes,
        'speech_dim': speech_dim,
        'eeg_dim': eeg_dim,
        'history': history
    }
    torch.save(checkpoint, 'multimodal_checkpoint.pth')
    print("Consolidated checkpoint saved as 'multimodal_checkpoint.pth'!")
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Multimodal Neurological Classifier')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix plot saved as 'confusion_matrix.png'!")
    plt.close()

if __name__ == "__main__":
    if not os.path.exists('features_dataset.pt'):
        print("features_dataset.pt not found! Running generate_mock_data.py first...")
        import subprocess
        subprocess.run(["python", "generate_mock_data.py"])
        
    train_model(epochs=30, batch_size=16, lr=0.001)
