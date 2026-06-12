"""
Inference module for neurological disorder detection.
Loads the consolidated checkpoint and runs multimodal prediction on new .wav and .edf files.
"""

import os
import torch
import numpy as np
from speech_features import extract_speech_features
from eeg_features import extract_eeg_features
from model import MultimodalAttentionClassifier

def load_prediction_pipeline(checkpoint_path='multimodal_checkpoint.pth'):
    """
    Loads the trained model, scalers, and classes from the checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Instantiate model
    model = MultimodalAttentionClassifier(
        speech_dim=checkpoint['speech_dim'], 
        eeg_dim=checkpoint['eeg_dim'], 
        num_classes=len(checkpoint['classes'])
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return {
        'model': model,
        'speech_scaler': checkpoint['speech_scaler'],
        'eeg_scaler': checkpoint['eeg_scaler'],
        'classes': checkpoint['classes'],
        'device': device
    }

def predict_disorder(wav_path, edf_path, pipeline=None, checkpoint_path='multimodal_checkpoint.pth'):
    """
    Predicts the neurological disorder from a speech WAV file and EEG EDF file.
    
    Parameters:
        wav_path (str): Path to WAV speech file.
        edf_path (str): Path to EDF EEG file.
        pipeline (dict): Pre-loaded pipeline components (optional).
        checkpoint_path (str): Path to checkpoint if pipeline is not provided.
        
    Returns:
        dict: A dictionary containing:
            - predicted_class (str): Label of predicted disorder.
            - confidence (float): Probability score of predicted class.
            - probabilities (dict): Probability map for all classes.
            - speech_features_dict (dict): Raw extracted speech features.
            - eeg_features_dict (dict): Raw extracted EEG features.
            - speech_vector_scaled (np.ndarray): Scaled speech features for explainability.
            - eeg_vector_scaled (np.ndarray): Scaled EEG features for explainability.
    """
    if pipeline is None:
        pipeline = load_prediction_pipeline(checkpoint_path)
        
    model = pipeline['model']
    speech_scaler = pipeline['speech_scaler']
    eeg_scaler = pipeline['eeg_scaler']
    classes = pipeline['classes']
    device = pipeline['device']
    
    # 1. Extract Features
    s_dict, s_vector = extract_speech_features(wav_path)
    e_dict, e_vector = extract_eeg_features(edf_path)
    
    # 2. Scale Features (reshape to batch size 1)
    s_vector_scaled = speech_scaler.transform(s_vector.reshape(1, -1))
    eeg_vector_scaled = eeg_scaler.transform(e_vector.reshape(1, -1))
    
    # Convert to Tensors
    s_tensor = torch.tensor(s_vector_scaled, dtype=torch.float32).to(device)
    e_tensor = torch.tensor(eeg_vector_scaled, dtype=torch.float32).to(device)
    
    # 3. Model Inference
    with torch.no_grad():
        logits = model(s_tensor, e_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
    # Handle single element batch array conversion
    if probs.ndim == 0:
        probs = np.array([probs])
        
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])
    
    prob_map = {classes[i]: float(probs[i]) for i in range(len(classes))}
    
    return {
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': prob_map,
        'speech_features_dict': s_dict,
        'eeg_features_dict': e_dict,
        'speech_vector_scaled': s_vector_scaled,
        'eeg_vector_scaled': eeg_vector_scaled
    }

if __name__ == "__main__":
    # Test inference on a generated sample
    print("Testing inference pipeline...")
    wav_file = "samples/sample_parkinsons_disease.wav"
    edf_file = "samples/sample_parkinsons_disease.edf"
    
    if os.path.exists(wav_file) and os.path.exists(edf_file):
        results = predict_disorder(wav_file, edf_file)
        print("\nINFERENCE RESULTS:")
        print(f"Predicted Class: {results['predicted_class']}")
        print(f"Confidence:      {results['confidence']:.4f}")
        print("Class Probabilities:")
        for cls, prob in results['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")
    else:
        print("Sample files not found. Run generate_mock_data.py first.")
