"""
Deep Learning Model module.
Defines the Attention-Based Multimodal Deep Learning Framework.
Integrates Speech and EEG dense encoders, multi-head cross-attention, and classifier.
"""

import torch
import torch.nn as nn

class DenseEncoder(nn.Module):
    """
    Encodes feature vectors into a 64-dimensional latent embedding.
    """
    def __init__(self, input_dim, hidden_dim=128, embed_dim=64, dropout=0.3):
        super(DenseEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.encoder(x)

class CrossAttentionFusion(nn.Module):
    """
    Performs bilateral multi-head cross-attention between EEG and Speech embeddings,
    combines them with residual connections, and projects to a fused space.
    """
    def __init__(self, embed_dim=64, num_heads=4, fusion_dim=128, dropout=0.3):
        super(CrossAttentionFusion, self).__init__()
        # Speech-to-EEG Attention: Speech queries EEG (keys/values)
        self.speech_to_eeg_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        # EEG-to-Speech Attention: EEG queries Speech (keys/values)
        self.eeg_to_speech_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Linear layer to fuse original + cross-attended representations: 64 * 4 = 256
        self.fusion_projection = nn.Sequential(
            nn.Linear(embed_dim * 4, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, speech_emb, eeg_emb):
        """
        Parameters:
            speech_emb (Tensor): Shape (B, embed_dim)
            eeg_emb (Tensor): Shape (B, embed_dim)
        """
        # Reshape to sequence form: (B, seq_len, embed_dim) with seq_len=1
        q_speech = speech_emb.unsqueeze(1)
        q_eeg = eeg_emb.unsqueeze(1)
        
        # 1. Speech-to-EEG Attention
        # Query: speech, Key/Value: eeg
        attn_s2e, _ = self.speech_to_eeg_attn(query=q_speech, key=q_eeg, value=q_eeg)
        
        # 2. EEG-to-Speech Attention
        # Query: eeg, Key/Value: speech
        attn_e2s, _ = self.eeg_to_speech_attn(query=q_eeg, key=q_speech, value=q_speech)
        
        # Squeeze sequence dimension: (B, 1, embed_dim) -> (B, embed_dim)
        attn_s2e = attn_s2e.squeeze(1)
        attn_e2s = attn_e2s.squeeze(1)
        
        # Concatenate original embeddings and cross-attended representations
        # Shape: (B, embed_dim * 4) = (B, 256)
        concat_feats = torch.cat([speech_emb, eeg_emb, attn_s2e, attn_e2s], dim=-1)
        
        # Project to fusion dimension
        fused_embedding = self.fusion_projection(concat_feats)
        return fused_embedding

class MultimodalAttentionClassifier(nn.Module):
    """
    Complete Multimodal Deep Learning Model for Neurological Disorder Detection.
    """
    def __init__(self, speech_dim, eeg_dim, embed_dim=64, num_heads=4, fusion_dim=128, num_classes=4, dropout=0.3):
        super(MultimodalAttentionClassifier, self).__init__()
        # Sub-branch encoders
        self.speech_encoder = DenseEncoder(input_dim=speech_dim, hidden_dim=128, embed_dim=embed_dim, dropout=dropout)
        self.eeg_encoder = DenseEncoder(input_dim=eeg_dim, hidden_dim=256, embed_dim=embed_dim, dropout=dropout)
        
        # Attention fusion module
        self.attention_fusion = CrossAttentionFusion(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            fusion_dim=fusion_dim, 
            dropout=dropout
        )
        
        # Final Deep Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes) # Outputs raw logits (Softmax applied during inference / loss)
        )
        
    def forward(self, speech_x, eeg_x):
        # 1. Extract embeddings
        speech_emb = self.speech_encoder(speech_x)
        eeg_emb = self.eeg_encoder(eeg_x)
        
        # 2. Perform cross-attention fusion
        fused_emb = self.attention_fusion(speech_emb, eeg_emb)
        
        # 3. Classify
        logits = self.classifier(fused_emb)
        return logits

if __name__ == "__main__":
    # Test network compilation and shapes
    print("Testing neural network compilation...")
    speech_dim = 66
    eeg_dim = 494
    
    # Create test model
    model = MultimodalAttentionClassifier(speech_dim=speech_dim, eeg_dim=eeg_dim)
    print("Model compiled successfully!")
    print(model)
    
    # Feed mock batch (batch_size=8)
    speech_batch = torch.randn(8, speech_dim)
    eeg_batch = torch.randn(8, eeg_dim)
    
    output = model(speech_batch, eeg_batch)
    print(f"Input speech shape: {speech_batch.shape}")
    print(f"Input EEG shape: {eeg_batch.shape}")
    print(f"Output logits shape: {output.shape} (Expected: [8, 4])")
