# File: models/bilstm/model/bilstm_model.py
"""
PyTorch implementation of Bi-LSTM classifier with optional pretrained GloVe embeddings.

Usage:
    model = BiLSTMClassifier(vocab_size, embed_dim=300, hidden_size=128, num_classes=3, embedding_weights=embedding_matrix)

If embedding_weights is provided (numpy array), they will be loaded and set as weights for the embedding layer.
"""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_size=128, num_classes=3, embedding_weights=None, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_weights is not None:
            # embedding_weights expected as torch.FloatTensor of shape (vocab_size, embed_dim)
            self.embedding.weight.data.copy_(embedding_weights)
            # It's often better to allow fine-tuning; if you want frozen embeddings set: self.embedding.weight.requires_grad = False

        self.encoder = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)
        emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        outputs, (hn, cn) = self.encoder(emb)
        # outputs: (batch, seq_len, hidden*2)
        # simple pooling: use last timestep output or mean-pooling
        # We'll use mean-pooling over valid tokens if attention_mask provided
        if attention_mask is None:
            pooled = outputs.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            pooled = (outputs * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-9))

        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits

