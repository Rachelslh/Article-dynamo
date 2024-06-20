import torch
import torch.nn as nn


torch.manual_seed(32)

class TransformerDecoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Embedding layer
        self.embedding_table = nn.Embedding(num_embeddings, embedding_dim)
        
    def forward(self, idx, targets):
        logits = self.embedding_table(idx)
        
        return logits