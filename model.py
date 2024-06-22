import torch
import torch.nn as nn
from torch.nn.functional import softmax


torch.manual_seed(32)

class TransformerDecoder(nn.Module):
    def __init__(self, num_tokens: int, embedding_dim: int, block_size: int, heads: int, head_size: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        
        # Model layers
        self.embedding_table = nn.Embedding(num_tokens, embedding_dim)
        self.positional_encodings_table = nn.Embedding(num_tokens, embedding_dim)
        self.attention_block = MultiHeadAttention(heads, head_size, block_size, embedding_dim)
        self.linear_head = nn.Linear(embedding_dim, num_tokens)
        
        self.loss_func = nn.CrossEntropyLoss()
        
    def forward(self, tokens, targets):
        # tokens is of shape [B, T], targets shape [B, T]
        emb_input = self.embedding_table(tokens) # [B, T, emb_d]
        # Add positional encoding
        pos_emb = self.positional_encodings_table((torch.arange(self.block_size))) # [T, emb_d]
        emb_input += pos_emb
        
        logits = self.linear_head(emb_input)
        # Apply cross-entropy loss
        loss = self.loss_func(logits, targets)
        return logits, loss
    
    def training_step(self, idx, targets):
        logits, loss = self.forward(idx, targets)
        

class ScaledSelfAttentionHead(nn.Module):
    def __init__(self, head_size: int, block_size: int, emb_d: int) -> None:
        self.head_size = head_size
        self.block_Size = block_size
        
        self.register_buffer('tril', torch.tril(torch.ones((self.block_size, self.block_size))))
        self.key = nn.Linear(emb_d, self.head_size)
        self.query = nn.Linear(emb_d, self.head_size)
        self.value = nn.Linear(emb_d, self.head_size)
    
    def forward(self, inputs):
        # B, T, emb_d = inputs.shape
        k = self.key(inputs)   # [B, T, head_size]
        q = self.query(inputs) # [B, T, head_size]
        v = self.value(inputs) # [B, T, head_size]
        
        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5 # [B, T, head_size] * [B, T, head_Size] -> [B, T, head_size] * [B, head_Size, T] = [B, T, T]
        
        weights = weights.masked_fill((self.tril == 0), float('-inf'))
        weights = softmax(weights, dim=1)
        input_w_past_attention = weights @ v # [B, T, T] * [B, T, head_size] = [B, T, head_size]
        
        return input_w_past_attention
    
    
class MultiHeadAttention(nn.Module): 
    def __init__(self, heads: int, head_size: int, block_size: int, emb_d: int) -> None:
        self.heads = heads
        self.head_size = head_size
        self.block_Size = block_size
        
        self.self_attention_blocks = [ScaledSelfAttentionHead(head_size, block_size, emb_d) for _ in range(heads)]
    
    def forward(self, inputs):
        outputs = [attention(inputs) for attention in self.self_attention_blocks]
        
        return torch.cat(outputs, -1)
        