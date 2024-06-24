import torch
import torch.nn as nn
from torch.nn.functional import softmax
from lightning import LightningModule

torch.manual_seed(32)

class TransformerDecoder(LightningModule):
    def __init__(self, num_tokens: int, embedding_dim: int, block_size: int, heads: int, head_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        
        # Model layers
        self.embedding_table = nn.Embedding(num_tokens, embedding_dim)
        self.positional_encodings_table = nn.Embedding(num_tokens, embedding_dim)
        self.attention_block = MultiHeadAttention(heads, head_size, block_size, embedding_dim)
        self.linear_head = nn.Linear(embedding_dim, num_tokens)
        
        self.loss_func = nn.CrossEntropyLoss()
        
    def forward(self, tokens, targets):
        B, T = tokens.shape
        # tokens is of shape [B, T], targets shape [B, T]
        emb_input = self.embedding_table(tokens) # [B, T, emb_d]
        # Add positional encoding
        pos_emb = self.positional_encodings_table((torch.arange(self.block_size, device='mps'))) # [T, emb_d]
        emb_input += pos_emb
        
        emb_output = self.attention_block(emb_input)
        
        logits = self.linear_head(emb_output) # [B, T, C=num_tokens]
        logits = logits.view(B * T, -1)
        targets = targets.view(B * T,)
        # Apply cross-entropy loss
        loss = self.loss_func(logits, targets)
        return logits, loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.forward(x, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.forward(x, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

class ScaledSelfAttentionHead(nn.Module):
    def __init__(self, head_size: int, block_size: int, emb_d: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head_size = head_size
        self.block_size = block_size
        
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
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
    def __init__(self, heads: int, head_size: int, block_size: int, emb_d: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.head_size = head_size
        self.block_Size = block_size
        
        self.self_attention_blocks = nn.ModuleList([ScaledSelfAttentionHead(head_size, block_size, emb_d) for _ in range(heads)])
        self.ln_layer = nn.Linear(heads*head_size, emb_d)
        
    def forward(self, inputs):
        outputs = [attention(inputs) for attention in self.self_attention_blocks]
        out = torch.cat(outputs, -1)
        return self.ln_layer(out)
        