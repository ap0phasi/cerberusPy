import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import math

# Positional Encoder
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        
        x = x + self.pe[:x.size(0)]
        return self.dropout(x) 


class ChannelWiseMultiHeadAttention(nn.Module):
    def __init__(self, channels, feature_length, dropout_rate, num_heads):
        super(ChannelWiseMultiHeadAttention, self).__init__()
        self.channels = channels
        self.feature_length = feature_length
        
        # Determine the projection dimension
        if feature_length % num_heads != 0:
            # Our feature length must be divisible by 2 times heads
            self.proj_dim = max(2 * num_heads, ((feature_length + num_heads - 1) // num_heads) * num_heads)
            self.proj_layer = nn.Linear(feature_length, self.proj_dim)
        else:
            self.proj_dim = feature_length
            self.proj_layer = None

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.proj_dim, num_heads=num_heads)
        
        self.pos_encoder = PositionalEncoding(d_model = self.proj_dim, dropout=dropout_rate)

    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length, feature_length]
        batch_size, channels, seq_length, feature_length = x.size()

        # Apply projection if necessary
        if self.proj_layer is not None:
            x = x.view(-1, feature_length)
            x = self.proj_layer(x)
            x = x.view(batch_size, channels, seq_length, self.proj_dim)

        # Reshape: Merge batch and channels dimension
        x = x.view(batch_size * channels, seq_length, self.proj_dim)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Apply MultiHeadAttention (assuming query, key, value are the same)
        attn_output, _ = self.multihead_attn(x, x, x)

        # Reshape: Split batch and channels dimension
        attn_output = attn_output.view(batch_size, channels, seq_length, self.proj_dim)
        
        # If projection was applied, add a layer to project back to original feature_length
        if self.proj_layer is not None:
            attn_output = attn_output.view(-1, self.proj_dim)
            # Assuming a simple linear layer for reverse projection (this could be improved)
            reverse_proj_layer = nn.Linear(self.proj_dim, feature_length)
            attn_output = reverse_proj_layer(attn_output)
            attn_output = attn_output.view(batch_size, channels, seq_length, feature_length)

        return attn_output
    
if __name__=="__main__":
    # Example usage
    batches = 10
    channels = 3
    seq_length = 32
    feature_length = 3

    model = ChannelWiseMultiHeadAttention(channels=2,feature_length=feature_length, num_heads= 4, dropout_rate=0.0)

    input_tensor = torch.randn(batches, channels, seq_length, feature_length) 
    
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")