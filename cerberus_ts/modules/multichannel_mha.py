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
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_length, num_heads=num_heads)
        
        self.pos_encoder = PositionalEncoding(d_model = feature_length, dropout=dropout_rate)

    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length, feature_length]
        batch_size, channels, seq_length, feature_length = x.size()

        # Reshape: Merge batch and channels dimension
        x = x.view(batch_size * channels, seq_length, feature_length)
        
        # Apply positional encoding
        x = self.pos_encoder(x)

        # Apply MultiHeadAttention (assuming query, key, value are the same)
        attn_output, _ = self.multihead_attn(x, x, x)

        # Reshape: Split batch and channels dimension
        attn_output = attn_output.view(batch_size, channels, seq_length, feature_length)

        return attn_output
    
if __name__=="__main__":
    # Example usage
    batches = 10
    channels = 3
    seq_length = 32
    feature_length = 16

    model = ChannelWiseMultiHeadAttention(channels=2,feature_length=feature_length, num_heads= 4, dropout_rate=0.0)

    input_tensor = torch.randn(batches, channels, seq_length, feature_length) 
    
    pos_encoder = PositionalEncoding(d_model = 16, dropout=0.0)
    mha_test = MultiheadAttention(embed_dim=16, num_heads=8)
    single_channel = input_tensor[:,0,:,:]
    in_channel = pos_encoder(single_channel)
    
    output_tensor = mha_test(in_channel,in_channel,in_channel)[0]
    
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")