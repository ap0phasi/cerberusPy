import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import math

from mamba_ssm import Mamba

class ChannelWiseMamba(nn.Module):
    def __init__(self, channels, feature_length, dropout_rate, d_state = 16, d_conv=4, expand=2):
        super(ChannelWiseMamba, self).__init__()
        self.channels = channels
        self.feature_length = feature_length
        self.dropout = nn.Dropout(p=dropout_rate)

        self.mamba = Mamba(
            d_model=feature_length, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )

    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length, feature_length]
        batch_size, channels, seq_length, feature_length = x.size()

        outs = []
        # Apply Mamba to each channel
        for channel in range(channels):
            outs.append(self.mamba(x[:,channel,:,:]))
            
        mamba_output = torch.stack(outs, dim = 1)
        return mamba_output
    
if __name__=="__main__":
    # Example usage
    batches = 10
    channels = 3
    seq_length = 32
    feature_length = 3

    model = ChannelWiseMamba(channels=2,feature_length=feature_length, num_heads= 4, dropout_rate=0.0)

    input_tensor = torch.randn(batches, channels, seq_length, feature_length) 
    
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")