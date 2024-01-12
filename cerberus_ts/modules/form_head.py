import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_processor import ConvProcessor
from .attention_aggregator import AttentionAggregator

class FormHead(nn.Module):
    def __init__(self, channels, seq_length, feature_length, d_neck, dropout_rate, layers, *args, **kwargs):
        super(FormHead, self).__init__()

        # Assign additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Establish Layers
        self.processor = ConvProcessor(channels, self.out_channels, seq_length, feature_length, self.kernel_size, dropout_rate, layers)
        self.aggregator = AttentionAggregator(channels, seq_length, feature_length, d_neck)
        
    def forward(self, x):
        
        x = self.processor(x)
        x = self.aggregator(x)
        
        return(x)
    

if __name__=="__main__":
    # Example usage
    batches = 10
    channels = 3
    seq_length = 32
    feature_length = 32

    out_channels = 12  # Should be a multiple of channels for split averaging
    kernel_size = 3
    dropout_rate = 0.5
    layers = ["conv", "conv"]
    
    d_neck = 24
    
    model = FormHead(channels, seq_length, feature_length, d_neck, dropout_rate, layers, out_channels, kernel_size)
    
    input_tensor = torch.randn(batches, channels, seq_length, feature_length)  # Input dimensions: [batches, channels, seq_length, feature_len]
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")