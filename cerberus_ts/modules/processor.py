import torch
import torch.nn as nn

from .multichannel_mha import ChannelWiseMultiHeadAttention

class InputProcessor(nn.Module):
    def __init__(self, in_channels, out_channels, seq_length, feature_length, dropout_rate, layers, *args, **kwargs):
        super(InputProcessor, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Create a sequential model based on the layers provided
        self.layers = nn.Sequential(*[self._create_processor_module(layer, in_channels if i == 0 else out_channels, 
                                                            out_channels, seq_length, feature_length, dropout_rate, *args, **kwargs) 
                                    for i, layer in enumerate(layers)])
        
    def _create_processor_module(self, type, in_channels, out_channels, seq_length, feature_length, dropout_rate, *args, **kwargs):
        if type == "conv":
            padding = (kwargs['kernel_size'] - 1) // 2
            # Convolutional layer followed by LayerNorm, ReLU, and Dropout
            proc = nn.Conv2d(in_channels, out_channels, kwargs['kernel_size'], padding=padding)
            norm = nn.LayerNorm([out_channels, seq_length, feature_length])
        elif type == "mha":
            proc = ChannelWiseMultiHeadAttention(channels=in_channels, feature_length=feature_length, num_heads = kwargs['num_heads'], dropout_rate=0.0)
            norm = nn.LayerNorm([in_channels, seq_length, feature_length])
        relu = nn.LeakyReLU()
        dropout = nn.Dropout(dropout_rate)
        return nn.Sequential(proc, norm, relu, dropout)
    
    def forward(self, x):
        # Store the original input for the residual connection
        original_x = x  # original_x has dimensions [batches, channels, seq_length, feature_length]

        # Apply the layers
        x = self.layers(x)  # After layers, x has dimensions [batches, out_channels, seq_length, feature_length]

        n, c, h, w = x.size()

        # Check if out_channels is a multiple of in_channels
        if c % self.in_channels != 0:
            raise ValueError("out_channels must be a multiple of in_channels for split averaging.")

        # Reshape and average
        x = x.view(n, self.in_channels, c // self.in_channels, h, w)  # Reshaped to [batches, in_channels, splits, seq_length, feature_length]
        x = x.mean(dim=2)  # Averaged across splits, resulting in [batches, in_channels, seq_length, feature_length]

        # Add the original tensor to the output
        x = x + original_x  # Adding back the original tensor, dimensions remain [batches, in_channels, seq_length, feature_length]
        return x

if __name__=="__main__":
    # Example usage
    batches = 10
    channels = 3
    seq_length = 32
    feature_length = 8

    out_channels = 3 # Should be a multiple of channels for split averaging
    kernel_size = 3
    dropout_rate = 0.5
    layers = ["mha"]
    layers = ["conv","conv"]
    layers = ["mha","conv","conv"]
    num_heads = 4

    model = InputProcessor(channels, out_channels, seq_length, feature_length, dropout_rate, layers, kernel_size = kernel_size, num_heads = num_heads)

    input_tensor = torch.randn(batches, channels, seq_length, feature_length)  # Input dimensions: [batches, channels, seq_length, feature_len]
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")