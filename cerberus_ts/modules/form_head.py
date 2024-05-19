import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FormHead_Base(nn.Module):
    def __init__(self, length, d_features, d_neck, dropout_rate, layers, *args, **kwargs):
        super(FormHead_Base, self).__init__()
        
        def round_up_to_odd(f):
            return int(np.ceil(f) // 2 * 2 + 1)
        
        kernel_size = round_up_to_odd(length // 4)  # Can be any odd number
        padding_size = (kernel_size - 1) // 2  # Ensuring output size equals input size
        
        head_layers = layers
        hsize = kwargs['out_channels']
        size = length
        
        if head_layers is None:
            head_layers = ["conv"]  # Default to a single convolutional layer if not specified
            
        # Handle hsize as either a single value or a list
        if not isinstance(hsize, list):
            hsize = [hsize] * len(head_layers)  # Repeat the single hsize value for each layer
            
        self.layers = nn.ModuleList()
    
        current_channels = d_features * 2 # We double this because of how we set up the 2D coil norm

        for idx, layer_type in enumerate(head_layers):
            layer_hsize = hsize[idx]  # hsize for the current layer
            if layer_type == "conv":
                conv_layer = nn.Conv1d(in_channels = current_channels, out_channels =  layer_hsize, kernel_size = kernel_size, stride=1, padding = padding_size)
                self.layers.append(conv_layer)
                # Batch Normalization layer
                bn_layer = nn.BatchNorm1d(layer_hsize)  # Make sure to use the correct number of features
                self.layers.append(bn_layer)

                # Activation layer
                self.layers.append(nn.LeakyReLU())
                
                # Linear layers
                self.layers.append(nn.Linear(length, length))
                self.layers.append(nn.Linear(length, length))
                self.layers.append(nn.Linear(length, length))
                
                current_channels = layer_hsize
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        self.fc = nn.Linear(current_channels, d_neck)

    def forward(self, x):
        batch, length, dim = x.shape
        
        # Shape for processing
        x = x.view(batch, dim, length)
        for layer in self.layers:
            x = layer(x)

        # Reshape for final step
        x = x.permute(0,2,1)
        
        # Get into size needed for neck
        x = self.fc(x)
        return x
    
class FormHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FormHead, self).__init__()
        
        self.head = FormHead_Base(*args, **kwargs)

    def forward(self, x):
        return self.head(x)


if __name__=="__main__":
    # Example usage
    batches = 10
    channels = 3
    seq_length = 32
    feature_length = 8

    out_channels = 12
    dropout_rate = 0.5
    layers = ["conv", "conv"]
    
    d_neck = 24
    
    model = FormHead_Base( 
                     length=seq_length, 
                     d_features=feature_length // 2, 
                     d_neck=d_neck, 
                     dropout_rate=dropout_rate, 
                     layers=layers, 
                     out_channels = out_channels).to("cuda")
    
    input_tensor = torch.randn(batches, seq_length, feature_length).to("cuda")  # Input dimensions: [batches, channels, seq_length, feature_len]
    output_tensor = model(input_tensor).to("cpu")

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    
    print(output_tensor)