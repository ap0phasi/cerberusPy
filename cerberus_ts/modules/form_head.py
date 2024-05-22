import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, length_in, length_out):
        super(Conv1dBlock, self).__init__()
        kernel_size = round_up_to_odd(length_in // 4)  # Can be any odd number
        padding_size = (kernel_size - 1) // 2  # Ensuring output size equals input size

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride = 1, padding = padding_size)
        # self.conv1_spec = nn.utils.spectral_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding_size))
        self.bn_layer1 = nn.BatchNorm1d(out_channels)
        # self.inorm = nn.InstanceNorm1d(out_channels)
        # self.gn = nn.GroupNorm(num_groups=out_channels // 2, num_channels=out_channels)
        self.fc1 = nn.Linear(length_in, length_in)
        self.fc_ln1 = nn.LayerNorm(length_in)
        self.fc2 = nn.Linear(length_in, length_in)
        self.fc_ln2 = nn.LayerNorm(length_in)
        self.fc3 = nn.Linear(length_in, length_out)
        self.fc_ln3 = nn.LayerNorm(length_out)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_layer1(x)
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc_ln1(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.fc_ln2(x)
        x = self.fc3(x)
        x = self.fc_ln3(x)
        return x

class FormHead_Base(nn.Module):
    def __init__(self, length, d_features, d_neck, dropout_rate, layers, *args, **kwargs):
        super(FormHead_Base, self).__init__()
        
        head_layers = layers
        hsize = kwargs['out_channels']
        
        if head_layers is None:
            head_layers = ["conv"]  # Default to a single convolutional layer if not specified
            
        # Handle hsize as either a single value or a list
        if not isinstance(hsize, list):
            hsize = [hsize] * len(head_layers)  # Repeat the single hsize value for each layer
            
        self.layers = nn.ModuleList()
    
        self.current_channels = d_features * 2 # We double this because of how we set up the 2D coil norm

        for idx, layer_type in enumerate(head_layers):
            layer_hsize = hsize[idx]  # hsize for the current layer
            if layer_type == "conv":
                self.layers.append(Conv1dBlock(in_channels = self.current_channels, out_channels =  layer_hsize, length_in=length, length_out= length))
                
                self.current_channels = layer_hsize
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
        
        self.fc = nn.Linear(self.current_channels, d_neck)

    def forward(self, x):
        # Shape for processing
        x = x.permute(0, 2, 1)
        
        for layer in self.layers:
            x = layer(x)

        # Reshape for final step
        x = x.permute(0, 2, 1)
        
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