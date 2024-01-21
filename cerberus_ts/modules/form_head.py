import torch
import torch.nn as nn
import torch.nn.functional as F

from .processor import InputProcessor
from .attention_aggregator import AttentionAggregator

# Load in configuration
from ..utils.cerberus_config import CerberusConfig

class FormHead_Preserve(nn.Module):
    def __init__(self, channels, seq_length, feature_length, d_neck, dropout_rate, layers, *args, **kwargs):
        super(FormHead_Preserve, self).__init__()

        # Assign additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Establish Layers
        self.processor = InputProcessor(in_channels = channels, seq_length=seq_length, feature_length=feature_length, dropout_rate=dropout_rate, layers=layers, **kwargs)
        self.aggregator = AttentionAggregator(channels, seq_length, feature_length, d_neck)
        
    def forward(self, x):
        
        x = self.processor(x)
        x = self.aggregator(x)
        
        return(x)

def ksize(size):
    return max([1, round(size / 9)])

class FormHead_Flatten(nn.Module):
    def __init__(self, channels, seq_length, feature_length, d_neck, dropout_rate, layers, *args, **kwargs):
        super(FormHead_Flatten, self).__init__()
        
        pool_size = 1
        head_layers = layers
        hsize = kwargs['out_channels']
        size = seq_length
        
        if head_layers is None:
            head_layers = ["conv"]  # Default to a single convolutional layer if not specified
            
        # Handle hsize as either a single value or a list
        if not isinstance(hsize, list):
            hsize = [hsize] * len(head_layers)  # Repeat the single hsize value for each layer
            
        # Handle pool_size as either a single value or a list
        if not isinstance(pool_size, list):
            pool_size = [pool_size] * len(head_layers)  # Repeat the single hsize value for each layer

        self.layers = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        current_channels = channels

        for idx, layer_type in enumerate(head_layers):
            layer_hsize = hsize[idx]  # hsize for the current layer
            layer_pool_size = pool_size[idx]
            if layer_type == "conv":
                conv_layer = nn.Conv2d(current_channels, layer_hsize, kernel_size=(ksize(size), ksize(feature_length)))
                self.layers.append(conv_layer)
                # Batch Normalization layer
                bn_layer = nn.BatchNorm2d(layer_hsize)  # Make sure to use the correct number of features
                self.layers.append(bn_layer)

                # Activation layer
                self.layers.append(nn.LeakyReLU())
                self.layers.append(nn.MaxPool2d(layer_pool_size,layer_pool_size))
                current_channels = layer_hsize
                
                # Calculate the output size after convolution and pooling
                size = (size- ksize(size) + 1) // layer_pool_size  # Assuming stride of 1 and pool of 2
                feature_length = max([1, (feature_length - ksize(feature_length) + 1) // layer_pool_size])
                linear_input_size = size * feature_length * layer_hsize
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        self.fc = nn.Linear(linear_input_size, d_neck)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc(x))
        return x
    
class FormHead(nn.Module):
    def __init__(self, option=None, *args, **kwargs):
        super(FormHead, self).__init__()

        # Use the current value of CerberusConfig.processor_type if option is not provided
        if option is None:
            option = CerberusConfig.processor_type
            
        if option == 'flatten':
            self.head = FormHead_Flatten(*args, **kwargs)
        elif option == 'preserve':
            self.head = FormHead_Preserve(*args, **kwargs)
        else:
            raise ValueError("Invalid option for FormHead")

    def forward(self, x):
        return self.head(x)


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
    
    model = FormHead(channels = channels, 
                     seq_length=seq_length, 
                     feature_length=feature_length, 
                     d_neck=d_neck, 
                     dropout_rate=dropout_rate, 
                     layers=layers, 
                     out_channels = out_channels , 
                     kernel_size = kernel_size)
    
    input_tensor = torch.randn(batches, channels, seq_length, feature_length)  # Input dimensions: [batches, channels, seq_length, feature_len]
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")