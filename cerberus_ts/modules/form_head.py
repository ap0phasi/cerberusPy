import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_processor import ConvProcessor
from .attention_aggregator import AttentionAggregator

class FormHead_Option1(nn.Module):
    def __init__(self, channels, seq_length, feature_length, d_neck, dropout_rate, layers, *args, **kwargs):
        super(FormHead_Option1, self).__init__()

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

def ksize(size):
    return max([1, round(size / 9)])

class FormHead_Option2(nn.Module):
    def __init__(self, channels, seq_length, feature_length, d_neck, dropout_rate, layers, *args, **kwargs):
        super(FormHead_Option2, self).__init__()
        
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
    
# To Select one of the FormHead options
class FormHead(nn.Module):
    def __init__(self, option = 'option1', *args, **kwargs):
        super(FormHead, self).__init__()
        if option == 'option1':
            self.head = FormHead_Option1(*args, **kwargs)
        elif option == 'option2':
            self.head = FormHead_Option2(*args, **kwargs)
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