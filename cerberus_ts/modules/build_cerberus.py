import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn import MultiheadAttention

def ksize(size):
    return max([1, round(size / 9)])

class MultiChannelMHA(nn.Module):
    def __init__(self, num_channels, size, feature_len, hsize=128, num_heads=4):
        super(MultiChannelMHA, self).__init__()
        self.mha_layers = nn.ModuleList([MultiheadAttention(embed_dim=hsize, num_heads=num_heads) for _ in range(num_channels)])
        self.hsize = hsize
        self.flatten_dim = size * feature_len
        self.num_channels = num_channels

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        channel_wise_outputs = []

        for i in range(self.num_channels):
            mha_layer = self.mha_layers[i]
            
            # Process each channel with its MHA layer
            channel_data = x[:, i, :, :].reshape(batch_size, self.flatten_dim)
            channel_data = channel_data.unsqueeze(-1).expand(-1, -1, self.hsize)
            channel_data = channel_data.permute(1, 0, 2)  # Reshape to [sequence_length, batch_size, embed_dim]

            mha_output, _ = mha_layer(channel_data, channel_data, channel_data)
            mha_output = mha_output.permute(1, 0, 2)  # Reshape back to [batch_size, sequence_length, embed_dim]

            # Aggregate outputs of all heads
            channel_wise_outputs.append(mha_output[:, :, 0])

        aggregated_output = torch.stack(channel_wise_outputs, dim=1)
        return aggregated_output

class FormHead(nn.Module):
    def __init__(self, size, feature_len, csize = 128, hsize=128, pool_size = 1, head_layers=None, channels=2):
        super(FormHead, self).__init__()

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
                conv_layer = nn.Conv2d(current_channels, layer_hsize, kernel_size=(ksize(size), ksize(feature_len)))
                self.layers.append(conv_layer)
                self.layers.append(nn.LeakyReLU())
                self.layers.append(nn.MaxPool2d(layer_pool_size,layer_pool_size))
                current_channels = layer_hsize
                
                # Calculate the output size after convolution and pooling
                size = (size - ksize(size) + 1) // layer_pool_size  # Assuming stride of 1 and pool of 2
                feature_len = max([1, (feature_len - ksize(feature_len) + 1) // layer_pool_size])
                linear_input_size = size * feature_len * layer_hsize
            elif layer_type == "mha":
                # Assuming a certain configuration for MultiheadAttention. Adjust as necessary.
                mha_layer = MultiChannelMHA(channels, size, feature_len, layer_hsize, num_heads=4)
                self.layers.append(mha_layer)
                # Note: MHA layer configuration will depend on your specific requirements
                linear_input_size = size * feature_len * channels
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        self.fc = nn.Linear(linear_input_size, csize)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc(x))
        return x

class Foresight(nn.Module):
    def __init__(self, sizes, feature_indexes, csize=128, hsize = 128, pool_size = 1, eventualities = 10, head_layers=None, dropout = 0.0):
        super(Foresight,self).__init__()
        
        call_size = sizes['call']
        res_size = sizes['response']
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])
        context_dims = [[sizes[key], len(feature_indexes[key])]  for key in sizes if 'context' in key]

        self.call_head = FormHead(call_size, call_fl, csize, hsize, pool_size, head_layers)
        self.context_heads = nn.ModuleList([FormHead(icl[0], icl[1], csize, hsize, pool_size, head_layers) for icl in context_dims])
        self.response_head = FormHead(res_size, res_fl, csize, hsize, pool_size, head_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(csize * (2 + len(context_dims)), csize * 16)
        self.fc2 = nn.Linear(csize * 16, csize * 8)
    
        # Parameters for reshaping the output of the linear layer
        self.reshape_channels = csize * 8  # Number of channels after reshaping
        self.reshape_height = res_size
        self.reshape_width = res_fl

        self.expander = nn.Linear(csize * 8 + call_fl, self.reshape_channels * self.reshape_height * self.reshape_width)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(csize * 8, csize * 16, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, 
                               output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(csize * 16, eventualities, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, 
                               output_padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x_call, x_contexts, x_response, x_lastknown):
        # Produce call, context, and masked response heads
        call_head_out = self.call_head(x_call)
        context_heads_out = [head(x) for head, x in zip(self.context_heads, x_contexts)]
        response_head_out = self.response_head(x_response)
        
        # Use last known value
        last_known = x_lastknown
        
        necks = torch.cat([call_head_out] + context_heads_out + [response_head_out], dim=1)
        necks = F.leaky_relu(self.dropout(self.fc1(necks)))
        necks = F.leaky_relu(self.dropout(self.fc2(necks)))
        necks = F.leaky_relu(self.expander(torch.cat([necks] + [last_known], dim=1)))
        necks = necks.view(-1, self.reshape_channels, self.reshape_height, self.reshape_width)
        out = self.decoder(necks)
        return out
        

class Cerberus(nn.Module):
    def __init__(self, sizes, feature_indexes, csize=128, hsize = 128, pool_size = 1, foresight = None, eventualities = 10, head_layers=None, dropout = 0.0):
        super(Cerberus, self).__init__()
        self.foresight = foresight
        
        call_size = sizes['call']
        res_size = sizes['response']
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])
        context_dims = [[sizes[key], len(feature_indexes[key])]  for key in sizes if 'context' in key]

        self.call_head = FormHead(call_size, call_fl, csize, hsize, pool_size, head_layers)
        self.context_heads = nn.ModuleList([FormHead(icl[0], icl[1], csize, hsize, pool_size, head_layers) for icl in context_dims])
        self.response_head = FormHead(res_size, res_fl, csize, hsize, pool_size, head_layers)
        
        if self.foresight is not None:
            self.foresight_head = FormHead(res_size, res_fl, csize, hsize, pool_size, head_layers, channels=eventualities)
            num_noncontext = 3
        else:
            num_noncontext = 2

        self.dropout = nn.Dropout(dropout)
 
        self.fc1 = nn.Linear(csize * (num_noncontext + len(context_dims)), csize * 16)
        self.fc2 = nn.Linear(csize * 16, csize * 8)
        self.fc3 = nn.Linear(csize * 8 + call_fl, csize * 4)
        self.fc4 = nn.Linear(csize * 4, csize)
        self.fc5 = nn.Linear(csize, csize // 2)
        self.out = nn.Linear(csize // 2, res_fl)

    def forward(self, x_call, x_contexts, x_response, x_lastknown):
        
        # If foresight is provided
        if self.foresight is not None:
            # Produce foresight
            foresight_out = self.foresight(x_call, x_contexts, x_response, x_lastknown)
            foresight_head_out = self.foresight_head(foresight_out)
        
        # Produce call, context, and masked response heads
        call_head_out = self.call_head(x_call)
        context_heads_out = [head(x) for head, x in zip(self.context_heads, x_contexts)]
        response_head_out = self.response_head(x_response)
        
        # Use last known value
        last_known = x_lastknown
        
        if self.foresight is not None:
            necks = torch.cat([call_head_out] + context_heads_out + [response_head_out] + [foresight_head_out], dim=1)
        else:
            necks = torch.cat([call_head_out] + context_heads_out + [response_head_out], dim=1)
            
        necks = F.leaky_relu(self.dropout(self.fc1(necks)))
        necks = F.leaky_relu(self.dropout(self.fc2(necks)))
        body = F.leaky_relu(self.dropout(self.fc3(torch.cat([necks] + [last_known], dim=1))))
        body = F.leaky_relu(self.dropout(self.fc4(body)))
        body = F.leaky_relu(self.dropout(self.fc5(body)))
        body = torch.sigmoid(self.out(body))
        return body
    
    
from accelerate import Accelerator
import torch

def train_cerberus(model, prepared_dataloaders, num_epochs, learning_rate = 0.001):
    # Define a loss function
    criterion = torch.nn.MSELoss()

    # Initialize the Accelerator
    accelerator = Accelerator()

    # Prepare the model and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Iterator for each prepared DataLoader
        iterators = [iter(dataloader) for dataloader in prepared_dataloaders]

        while True:
            try:
                # Collect batches from each DataLoader
                batches = [next(iterator) for iterator in iterators]

                # Prepare data for the model
                calls_batch = next(batch[0] for batch in batches)
                contexts_batch = [batch[1] for batch in batches]
                responses_batch = next(batch[2] for batch in batches)
                last_knowns_batch = next(batch[3] for batch in batches)
                y_batch = next(batch[4] for batch in batches)

                # Forward and backward passes
                with accelerator.accumulate(model):
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(calls_batch, contexts_batch, responses_batch, last_knowns_batch)
                    loss = criterion(outputs, y_batch)

                    # Backward pass and optimize
                    accelerator.backward(loss)
                    optimizer.step()

                    running_loss += loss.item()

            except StopIteration:
                # End of epoch
                break

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / sum(len(d) for d in prepared_dataloaders)}")
        
    return model

class EventualityMSELoss(nn.Module):
    def __init__(self):
        super(EventualityMSELoss, self).__init__()

    def forward(self, output, observed):
        cum_error = 0
        # Separate the mean and variance
        for iev in range(output.shape[1]):
            nll = (observed - output[:,iev,:,:]) ** 2
            cum_error += nll.mean()
        return cum_error

def train_foresight(foresight, prepared_dataloaders, num_epochs, learning_rate = 0.001):
    # Define a loss function
    criterion = EventualityMSELoss()

    # Initialize the Accelerator
    accelerator = Accelerator()

    # Prepare the model and optimizer
    optimizer = torch.optim.Adam(foresight.parameters(), lr=learning_rate)
    foresight, optimizer = accelerator.prepare(foresight, optimizer)

    # Training Loop
    for epoch in range(num_epochs):
        foresight.train()
        running_loss = 0.0

        # Iterator for each prepared DataLoader
        iterators = [iter(dataloader) for dataloader in prepared_dataloaders]

        while True:
            try:
                # Collect batches from each DataLoader
                batches = [next(iterator) for iterator in iterators]
                
                # Prepare data for the model
                calls_batch = next(batch[0] for batch in batches)
                contexts_batch = [batch[1] for batch in batches]
                responses_batch = next(batch[2] for batch in batches)
                last_knowns_batch = next(batch[3] for batch in batches)
                unmasked_batch = next(batch[5] for batch in batches) # Unmasked is stored in the fifth entry
                
                # Forward and backward passes
                with accelerator.accumulate(foresight):
                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = foresight(calls_batch, contexts_batch, responses_batch, last_knowns_batch)
                    loss = criterion(outputs, unmasked_batch)

                    # Backward pass and optimize
                    accelerator.backward(loss)
                    optimizer.step()

                    running_loss += loss.item()

            except StopIteration:
                # End of epoch
                break

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / sum(len(d) for d in prepared_dataloaders)}")
        
    # Freeze all foresight weights
    for param in foresight.parameters():
        param.requires_grad = False
        
    return foresight