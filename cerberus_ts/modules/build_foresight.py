import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .form_neck import FormNeck

from .training_warmup import LinearWarmupScheduler

# Load in configuration
from ..utils.cerberus_config import CerberusConfig

class Foresight(nn.Module):
    def __init__(self, sizes, feature_indexes, d_neck, head_layers, body_layer_sizes, dropout_rate=0.0, eventualities=10, expander_sizes=[128, 256], last_known_loc=0, *args, **kwargs):
        super(Foresight, self).__init__()

        self.form_necks = FormNeck(sizes, feature_indexes, d_neck, head_layers, dropout_rate, **kwargs)
        self.dropout = nn.Dropout(dropout_rate)
        self.last_known_loc = last_known_loc

        num_contexts = len([key for key in sizes if 'context' in key])
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])
        res_size = sizes['response']

        combined_neck_size = d_neck * (2 + num_contexts)

        # Sequentially build the body
        body_layers = []
        
        last_size = combined_neck_size
        
        if self.last_known_loc == 0:
            last_size += call_fl

        for i, size in enumerate(body_layer_sizes):
            if i+1 == self.last_known_loc:
                # Adjust last_size to include x_lastknown size
                last_size += call_fl

            body_layers.append(nn.Linear(last_size, size))
            
            last_size = size

        self.body = nn.Sequential(*body_layers)
        
        self.reshape_channels = expander_sizes[0]
        self.reshape_height = res_size
        self.reshape_width = res_fl

        self.expander = nn.Linear(last_size, self.reshape_channels * self.reshape_height * self.reshape_width)
        
        # Decoder layers
        decoder_layers = []
        in_channels = self.reshape_channels
        for out_channels in expander_sizes:
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            decoder_layers.append(nn.LeakyReLU())
            in_channels = out_channels

        decoder_layers.append(nn.ConvTranspose2d(in_channels, eventualities, kernel_size=3, stride=1, padding=1))
        
        if not CerberusConfig.foresight_residual:
            decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x_call, x_contexts, x_response, x_lastknown):
        necks = self.form_necks(x_call, x_contexts, x_response)

        if self.last_known_loc == 0:
            # Original behavior
            combined_input = torch.cat([necks, x_lastknown], dim=1)
        else:
            combined_input = necks
        
        combined_input = self.dropout(combined_input)

        # Process through body layers
        for i, layer in enumerate(self.body):
            if i+1 == self.last_known_loc:
                combined_input = torch.cat([combined_input, x_lastknown], dim=1)
            combined_input = layer(combined_input)
            
            combined_input = F.leaky_relu(combined_input)

        necks = F.leaky_relu(self.expander(combined_input))
        necks = necks.view(-1, self.reshape_channels, self.reshape_height, self.reshape_width)
        out = self.decoder(necks)
        
        if CerberusConfig.foresight_residual:
            out = out + x_response[:,0:1,:,:]

        return out


from accelerate import Accelerator
import torch

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


def train_foresight(foresight, prepared_dataloaders, num_epochs, learning_rate=0.001, warmup_steps=100, base_lr=0.001, weight_decay = 0.0):
    # Define a loss function
    criterion = EventualityMSELoss()

    # Initialize the Accelerator
    accelerator = Accelerator()

    # Prepare the model and optimizer
    optimizer = torch.optim.AdamW(foresight.parameters(), lr=base_lr, weight_decay = weight_decay)
    foresight, optimizer = accelerator.prepare(foresight, optimizer)
    
    # Initialize the learning rate scheduler with warmup
    lr_scheduler = LinearWarmupScheduler(optimizer, warmup_steps, base_lr, learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        foresight.train()
        running_loss = 0.0

        # Set up iterator from prepared dataloader
        iterator = iter(prepared_dataloaders)

        step = 0  # Initialize step count for warmup updates
        while True:
            try:
                # Collect batch
                batch = next(iterator)
                
                # Prepare data for the model
                calls_batch = batch[0]
                responses_batch = batch[1]
                last_knowns_batch = batch[2]
                # Unmasked is index 4
                unmasked_batch = batch[4] 
                
                contexts_batch = batch[5:]
                
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
                    
                    # Step the learning rate scheduler
                    lr_scheduler.step(step)
                    step += 1

            except StopIteration:
                # End of epoch
                break

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / sum(len(d) for d in prepared_dataloaders)}")
        
    # Freeze all foresight weights
    for param in foresight.parameters():
        param.requires_grad = False
        
    return foresight