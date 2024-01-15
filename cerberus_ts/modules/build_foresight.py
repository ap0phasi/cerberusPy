import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .form_neck import FormNeck

from .training_warmup import LinearWarmupScheduler

# Load in configuration
from ..utils.cerberus_config import CerberusConfig

class Foresight(nn.Module):
    def __init__(self, sizes, feature_indexes, d_neck, head_layers, body_layer_sizes, dropout_rate=0.0, eventualities = 10, expander_sizes = [128, 256], *args, **kwargs):
        super(Foresight, self).__init__()

        self.form_necks = FormNeck(sizes, feature_indexes, d_neck, head_layers, dropout_rate, **kwargs)
        self.dropout = nn.Dropout(dropout_rate)

        num_contexts = len([key for key in sizes if 'context' in key])
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])
        res_size = sizes['response']

        combined_neck_size = d_neck * (2 + num_contexts) + call_fl

        # Sequentially build the body of Cerberus
        body_layers = []
        last_size = combined_neck_size

        for size in body_layer_sizes:
            body_layers.append(nn.Linear(last_size, size))
            body_layers.append(nn.LeakyReLU())
            last_size = size

        self.body = nn.Sequential(*body_layers)
        
        # Parameters for reshaping the output of the linear layer
        self.reshape_channels = expander_sizes[0]  # Assuming the first element is used for reshaping
        self.reshape_height = res_size
        self.reshape_width = res_fl

        self.expander = nn.Linear(last_size, self.reshape_channels * self.reshape_height * self.reshape_width)
        
        # Dynamically create the decoder layers based on expander_args
        decoder_layers = []
        in_channels = self.reshape_channels
        for out_channels in expander_sizes:
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            decoder_layers.append(nn.LeakyReLU())
            in_channels = out_channels
        
        # Add the final layer transitioning to 'eventualities' channels with Sigmoid activation
        decoder_layers.append(nn.ConvTranspose2d(in_channels, eventualities, kernel_size=3, stride=1, padding=1))
        
        # If we are using a residual connection, we don't want the last decoder layer to be sigmoid
        if not CerberusConfig.foresight_residual:
            decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x_call, x_contexts, x_response, x_lastknown):
        # Use FormNeck to create necks
        necks = self.form_necks(x_call, x_contexts, x_response)

        # Concatenate the last known value to the necks
        combined_input = torch.cat([necks, x_lastknown], dim=1)
        combined_input = self.dropout(combined_input)
        body = self.body(combined_input)
        necks = F.leaky_relu(self.expander(body))
        necks = necks.view(-1, self.reshape_channels, self.reshape_height, self.reshape_width)
        out = self.decoder(necks)
        
        if CerberusConfig.foresight_residual:
            # Add Residual Connection
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


def train_foresight(foresight, prepared_dataloaders, num_epochs, learning_rate=0.001, warmup_steps=100, base_lr=1e-6):
    # Define a loss function
    criterion = EventualityMSELoss()

    # Initialize the Accelerator
    accelerator = Accelerator()

    # Prepare the model and optimizer
    optimizer = torch.optim.AdamW(foresight.parameters(), lr=base_lr)
    foresight, optimizer = accelerator.prepare(foresight, optimizer)
    
    # Initialize the learning rate scheduler with warmup
    lr_scheduler = LinearWarmupScheduler(optimizer, warmup_steps, base_lr, learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        foresight.train()
        running_loss = 0.0

        # Iterator for each prepared DataLoader
        iterators = [iter(dataloader) for dataloader in prepared_dataloaders]

        step = 0  # Initialize step count for warmup updates
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