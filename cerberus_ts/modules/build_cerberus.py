import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .form_neck import FormNeck
from .form_head import FormHead

from .training_warmup import LinearWarmupScheduler

import torch
import torch.nn as nn

class Cerberus(nn.Module):
    def __init__(self, sizes, feature_indexes, d_neck, head_layers, body_layer_sizes, dropout_rate=0.0, foresight=None, eventualities=10, last_known_loc=0, *args, **kwargs):
        super(Cerberus, self).__init__()
        
        self.foresight = foresight
        self.last_known_loc = last_known_loc

        self.form_necks = FormNeck(sizes, feature_indexes, d_neck, head_layers, dropout_rate, **kwargs)
        self.dropout = nn.Dropout(dropout_rate)

        num_contexts = len([key for key in sizes if 'context' in key])
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])
        res_size = sizes['response']
        call_size = sizes['call']
        
        concatenated_size = sum([value for key, value in sizes.items()])
        
        if self.foresight is not None:
            
            self.foresight_head = FormHead(
                        channels = eventualities, 
                        seq_length = res_size,
                        feature_length = res_fl,
                        d_neck = d_neck, 
                        dropout_rate = dropout_rate, 
                        layers = head_layers, 
                        out_channels = kwargs['out_channels'])
            concatenated_size += res_size

        # The length of last_known is the same size as the call_fl, so we need a layer to get that to the neck size
        self.last_known_process = nn.Linear(call_fl, d_neck)
        
        # Process down to a dimension of 1. 
        current_length = concatenated_size
        
        self.body = nn.ModuleList()
        
        for idx, layer_size in enumerate(body_layer_sizes):
                self.body.append(nn.Linear(current_length, layer_size))
                current_length = layer_size
                
                # When we append lastknown it will increase the size by 1
                if idx + 1 == self.last_known_loc:
                    current_length += 1
                
        self.body.append(nn.Linear(current_length, 1))
        
        self.final_agg  = nn.Linear(d_neck, res_fl * 2)

    def forward(self, x_call, x_contexts, x_response, x_lastknown):
        necks = self.form_necks(x_call, x_contexts, x_response)

        # process last known to correct size
        processed_lastknown = self.last_known_process(x_lastknown)
        
        necks = necks.permute(0, 2, 1)
        
        if self.last_known_loc == 0:
            # Original behavior
            combined_input = torch.cat([necks, processed_lastknown.unsqueeze(-1)], dim=2)
        else:
            combined_input = necks

        if self.foresight is not None:
            foresight_out = self.foresight(x_call, x_contexts, x_response, x_lastknown)
            foresight_head_out = self.foresight_head(foresight_out)
            combined_input = torch.cat([combined_input, foresight_head_out], dim=2)

        combined_input = self.dropout(combined_input)
        
        # Process through body layers
        for i, layer in enumerate(self.body):
            #print(f"Iteration: {i}, combined input size: {combined_input.shape}")
            if i == self.last_known_loc:
                # Concatenate x_lastknown at the specified layer
                combined_input = torch.cat([combined_input, processed_lastknown.unsqueeze(-1)], dim=2)
                
            combined_input = layer(combined_input)
            
            if i < len(self.body) - 1:
                combined_input = F.leaky_relu(combined_input)
            
        # Now we want to go back to operations along the feature length    
        combined_input = combined_input.permute(0, 2, 1)
            
        # Process down into res_fl, then softmax
        combined_input = self.final_agg(combined_input)
        #print(f"combined_input shape: {combined_input.shape}")    
        out = torch.softmax(combined_input.squeeze(1), dim = 1)
        return out

from accelerate import Accelerator
import torch

def train_cerberus(model, prepared_dataloaders, num_epochs, learning_rate=0.001, warmup_steps=100, base_lr=0.001, weight_decay = 0.0):
    # Define a loss function
    criterion = torch.nn.MSELoss()

    # Initialize the Accelerator
    accelerator = Accelerator()

    # Prepare the model and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay = weight_decay)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # Initialize the learning rate scheduler with warmup
    lr_scheduler = LinearWarmupScheduler(optimizer, warmup_steps, base_lr, learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
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
                y_batch = batch[3]
                
                # unmasked is on index 4, everything after that is contexts
                contexts_batch = batch[5:]

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
                    
                    # Step the learning rate scheduler
                    lr_scheduler.step(step)
                    step += 1
                
            except StopIteration:
                # End of epoch
                break

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / sum(len(d) for d in prepared_dataloaders)}")
        
    return model