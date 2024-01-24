import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .form_neck import FormNeck
from .form_head import FormHead

from .training_warmup import LinearWarmupScheduler

class Cerberus(nn.Module):
    def __init__(self, sizes, feature_indexes, d_neck, head_layers, body_layer_sizes, dropout_rate=0.0, foresight = None, eventualities = 10, *args, **kwargs):
        super(Cerberus, self).__init__()
        
        self.foresight = foresight

        self.form_necks = FormNeck(sizes, feature_indexes, d_neck, head_layers, dropout_rate, **kwargs)
        self.dropout = nn.Dropout(dropout_rate)

        num_contexts = len([key for key in sizes if 'context' in key])
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])
        
        # Check if foresight is provided, if so, process it. This will be appended with everything else
        if self.foresight is not None:
            res_size = sizes['response']
            
            self.foresight_head = FormHead(
                        channels = eventualities, 
                        seq_length = res_size,
                        feature_length = res_fl,
                        d_neck = d_neck, 
                        dropout_rate = dropout_rate, 
                        layers = head_layers, 
                        out_channels = kwargs['out_channels'], 
                        kernel_size = kwargs['kernel_size'])
            num_noncontext = 3
        else:
            num_noncontext = 2

        # The size of all the combined necks will be based on the number of contexts, the call and response, as well as the last known
        combined_neck_size = d_neck * (num_noncontext + num_contexts) + call_fl

        # Sequentially build the body of Cerberus
        body_layers = []
        last_size = combined_neck_size

        for size in body_layer_sizes:
            body_layers.append(nn.Linear(last_size, size))
            
            # Batch Normalization layer
            bn_layer = nn.BatchNorm1d(size)  # size corresponds to the number of features in the linear layer
            body_layers.append(bn_layer)

            # Activation layer
            body_layers.append(nn.LeakyReLU())
            last_size = size

        body_layers.append(nn.Linear(last_size, res_fl))
        self.body = nn.Sequential(*body_layers)

    def forward(self, x_call, x_contexts, x_response, x_lastknown):
        # Use FormNeck to create necks
        necks = self.form_necks(x_call, x_contexts, x_response)
        
        # Concatenate the last known value to the necks
        combined_input = torch.cat([necks, x_lastknown], dim=1)
        
        # If foresight is provided
        if self.foresight is not None:
            # Produce foresight
            foresight_out = self.foresight(x_call, x_contexts, x_response, x_lastknown)
            foresight_head_out = self.foresight_head(foresight_out)
            combined_input = torch.cat([combined_input, foresight_head_out], dim = 1)

        # Apply dropout to combined head
        combined_input = self.dropout(combined_input)
        
        out = torch.sigmoid(self.body(combined_input))
        return out


from accelerate import Accelerator
import torch

def train_cerberus(model, prepared_dataloaders, num_epochs, learning_rate=0.001, warmup_steps=100, base_lr=1e-6, weight_decay = 0.0):
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
                
                # unmaked is on index 4, everything after that is contexts
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