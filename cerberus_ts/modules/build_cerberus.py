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

        # The size of the combined necks is the neck dimension times the number of contexts, plus a call and a response
        combined_neck_size = d_neck * (num_noncontext + num_contexts)

        # Sequentially build the body of Cerberus
        body_layers = []
        last_size = combined_neck_size

        for size in body_layer_sizes:
            body_layers.append(nn.Linear(last_size, size))
            body_layers.append(nn.LeakyReLU())
            last_size = size

        self.body = nn.Sequential(*body_layers)
        
        # We will be concatenating to the last layer of the body, so the last size will need to include those features.
        self.feet = nn.Linear(last_size + call_fl, res_fl)

    def forward(self, x_call, x_contexts, x_response, x_lastknown):
        # Use FormNeck to create necks
        necks = self.form_necks(x_call, x_contexts, x_response)
        
        combined_input = necks
        
        # If foresight is provided
        if self.foresight is not None:
            # Produce foresight
            foresight_out = self.foresight(x_call, x_contexts, x_response, x_lastknown)
            foresight_head_out = self.foresight_head(foresight_out)
            combined_input = torch.cat([combined_input, foresight_head_out], dim = 1)

        # Apply dropout to combined head
        combined_input = self.dropout(combined_input)
        
        body_out = torch.sigmoid(self.body(combined_input))
        
        # We will include the last known at the end of the body
        body_out = torch.cat([body_out, x_lastknown], dim=1)
        
        out = torch.sigmoid(self.feet(body_out))
        return out


from accelerate import Accelerator
import torch

def train_cerberus(model, prepared_dataloaders, num_epochs, learning_rate=0.001, warmup_steps=100, base_lr=0.001):
    # Define a loss function
    criterion = torch.nn.MSELoss()

    # Initialize the Accelerator
    accelerator = Accelerator()

    # Prepare the model and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    # Initialize the learning rate scheduler with warmup
    lr_scheduler = LinearWarmupScheduler(optimizer, warmup_steps, base_lr, learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
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
                    
                    # Step the learning rate scheduler
                    lr_scheduler.step(step)
                    step += 1
                
            except StopIteration:
                # End of epoch
                break

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / sum(len(d) for d in prepared_dataloaders)}")
        
    return model