import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .form_neck import FormNeck

class Cerberus(nn.Module):
    def __init__(self, sizes, feature_indexes, d_neck, head_layers, body_layer_sizes, dropout_rate=0.0, *args, **kwargs):
        super(Cerberus, self).__init__()

        self.form_necks = FormNeck(sizes, feature_indexes, d_neck, head_layers, dropout_rate, **kwargs)
        self.dropout = nn.Dropout(dropout_rate)

        num_contexts = len([key for key in sizes if 'context' in key])
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])

        combined_neck_size = d_neck * (2 + num_contexts) + call_fl

        # Sequentially build the body of Cerberus
        body_layers = []
        last_size = combined_neck_size

        for size in body_layer_sizes:
            body_layers.append(nn.Linear(last_size, size))
            body_layers.append(nn.LeakyReLU())
            last_size = size

        body_layers.append(nn.Linear(last_size, res_fl))
        self.body = nn.Sequential(*body_layers)

    def forward(self, x_call, x_contexts, x_response, x_lastknown):
        # Use FormNeck to create necks
        necks = self.form_necks(x_call, x_contexts, x_response)

        # Concatenate the last known value to the necks
        combined_input = torch.cat([necks, x_lastknown], dim=1)
        combined_input = self.dropout(combined_input)
        return self.body(combined_input)


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