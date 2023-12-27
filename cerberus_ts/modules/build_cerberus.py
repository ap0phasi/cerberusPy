import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def ksize(size):
    return max([2, round(size / 9)])

class FormHead(nn.Module):
    def __init__(self, size, feature_len, csize=128, channels=2):
        super(FormHead, self).__init__()
        self.conv = nn.Conv2d(channels, csize, kernel_size=(ksize(size), ksize(feature_len)))
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the output size after convolution and pooling
        conv_output_size = (size - ksize(size) + 1) // 2  # Assuming stride of 1 in conv and 2 in pool
        conv_output_flen = (feature_len - ksize(feature_len) + 1) // 2
        linear_input_size = conv_output_size * conv_output_flen * csize
        # print(linear_input_size)
        self.fc = nn.Linear(linear_input_size, csize)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc(x))
        return x

class Foresight(nn.Module):
    def __init__(self, sizes, feature_indexes, eventualities = 10, csize=128):
        super(Foresight,self).__init__()
        
        call_size = sizes['call']
        res_size = sizes['response']
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])
        context_dims = [[sizes[key], len(feature_indexes[key])]  for key in sizes if 'context' in key]

        self.call_head = FormHead(call_size, call_fl, csize)
        self.context_heads = nn.ModuleList([FormHead(icl[0], icl[1], csize) for icl in context_dims])
        self.response_head = FormHead(res_size, res_fl, csize)
        
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
        necks = F.leaky_relu(self.fc1(necks))
        necks = F.leaky_relu(self.fc2(necks))
        necks = F.leaky_relu(self.expander(torch.cat([necks] + [last_known], dim=1)))
        necks = necks.view(-1, self.reshape_channels, self.reshape_height, self.reshape_width)
        out = self.decoder(necks)
        return out
        
    def forward(self, x_call, x_contexts, x_response, x_lastknown):
        # Produce call, context, and masked response heads
        call_head_out = self.call_head(x_call)
        context_heads_out = [head(x) for head, x in zip(self.context_heads, x_contexts)]
        response_head_out = self.response_head(x_response)
        
        # Use last known value
        last_known = x_lastknown
        
        necks = torch.cat([call_head_out] + context_heads_out + [response_head_out], dim=1)
        necks = F.leaky_relu(self.fc1(necks))
        necks = F.leaky_relu(self.fc2(necks))
        necks = F.leaky_relu(self.expander(torch.cat([necks] + [last_known], dim=1)))
        necks = necks.view(-1, self.reshape_channels, self.reshape_height, self.reshape_width)
        out = self.decoder(necks)
        return out

class Cerberus(nn.Module):
    def __init__(self, sizes, feature_indexes, csize=128, foresight = None, eventualities = 10):
        super(Cerberus, self).__init__()
        self.foresight = foresight
        
        call_size = sizes['call']
        res_size = sizes['response']
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])
        context_dims = [[sizes[key], len(feature_indexes[key])]  for key in sizes if 'context' in key]

        self.call_head = FormHead(call_size, call_fl, csize)
        self.context_heads = nn.ModuleList([FormHead(icl[0], icl[1], csize) for icl in context_dims])
        self.response_head = FormHead(res_size, res_fl, csize)
        
        if self.foresight is not None:
            self.foresight_head = FormHead(res_size, res_fl, csize, channels=eventualities)
            num_noncontext = 3
        else:
            num_noncontext = 2

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
            
        necks = F.leaky_relu(self.fc1(necks))
        necks = F.leaky_relu(self.fc2(necks))
        body = F.leaky_relu(self.fc3(torch.cat([necks] + [last_known], dim=1)))
        body = F.leaky_relu(self.fc4(body))
        body = F.leaky_relu(self.fc5(body))
        body = torch.sigmoid(self.out(body))
        return body
    
    
from accelerate import Accelerator
import torch

def train_cerberus(model, prepared_dataloaders, num_epochs):
    # Define a loss function
    criterion = torch.nn.MSELoss()

    # Initialize the Accelerator
    accelerator = Accelerator()

    # Prepare the model and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

def train_foresight(foresight, prepared_dataloaders, num_epochs):
    # Define a loss function
    criterion = EventualityMSELoss()

    # Initialize the Accelerator
    accelerator = Accelerator()

    # Prepare the model and optimizer
    optimizer = torch.optim.Adam(foresight.parameters(), lr=0.001)
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