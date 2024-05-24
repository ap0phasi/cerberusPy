import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .form_head import FormHead

class FormNeck(nn.Module):
    def __init__(self, sizes, feature_indexes, d_neck, head_layers, dropout_rate = 0.0, *args, **kwargs):
        super(FormNeck,self).__init__()
        
        call_length = sizes['call']
        res_length = sizes['response']
        call_features = len(feature_indexes['call'])
        res_features = len(feature_indexes['response'])
        context_dims = [[sizes[key], len(feature_indexes[key])]  for key in sizes if 'context' in key]
            
        partial_head = partial(FormHead,
                         d_neck = call_features * 2, 
                         dropout_rate = dropout_rate, 
                         length_out = call_length,
                         layers = head_layers, 
                         **kwargs)
        
        # Process the head. We can allow for a bias here
        self.call_head = partial_head(length_in = call_length, d_features = call_features, d_neck = d_neck, bias = True)
        
        # We want to get the contexts and responses into the same size as the features
        self.context_heads = nn.ModuleList([ partial_head(length_in = icl[0], d_features = icl[1], bias = True) for icl in context_dims ])
        # We do not want biases because we want an empty response to make no change
        self.response_head = partial_head(length_in = res_length, d_features = res_features, bias = True)
        
        # Multi Headed Attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=call_features * 2, num_heads= 2)
        
    def forward(self, x_call, x_contexts, x_response):
        # Produce context and masked response heads
        context_heads_out = [head(x) for head, x in zip(self.context_heads, x_contexts)]
        response_head_out = self.response_head(x_response)
        
        # # Add up the response and context heads
        # modification_tensor = torch.stack(context_heads_out + [response_head_out],dim=2).sum(dim=2)#.permute(1,0,2)
        
        # # Option 1: Modify the call tensor
        # call_mod = x_call + modification_tensor #[batch, length, features]
        
        # # Option 2:
        # call_mod, _ = self.multihead_attention(x_call.permute(1,0,2), modification_tensor, modification_tensor)
        # call_mod = call_mod.permute(1,0,2)
        
        # Option 3:
        context_mod = torch.stack(context_heads_out,dim=2).sum(dim=2)
        base_tensor = (x_call + context_mod).permute(1,0,2)
        modification_tensor = response_head_out.permute(1,0,2)
        call_mod, _ = self.multihead_attention(base_tensor, modification_tensor, modification_tensor)
        call_mod = call_mod.permute(1,0,2)
        

        # Normalize the new call_mod 
        call_mod_norm = torch.softmax(call_mod, dim = 2) 
        #print(f"Normed Call shape: {call_mod_norm.shape}")
    
        call_head_out = self.call_head(call_mod_norm)
        # print(f"Neck shape {necks.shape}")
        return call_head_out