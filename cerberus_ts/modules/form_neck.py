import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .form_head import FormHead

class FormNeck(nn.Module):
    def __init__(self, sizes, feature_indexes, d_neck, head_layers, dropout_rate = 0.0, *args, **kwargs):
        super(FormNeck,self).__init__()
        
        call_size = sizes['call']
        res_size = sizes['response']
        call_fl = len(feature_indexes['call'])
        res_fl = len(feature_indexes['response'])
        context_dims = [[sizes[key], len(feature_indexes[key])]  for key in sizes if 'context' in key]
        
        partial_head = partial(FormHead, channels = 2, 
                         d_neck = d_neck, 
                         dropout_rate = dropout_rate, 
                         layers = head_layers, 
                         **kwargs)
        
        self.call_head = partial_head(seq_length = call_size, feature_length = call_fl)
        self.context_heads = nn.ModuleList([ partial_head(seq_length = icl[0], feature_length = icl[1]) for icl in context_dims ])
        self.response_head = partial_head(seq_length = res_size, feature_length = res_fl)

        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x_call, x_contexts, x_response):
        # Produce call, context, and masked response heads
        call_head_out = self.call_head(x_call)
        context_heads_out = [head(x) for head, x in zip(self.context_heads, x_contexts)]
        response_head_out = self.response_head(x_response)
        
        # Concatenate all the necks together
        necks = torch.cat([call_head_out] + context_heads_out + [response_head_out], dim=1)
        return necks