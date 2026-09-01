from model.models import *
import torch
import copy
from typing import List, Dict
import numpy as np
import os

def create_clients(opt):
    """
    Create clients based on the given options.
    """
    clients = []
    for i in range(opt.num_clients):
        client = Client(i, opt)
        clients.append(client)
    
    return clients

def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], 2*len(weights))

    return weights_avg

def concatenate_tensors(tensor_list):
    """
    Concatenate a list of tensors along the first dimension
    
    Args:
        tensor_list: List of tensors to concatenate
        
    Returns:
        concatenated: A single tensor
    """
    if not tensor_list:
        return None
    
    # Check if all tensors have the same shape except for the first dimension
    first_shape = tensor_list[0].shape[1:]
    for tensor in tensor_list:
        if tensor.shape[1:] != first_shape:
            raise ValueError(f"Tensors have incompatible shapes: {tensor.shape} vs {tensor_list[0].shape}")
    
    # Concatenate along the first dimension
    return torch.cat(tensor_list, dim=0)
