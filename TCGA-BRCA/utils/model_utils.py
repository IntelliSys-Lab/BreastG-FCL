import torch

from model.models import Client, add_laplace_noise, flat, to_np, to_tensor, write_pickle
from utils.server_utils import average_weights

def create_clients(opt):
    """
    Create clients based on the given options.
    """
    clients = []
    for i in range(opt.num_clients):
        client = Client(i, opt)
        clients.append(client)

    if clients:
        initial_weights = clients[0].get_weights()
        for client in clients[1:]:
            client.set_weights(initial_weights)
    return clients

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
