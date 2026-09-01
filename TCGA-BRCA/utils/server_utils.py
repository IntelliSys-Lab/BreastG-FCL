import torch
import copy
from typing import List, Dict

def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Average the weights across multiple clients
    
    Args:
        weights: List of state dictionaries, one per client
        
    Returns:
        averaged_weights: A single state dictionary with averaged weights
    """
    # Return None if no weights provided
    if not weights:
        return None
    
    # Start with a deep copy of the first client's weights
    weights_avg = copy.deepcopy(weights[0])
    
    # Add weights from other clients
    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        
        # Compute average
        weights_avg[key] = torch.div(weights_avg[key], len(weights))
    
    return weights_avg