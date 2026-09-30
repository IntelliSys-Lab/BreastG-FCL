"""Compatibility imports for the active E/F/G client implementation.

``Client`` is the same class used by the Ray training entry point. Graph rows
must be supplied to its training and evaluation methods; the obsolete
label-conditioned encoder and graph-mapping generator are no longer exposed.
"""

import pickle

import numpy as np
import torch

from model.client import ModifiedClient as Client


def to_np(x):
    """Convert torch tensor to numpy array"""
    return x.detach().cpu().numpy()


def to_tensor(x, device="cuda"):
    """Convert numpy array or tensor to tensor on specified device"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x

def add_laplace_noise(data, scale):
    noise = np.random.laplace(0, scale, data.shape)
    noise = np.array(noise, dtype=np.float32)
    noisy_data = data + to_tensor(noise)
    return noisy_data

def flat(x):
    """Flatten first two dimensions of tensor"""
    if x.dim() <= 1:  # Handle 1D or 0D tensors
        return x
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])


def write_pickle(data, name):
    """Write data to pickle file"""
    with open(name, "wb") as f:
        pickle.dump(data, f)
