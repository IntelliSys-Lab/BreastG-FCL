"""Transport-independent client operations with isolated random-number state."""

from contextlib import contextmanager
import hashlib
import json
from numbers import Integral
import random
import threading

import numpy as np
import torch


# NVFlare executors may share a process. Serialize seeded contexts so one
# operation cannot overwrite another operation's process-global RNG state.
_RNG_LOCK = threading.RLock()
_OPERATIONS = frozenset(("encode", "train", "test"))


def operation_seed(base_seed, task, round_index, client_id, operation):
    """Derive a stable 32-bit seed independently of worker assignment or order."""
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown client operation: {operation!r}")
    identity = json.dumps(
        [int(base_seed), int(task), int(round_index), int(client_id), operation],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(identity, digest_size=4).digest(), "big")


def snapshot_rng(device=None):
    """Save Python, NumPy, CPU Torch, and optionally one CUDA device's RNG."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda_device": None,
        "cuda": None,
    }
    if device is not None:
        device = torch.device(device)
        if device.type == "cuda":
            index = torch.cuda.current_device() if device.index is None else device.index
            state["cuda_device"] = index
            state["cuda"] = torch.cuda.get_rng_state(index).clone()
    return state


def restore_rng(state):
    """Restore a snapshot without touching unrelated CUDA devices."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda_device"] is not None:
        torch.cuda.set_rng_state(state["cuda"], state["cuda_device"])


@contextmanager
def seeded_rng(seed=None, device=None):
    """Run with an optional local seed and restore the caller's RNG on exit."""
    with _RNG_LOCK:
        state = snapshot_rng(device)
        try:
            if seed is not None:
                seed = int(seed)
                random.seed(seed)
                np.random.seed(seed % (1 << 32))
                # torch.manual_seed also seeds every CUDA device; use the CPU
                # generator directly, then seed only the client's CUDA device.
                torch.random.default_generator.manual_seed(seed % (1 << 64))
                if state["cuda_device"] is not None:
                    with torch.cuda.device(state["cuda_device"]):
                        torch.cuda.manual_seed(seed % (1 << 64))
            yield
        finally:
            restore_rng(state)


def execute_client(operation, client, task, relational_graphs, dataloader,
                   epochs=1, seed=None):
    """Execute the same E/F/G operation under Ray or NVFlare.

    ``encode`` returns latent/graph batches; ``train`` returns E/F/G weights
    and client optimizer/task state; ``test`` returns the client's metrics.
    Model state is intentionally updated by training, while process RNG state
    is always restored, including when a client operation raises an exception.
    """
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown client operation: {operation!r}")
    if operation == "train" and (not isinstance(epochs, Integral) or epochs < 1):
        raise ValueError("Client training requires a positive integer epoch count")
    with seeded_rng(seed, getattr(client, "device", None)):
        if operation == "encode":
            return client.generate_encodings(task, relational_graphs, dataloader, False)
        if operation == "test":
            return client.test(task, dataloader, relational_graphs)
        for epoch in range(epochs):
            client.learn(epoch, task, relational_graphs, dataloader, False)
        result = client.get_weights()
        result["training_state"] = client.get_training_state()
        return result
