import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import copy
import numpy as np

logger = logging.getLogger('GFedCL')

class Identity(nn.Module):
    """Simple identity module"""
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

#-----------------------------
# For GFedCL - Updated for TCGA-BRCA
#-----------------------------
class GNet(nn.Module):
    """
    Graph Network - takes client relationship vector and generates embedding
    """
    def __init__(self, opt):
        super(GNet, self).__init__()
        self.num_clients = opt.num_clients
        self.hidden_dim = opt.nh
        self.output_dim = opt.nt
        
        # Simple network with minimal operations
        self.net = nn.Sequential(
            nn.Linear(self.num_clients, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, x):
        # Clone input to avoid any in-place operations
        x_copy = x.clone()
        
        # Ensure input is float32
        if x_copy.dtype != torch.float32:
            x_copy = x_copy.float()
        
        # Always use 2D tensors for network operations
        if x_copy.dim() > 2:
            batch_shape = x_copy.shape[:-1]
            x_flat = x_copy.reshape(-1, x_copy.size(-1))
            output = self.net(x_flat)
            # Reshape back to original batch dimensions
            return output.reshape(*batch_shape, self.output_dim)
        else:
            return self.net(x_copy)
        
class FeatureEncoder(nn.Module):
    """
    Feature encoder for TCGA-BRCA RNA-seq vectors.
    It processes tabular expression features, labels, and graph embeddings.
    """
    def __init__(self, opt):
        super(FeatureEncoder, self).__init__()
        self.input_dim = opt.input_dim
        self.hidden_dim = opt.nh
        self.graph_dim = opt.nt
        self.num_classes = opt.num_classes

        self.data_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2) if not opt.no_bn else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(opt.p),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if not opt.no_bn else nn.Identity(),
            nn.ReLU(),
        )

        self.label_embedding = nn.Embedding(self.num_classes, self.hidden_dim // 4)
        self.graph_processor = nn.Sequential(
            nn.Linear(self.graph_dim, self.hidden_dim // 4),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim // 4 + self.hidden_dim // 4, self.hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, labels, graph_embed=None):
        x_copy = x.clone().float()
        if x_copy.dim() > 2:
            x_copy = x_copy.reshape(x_copy.size(0), -1)
        labels_copy = labels.clone().detach()
        batch_size = x_copy.size(0)

        data_features = self.data_encoder(x_copy)

        if labels_copy.dim() == 1:
            label_features = self.label_embedding(labels_copy.long())
        else:
            _, indices = torch.max(labels_copy, dim=1)
            label_features = self.label_embedding(indices.long())

        if graph_embed is not None:
            graph_copy = graph_embed.clone().float()
            if graph_copy.size(0) == 1 and batch_size > 1:
                graph_copy = graph_copy.expand(batch_size, -1)
            graph_features = self.graph_processor(graph_copy)
        else:
            graph_features = torch.zeros(batch_size, self.hidden_dim // 4, device=x_copy.device)

        combined = torch.cat([data_features, label_features, graph_features], dim=1)
        return self.fusion(combined)

class GraphDNet(nn.Module):
    """
    Graph Discriminator - reconstructs graph embedding from encoder latent space
    """
    def __init__(self, opt):
        super(GraphDNet, self).__init__()
        self.input_dim = opt.nh
        self.hidden_dim = opt.nh
        self.output_dim = opt.nt
        
        # Simple network with minimal operations
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def forward(self, x):
        # Clone input to avoid in-place modifications
        x_copy = x.clone()
        
        # Always use 2D tensors for network operations
        if x_copy.dim() > 2:
            batch_shape = x_copy.shape[:-1]
            x_flat = x_copy.reshape(-1, x_copy.size(-1))
            output = self.net(x_flat)
            # Reshape back to original batch dimensions
            return output.reshape(*batch_shape, self.output_dim)
        else:
            return self.net(x_copy)

class PredNet(nn.Module):
    """
    Prediction Network - classifies encoded features
    Updated for TCGA-BRCA
    """
    def __init__(self, opt):
        super(PredNet, self).__init__()
        self.input_dim = opt.nh
        self.hidden_dim = opt.nh
        self.num_classes = opt.num_classes
        
        # Enhanced classifier for more complex dataset
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
    
    def forward(self, x, return_softmax=False):
        # Clone input to avoid in-place operations
        x_copy = x.clone()
        
        # Always use 2D tensors for network operations
        original_shape = x_copy.shape
        if x_copy.dim() > 2:
            x_flat = x_copy.reshape(-1, x_copy.size(-1))
        else:
            x_flat = x_copy
        
        # Forward pass
        logits = self.net(x_flat)
        
        # Get softmax probabilities
        softmax_probs = F.softmax(logits, dim=1)
        
        # Get log probabilities (add small epsilon to avoid log(0))
        log_probs = torch.log(softmax_probs + 1e-10)
        
        # Reshape outputs if needed
        if x_copy.dim() > 2:
            new_shape = original_shape[:-1] + (self.num_classes,)
            log_probs = log_probs.reshape(*new_shape)
            softmax_probs = softmax_probs.reshape(*new_shape)
        
        if return_softmax:
            return log_probs, softmax_probs
        else:
            return log_probs

#-----------------------------
class BreastGraphGenerator(nn.Module):
    """BreastG-FCL disease-aware relational graph from Section III-C.

    Only graph construction is changed from GFedCL. For task ``k`` this class
    computes spatial attention from client morphology summaries, temporal
    attention from a sliding window of DCE summaries, and multiplicatively
    fuses the two attention matrices.

    The paper does not publish the architecture or learned parameters of
    ``a_theta_s`` and ``b_theta_t``.  This source-reproducible implementation
    therefore uses a fixed, seeded-free negative squared-distance scorer for
    both functions; the softmax, temporal window, and multiplicative fusion
    follow the published equations literally.
    """

    def __init__(self, opt):
        super(BreastGraphGenerator, self).__init__()
        self.opt = opt
        self.num_clients = int(opt.num_clients)
        self.temporal_window = max(1, int(opt.temporal_window))
        self.temperature = max(float(opt.attention_temperature), 1e-8)
        self.epsilon = max(float(opt.graph_epsilon), 0.0)
        self.temporal_history = []
        self.last_spatial_attention = None
        self.last_temporal_patterns = None
        self.last_temporal_window = None

    def _validate_client_summaries(self, values, name):
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != self.num_clients:
            raise ValueError(
                f"{name} must have shape [{self.num_clients}, feature_dim], "
                f"got {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite values")
        return matrix

    def _attention(self, values, name):
        """Compute softmax_j a(s_i, s_j) with a fixed RBF-style scorer."""
        matrix = self._validate_client_summaries(values, name)
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        standardized = (matrix - mean) / np.maximum(std, 1e-6)
        differences = standardized[:, None, :] - standardized[None, :, :]
        scores = -np.mean(differences * differences, axis=-1) / self.temperature
        scores -= scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        return weights.astype(np.float32)

    def _windowed_temporal_summaries(self, temporal_features, task_id):
        current = self._validate_client_summaries(
            temporal_features,
            "DCE temporal summaries",
        )
        if task_id == len(self.temporal_history):
            self.temporal_history.append(current)
        elif 0 <= task_id < len(self.temporal_history):
            self.temporal_history[task_id] = current
        else:
            raise ValueError(
                "BreastG-FCL graph tasks must be generated sequentially; "
                f"received task {task_id} with {len(self.temporal_history)} tasks stored"
            )
        start = max(0, task_id - self.temporal_window + 1)
        return np.concatenate(self.temporal_history[start : task_id + 1], axis=1)

    def learn(
        self,
        epochs,
        model_updates=None,
        task_id=None,
        spatial_features=None,
        temporal_features=None,
    ):
        del epochs, model_updates
        task_id = 0 if task_id is None else int(task_id)
        if spatial_features is None:
            raise RuntimeError("Spatial morphology summaries are required")
        spatial_attention = self._attention(
            spatial_features,
            "spatial morphology summaries",
        )
        self.last_spatial_attention = spatial_attention

        if temporal_features is None:
            raise RuntimeError("DCE kinetic summaries are required")
        temporal_window = self._windowed_temporal_summaries(
            temporal_features,
            task_id,
        )
        temporal_attention = self._attention(
            temporal_window,
            "windowed DCE temporal summaries",
        )
        self.last_temporal_window = temporal_window
        self.last_temporal_patterns = temporal_attention

        product = spatial_attention * temporal_attention
        denominator = product.sum(axis=1, keepdims=True) + self.epsilon
        return (product / denominator).astype(np.float32)

    def forward(self, spatial_features, temporal_features=None, task_id=0):
        graph = self.learn(
            epochs=0,
            task_id=task_id,
            spatial_features=spatial_features,
            temporal_features=temporal_features,
        )
        return torch.as_tensor(graph, dtype=torch.float32)

def tensor_memory_in_MB(tensor):
    return tensor.element_size() * tensor.nelement() / (1024 ** 2)
