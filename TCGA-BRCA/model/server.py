import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import copy
import logging

from model.modules import GraphDNet

logger = logging.getLogger('GFedCL')

class Server:
    """
    Server Class for GFedCL
    
    The server:
    1. Maintains a global discriminator
    2. Receives encoded samples from clients
    3. Trains the discriminator on those encodings
    4. Sends updated discriminator back to clients
    """
    def __init__(self, opt):
        """
        Initialize the server with a global discriminator
        
        Args:
            opt: Configuration options
        """
        self.opt = opt
        self.device = torch.device(opt.device)
        
        # Initialize the global discriminator
        self.global_discriminator = GraphDNet(opt).to(self.device)
        
        # Set up optimizer
        self.optimizer_D = optim.Adam(
            self.global_discriminator.parameters(), 
            lr=opt.lr_d, 
            betas=(opt.beta1, 0.999)
        )
        
        # Set up learning rate scheduler
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_D, 
            gamma=0.5 ** (1 / 100)
        )
    
    def train_discriminator(self, encoded_samples, graph_embeddings):
        """
        Train the global discriminator using encoded samples from clients
        
        Args:
            encoded_samples: List of encoded samples from clients
            graph_embeddings: List of graph embeddings for each client
            
        Returns:
            loss_D: Discriminator loss
        """
        self.global_discriminator.train()
        
        # Concatenate all encoded samples and graph embeddings
        e_seq = torch.cat(encoded_samples, dim=0)
        z_seq = torch.cat(graph_embeddings, dim=0)
        
        # Ensure both tensors are on the same device
        e_seq = e_seq.to(self.device)
        z_seq = z_seq.to(self.device)
        
        # Forward pass through discriminator
        d_seq = self.global_discriminator(e_seq.detach())  # Detach to avoid gradient flow to encoder
        
        # Handle dimension mismatch by adjusting the target tensor
        if d_seq.dim() > 2:
            # Flatten batch dimensions for predicted tensor
            predicted = d_seq.reshape(-1, d_seq.size(-1))
        else:
            predicted = d_seq
            
        # Handle graph embeddings - ensure proper shape matching
        if z_seq.dim() > 2:
            # Flatten graph embeddings if needed
            target_flat = z_seq.reshape(-1, z_seq.size(-1))
        else:
            target_flat = z_seq
            
        # Ensure the target tensor matches the predicted tensor size
        predicted_batch_size = predicted.size(0)
        target_batch_size = target_flat.size(0)
        
        if predicted_batch_size != target_batch_size:
            # Calculate how many times we need to repeat the target
            if predicted_batch_size % target_batch_size == 0:
                # Perfect division - repeat the target
                repeat_factor = predicted_batch_size // target_batch_size
                target = target_flat.repeat(repeat_factor, 1)
            else:
                # Imperfect division - handle by truncating or padding
                if predicted_batch_size > target_batch_size:
                    # Repeat and then truncate
                    repeat_factor = (predicted_batch_size // target_batch_size) + 1
                    repeated_target = target_flat.repeat(repeat_factor, 1)
                    target = repeated_target[:predicted_batch_size]
                else:
                    # Truncate target to match predicted size
                    target = target_flat[:predicted_batch_size]
        else:
            target = target_flat
            
        # Ensure target tensor has the same shape as predicted tensor
        if target.size() != predicted.size():
            logger.warning(f"Shape mismatch after adjustment: predicted {predicted.size()} vs target {target.size()}")
            # Final fallback: use only the minimum batch size
            min_batch_size = min(predicted.size(0), target.size(0))
            predicted = predicted[:min_batch_size]
            target = target[:min_batch_size]
            
        # Compute MSE loss
        loss_D = F.mse_loss(predicted, target, reduction='mean')
        
        # Backward pass and optimization
        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()
        
        logger.debug(f"Discriminator training: predicted shape {predicted.size()}, target shape {target.size()}, loss {loss_D.item():.4f}")
        
        return loss_D.item()
    
    def get_discriminator(self):
        """
        Get the global discriminator state_dict
        
        Returns:
            state_dict: The state dictionary of the global discriminator
        """
        return copy.deepcopy(self.global_discriminator.state_dict())
    
    def set_discriminator(self, state_dict):
        """
        Update the global discriminator with a new state_dict
        
        Args:
            state_dict: The state dictionary to load
        """
        self.global_discriminator.load_state_dict(state_dict)
    
    def update_learning_rate(self):
        """
        Update the learning rate using the scheduler
        """
        self.lr_scheduler_D.step()
    
    def evaluate_discriminator(self, encoded_samples, graph_embeddings):
        """
        Evaluate the discriminator without training
        
        Args:
            encoded_samples: List of encoded samples from clients
            graph_embeddings: List of graph embeddings for each client
            
        Returns:
            loss_D: Discriminator loss
        """
        self.global_discriminator.eval()
        
        # Concatenate all encoded samples and graph embeddings
        e_seq = torch.cat(encoded_samples, dim=0)
        z_seq = torch.cat(graph_embeddings, dim=0)
        
        # Ensure both tensors are on the same device
        e_seq = e_seq.to(self.device)
        z_seq = z_seq.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            d_seq = self.global_discriminator(e_seq)
            
            # Handle dimension mismatch same way as in training
            if d_seq.dim() > 2:
                predicted = d_seq.reshape(-1, d_seq.size(-1))
            else:
                predicted = d_seq
                
            if z_seq.dim() > 2:
                target_flat = z_seq.reshape(-1, z_seq.size(-1))
            else:
                target_flat = z_seq
                
            # Handle size mismatch
            predicted_batch_size = predicted.size(0)
            target_batch_size = target_flat.size(0)
            
            if predicted_batch_size != target_batch_size:
                if predicted_batch_size % target_batch_size == 0:
                    repeat_factor = predicted_batch_size // target_batch_size
                    target = target_flat.repeat(repeat_factor, 1)
                else:
                    if predicted_batch_size > target_batch_size:
                        repeat_factor = (predicted_batch_size // target_batch_size) + 1
                        repeated_target = target_flat.repeat(repeat_factor, 1)
                        target = repeated_target[:predicted_batch_size]
                    else:
                        target = target_flat[:predicted_batch_size]
            else:
                target = target_flat
                
            if target.size() != predicted.size():
                min_batch_size = min(predicted.size(0), target.size(0))
                predicted = predicted[:min_batch_size]
                target = target[:min_batch_size]
                
            # Compute MSE loss
            loss_D = F.mse_loss(predicted, target, reduction='mean')
            
        return loss_D.item()