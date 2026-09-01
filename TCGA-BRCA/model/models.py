import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import numpy as np
from model.modules import *
import pickle
import copy
import logging

logger = logging.getLogger('GFedCL')

# ===========================================================================================================
# Utility functions

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


# ======================================================================================================================

# the base model
class BaseModel(nn.Module):
    def __init__(self, client_id, opt):
        super(BaseModel, self).__init__()
        # set output format
        np.set_printoptions(suppress=True, precision=6)
        self.client_id = client_id
        self.opt = opt
        self.device = opt.device
        self.batch_size = opt.batch_size

        # visualization
        self.use_visdom = opt.use_visdom
        self.use_g_encode = opt.use_g_encode

        # Setup logging and output directories
        self.train_log = self.opt.outf + "/loss.log"
        self.model_path = opt.outf + "/model.pth"
        self.out_pic_f = opt.outf + "/plt_pic"
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)
        if not os.path.exists(self.out_pic_f):
            os.mkdir(self.out_pic_f)
        with open(self.train_log, "w") as f:
            f.write("log start!\n")

    def getId(self): 
        """Return client ID"""
        return self.client_id
    
    def learn(self, epoch, dataloader):
        """
        Basic learning loop for a single epoch
        
        Args:
            epoch: Current epoch number
            dataloader: DataLoader for training data
        """
        self.train()
        if self.use_g_encode:
            self.netG.eval()
        self.epoch = epoch
        loss_values = {loss: 0 for loss in self.loss_names}

        count = 0
        for data in dataloader:
            count += 1
            self.__set_input__(data)
            self.__train_forward__()
            new_loss_values = self.__optimize__()

            # for the loss visualization
            for key, loss in new_loss_values.items():
                loss_values[key] += loss

        # Calculate average loss
        for key in loss_values.keys():
            if count > 0:
                loss_values[key] /= count

        # Log loss periodically
        if (self.epoch + 1) % 10 == 0:
            logger.info("Client %d, epoch %d: %s", self.client_id, self.epoch, loss_values)

        # Learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
            
        return loss_values

    def test(self, task_id, dataloader):
        """
        Test the model on a dataset
        
        Args:
            task_id: Current task ID
            dataloader: DataLoader containing test samples
            
        Returns:
            dict: Dictionary containing metrics
        """
        self.eval()
        
        # Track metrics
        correct = 0
        total = 0
        total_loss = 0.0
        
        # Test loop
        with torch.no_grad():
            for data in dataloader:
                # Set input data
                self.__set_input__(data, generate=False, train=False)
                
                # Forward pass
                self.__test_forward__()
                
                # Calculate metrics
                total += self.y_seq.size(0)
                predictions = torch.argmax(self.f_seq, dim=-1) if self.f_seq.dim() > 1 else self.f_seq
                correct += (predictions == self.y_seq).sum().item()
                
                # Calculate loss
                if self.f_seq.dim() == 3:
                    f_seq_flat = self.f_seq.view(-1, self.opt.num_classes)
                    y_seq_flat = self.y_seq.reshape(-1)
                else:
                    f_seq_flat = self.f_seq
                    y_seq_flat = self.y_seq
                    
                loss = F.nll_loss(f_seq_flat, y_seq_flat.long())
                total_loss += loss.item() * self.y_seq.size(0)
        
        # Calculate final metrics
        accuracy = 100.0 * correct / total if total > 0 else 0
        avg_loss = total_loss / total if total > 0 else 0
        
        logger.info(f"Client {self.client_id}, Task {task_id} Test - Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        
        return {
            "loss": avg_loss,
            "acc": accuracy
        }

    def save(self, suffix=''):
        """Save model to disk"""
        if suffix:
            path = self.model_path.replace('.pth', f'_{suffix}.pth')
        else:
            path = self.model_path
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def __set_input__(self, data, generate=False, train=True):
        """
        Sets the input data for model training/testing.
        
        Args:
            data: Tuple of (inputs, targets) from dataloader
            generate: Whether to generate synthetic samples
            train: Whether in training mode
        """
        # DataLoader in PyTorch returns a list/tuple where:
        # data[0] = images (batch_size, channels, height, width)
        # data[1] = labels (batch_size)
        
        # Unpack the data
        inputs, targets = data
        
        # Move to device
        self.x_seq = inputs.to(self.device)
        self.y_seq = targets.to(self.device)
        
        self.tmp_batch_size = self.x_seq.size(0)

        # Create synthetic data if needed (for continual learning)
        if generate and hasattr(self, 'task_ID') and self.task_ID > 0:
            # Create synthetic data for replay
            if len(self.x_seq.shape) == 4:  # [batch_size, channels, height, width]
                # Generate random noise with the same shape as the image data
                noise = torch.randn_like(self.x_seq, device=self.device)
                self.x_seq_synthetic = noise
            else:
                # Create random noise with the same shape as flattened data
                noise = torch.randn_like(self.x_seq, device=self.device)
                self.x_seq_synthetic = noise
        
        # No need to create one-hot encoding for base model
        # Create a dummy z_seq for compatibility with the Client class
        # Base-class placeholder; graph-aware clients replace this value.
        self.z_seq = torch.zeros(1, self.opt.nt, device=self.device)

    def __train_forward__(self):
        """Forward pass during training - implemented by subclasses"""
        pass

    def __test_forward__(self):
        """
        Forward pass during testing - default implementation
        """
        self.z_seq = self.netG(self.one_hot_seq)
        self.e_seq, _ = self.netE(self.x_seq, self.y_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        _, softmax_output = self.netF(self.e_seq, return_softmax=True)
        self.g_seq = torch.argmax(softmax_output, dim=-1)  # class predictions

    def __optimize__(self):
        """Default optimization step for GAN-based models"""
        
        loss_value = dict()

        self.loss_D = self.__loss_D__()
        self.optimizer_D.zero_grad()
        self.loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        loss_value["D"] = self.loss_D.item()

        # Train encoder and predictor
        
        self.loss_E, self.loss_E_pred, self.loss_E_gan = self.__loss_EF__()
        #print(f"loss_E: {self.loss_E}, loss_E_pred: {self.loss_E_pred}, loss_E_gan: {self.loss_E_gan}")
        self.optimizer_EF.zero_grad()
        self.loss_E.backward()
        self.optimizer_EF.step()
        
        loss_value["E_pred"] = self.loss_E_pred.item()
        loss_value["E_gan"] = self.loss_E_gan.item()
        
        return loss_value

    def __loss_D__(self):
        """Discriminator loss - implemented by subclasses"""
        pass

    def __loss_EF__(self):
        """Encoder and Predictor loss - implemented by subclasses"""
        pass

    def __log_write__(self, message):
        """Write message to log file"""
        logger.info(message)
        with open(self.train_log, "a") as f:
            f.write(message + "\n")

    def __init_weight__(self, net=None):
        """Initialize weights for the network"""
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

class Client(BaseModel):
    """
    GFedCL Client Model Implementation
    
    Manages the training process for a client in the GFedCL framework, handling:
    - Graph-based knowledge transfer
    - Continual learning with synthetic sample generation
    - Custom loss functions for graph-based federated learning
    """
    def __init__(self, client_id, opt):
        super(Client, self).__init__(client_id, opt)
        
        # Initialize the neural networks
        self.netE = FeatureEncoder(opt).to(opt.device)  # Encoder-decoder
        self.netF = PredNet(opt).to(opt.device)                # Classifier
        self.netG = GNet(opt).to(opt.device)                   # Graph embedding generator 
        self.netD = GraphDNet(opt).to(opt.device)              # Graph discriminator
        
        # Initialize weights
        self.__init_weight__()
        
        # Set up optimizers
        # Combine Encoder and Predictor parameters for joint optimization
        EF_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_EF = optim.Adam(
            EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999)
        )
        
        # Discriminator optimizer
        self.optimizer_D = optim.Adam(
            self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999)
        )
        
        # Set up learning rate schedulers
        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_EF, gamma=0.5 ** (1 / 100)
        )
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_D, gamma=0.5 ** (1 / 100)
        )

        self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        
        # Define loss names for tracking
        self.loss_names = ["E_pred", "E_gan", "D"]
            
        # Initialize tracking variables
        self.relation_graph = None
        self.task_ID = None

    def learn(self, epoch, task, relational_graphs, dataloader, generate=False):
        """
        Train the client model using relational graph and dataloader
        
        Args:
            epoch: Current epoch number
            task: Current task ID
            relational_graphs: Relational graphs for all tasks
            dataloader: DataLoader containing samples
            generate: Flag to control synthetic sample generation
            
        Returns:
            dict: Loss values for monitoring
        """
        torch.autograd.set_detect_anomaly(True)
        # Set model to training mode
        self.train()
            
        # Store the task ID and relational graph for use in training
        self.task_ID = task
        self.relational_graph = relational_graphs
        self.epoch = epoch
        self.generate = generate
        
        # Initialize loss tracking
        loss_values = {loss: 0 for loss in self.loss_names}
        count = 0
        
        # Training loop
        for data in dataloader:
            count += 1
            
            # Set the input data - this will also create synthetic data if needed
            self.__set_input__(data, self.generate)
            
            # Forward pass
            self.__train_forward__()
            
            # Calculate losses and update weights
            new_loss_values = self.__optimize__()

            # Track loss values
            for key, loss in new_loss_values.items():
                loss_values[key] += loss

        # Calculate average loss
        if count > 0:
            for key in loss_values.keys():
                loss_values[key] /= count

        # Log progress periodically
        #
        status_msg = f"Client {self.client_id}, Task {task}, Epoch {self.epoch}"
        if self.generate:
            status_msg += " (with synthetic data)"
        status_msg += f": {loss_values}"
        logger.info(status_msg)

        # Apply learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
            
        return loss_values

    def __set_input__(self, data, generate=False, train=True):
        """
        Sets the input data for model training/testing.
        
        Args:
            data: Tuple of (inputs, targets) from dataloader
            generate: Whether to generate synthetic samples
            train: Whether in training mode
        """
        # DataLoader in PyTorch returns a list/tuple where:
        # data[0] = images (batch_size, channels, height, width)
        # data[1] = labels (batch_size)
        
        # Unpack the data
        inputs, targets = data
        
        # Move to device
        self.x_seq = inputs.to(self.device)
        self.y_seq = targets.to(self.device)
        
        self.tmp_batch_size = self.x_seq.size(0)

        # Create synthetic data if needed (for continual learning)
        if generate and hasattr(self, 'task_ID') and self.task_ID > 0:
            # Create random noise with the same shape as flattened data
            noise = torch.randn_like(self.x_seq, device=self.device)
            self.x_seq_synthetic = noise
        
        # For graph-based models, create one-hot client encoding
        if hasattr(self, 'client_id') and hasattr(self, 'opt') and hasattr(self.opt, 'num_clients'):
            # Create one-hot encoding for client ID
            one_hot = torch.zeros(1, self.opt.num_clients, device=self.device)
            one_hot[0, self.client_id] = 1.0
            self.one_hot_seq = one_hot
            
            # Extract client relations from graph if available
            if hasattr(self, 'relational_graph') and hasattr(self, 'task_ID'):
                try:
                    # Get client's row from the relational graph for current task
                    # Updated to handle the new relational_graph structure [task]
                    if isinstance(self.relational_graph, list) and len(self.relational_graph) > self.task_ID:
                        graph = self.relational_graph[self.task_ID]
                        
                        # Extract client-specific relationships
                        if isinstance(graph, np.ndarray) and graph.shape[0] > self.client_id:
                            client_relations = torch.tensor(graph[self.client_id], device=self.device)
                            self.client_relations = client_relations.unsqueeze(0)  # Add batch dimension
                        else:
                            # If the graph doesn't have the expected structure, use one-hot encoding
                            self.client_relations = self.one_hot_seq
                    else:
                        # If there's no graph for this task, use one-hot encoding
                        self.client_relations = self.one_hot_seq
                except Exception as e:
                    logger.error(f"Error accessing relational graph: {str(e)}")
                    # Fallback to identity relationships
                    self.client_relations = self.one_hot_seq

    def __train_forward__(self):
        """
        Forward pass during training - fixes gradient issues with label tensors
        """
        # Determine graph embedding source
        if hasattr(self, 'client_relations'):
            graph_embedding = self.client_relations.clone()
        else:
            graph_embedding = self.one_hot_seq.clone()
        
        # Use appropriate input data
        if self.generate and hasattr(self, 'task_ID') and self.task_ID > 0 and hasattr(self, 'x_seq_synthetic'):
            input_data = self.x_seq_synthetic
            self.x_seq_processed = self.x_seq_synthetic
        else:
            input_data = self.x_seq
            self.x_seq_processed = self.x_seq
        
        # Forward pass through networks - no gradient requirements on labels
        self.z_seq = self.netG(graph_embedding)
        
        # Don't clone labels with requires_grad
        self.e_seq= self.netE(input_data, self.y_seq, self.z_seq)
        self.e_seq_noised = add_laplace_noise(self.e_seq, self.opt.b)
        self.f_seq = self.netF(self.e_seq_noised)
        self.d_seq = self.netD(self.e_seq_noised.clone())

    def __test_forward__(self):
        """
        Forward pass during testing
        """
        # Create graph embedding
        if hasattr(self, 'client_relations'):
            graph_embedding = self.client_relations
        else:
            graph_embedding = self.one_hot_seq
            
        # Generate graph embeddings
        self.z_seq = self.netG(graph_embedding)
        
        # Encode the data
        self.e_seq = self.netE(self.x_seq, self.y_seq, self.z_seq)
        
        # Generate predictions
        self.f_seq, self.f_seq_softmax = self.netF(self.e_seq, return_softmax=True)
        
        # Get class predictions
        self.g_seq = torch.argmax(self.f_seq_softmax, dim=-1)

    def __loss_D__(self):
        """
        Discriminator loss calculation - with dimension checks and without in-place operations
        """
        # Get predicted embedding (from D)
        if self.d_seq.dim() > 2:
            # Flatten batch dimensions
            predicted = self.d_seq.reshape(-1, self.d_seq.size(-1))
        else:
            predicted = self.d_seq
        
        target = self.z_seq.repeat(predicted.size(0) // self.z_seq.size(0), 1)
        # Compute MSE loss without in-place operations        
        return F.mse_loss(predicted, target, reduction='mean')

    def __loss_EF__(self):
        """
        Encoder and predictor loss calculation - with dimension checks and without in-place operations
        """
        # Create a new loss computation to avoid in-place operations
        batch_size = self.e_seq.size(0) if self.e_seq.dim() <= 2 else self.e_seq.size(0) * self.e_seq.size(1)
        
        if self.z_seq.size(0) == 1 and batch_size > 1:
            target = self.z_seq.clone().expand(batch_size, -1)
        else:
            target = self.z_seq
        
        if self.d_seq.dim() > 2:
            predicted = self.d_seq.detach().reshape(-1, self.d_seq.size(-1))
        else:
            predicted = self.d_seq.detach()
        
        # Compute GAN loss as negative MSE
        criterion = nn.MSELoss()
        try:
            loss_E_gan = -criterion(predicted, target)
        except Exception as e:
            logger.error(f"Error computing GAN loss: {str(e)}")
            loss_E_gan = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Classification loss
        if self.f_seq.dim() <= 1:
            loss_E_pred = F.nll_loss(self.f_seq.unsqueeze(0), self.y_seq.unsqueeze(0))
        elif self.f_seq.dim() == 2:
            loss_E_pred = F.nll_loss(self.f_seq, self.y_seq.long())
        else:
            # Multi-dimensional case - flatten first
            f_flat = self.f_seq.reshape(-1, self.f_seq.size(-1))
            y_flat = self.y_seq.view(-1)
            loss_E_pred = F.nll_loss(f_flat, y_flat.long())
        

        loss_E = loss_E_gan * self.opt.lambda_gan + loss_E_pred

        return loss_E, loss_E_pred, loss_E_gan
    
    def get_weights(self):
        """
        Get the model weights as state dictionaries
        
        Returns:
            dict: Model weights
        """
        return {
            'encoder': copy.deepcopy(self.netE.state_dict()),
            'predictor': copy.deepcopy(self.netF.state_dict()),
            'discriminator': copy.deepcopy(self.netD.state_dict()),
            'graph_generator': copy.deepcopy(self.netG.state_dict())
        }
    
    def set_weights(self, weights):
        """
        Set the model weights from state dictionaries
        
        Args:
            weights: Dictionary containing model weights
        """
        if 'encoder' in weights:
            self.netE.load_state_dict(weights['autoencoder'])
        if 'predictor' in weights:
            self.netF.load_state_dict(weights['predictor'])
        if 'discriminator' in weights:
            self.netD.load_state_dict(weights['discriminator'])
        if 'graph_generator' in weights:
            self.netG.load_state_dict(weights['graph_generator'])
