import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from model.modules import *
import copy
import logging

logger = logging.getLogger('GFedCL')

class ModifiedClient(nn.Module):
    """
    Modified GFedCL Client that works with a centralized server discriminator
    
    Main changes:
    1. No local discriminator
    2. Uses server's discriminator for loss computation
    3. Sends encoded samples to server for discriminator training
    4. Can generate encodings without training for the server's discriminator
    """
    def __init__(self, client_id, opt):
        super(ModifiedClient, self).__init__()
        
        # set output format
        np.set_printoptions(suppress=True, precision=6)
        self.client_id = client_id
        self.opt = opt
        self.device = opt.device
        self.batch_size = opt.batch_size

        # visualization
        self.use_visdom = opt.use_visdom
        self.use_g_encode = opt.use_g_encode

        # Initialize the neural networks
        self.netE = FeatureEncoder(opt).to(opt.device)  # Encoder-decoder
        self.netF = PredNet(opt).to(opt.device)        # Classifier
        self.netG = GNet(opt).to(opt.device)           # Graph embedding generator 
        
        # Note: No local discriminator (netD) anymore
        
        # Initialize weights
        self.__init_weight__(self.netE)
        self.__init_weight__(self.netF)
        self.__init_weight__(self.netG)
        
        # Set up optimizers - only for encoder and predictor
        EF_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_EF = optim.Adam(
            EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999)
        )
        
        # Set up learning rate schedulers
        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_EF, gamma=0.5 ** (1 / 100)
        )

        self.lr_schedulers = [self.lr_scheduler_EF]
        
        # Define loss names for tracking
        self.loss_names = ["E_pred", "E_gan"]
            
        # Initialize tracking variables
        self.relational_graph = None
        self.task_ID = None
        self.server_discriminator = None  # Will hold the server's discriminator for inference
    
    def getId(self): 
        """Return client ID"""
        return self.client_id
        
    def set_server_discriminator(self, discriminator_state_dict):
        """
        Set the server's discriminator for loss computation
        
        Args:
            discriminator_state_dict: State dict of server's discriminator
        """
        if self.server_discriminator is None:
            # Initialize a local copy of the discriminator
            self.server_discriminator = GraphDNet(self.opt).to(self.device)
        
        # Load state dict
        self.server_discriminator.load_state_dict(discriminator_state_dict)
        
        # Set to eval mode since we don't train it locally
        self.server_discriminator.eval()
    
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
            dict: Loss values and encoded samples for server
        """
        # Ensure we have a server discriminator to compute losses
        if self.server_discriminator is None:
            logger.error(f"Client {self.client_id}: No server discriminator available for training")
            return None
        
        # Set model to training mode
        self.train()
            
        # Store the task ID and relational graph for use in training
        self.epoch = epoch
        self.generate = generate
        self.task_ID = task
        self.relational_graph = relational_graphs
        
        # Initialize loss tracking
        loss_values = {loss: 0 for loss in self.loss_names}
        count = 0
        
        # Lists to collect encoded samples and graph embeddings
        collected_encodings = []
        collected_graph_embeddings = []
        
        # Training loop
        for data in dataloader:
            count += 1
            
            # Set the input data - this will also create synthetic data if needed
            self.__set_input__(data, self.generate)
            
            # Forward pass
            self.__train_forward__()
            
            # Calculate losses and update weights for encoder and predictor
            new_loss_values = self.__optimize__()

            # Track loss values
            for key, loss in new_loss_values.items():
                loss_values[key] += loss
                
            # Collect encoded samples and graph embeddings for server training
            collected_encodings.append(self.e_seq.detach().clone())
            collected_graph_embeddings.append(self.z_seq.detach().clone())

        # Calculate average loss
        if count > 0:
            for key in loss_values.keys():
                loss_values[key] /= count

        # Log progress periodically
        status_msg = f"Client {self.client_id}, Task {task}, Epoch {self.epoch}"
        if self.generate:
            status_msg += " (with synthetic data)"
        status_msg += f": {loss_values}"
        logger.info(status_msg)

        # Apply learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
            
        # Return loss values and collected data for server training
        return {
            'loss_values': loss_values,
            'encodings': collected_encodings,
            'graph_embeddings': collected_graph_embeddings
        }
    
    def test(self, task_id, dataloader, relational_graphs):
        """
        Test the model on a dataset
        
        Args:
            task_id: Current task ID
            dataloader: DataLoader containing test samples
            relational_graphs: Task-indexed relational graphs used for evaluation
            
        Returns:
            dict: Dictionary containing metrics
        """
        self.eval()
        self.task_ID = task_id
        self.relational_graph = relational_graphs
        
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
        
        # Create synthetic data if needed (for continual learning)
        if generate and hasattr(self, 'task_ID') and self.task_ID > 0:
            # Create random noise with the same shape as flattened data
            noise = torch.randn_like(self.x_seq, device=self.device)
            self.x_seq_synthetic = noise

        if not isinstance(self.relational_graph, list):
            raise ValueError("relational_graph must be a task-indexed list")
        if self.task_ID is None or not 0 <= self.task_ID < len(self.relational_graph):
            raise ValueError(f"Missing relational graph for task {self.task_ID}")

        graph = np.array(
            self.relational_graph[self.task_ID], dtype=np.float32, copy=True
        )
        expected_shape = (self.opt.num_clients, self.opt.num_clients)
        if graph.shape != expected_shape or not np.isfinite(graph).all():
            raise ValueError(
                f"Task {self.task_ID} relational graph must be a finite matrix "
                f"with shape {expected_shape}; got {graph.shape}"
            )
        self.client_relations = torch.as_tensor(
            graph[self.client_id], device=self.device, dtype=torch.float32
        ).unsqueeze(0)

    def __train_forward__(self):
        """
        Forward pass during training - fixes gradient issues with label tensors
        """
        graph_embedding = self.client_relations.clone()
        
        # Use appropriate input data
        if self.generate and hasattr(self, 'task_ID') and self.task_ID > 0 and hasattr(self, 'x_seq_synthetic'):
            input_data = self.x_seq_synthetic
            self.x_seq_processed = self.x_seq_synthetic
        else:
            input_data = self.x_seq
            self.x_seq_processed = self.x_seq
        
        # Forward pass through networks
        self.z_seq = self.netG(graph_embedding)
        
        # Don't clone labels with requires_grad
        self.e_seq = self.netE(input_data, self.y_seq, self.z_seq)
        self.f_seq = self.netF(self.e_seq)
        
        # Now use the server discriminator for inference only
        with torch.no_grad():
            self.d_seq = self.server_discriminator(self.e_seq.clone())

    def __test_forward__(self):
        """
        Forward pass during testing
        """
        graph_embedding = self.client_relations
            
        # Generate graph embeddings
        self.z_seq = self.netG(graph_embedding)
        
        # Encode the data
        self.e_seq = self.netE(self.x_seq, self.y_seq, self.z_seq)
        
        # Generate predictions
        self.f_seq, self.f_seq_softmax = self.netF(self.e_seq, return_softmax=True)
        
        # Get class predictions
        self.g_seq = torch.argmax(self.f_seq_softmax, dim=-1)

    def __optimize__(self):
        """
        Optimize encoder and predictor models
        
        Returns:
            dict: Loss values
        """
        loss_value = dict()
        
        # We only optimize encoder and predictor, not the discriminator
        self.loss_E, self.loss_E_pred, self.loss_E_gan = self.__loss_EF__()
        self.optimizer_EF.zero_grad()
        self.loss_E.backward()
        self.optimizer_EF.step()
        
        loss_value["E_pred"] = self.loss_E_pred.item()
        loss_value["E_gan"] = self.loss_E_gan.item()
        
        return loss_value
        
    def generate_encodings(self, task, relational_graphs, dataloader, generate=False):
        # Ensure we have a server discriminator
        if self.server_discriminator is None:
            logger.error(f"Client {self.client_id}: No server discriminator available")
            return {'encodings': [], 'graph_embeddings': []}
        
        # Set model to eval mode to prevent batch norm, dropout effects
        self.eval()
            
        # Store the task ID and relational graph for use
        self.task_ID = task
        self.relational_graph = relational_graphs
        
        # Lists to collect encoded samples and graph embeddings
        collected_encodings = []
        collected_graph_embeddings = []
        
        # Forward pass through dataloader without training
        with torch.no_grad():
            for data in dataloader:
                # Set the input data
                self.__set_input__(data, generate)
                
                graph_embedding = self.client_relations.clone()
                
                # Ensure graph_embedding is float32
                if graph_embedding.dtype != torch.float32:
                    graph_embedding = graph_embedding.float()
                
                # Use appropriate input data
                if generate and hasattr(self, 'task_ID') and self.task_ID > 0 and hasattr(self, 'x_seq_synthetic'):
                    input_data = self.x_seq_synthetic
                else:
                    input_data = self.x_seq
                
                # Forward pass through networks
                self.z_seq = self.netG(graph_embedding)
                self.e_seq = self.netE(input_data, self.y_seq, self.z_seq)
                
                # Collect encoded samples and graph embeddings
                collected_encodings.append(self.e_seq.clone())
                collected_graph_embeddings.append(self.z_seq.clone())
        
        # Return to train mode
        self.train()
        
        # Return the collected data
        return {
            'encodings': collected_encodings,
            'graph_embeddings': collected_graph_embeddings
        }

    def __loss_EF__(self):
        """
        Encoder and predictor loss calculation
        """
        # Create a new loss computation
        batch_size = self.e_seq.size(0) if self.e_seq.dim() <= 2 else self.e_seq.size(0) * self.e_seq.size(1)
        
        if self.z_seq.size(0) == 1 and batch_size > 1:
            target = self.z_seq.clone().expand(batch_size, -1)
        else:
            target = self.z_seq
        
        if self.d_seq.dim() > 2:
            predicted = self.d_seq.reshape(-1, self.d_seq.size(-1))
        else:
            predicted = self.d_seq
        
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
            'graph_generator': copy.deepcopy(self.netG.state_dict())
        }
    
    def set_weights(self, weights):
        """
        Set the model weights from state dictionaries
        
        Args:
            weights: Dictionary containing model weights
        """
        if 'encoder' in weights:
            self.netE.load_state_dict(weights['encoder'])
        if 'predictor' in weights:
            self.netF.load_state_dict(weights['predictor'])
        if 'graph_generator' in weights and hasattr(self, 'netG'):
            self.netG.load_state_dict(weights['graph_generator'])
    
    def __init_weight__(self, net=None):
        """Initialize weights for the network"""
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)
