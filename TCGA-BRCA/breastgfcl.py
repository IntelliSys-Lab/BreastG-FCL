import torch
import numpy as np
import random
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
import copy
import csv
import ray

# Initialize Ray - ignore reinit error in case Ray is already running
ray.init(ignore_reinit_error=True)

from model.modules import BreastGraphGenerator
# from utils.dp_ks_analysis import DifferentialPrivacyAnalyzer
from utils.visualization_utils import *
from utils.log_utils import *
from utils.plot_utils import *
from utils.evaluation_utils import *
# from utils.communication_tracker import CommunicationTracker
# from utils.quality_evaluator import QualityEvaluator

# Import our Server and ModifiedClient classes
from model.server import Server
from model.client import ModifiedClient

logger = logging.getLogger('GFedCL')

def to_tensor(x, device="cuda"):
    """Convert numpy array or tensor to tensor on specified device"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x

def add_laplace_noise(data, scale):
    """Add Laplace noise to data (works with both tensors and numpy arrays)"""
    if isinstance(data, np.ndarray):
        # For numpy arrays
        noise = np.random.laplace(0, scale, data.shape)
        noise = np.array(noise, dtype=np.float32)
        noisy_data = data + noise
        return noisy_data
    else:
        # For tensors
        device = data.device if hasattr(data, 'device') else 'cuda'
        noise = np.random.laplace(0, scale, data.shape)
        noise = np.array(noise, dtype=np.float32)
        noisy_data = data + to_tensor(noise, device)
        return noisy_data

def add_laplace_noise_to_graph(relational_graph, scale, normalize=True):
    """
    Add Laplace noise to relational graph while maintaining graph properties
    
    Args:
        relational_graph: numpy array representing the graph
        scale: Scale parameter for Laplace noise
        normalize: Whether to normalize the graph after adding noise
        
    Returns:
        noisy_graph: Graph with added Laplace noise
    """
    # Add Laplace noise
    noisy_graph = add_laplace_noise(relational_graph, scale)
    
    # Ensure non-negative values (attention scores should be non-negative)
    noisy_graph = np.maximum(noisy_graph, 0)
    
    if normalize:
        # Normalize rows to sum to 1 (maintain attention property)
        row_sums = noisy_graph.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.maximum(row_sums, 1e-8)
        noisy_graph = noisy_graph / row_sums
    
    return noisy_graph.astype(np.float32)

@ray.remote
def generate_encodings_remote(client, task, relational_graphs, dataloader, generate_synthetic=False):
    """
    Generate encodings from a client in parallel
    """
    logger.info(f"Generating encodings from client {client.getId()} for task {task}")
    
    # Convert relational graphs to float32 if they're numpy arrays
    if isinstance(relational_graphs, list):
        for i in range(len(relational_graphs)):
            if isinstance(relational_graphs[i], np.ndarray):
                relational_graphs[i] = relational_graphs[i].astype(np.float32)
                
    result = client.generate_encodings(task, relational_graphs, dataloader, generate_synthetic)
    return result

@ray.remote
def train_client_remote(client, task, relational_graphs, dataloader, epochs, generate_synthetic=False):
    """
    Train a client in parallel
    
    Args:
        client: Client object
        task: Task ID
        relational_graphs: Relational graphs for all tasks
        dataloader: DataLoader for the client
        epochs: Number of epochs to train
        generate_synthetic: Whether to use synthetic samples
        
    Returns:
        dict: Client weights after training
    """
    logger.info(f"Training client {client.getId()} for task {task}")
    for epoch in range(epochs):
        result = client.learn(epoch, task, relational_graphs, dataloader, generate_synthetic)
    
    # Return client weights
    return client.get_weights()

@ray.remote
def test_client_remote(client, task, dataloader, relational_graphs):
    """
    Test a client in parallel
    
    Args:
        client: Client object
        task: Task ID  
        dataloader: DataLoader for testing
        relational_graphs: Task-indexed relational graphs
        
    Returns:
        dict: Test metrics
    """
    logger.info(f"Testing client {client.getId()} for task {task}")
    metrics = client.test(task, dataloader, relational_graphs)
    return metrics

def create_modified_clients(opt):
    """
    Create modified clients based on the given options
    
    Args:
        opt: Configuration options
        
    Returns:
        list: List of client objects
    """
    clients = []
    for i in range(opt.num_clients):
        client = ModifiedClient(i, opt)
        clients.append(client)
    
    return clients

class ParallelServerGFedCL:
    def __init__(self, opt):
        self.opt = opt
        # Handle device safely
        if torch.cuda.is_available() and opt.device == 'cuda':
            self.device = torch.device('cuda')
            # Print CUDA device info for debugging
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
            
        # Update opt.device to match actual device being used
        opt.device = str(self.device)
        
        # Initialize the server
        logger.info("Initializing server with global discriminator...")
        self.server = Server(opt)
        
        logger.info("Initializing BreastG-FCL disease-aware graph generator...")
        self.dygat = BreastGraphGenerator(opt).to(self.device)

        # Load and partition TCGA-BRCA dataset
        logger.info("Setting up TCGA-BRCA dataloaders...")
        from utils.dataset_utils import setup_tcga_brca_loaders
        self.dataloaders = setup_tcga_brca_loaders(opt)
        logger.info("TCGA-BRCA dataloaders prepared successfully")
        
        # Create modified clients
        logger.info("Creating modified clients...")
        self.clients = create_modified_clients(opt)
        logger.info(f"Created {len(self.clients)} clients")
        
        # Initialize DP analyzer if differential privacy is enabled
        # self.dp_analyzer = DifferentialPrivacyAnalyzer(
        # epsilon=self.opt.epsilon,
        # sensitivity=self.opt.sensitivity,
        # output_dir=os.path.join(self.opt.output_dir, 'dp_analysis')
        # )

        # Initialize communication tracker
        # self.comm_tracker = CommunicationTracker()
        # logger.info("Initialized communication tracker")

        # Initialize quality evaluator (FID/IS) if enabled
        # self.quality_evaluator = QualityEvaluator(opt)
        
    def _generate_relational_graph(self, task):
        """Generate and persist the complete BreastG-FCL task graph."""
        logger.info(
            "Generating relational graph: TCIA spatial attention + DCE temporal attention"
        )
        spatial_by_task = getattr(self.opt, "client_spatial_features", None)
        temporal_by_task = getattr(self.opt, "client_temporal_features", None)
        if spatial_by_task is None:
            raise RuntimeError(
                "TCIA spatial summaries are required; provide --tcia-mri-features-path"
            )
        if temporal_by_task is None:
            raise RuntimeError(
                "DCE temporal summaries are required; provide --tcia-dce-kinetics-path"
            )

        clean_graph = self.dygat.learn(
            self.opt.gat_epochs,
            model_updates=None,
            task_id=task,
            spatial_features=spatial_by_task[task],
            temporal_features=temporal_by_task[task],
        )
        expected_shape = (self.opt.num_clients, self.opt.num_clients)
        if clean_graph.shape != expected_shape:
            raise ValueError(
                f"Relational graph must have shape {expected_shape}, got {clean_graph.shape}"
            )
        if not np.isfinite(clean_graph).all():
            raise ValueError("Relational graph contains non-finite values")

        graph_dir = os.path.join(self.opt.output_dir, "relational_graphs")
        os.makedirs(graph_dir, exist_ok=True)
        artifacts = {
            "fused": clean_graph,
            "spatial": self.dygat.last_spatial_attention,
            "temporal": self.dygat.last_temporal_patterns,
            "temporal_window": self.dygat.last_temporal_window,
        }
        for name, values in artifacts.items():
            np.save(
                os.path.join(graph_dir, f"task_{task + 1}_{name}.npy"),
                values,
            )

        logger.info(
            "Adding Laplace noise to relational graph with scale %s",
            self.opt.b,
        )
        private_graph = add_laplace_noise_to_graph(
            clean_graph,
            scale=self.opt.b,
            normalize=True,
        )
        noise_magnitude = np.abs(private_graph - clean_graph)
        logger.info(
            "Noise statistics - Mean: %.4f, Max: %.4f, Std: %.4f",
            np.mean(noise_magnitude),
            np.max(noise_magnitude),
            np.std(noise_magnitude),
        )
        return private_graph

    def train_GFedCL(self):
        logger.info('Starting Parallel Server-based GFedCL training for TCGA-BRCA...')
        
        relational_graphs = [None for _ in range(self.opt.num_task)]
        
        # Track accuracy for each round and task
        round_accuracy = []
        round_labels = []
        
        # NEW: Track test accuracy on all previous tasks during each round
        all_tasks_accuracy = []  # Will store data for all rounds and all tasks
        
        # Training loop for each task
        for task in range(self.opt.num_task):
            logger.info(f'Training for task {task+1}/{self.opt.num_task}')
            
            # Clear CUDA cache between tasks if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            relational_graphs[task] = self._generate_relational_graph(task)
    
            # Main training loop with the server and modified clients
            for r in range(self.opt.num_rounds):
                logger.info(f'Round {r+1}/{self.opt.num_rounds}')

                # Start tracking this round
                # self.comm_tracker.start_round(task, r)
                
                # PHASE 1: Collect encodings from all clients with current encoders (no training)
                logger.info(f'Phase 1: Collecting encodings from all clients in parallel')
                all_encodings = []
                all_graph_embeddings = []
                
                # First, distribute the server's current discriminator to all clients
                server_discriminator = self.server.get_discriminator()
                for client in self.clients:
                    client.set_server_discriminator(server_discriminator)
                    # self.comm_tracker.add_model_weights_communication(
                    #     server_discriminator,
                    #     direction='download',
                    #     model_type='discriminator_weights'
                    # )
                
                # Collect encodings from each client without training (batched)
                encoding_results = []

                # Launch encoding generation for current task
                for i in range(0, len(self.clients), self.opt.ray_max_in_flight):
                    batch_clients = self.clients[i : i + self.opt.ray_max_in_flight]
                    futures = []
                    for j, client in enumerate(batch_clients):
                        client_id = i + j
                        futures.append(
                            generate_encodings_remote.options(
                                num_gpus=self.opt.ray_num_gpus_per_task,
                                num_cpus=self.opt.ray_num_cpus_per_task,
                            ).remote(
                                client,
                                task,
                                relational_graphs,
                                self.dataloaders[client_id][task]['train'],
                                False,
                            )
                        )
                    encoding_results.extend(ray.get(futures))

                # Launch encoding generation for previous task if applicable
                if self.opt.replay:
                    logger.info(f'Using replay for previous task {task-1} encodings')
                    if task >= 1:
                        for i in range(0, len(self.clients), self.opt.ray_max_in_flight):
                            batch_clients = self.clients[i : i + self.opt.ray_max_in_flight]
                            futures = []
                            for j, client in enumerate(batch_clients):
                                client_id = i + j
                                futures.append(
                                    generate_encodings_remote.options(
                                        num_gpus=self.opt.ray_num_gpus_per_task,
                                        num_cpus=self.opt.ray_num_cpus_per_task,
                                    ).remote(
                                        client,
                                        task - 1,
                                        relational_graphs,
                                        self.dataloaders[client_id][task - 1]['train'],
                                        True,
                                    )
                                )
                            encoding_results.extend(ray.get(futures))
                
                # Process results
                for result in encoding_results:
                    all_encodings.extend(result['encodings'])
                    all_graph_embeddings.extend(result['graph_embeddings'])

                # Track encodings uploaded to server
                # self.comm_tracker.add_encodings_communication(
                #     all_encodings,
                #     all_graph_embeddings
                # )
                
                # PHASE 2: Train the server's discriminator with collected encodings
                logger.info(f'Phase 2: Training server discriminator with {len(all_encodings)} samples')
                if len(all_encodings) > 0:
                    # Apply Laplace noise to the encodings for differential privacy
                    all_encodings_noised = []
                    for encoding in all_encodings:
                        # Make sure the encoding is on the correct device
                        encoding = encoding.to(self.device)
                        # Add Laplace noise with scale parameter self.opt.b
                        noised_encoding = add_laplace_noise(encoding, self.opt.b)
                        all_encodings_noised.append(noised_encoding)
                    server_loss = self.server.train_discriminator(all_encodings_noised, all_graph_embeddings)
                    logger.info(f'Server discriminator loss: {server_loss:.4f}')
                else:
                    logger.warning("No encoded samples collected for server training")
                
                # PHASE 3: Distribute the updated discriminator to clients and train them in parallel
                logger.info(f'Phase 3: Training clients with updated server discriminator in parallel')
                
                # Distribute the updated discriminator to all clients
                server_discriminator = self.server.get_discriminator()
                for client in self.clients:
                    client.set_server_discriminator(server_discriminator)
                    # self.comm_tracker.add_model_weights_communication(
                    #     server_discriminator,
                    #     direction='download',
                    #     model_type='discriminator_weights'
                    # )
                
                # Train clients with the updated discriminator in parallel (batched)
                logger.info(f'Waiting for client training to complete...')
                training_results = []

                # Launch client training for current task
                for i in range(0, len(self.clients), self.opt.ray_max_in_flight):
                    batch_clients = self.clients[i : i + self.opt.ray_max_in_flight]
                    futures = []
                    for j, client in enumerate(batch_clients):
                        client_id = i + j
                        futures.append(
                            train_client_remote.options(
                                num_gpus=self.opt.ray_num_gpus_per_task,
                                num_cpus=self.opt.ray_num_cpus_per_task,
                            ).remote(
                                client,
                                task,
                                relational_graphs,
                                self.dataloaders[client_id][task]['train'],
                                self.opt.num_local_epochs,
                                False,
                            )
                        )
                    training_results.extend(ray.get(futures))

                if self.opt.replay:
                    logger.info(f'Using replay for previous task {task-1} training')
                    if task >= 1:
                        for i in range(0, len(self.clients), self.opt.ray_max_in_flight):
                            batch_clients = self.clients[i : i + self.opt.ray_max_in_flight]
                            futures = []
                            for j, client in enumerate(batch_clients):
                                client_id = i + j
                                futures.append(
                                    train_client_remote.options(
                                        num_gpus=self.opt.ray_num_gpus_per_task,
                                        num_cpus=self.opt.ray_num_cpus_per_task,
                                    ).remote(
                                        client,
                                        task - 1,
                                        relational_graphs,
                                        self.dataloaders[client_id][task - 1]['train'],
                                        self.opt.num_local_epochs,
                                        True,
                                    )
                                )
                            ray.get(futures)
                
                # Process results and extract weights
                encoder_weights = []
                predictor_weights = []
                
                for result in training_results:
                    encoder_weights.append(result['encoder'])
                    predictor_weights.append(result['predictor'])
                    # self.comm_tracker.add_model_weights_communication(
                    #     result['encoder'],
                    #     direction='upload',
                    #     model_type='encoder_weights'
                    # )
                    # self.comm_tracker.add_model_weights_communication(
                    #     result['predictor'],
                    #     direction='upload',
                    #     model_type='predictor_weights'
                    # )
                
                # Update server's learning rate
                self.server.update_learning_rate()
                
                # Average weights across clients using utility function from server_utils.py
                from utils.server_utils import average_weights
                global_encoder = average_weights(encoder_weights)
                global_predictor = average_weights(predictor_weights)

                # Update the local models with the averaged weights
                for client in self.clients:
                    client.set_weights({
                        'encoder': global_encoder,
                        'predictor': global_predictor
                    })
                    # self.comm_tracker.add_model_weights_communication(
                    #     global_encoder,
                    #     direction='download',
                    #     model_type='encoder_weights'
                    # )
                    # self.comm_tracker.add_model_weights_communication(
                    #     global_predictor,
                    #     direction='download',
                    #     model_type='predictor_weights'
                    # )

                # End round tracking
                # self.comm_tracker.end_round()
                
                # Evaluate current performance for all clients on this task in parallel
                logger.info(f'Evaluating clients for task {task+1} in parallel...')
                test_results = []
                for i in range(0, len(self.clients), self.opt.ray_max_in_flight):
                    batch_clients = self.clients[i : i + self.opt.ray_max_in_flight]
                    futures = []
                    for j, client in enumerate(batch_clients):
                        client_id = i + j
                        futures.append(
                            test_client_remote.options(
                                num_gpus=self.opt.ray_num_gpus_per_task,
                                num_cpus=self.opt.ray_num_cpus_per_task,
                            ).remote(
                                client,
                                task,
                                self.dataloaders[client_id][task]['test'],
                                relational_graphs,
                            )
                        )
                    test_results.extend(ray.get(futures))
                task_accuracies = [result["acc"] for result in test_results]
                
                # Calculate average accuracy across all clients for this round
                avg_accuracy = sum(task_accuracies) / len(task_accuracies)
                round_accuracy.append(avg_accuracy)
                round_labels.append(f"Task {task+1}, Round {r+1}")
                
                logger.info(f"Task {task+1}, Round {r+1} - Average Accuracy: {avg_accuracy:.2f}%")
                
                # NEW: Collect data for all tasks (including current and previous) during this round
                round_all_tasks_data = {
                    'round': f"Task {task+1}, Round {r+1}",
                    'current_task': task,
                    'round_number': r+1,
                    'tasks': {}
                }
                
                # Add current task accuracy
                round_all_tasks_data['tasks'][task] = avg_accuracy
                
                # Evaluate on all previous tasks for this round
                if task > 0:
                    logger.info(f"Evaluating performance on previous tasks during Task {task+1}, Round {r+1}...")
                    
                    for prev_task in range(task):
                        prev_task_results = []
                        for i in range(0, len(self.clients), self.opt.ray_max_in_flight):
                            batch_clients = self.clients[i : i + self.opt.ray_max_in_flight]
                            futures = []
                            for j, client in enumerate(batch_clients):
                                client_id = i + j
                                if prev_task in self.dataloaders[client_id]:
                                    futures.append(
                                        test_client_remote.options(
                                            num_gpus=self.opt.ray_num_gpus_per_task,
                                            num_cpus=self.opt.ray_num_cpus_per_task,
                                        ).remote(
                                            client,
                                            prev_task,
                                            self.dataloaders[client_id][prev_task]['test'],
                                            relational_graphs,
                                        )
                                    )

                            if futures:
                                prev_task_results.extend(ray.get(futures))
                        
                        # Collect test results for previous task
                        if prev_task_results:
                            prev_task_accuracies = [result["acc"] for result in prev_task_results]
                            avg_prev_accuracy = sum(prev_task_accuracies) / len(prev_task_accuracies)
                            
                            # Log and store the accuracy on this previous task
                            logger.info(f"  Task {task+1}, Round {r+1} - Accuracy on previous Task {prev_task+1}: {avg_prev_accuracy:.2f}%")
                            round_all_tasks_data['tasks'][prev_task] = avg_prev_accuracy
                
                # Add this round's data to our tracking
                all_tasks_accuracy.append(round_all_tasks_data)

                # Optional FID/IS evaluations after each round
                # self.quality_evaluator.evaluate_round(
                #     task, r, self.clients, self.dataloaders, relational_graphs
                # )

        # Save round accuracy to CSV
        csv_path = os.path.join(self.opt.output_dir, 'round_accuracy.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Round', 'Average Accuracy'])
            for i, label in enumerate(round_labels):
                writer.writerow([label, f"{round_accuracy[i]:.2f}"])
        
        logger.info(f"Saved round accuracy data to {csv_path}")
        
        # NEW: Save all tasks accuracy to CSV
        all_tasks_csv_path = os.path.join(self.opt.output_dir, 'all_tasks_accuracy.csv')
        with open(all_tasks_csv_path, 'w', newline='') as csvfile:
            # Define CSV header
            fieldnames = ['Round', 'Current Task']
            # Add columns for each task
            for t in range(self.opt.num_task):
                fieldnames.append(f'Task {t+1} Accuracy')
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data for each round
            for round_data in all_tasks_accuracy:
                row = {
                    'Round': round_data['round'],
                    'Current Task': round_data['current_task'] + 1  # Add 1 for 1-based indexing
                }
                
                # Add accuracy for each task
                for t in range(self.opt.num_task):
                    if t in round_data['tasks']:
                        row[f'Task {t+1} Accuracy'] = f"{round_data['tasks'][t]:.2f}"
                    else:
                        row[f'Task {t+1} Accuracy'] = ""
                
                writer.writerow(row)
        
        logger.info(f"Saved all tasks accuracy data to {all_tasks_csv_path}")

        # Save communication overhead data
        # comm_csv_path = os.path.join(self.opt.output_dir, 'communication_overhead.csv')
        # self.comm_tracker.save_to_csv(comm_csv_path)
        # logger.info(f"Saved communication overhead data to {comm_csv_path}")
        
        # Plot communication overhead
        # comm_plots_dir = os.path.join(self.opt.output_dir, 'communication_plots')
        # self.comm_tracker.plot_communication_overhead(comm_plots_dir)
        # logger.info(f"Saved communication overhead plots to {comm_plots_dir}")
        
        # comm_summary = self.comm_tracker.get_summary()
        # logger.info("===== COMMUNICATION SUMMARY =====")
        # for key, value in comm_summary.items():
        #     logger.info(f"{key}: {value}")
        # logger.info("===============================")
        
        # Test accuracy after training all tasks
        logger.info("Evaluating final model accuracy...")
        all_tasks_acc = evaluate_all_tasks(
            self.opt, self.clients, self.dataloaders, relational_graphs
        )
        
        # quality_summary = self.quality_evaluator.finalize()
        quality_summary = {}

        return all_tasks_acc, all_tasks_accuracy, quality_summary
    
    def visualize_attention_components(self, task, spatial_attention, temporal_patterns, combined_attention):
        """
        Visualize the components of the attention mechanism
        
        Args:
            task: Current task ID
            spatial_attention: Spatial attention matrix
            temporal_patterns: Temporal pattern similarity matrix (can be None)
            combined_attention: Combined attention matrix
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        # Create output directory if it doesn't exist
        vis_dir = os.path.join(self.opt.output_dir, 'attention_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Create figure
        if temporal_patterns is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot spatial attention
            sns.heatmap(spatial_attention, ax=axes[0], cmap='viridis', annot=False)
            axes[0].set_title(f'Task {task+1}: Spatial Attention')
            
            # Plot temporal patterns
            sns.heatmap(temporal_patterns, ax=axes[1], cmap='viridis', annot=False)
            axes[1].set_title(f'Task {task+1}: Temporal Patterns')
            
            # Plot combined attention
            sns.heatmap(combined_attention, ax=axes[2], cmap='viridis', annot=False)
            axes[2].set_title(f'Task {task+1}: Combined Attention')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Plot spatial attention
            sns.heatmap(spatial_attention, ax=axes[0], cmap='viridis', annot=False)
            axes[0].set_title(f'Task {task+1}: Spatial Attention')
            
            # Plot combined attention (same as spatial in this case)
            sns.heatmap(combined_attention, ax=axes[1], cmap='viridis', annot=False)
            axes[1].set_title(f'Task {task+1}: Combined Attention')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'attention_components_task{task+1}.png'), dpi=300)
        plt.close()
        
        logger.info(f"Task {task+1}: Saved attention component visualization")
    
    def visualize_dp_comparison(self, original_graphs, noisy_graphs):
        """
        Create a visualization comparing original and noisy relational graphs
        
        Args:
            original_graphs: List of original relational graphs
            noisy_graphs: List of relational graphs with Laplace noise
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        vis_dir = os.path.join(self.opt.output_dir, 'dp_comparison')
        os.makedirs(vis_dir, exist_ok=True)
        
        for task in range(self.opt.num_task):
            if original_graphs[task] is None:
                continue
                
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original graph
            sns.heatmap(original_graphs[task], ax=axes[0], cmap='viridis', 
                       vmin=0, vmax=1, annot=False)
            axes[0].set_title(f'Task {task+1}: Original Graph')
            
            # Noisy graph
            sns.heatmap(noisy_graphs[task], ax=axes[1], cmap='viridis', 
                       vmin=0, vmax=1, annot=False)
            axes[1].set_title(f'Task {task+1}: Graph with Laplace Noise (ε={self.opt.epsilon})')
            
            # Difference
            diff = np.abs(original_graphs[task] - noisy_graphs[task])
            sns.heatmap(diff, ax=axes[2], cmap='Reds', annot=False)
            axes[2].set_title(f'Task {task+1}: Absolute Difference')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'dp_comparison_task{task+1}.png'), dpi=300)
            plt.close()
            
        logger.info(f"Saved DP comparison visualizations to {vis_dir}")
