import torch
import numpy as np
import logging
import os
import sys

logger = logging.getLogger('GFedCL')

class CommunicationTracker:
    """
    Track communication overhead in federated learning
    """
    def __init__(self):
        self.round_data = []
        self.current_round_data = {
            'upload': 0,  # Client to server
            'download': 0,  # Server to client
            'upload_details': {
                'encodings': 0,
                'graph_embeddings': 0,
                'encoder_weights': 0,
                'predictor_weights': 0,
                'discriminator_weights': 0,
                'graph_generator_weights': 0,
                'classifier_updates': 0,
                'other': 0
            },
            'download_details': {
                'discriminator_weights': 0,
                'encoder_weights': 0,
                'predictor_weights': 0,
                'graph_generator_weights': 0,
                'relational_graph': 0,
                'other': 0
            }
        }
    
    def start_round(self, task_id, round_id):
        """Start tracking a new round"""
        self.current_round_data = {
            'task_id': task_id,
            'round_id': round_id,
            'upload': 0,
            'download': 0,
            'upload_details': {
                'encodings': 0,
                'graph_embeddings': 0,
                'encoder_weights': 0,
                'predictor_weights': 0,
                'discriminator_weights': 0,
                'graph_generator_weights': 0,
                'classifier_updates': 0,
                'other': 0
            },
            'download_details': {
                'discriminator_weights': 0,
                'encoder_weights': 0,
                'predictor_weights': 0,
                'graph_generator_weights': 0,
                'relational_graph': 0,
                'other': 0
            }
        }
    
    def add_model_updates_communication(self, model_updates, num_clients):
        """
        Add communication overhead for model updates sent to GAT for relational graph generation
        
        Args:
            model_updates: Dictionary mapping client IDs to their model updates
            num_clients: Total number of clients
        """
        total_size = 0
        
        for client_id in range(num_clients):
            client_updates = model_updates[client_id]
            
            # Calculate size of updates for this client
            client_update_size = 0
            for param_name, param_tensor in client_updates.items():
                if isinstance(param_tensor, torch.Tensor):
                    client_update_size += param_tensor.element_size() * param_tensor.nelement()
                elif isinstance(param_tensor, np.ndarray):
                    client_update_size += param_tensor.itemsize * param_tensor.size
            
            total_size += client_update_size
        
        # These updates are sent from clients to server for GAT processing
        self.current_round_data['upload'] += total_size
        self.current_round_data['upload_details']['classifier_updates'] += total_size
        
        logger.info(f"Added {total_size / (1024*1024):.2f} MB of classifier updates communication")

    def add_tensor_communication(self, tensor, direction='upload', category='other'):
        """
        Add communication overhead for a tensor
        
        Args:
            tensor: PyTorch tensor or numpy array
            direction: 'upload' (client->server) or 'download' (server->client)
            category: Type of data being communicated
        """
        if tensor is None:
            return
            
        # Calculate size in bytes
        if isinstance(tensor, torch.Tensor):
            # PyTorch tensor: element_size() * number of elements
            size_bytes = tensor.element_size() * tensor.nelement()
        elif isinstance(tensor, np.ndarray):
            # NumPy array: itemsize * size
            size_bytes = tensor.itemsize * tensor.size
        elif isinstance(tensor, (list, tuple)):
            # List of tensors
            size_bytes = sum(self._get_tensor_size(t) for t in tensor)
        else:
            # For other types, estimate using sys.getsizeof
            import sys
            size_bytes = sys.getsizeof(tensor)
        
        # Update tracking
        self.current_round_data[direction] += size_bytes
        
        # Update detailed tracking based on direction
        if direction == 'upload' and category in self.current_round_data['upload_details']:
            self.current_round_data['upload_details'][category] += size_bytes
        elif direction == 'download' and category in self.current_round_data['download_details']:
            self.current_round_data['download_details'][category] += size_bytes
    
    def add_model_weights_communication(self, state_dict, direction='upload', model_type='model_weights'):
        """
        Add communication overhead for model weights
        
        Args:
            state_dict: Model state dictionary
            direction: 'upload' or 'download'
            model_type: Type of model weights (e.g., 'encoder_weights', 'predictor_weights', etc.)
        """
        total_size = 0
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                size_bytes = value.element_size() * value.nelement()
                total_size += size_bytes
        
        self.current_round_data[direction] += total_size
        
        # Update detailed tracking based on direction
        if direction == 'upload' and model_type in self.current_round_data['upload_details']:
            self.current_round_data['upload_details'][model_type] += total_size
        elif direction == 'download' and model_type in self.current_round_data['download_details']:
            self.current_round_data['download_details'][model_type] += total_size
    
    def add_encodings_communication(self, encodings, graph_embeddings):
        """
        Add communication overhead for encodings and graph embeddings
        
        Args:
            encodings: List of encoding tensors
            graph_embeddings: List of graph embedding tensors
        """
        # Calculate encodings size
        encodings_size = 0
        for encoding in encodings:
            if isinstance(encoding, torch.Tensor):
                encodings_size += encoding.element_size() * encoding.nelement()
        
        # Calculate graph embeddings size
        embeddings_size = 0
        for embedding in graph_embeddings:
            if isinstance(embedding, torch.Tensor):
                embeddings_size += embedding.element_size() * embedding.nelement()
        
        # These are uploaded from clients to server
        self.current_round_data['upload'] += encodings_size + embeddings_size
        self.current_round_data['upload_details']['encodings'] += encodings_size
        self.current_round_data['upload_details']['graph_embeddings'] += embeddings_size
    
    def end_round(self):
        """Finish tracking the current round and store results"""
        # Convert bytes to MB for easier reading
        round_data_mb = {
            'task_id': self.current_round_data['task_id'],
            'round_id': self.current_round_data['round_id'],
            'upload_mb': self.current_round_data['upload'] / (1024 * 1024),
            'download_mb': self.current_round_data['download'] / (1024 * 1024),
            'total_mb': (self.current_round_data['upload'] + self.current_round_data['download']) / (1024 * 1024),
            'upload_details': {
                k: v / (1024 * 1024) for k, v in self.current_round_data['upload_details'].items()
            },
            'download_details': {
                k: v / (1024 * 1024) for k, v in self.current_round_data['download_details'].items()
            }
        }
        
        self.round_data.append(round_data_mb)
        
        # Log the communication overhead with detailed breakdown
        logger.info(f"Communication overhead for Task {round_data_mb['task_id']+1}, "
                   f"Round {round_data_mb['round_id']+1}:")
        logger.info(f"  Total: {round_data_mb['total_mb']:.2f} MB")
        
        # Log upload details
        logger.info(f"  Upload ({round_data_mb['upload_mb']:.2f} MB):")
        for category, size in round_data_mb['upload_details'].items():
            if size > 0:
                logger.info(f"    - {category}: {size:.2f} MB")
        
        # Log download details
        logger.info(f"  Download ({round_data_mb['download_mb']:.2f} MB):")
        for category, size in round_data_mb['download_details'].items():
            if size > 0:
                logger.info(f"    - {category}: {size:.2f} MB")
        
        return round_data_mb
    
    def get_summary(self):
        """Get summary statistics of communication overhead"""
        if not self.round_data:
            return None
        
        total_upload = sum(r['upload_mb'] for r in self.round_data)
        total_download = sum(r['download_mb'] for r in self.round_data)
        total_communication = sum(r['total_mb'] for r in self.round_data)
        
        # Calculate per-task statistics
        task_stats = {}
        for round_data in self.round_data:
            task_id = round_data['task_id']
            if task_id not in task_stats:
                task_stats[task_id] = {
                    'upload_mb': 0,
                    'download_mb': 0,
                    'total_mb': 0,
                    'rounds': 0
                }
            
            task_stats[task_id]['upload_mb'] += round_data['upload_mb']
            task_stats[task_id]['download_mb'] += round_data['download_mb']
            task_stats[task_id]['total_mb'] += round_data['total_mb']
            task_stats[task_id]['rounds'] += 1
        
        # Calculate category breakdown for uploads and downloads
        upload_category_totals = {}
        download_category_totals = {}
        
        for round_data in self.round_data:
            for category, size_mb in round_data['upload_details'].items():
                if category not in upload_category_totals:
                    upload_category_totals[category] = 0
                upload_category_totals[category] += size_mb
                
            for category, size_mb in round_data['download_details'].items():
                if category not in download_category_totals:
                    download_category_totals[category] = 0
                download_category_totals[category] += size_mb
        
        summary = {
            'total_upload_mb': total_upload,
            'total_download_mb': total_download,
            'total_communication_mb': total_communication,
            'avg_per_round_mb': total_communication / len(self.round_data) if self.round_data else 0,
            'task_statistics': task_stats,
            'upload_breakdown': upload_category_totals,
            'download_breakdown': download_category_totals,
            'num_rounds': len(self.round_data)
        }
        
        return summary
    
    def save_to_csv(self, filepath):
        """Save communication data to CSV file with detailed breakdown"""
        import csv
        
        with open(filepath, 'w', newline='') as csvfile:
            # Extended fieldnames for detailed tracking
            fieldnames = ['Task', 'Round', 'Upload_MB', 'Download_MB', 'Total_MB',
                         # Upload details
                         'Upload_Encodings_MB', 'Upload_Graph_Embeddings_MB', 
                         'Upload_Encoder_Weights_MB', 'Upload_Predictor_Weights_MB',
                         'Upload_Discriminator_Weights_MB', 'Upload_Graph_Generator_Weights_MB',
                         'Upload_Classifier_Updates_MB', 'Upload_Other_MB',
                         # Download details
                         'Download_Discriminator_Weights_MB', 'Download_Encoder_Weights_MB',
                         'Download_Predictor_Weights_MB', 'Download_Graph_Generator_Weights_MB',
                         'Download_Relational_Graph_MB', 'Download_Other_MB']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for round_data in self.round_data:
                row = {
                    'Task': round_data['task_id'] + 1,
                    'Round': round_data['round_id'] + 1,
                    'Upload_MB': f"{round_data['upload_mb']:.2f}",
                    'Download_MB': f"{round_data['download_mb']:.2f}",
                    'Total_MB': f"{round_data['total_mb']:.2f}",
                    # Upload details
                    'Upload_Encodings_MB': f"{round_data['upload_details'].get('encodings', 0):.2f}",
                    'Upload_Graph_Embeddings_MB': f"{round_data['upload_details'].get('graph_embeddings', 0):.2f}",
                    'Upload_Encoder_Weights_MB': f"{round_data['upload_details'].get('encoder_weights', 0):.2f}",
                    'Upload_Predictor_Weights_MB': f"{round_data['upload_details'].get('predictor_weights', 0):.2f}",
                    'Upload_Discriminator_Weights_MB': f"{round_data['upload_details'].get('discriminator_weights', 0):.2f}",
                    'Upload_Graph_Generator_Weights_MB': f"{round_data['upload_details'].get('graph_generator_weights', 0):.2f}",
                    'Upload_Classifier_Updates_MB': f"{round_data['upload_details'].get('classifier_updates', 0):.2f}",
                    'Upload_Other_MB': f"{round_data['upload_details'].get('other', 0):.2f}",
                    # Download details
                    'Download_Discriminator_Weights_MB': f"{round_data['download_details'].get('discriminator_weights', 0):.2f}",
                    'Download_Encoder_Weights_MB': f"{round_data['download_details'].get('encoder_weights', 0):.2f}",
                    'Download_Predictor_Weights_MB': f"{round_data['download_details'].get('predictor_weights', 0):.2f}",
                    'Download_Graph_Generator_Weights_MB': f"{round_data['download_details'].get('graph_generator_weights', 0):.2f}",
                    'Download_Relational_Graph_MB': f"{round_data['download_details'].get('relational_graph', 0):.2f}",
                    'Download_Other_MB': f"{round_data['download_details'].get('other', 0):.2f}"
                }
                writer.writerow(row)
        
        logger.info(f"Saved detailed communication overhead data to {filepath}")
    
    def plot_communication_overhead(self, output_dir):
        """Create visualizations of communication overhead with detailed breakdown"""
        import matplotlib.pyplot as plt
        import os
        
        if not self.round_data:
            logger.warning("No communication data to plot")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for plotting
        rounds = list(range(1, len(self.round_data) + 1))
        uploads = [r['upload_mb'] for r in self.round_data]
        downloads = [r['download_mb'] for r in self.round_data]
        totals = [r['total_mb'] for r in self.round_data]
        
        # Plot 1: Communication over rounds
        plt.figure(figsize=(12, 6))
        plt.plot(rounds, uploads, 'b-o', label='Upload (Client→Server)', linewidth=2, markersize=8)
        plt.plot(rounds, downloads, 'r-s', label='Download (Server→Client)', linewidth=2, markersize=8)
        plt.plot(rounds, totals, 'g-^', label='Total', linewidth=2, markersize=8)
        
        plt.xlabel('Communication Round', fontsize=12)
        plt.ylabel('Data Volume (MB)', fontsize=12)
        plt.title('Communication Overhead per Round', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add task boundaries
        current_task = self.round_data[0]['task_id']
        for i, round_data in enumerate(self.round_data[1:], 1):
            if round_data['task_id'] != current_task:
                plt.axvline(x=i + 0.5, color='gray', linestyle='--', alpha=0.5)
                current_task = round_data['task_id']
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'communication_per_round.png'), dpi=300)
        plt.close()
        
        # Plot 2: Upload breakdown by category
        summary = self.get_summary()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Upload breakdown
        upload_categories = list(summary['upload_breakdown'].keys())
        upload_sizes = list(summary['upload_breakdown'].values())
        
        colors1 = plt.cm.Set3(range(len(upload_categories)))
        ax1.pie(upload_sizes, labels=upload_categories, colors=colors1, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Upload Communication Breakdown', fontsize=14)
        
        # Download breakdown
        download_categories = list(summary['download_breakdown'].keys())
        download_sizes = list(summary['download_breakdown'].values())
        
        colors2 = plt.cm.Pastel1(range(len(download_categories)))
        ax2.pie(download_sizes, labels=download_categories, colors=colors2, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Download Communication Breakdown', fontsize=14)
        
        plt.suptitle('Communication Breakdown by Category', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'communication_breakdown.png'), dpi=300)
        plt.close()
        
        # Plot 3: Stacked bar chart showing detailed breakdown per round
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Prepare data for stacked bar chart
        upload_data = {cat: [] for cat in ['encodings', 'graph_embeddings', 'encoder_weights', 
                                           'predictor_weights', 'discriminator_weights', 'other']}
        download_data = {cat: [] for cat in ['discriminator_weights', 'encoder_weights', 
                                            'predictor_weights', 'relational_graph', 'other']}
        
        for round_data in self.round_data:
            for cat in upload_data:
                upload_data[cat].append(round_data['upload_details'].get(cat, 0))
            for cat in download_data:
                download_data[cat].append(round_data['download_details'].get(cat, 0))
        
        # Upload stacked bar
        bottom = np.zeros(len(rounds))
        for cat, values in upload_data.items():
            ax1.bar(rounds, values, bottom=bottom, label=cat)
            bottom += np.array(values)
        
        ax1.set_xlabel('Communication Round', fontsize=12)
        ax1.set_ylabel('Upload Volume (MB)', fontsize=12)
        ax1.set_title('Upload Breakdown per Round', fontsize=14)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Download stacked bar
        bottom = np.zeros(len(rounds))
        for cat, values in download_data.items():
            ax2.bar(rounds, values, bottom=bottom, label=cat)
            bottom += np.array(values)
        
        ax2.set_xlabel('Communication Round', fontsize=12)
        ax2.set_ylabel('Download Volume (MB)', fontsize=12)
        ax2.set_title('Download Breakdown per Round', fontsize=14)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Detailed Communication Breakdown per Round', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'communication_stacked.png'), dpi=300)
        plt.close()
        
        # Plot 4: Per-task summary
        task_ids = sorted(summary['task_statistics'].keys())
        task_uploads = [summary['task_statistics'][t]['upload_mb'] for t in task_ids]
        task_downloads = [summary['task_statistics'][t]['download_mb'] for t in task_ids]
        
        x = np.arange(len(task_ids))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width/2, task_uploads, width, label='Upload', color='skyblue')
        plt.bar(x + width/2, task_downloads, width, label='Download', color='lightcoral')
        
        plt.xlabel('Task', fontsize=12)
        plt.ylabel('Total Data Volume (MB)', fontsize=12)
        plt.title('Communication Overhead by Task', fontsize=14)
        plt.xticks(x, [f'Task {t+1}' for t in task_ids])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'communication_per_task.png'), dpi=300)
        plt.close()
        
        logger.info(f"Saved communication overhead plots to {output_dir}")
    
    def _get_tensor_size(self, tensor):
        """Helper to get tensor size in bytes"""
        if isinstance(tensor, torch.Tensor):
            return tensor.element_size() * tensor.nelement()
        elif isinstance(tensor, np.ndarray):
            return tensor.itemsize * tensor.size
        else:
            import sys
            return sys.getsizeof(tensor)