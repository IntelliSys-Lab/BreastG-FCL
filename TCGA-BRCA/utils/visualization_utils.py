import matplotlib
# Use Agg backend for environments without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import matplotlib.cm as cm
from matplotlib.colors import Normalize

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_VIS_DIR = os.path.join(PROJECT_DIR, 'dump', 'visualizations')

def create_enhanced_heatmap(relational_graph, task_id, output_dir=DEFAULT_VIS_DIR,
                           show_annotations=True, highlight_top_k=3, dpi=300, 
                           cmap='viridis', figsize=(10, 8)):
    """
    Create an enhanced heatmap visualization of the relational graph with attention scores.
    
    Args:
        relational_graph: numpy array with shape (num_clients, num_clients)
        task_id: The task identifier
        round_id: The round identifier
        output_dir: Directory to save the visualization
        show_annotations: Whether to show score annotations in cells
        highlight_top_k: Highlight top K connections per client
        dpi: Resolution of saved image
        cmap: Colormap to use
        figsize: Figure size (width, height) in inches
    
    Returns:
        str: Path to saved visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of clients
    num_clients = relational_graph.shape[0]
    
    # Create a figure with the specified size
    plt.figure(figsize=figsize)
    
    # Create a mask to highlight diagonal (self-connections)
    mask = np.zeros_like(relational_graph, dtype=bool)
    np.fill_diagonal(mask, True)
    
    # Create a highlight mask for top-k connections per client (outgoing)
    highlight_mask = np.zeros_like(relational_graph, dtype=bool)
    
    if highlight_top_k > 0:
        for i in range(num_clients):
            # Get top-k connections for each client (excluding self)
            row = relational_graph[i].copy()
            row[i] = 0  # Exclude self connection
            if np.any(row > 0):  # Check if there are any non-zero values
                top_indices = np.argsort(row)[-highlight_top_k:]
                highlight_mask[i, top_indices] = True
    
    # Create the heatmap using regular matplotlib functions
    im = plt.imshow(relational_graph, cmap=cmap, interpolation='nearest')
    plt.colorbar(im, label='Attention Score')
    
    # Add annotations if requested
    if show_annotations:
        for i in range(num_clients):
            for j in range(num_clients):
                # Only annotate cells with significant attention
                if relational_graph[i, j] > 0.05:
                    text_color = 'white' if relational_graph[i, j] > 0.5 else 'black'
                    plt.text(j, i, f'{relational_graph[i, j]:.2f}', 
                            ha='center', va='center', 
                            color=text_color, fontsize=9)
    
    # Add a black edge around diagonal cells (self-connections)
    for i in range(num_clients):
        plt.gca().add_patch(
            plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', lw=2)
        )
    
    # Add a red edge around top-k connections
    for i in range(num_clients):
        for j in range(num_clients):
            if highlight_mask[i, j]:
                plt.gca().add_patch(
                    plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', lw=1.5)
                )
    
    # Add client labels
    client_labels = [f"{i}" for i in range(num_clients)]
    plt.xticks(range(num_clients), client_labels)
    plt.yticks(range(num_clients), client_labels)
    
    # Add labels and title
    plt.xlabel('Target Client', fontsize=12)
    plt.ylabel('Source Client', fontsize=12)
    plt.title(f'Client Relationship Heatmap\nTask {task_id+1}', 
              fontsize=14)
    
    # Add subtitle explaining the visualization
    plt.figtext(0.5, 0.01, 
               '• Diagonal cells (outlined in black) represent self-attention\n'
               f'• Top {highlight_top_k} strongest connections per client outlined in red',
               ha='center', fontsize=9)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    file_path = os.path.join(output_dir, f'heatmap_task{task_id+1}.png')
    plt.savefig(file_path, bbox_inches='tight', dpi=dpi)
    plt.close()
    
    return file_path
   
