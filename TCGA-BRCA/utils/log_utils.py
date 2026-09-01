import logging
import os
import sys
import numpy as np
from utils.visualization_utils import create_enhanced_heatmap

def set_log_file(fname, file_only=False):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    if file_only:
        # we only output messages to file, and stdout/stderr receives nothing.
        # this feature is designed for executing the script via ssh:
        # since ssh has a windowing kind of flow control, i.e., if the controller does not read data from a
        # ssh channel and its buffer fills up, the execution machine will not be able to write anything into the
        # channel and the process will be set to sleeping (S) status until someone reads all data from the channel.
        # this is not desired since we do not want to read stdout/stderr from the controller machine.
        # so, here we use a simple solution: disable output to stdout/stderr and only output messages to log file.
        sys.stdout = sys.stderr = open(fname, 'w', buffering=1)
    else:
        # we output messages to both file and stdout/stderr
        import subprocess
        tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
        os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def save_heatmap(opt, task, relational_graphs):
    # Get logger
    logger = logging.getLogger('GFedCL')
    
    # Log properties of the generated graph
    attention_max = np.max(relational_graphs[task])
    attention_min = np.min(relational_graphs[task])
    attention_mean = np.mean(relational_graphs[task])
    attention_std = np.std(relational_graphs[task])
    
    logger.info('Generated attention-based relational graph with:')
    logger.info(f'  Max attention: {attention_max:.4f}')
    logger.info(f'  Min attention: {attention_min:.4f}')
    logger.info(f'  Mean attention: {attention_mean:.4f}')
    logger.info(f'  Std deviation: {attention_std:.4f}')
    
    # Visualize the generated relational graph
    vis_dir = os.path.join(opt.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    logger.info(f'Visualizing relational graph for task {task+1}')
    
    # Create enhanced heatmap visualization
    heatmap_path = create_enhanced_heatmap(
        relational_graphs[task],
        task,
        output_dir=vis_dir,
        show_annotations=True,
        highlight_top_k=3
    )
    logger.info(f'Enhanced heatmap visualization saved to {heatmap_path}')