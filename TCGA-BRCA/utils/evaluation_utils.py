import logging as logger

def evaluate_all_tasks(opt, clients, dataloaders, relational_graphs):
        """
        Evaluate all clients on all tasks to measure final performance
        
        Args:
            relational_graphs: Task-indexed relational graphs used by every client

        Returns:
            Dictionary of accuracy metrics for all clients and tasks
        """
        # Structure to store accuracy for each client on each task
        client_task_acc = {i: {} for i in range(opt.num_clients)}
        
        # For tracking average performance
        task_avg_acc = {t: 0.0 for t in range(opt.num_task)}
        client_avg_acc = {i: 0.0 for i in range(opt.num_clients)}
        overall_avg_acc = 0.0
        
        # Evaluate each client on each task
        for client_id in range(opt.num_clients):
            client = clients[client_id]
            total_acc = 0.0
            
            for task_id in range(opt.num_task):
                # Skip if the client doesn't have data for this task
                if task_id not in dataloaders[client_id]:
                    logger.warning(f"Client {client_id} has no data for task {task_id}")
                    continue
                
                # Test the client on this task
                test_metrics = client.test(
                    task_id,
                    dataloaders[client_id][task_id]['test'],
                    relational_graphs,
                )
                
                # Store accuracy
                acc = test_metrics["acc"]
                client_task_acc[client_id][task_id] = acc
                
                # Update averages
                task_avg_acc[task_id] += acc
                total_acc += acc
            
            # Compute average accuracy for this client
            if len(client_task_acc[client_id]) > 0:
                client_avg_acc[client_id] = total_acc / len(client_task_acc[client_id])
                
        # Compute per-task average across all clients
        for task_id in range(opt.num_task):
            clients_with_task = sum(1 for c in client_task_acc if task_id in client_task_acc[c])
            if clients_with_task > 0:
                task_avg_acc[task_id] /= clients_with_task
        
        # Compute overall average
        total_entries = sum(len(accs) for accs in client_task_acc.values())
        if total_entries > 0:
            overall_avg_acc = sum(sum(accs.values()) for accs in client_task_acc.values()) / total_entries
        
        # Log the results
        logger.info("===== Final Accuracy Results =====")
        logger.info(f"Overall average accuracy: {overall_avg_acc:.2f}%")
        logger.info("Average accuracy per task:")
        for task_id, acc in task_avg_acc.items():
            logger.info(f"  Task {task_id}: {acc:.2f}%")
        logger.info("Average accuracy per client:")
        for client_id, acc in client_avg_acc.items():
            logger.info(f"  Client {client_id}: {acc:.2f}%")
            
        # Return all metrics for plotting
        return {
            'client_task_acc': client_task_acc,
            'task_avg_acc': task_avg_acc,
            'client_avg_acc': client_avg_acc,
            'overall_avg_acc': overall_avg_acc
        }
