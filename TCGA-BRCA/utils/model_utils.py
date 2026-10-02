from model.client import ModifiedClient

def create_clients(opt):
    """
    Create clients based on the given options.
    """
    clients = []
    for i in range(opt.num_clients):
        client = ModifiedClient(i, opt)
        clients.append(client)

    if clients:
        initial_weights = clients[0].get_weights()
        for client in clients[1:]:
            client.set_weights(initial_weights)
    return clients
