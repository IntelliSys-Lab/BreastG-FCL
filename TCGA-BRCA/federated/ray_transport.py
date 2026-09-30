"""Ray scheduling for the shared federated client operations."""

from federated.runtime import execute_client, operation_seed


class RayTransport:
    def __init__(self, opt):
        # Importing the coordinator/runtime must not start a Ray cluster.
        import ray

        self.opt = opt
        self._ray = ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self._execute = ray.remote(execute_client)
        self._task = 0
        self._round_index = 0

    def set_round(self, task, round_index):
        self._task = int(task)
        self._round_index = int(round_index)

    def _submit(self, operation, client, task, relational_graphs, dataloader, epochs=1):
        seed = operation_seed(
            getattr(self.opt, "seed", 0), task, self._round_index,
            client.getId(), operation,
        )
        return self._execute.options(
            num_gpus=getattr(self.opt, "ray_num_gpus_per_task", 0.0) or 0.0,
            num_cpus=getattr(self.opt, "ray_num_cpus_per_task", 1.0),
        ).remote(operation, client, task, relational_graphs, dataloader, epochs, seed)

    def generate_encodings(self, client, task, graphs, loader):
        return self._submit("encode", client, task, graphs, loader)

    def train_client(self, client, task, graphs, loader, epochs):
        return self._submit("train", client, task, graphs, loader, epochs)

    def test_client(self, client, task, loader, graphs):
        return self._submit("test", client, task, graphs, loader)

    def gather(self, futures):
        return self._ray.get(futures)
