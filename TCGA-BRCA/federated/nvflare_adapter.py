"""NVFlare transport for the shared BreastG-FCL coordinator and client runtime.

Only prepared, site-local dataset bundles contain raw features. RPC requests
carry model/optimizer state, graph rows, task identifiers, and operation seeds.
"""

import copy
import logging
from concurrent.futures import Future, wait, FIRST_EXCEPTION
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nvflare.apis.controller_spec import Task, TaskCompletionStatus
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.utils.decomposers.flare_decomposers import register as register_flare_decomposers
from nvflare.app_common.decomposers.numpy_decomposers import register as register_numpy_decomposers
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.utils import fobs

from federated.runtime import execute_client, operation_seed, restore_rng, seeded_rng


# Register before the first task/result is serialized in either process.
fobs.register(TensorDecomposer)
register_numpy_decomposers()
register_flare_decomposers()

LOGGER = logging.getLogger(__name__)
TASK_NAME = "breastgfcl_rpc"
OPERATIONS = frozenset(("encode", "train", "test"))


def cpu_tree(value):
    """Detach transport/checkpoint tensors without pickle-based RPC decoding."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


def _bundle_path(fl_ctx, bundle_name):
    if not isinstance(bundle_name, str) or Path(bundle_name).name != bundle_name:
        raise ValueError("bundle_name must be a filename within app/data")
    job_id = fl_ctx.get_job_id()
    if job_id is None:
        raise RuntimeError("NVFlare job identity is missing")
    return Path(fl_ctx.get_workspace().get_app_dir(job_id)) / "data" / bundle_name


def _load_bundle(fl_ctx, bundle_name):
    # Bundles are locally prepared job artifacts, never untrusted RPC bytes.
    return torch.load(_bundle_path(fl_ctx, bundle_name), map_location="cpu", weights_only=False)


def _aborted(signal):
    return signal is not None and signal.triggered


def _validate_device(opt):
    device = torch.device(opt.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The prepared experiment requires CUDA, but this site has no CUDA device")
    return device


class _AbortAwareLoader:
    """Preserve loader behavior while checking transport cancellation per batch."""

    def __init__(self, loader, abort_signal):
        self.loader = loader
        self.abort_signal = abort_signal

    def __iter__(self):
        for batch in self.loader:
            if _aborted(self.abort_signal):
                raise RuntimeError("NVFlare client operation aborted")
            yield batch

    def __getattr__(self, name):
        return getattr(self.loader, name)


class ProxyClient:
    """Server-side client state; raw datasets remain at the corresponding site."""

    def __init__(self, client_id, opt, bundle, transport):
        self.client_id = int(client_id)
        self.opt = opt
        self.device = torch.device(opt.device)
        self.transport = transport
        self._weights = cpu_tree(bundle["weights"])
        self._training_state = cpu_tree(bundle["training_state"])
        self._task_metadata = {int(key): cpu_tree(value) for key, value in bundle["task_metadata"].items()}
        self._discriminator_state = None

    def getId(self):
        return self.client_id

    def register_task(self, task, dataloader):
        del dataloader
        counts_by_task = self._training_state.setdefault("task_label_counts", {})
        if task in counts_by_task:
            return
        metadata = self._task_metadata[task]
        counts = metadata["label_counts"].detach().cpu().long().clone()
        sizes = list(metadata["batch_sizes"])
        if (tuple(counts.shape) != (self.opt.num_classes,) or (counts < 0).any()
                or not sizes or any(not isinstance(size, int) or size < 1 for size in sizes)
                or int(counts.sum()) != sum(sizes)):
            raise ValueError(f"Invalid prepared replay metadata for client {self.client_id}, task {task}")
        counts_by_task[task] = counts
        self._training_state.setdefault("task_batch_sizes", {})[task] = sizes

    def set_server_discriminator(self, state):
        self._discriminator_state = cpu_tree(state)

    def get_weights(self):
        return cpu_tree(self._weights)

    def set_weights(self, weights):
        for key in ("encoder", "predictor", "generator"):
            if key in weights:
                self._weights[key] = cpu_tree(weights[key])

    def get_training_state(self):
        return cpu_tree(self._training_state)

    def set_training_state(self, state):
        self._training_state = cpu_tree(state)

    def test(self, task_id, dataloader, relational_graphs):
        # Final evaluation calls client.test directly in the shared coordinator.
        pending = self.transport.test_client(self, task_id, dataloader, relational_graphs)
        return self.transport.gather([pending])[0]


class NVFlareTransport:
    """Asynchronous, addressed task RPC with the same API as RayTransport."""

    def __init__(self, opt, controller, fl_ctx, abort_signal):
        self.opt = opt
        self.controller = controller
        self.fl_ctx = fl_ctx
        self.abort_signal = abort_signal
        self._round_index = 0
        self._task = 0
        self._timeout = int(getattr(opt, "nvflare_timeout", 300))
        if self._timeout < 1:
            raise ValueError("nvflare_timeout must be a positive integer")
        self._pending = {}
        self._lock = threading.Lock()

    def set_round(self, task, round_index):
        self._task = int(task)
        self._round_index = int(round_index)

    def _submit(self, operation, client, task, graphs, dataloader, epochs=1):
        del dataloader
        if _aborted(self.abort_signal):
            raise RuntimeError("NVFlare workflow aborted")
        client_id = client.getId()
        if not isinstance(client_id, int) or not 0 <= client_id < self.opt.num_clients:
            raise ValueError(f"Invalid client identity: {client_id}")
        site = f"site-{client_id + 1}"
        request_id = uuid.uuid4().hex
        request = {
            "request_id": request_id,
            "client_id": client_id,
            "operation": operation,
            "task": int(task),
            "graphs": cpu_tree(graphs),
            "epochs": int(epochs),
            "seed": operation_seed(self.opt.seed, task, self._round_index, client_id, operation),
            "weights": client.get_weights(),
            "training_state": client.get_training_state(),
            "discriminator": cpu_tree(client._discriminator_state),
        }
        future = Future()

        def on_result(client_task, fl_ctx):
            del fl_ctx
            if future.done():
                return
            try:
                if client_task.client.name != site:
                    raise RuntimeError(f"Expected result from {site}, received {client_task.client.name}")
                result = client_task.result
                if not isinstance(result, Shareable):
                    raise RuntimeError(f"{site} returned a non-Shareable result")
                if result.get_return_code() != ReturnCode.OK:
                    raise RuntimeError(
                        f"{site} {operation} failed: {result.get_return_code()}: {result.get('error', '')}"
                    )
                if (result.get("request_id") != request_id or result.get("client_id") != client_id
                        or result.get("operation") != operation or result.get("task") != task):
                    raise RuntimeError(f"{site} returned a mismatched RPC identity")
                future.set_result(result["payload"])
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)

        def on_done(task, fl_ctx):
            del fl_ctx
            if not future.done():
                future.set_exception(RuntimeError(
                    f"{site} {operation} ended without a result: {task.completion_status}"
                ))

        rpc_task = Task(
            name=TASK_NAME,
            data=Shareable({"payload": request}),
            timeout=self._timeout,
            result_received_cb=on_result,
            task_done_cb=on_done,
        )
        with self._lock:
            self._pending[future] = (rpc_task, time.monotonic() + self._timeout, site)
        try:
            self.controller.broadcast(
                task=rpc_task, fl_ctx=self.fl_ctx, targets=[site], min_responses=1,
                wait_time_after_min_received=0,
            )
        except Exception as exc:
            future.set_exception(exc)
        return future

    def generate_encodings(self, client, task, graphs, loader):
        return self._submit("encode", client, task, graphs, loader)

    def train_client(self, client, task, graphs, loader, epochs):
        return self._submit("train", client, task, graphs, loader, epochs)

    def test_client(self, client, task, loader, graphs):
        return self._submit("test", client, task, graphs, loader)

    def cancel_pending(self):
        with self._lock:
            pending = list(self._pending.items())
        for future, (task, _deadline, _site) in pending:
            if not future.done():
                future.cancel()
            if task.is_standing:
                self.controller.cancel_task(task, TaskCompletionStatus.ABORTED, self.fl_ctx)

    def gather(self, futures):
        futures = list(futures)
        try:
            while not all(future.done() for future in futures):
                if _aborted(self.abort_signal):
                    raise RuntimeError("NVFlare workflow aborted while waiting for clients")
                with self._lock:
                    pending = [(future, self._pending[future]) for future in futures if not future.done()]
                now = time.monotonic()
                for future, (_task, deadline, site) in pending:
                    if now >= deadline and not future.done():
                        raise TimeoutError(f"NVFlare task for {site} exceeded {self._timeout} seconds")
                done, _ = wait(futures, timeout=0.2, return_when=FIRST_EXCEPTION)
                for future in done:
                    if future.exception() is not None:
                        raise future.exception()
            return [future.result() for future in futures]
        except Exception:
            self.cancel_pending()
            raise
        finally:
            with self._lock:
                for future in futures:
                    self._pending.pop(future, None)


class BreastGFCLExecutor(Executor):
    """Load one site's prepared dataset and delegate RPCs to execute_client."""

    def __init__(self, bundle_name="site.pt"):
        super().__init__()
        self.bundle_name = bundle_name
        self._client = None
        self._dataloaders = None
        self._client_id = None
        self._lock = threading.Lock()

    def _initialize_site(self, fl_ctx):
        from model.client import ModifiedClient

        bundle = _load_bundle(fl_ctx, self.bundle_name)
        opt = SimpleNamespace(**bundle["opt"])
        _validate_device(opt)
        client_id = int(bundle["client_id"])
        if not 0 <= client_id < opt.num_clients:
            raise ValueError("Prepared site identity is outside the configured client roster")
        expected_site = f"site-{client_id + 1}"
        if fl_ctx.get_identity_name() != expected_site:
            raise ValueError(f"Prepared bundle belongs to {expected_site}, not {fl_ctx.get_identity_name()}")
        loaders = {}
        for task, splits in bundle["data"].items():
            task = int(task)
            loaders[task] = {}
            for split in ("train", "test"):
                values = splits[split]
                x, y = values["x"], values["y"]
                if len(x) != len(y):
                    raise ValueError(f"Site dataset sample/label mismatch for task {task}, {split}")
                loaders[task][split] = DataLoader(
                    TensorDataset(x, y), batch_size=opt.batch_size,
                    shuffle=bool(opt.shuffle) if split == "train" else False,
                    num_workers=getattr(opt, "num_workers", 0),
                    pin_memory=getattr(opt, "pin_memory", False),
                )
        if set(loaders) != set(range(opt.num_task)):
            raise ValueError("Prepared site dataset does not contain every configured task")
        self._client = ModifiedClient(client_id, opt)
        self._dataloaders = loaders
        self._client_id = client_id

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if task_name != TASK_NAME:
            return make_reply(ReturnCode.TASK_UNKNOWN)
        if _aborted(abort_signal):
            return make_reply(ReturnCode.TASK_ABORTED)
        try:
            # Model/D construction also consumes RNG. Protect it with the same
            # reentrant lock used by execute_client when simulator sites share
            # a process; the inner context still supplies the operation seed.
            with self._lock, seeded_rng():
                if self._client is None:
                    self._initialize_site(fl_ctx)
                request = shareable["payload"]
                if request["client_id"] != self._client_id:
                    raise ValueError("RPC client identity does not match this site's prepared data")
                if fl_ctx.get_identity_name() != f"site-{self._client_id + 1}":
                    raise ValueError("NVFlare executor identity changed during the run")
                operation = request["operation"]
                task = request["task"]
                if operation not in OPERATIONS or task not in self._dataloaders:
                    raise ValueError("Invalid RPC operation or task")
                client = self._client
                client.set_weights(request["weights"])
                client.set_training_state(request["training_state"])
                if request["discriminator"] is not None:
                    client.set_server_discriminator(request["discriminator"])
                elif operation == "train":
                    raise ValueError("Client training requires server discriminator state")
                loader = _AbortAwareLoader(
                    self._dataloaders[task]["test" if operation == "test" else "train"], abort_signal,
                )
                result = execute_client(
                    operation, client, task, request["graphs"], loader,
                    epochs=request["epochs"], seed=request["seed"],
                )
                if _aborted(abort_signal):
                    return make_reply(ReturnCode.TASK_ABORTED)
                return Shareable({
                    "request_id": request["request_id"], "client_id": self._client_id,
                    "operation": operation, "task": task, "payload": cpu_tree(result),
                })
        except Exception as exc:
            if _aborted(abort_signal):
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_exception(fl_ctx, f"BreastG-FCL RPC failed: {exc}")
            reply = make_reply(ReturnCode.EXECUTION_EXCEPTION)
            reply["error"] = str(exc)
            return reply


class BreastGFCLController(Controller):
    """Run the unchanged shared coordinator using proxies and NVFlare RPCs."""

    def __init__(self, bundle_name="server.pt"):
        super().__init__()
        self.bundle_name = bundle_name
        self._transport = None

    def start_controller(self, fl_ctx):
        pass

    def stop_controller(self, fl_ctx):
        if self._transport is not None:
            self._transport.cancel_pending()

    def control_flow(self, abort_signal, fl_ctx):
        from breastgfcl import ParallelServerGFedCL
        from model.modules import BreastGraphGenerator
        from model.server import Server

        try:
            if _aborted(abort_signal):
                raise RuntimeError("NVFlare workflow aborted before initialization")
            bundle = _load_bundle(fl_ctx, self.bundle_name)
            opt = SimpleNamespace(**bundle["opt"])
            _validate_device(opt)
            if len(bundle["clients"]) != opt.num_clients:
                raise ValueError("Prepared server bundle does not contain every client")
            expected_sites = {f"site-{index + 1}" for index in range(opt.num_clients)}
            actual_sites = [client.name for client in fl_ctx.get_engine().get_clients()]
            if len(actual_sites) != len(set(actual_sites)) or set(actual_sites) != expected_sites:
                raise RuntimeError(f"NVFlare roster mismatch: expected {sorted(expected_sites)}, got {actual_sites}")
            server = Server(opt)
            server.set_discriminator(bundle["discriminator"])
            graph_generator = BreastGraphGenerator(opt).to(opt.device)
            graph_generator.load_state_dict(bundle["graph_state"])
            transport = NVFlareTransport(opt, self, fl_ctx, abort_signal)
            self._transport = transport
            clients = [ProxyClient(i, opt, state, transport) for i, state in enumerate(bundle["clients"])]
            dummy_loaders = {
                i: {task: {"train": None, "test": None} for task in range(opt.num_task)}
                for i in range(opt.num_clients)
            }
            workflow = ParallelServerGFedCL.from_components(
                opt, server, graph_generator, clients, dummy_loaders, transport,
            )
            output_dir = Path(opt.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Initialization above must not change graph/privacy randomness.
            restore_rng(bundle["rng_state"])
            metrics, rounds, _quality = workflow.train_GFedCL()
            if _aborted(abort_signal):
                raise RuntimeError("NVFlare workflow aborted before final state export")
            final = {
                "weights": clients[0].get_weights(),
                "discriminator": cpu_tree(server.get_discriminator()),
                "discriminator_optimizer": cpu_tree(server.optimizer_D.state_dict()),
                "discriminator_scheduler": cpu_tree(server.lr_scheduler_D.state_dict()),
                "graph_state": cpu_tree(graph_generator.state_dict()),
                "training_states": [client.get_training_state() for client in clients],
                "metrics": metrics,
                "rounds": rounds,
            }
            torch.save(final, output_dir / "final_state.pt")
            LOGGER.info("NVFlare shared workflow completed; saved %s", output_dir / "final_state.pt")
        except Exception as exc:
            if self._transport is not None:
                self._transport.cancel_pending()
            self.log_exception(fl_ctx, f"BreastG-FCL controller failed: {exc}")
            self.system_panic(f"BreastG-FCL workflow failed: {exc}", fl_ctx)
            raise
