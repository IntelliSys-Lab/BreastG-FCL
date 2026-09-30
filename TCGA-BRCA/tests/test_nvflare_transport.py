import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from nvflare.apis.fl_constant import ReturnCode
    from nvflare.apis.shareable import Shareable, make_reply
    from nvflare.apis.signal import Signal
    from nvflare.fuel.utils import fobs
    from federated import nvflare_adapter as adapter
    HAS_NVFLARE = True
except ModuleNotFoundError as error:
    if error.name != "nvflare" and not error.name.startswith("nvflare."):
        raise
    HAS_NVFLARE = False

from federated.runtime import execute_client
from model.client import ModifiedClient
from model.server import Server


class UnreadableLoader:
    def __iter__(self):
        raise AssertionError("A server proxy must never read raw site data")


class RecordingController:
    def __init__(self):
        self.requests = []
        self.cancelled = []

    def broadcast(self, **kwargs):
        kwargs["task"].is_standing = True
        self.requests.append(kwargs)

    def cancel_task(self, task, completion_status, fl_ctx):
        self.cancelled.append(task)
        task.is_standing = False


@unittest.skipUnless(HAS_NVFLARE, "NVFlare is an optional transport dependency")
class NVFlareTransportTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(107)
        self.opt = SimpleNamespace(
            device="cpu", batch_size=4, input_dim=8, nh=16, ni=16, nt=4,
            nd_out=4, num_clients=4, num_classes=2, num_task=3, noise_dim=5,
            no_bn=False, p=0.2, lr_e=0.001, lr_f=0.002, lr_g=0.003,
            lr_d=0.004, beta1=0.9, beta2=0.999, lambda_gan=0.5,
            replay=True, shuffle=True, seed=71, nvflare_timeout=1,
        )
        self.inputs = torch.randn(4, 8)
        self.labels = torch.tensor([0, 1, 0, 1])
        self.loader = DataLoader(TensorDataset(self.inputs, self.labels), batch_size=4, shuffle=True)
        self.graphs = [np.roll(np.eye(4, dtype=np.float32), task, axis=1) for task in range(3)]
        self.client = ModifiedClient(0, self.opt)
        self.discriminator = Server(self.opt).get_discriminator()
        self.client.set_server_discriminator(self.discriminator)
        self.proxy_bundle = {
            "weights": self.client.get_weights(),
            "training_state": self.client.get_training_state(),
            "task_metadata": {
                task: {"label_counts": torch.tensor([2, 2]), "batch_sizes": [4]}
                for task in range(3)
            },
        }
        for task in range(3):
            self.client.register_task(task, self.loader)

    def assert_nested_equal(self, first, second):
        if torch.is_tensor(first):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def proxy(self, transport=None, bundle=None):
        client = adapter.ProxyClient(0, self.opt, self.proxy_bundle if bundle is None else bundle, transport)
        client.set_server_discriminator(self.discriminator)
        return client

    def transport(self):
        controller = RecordingController()
        signal = Signal()
        transport = adapter.NVFlareTransport(self.opt, controller, Mock(), signal)
        return transport, controller, signal

    def request(self, operation="train"):
        return {
            "request_id": "unit-test-request", "client_id": 0,
            "operation": operation, "task": 2, "epochs": 1, "seed": 711,
            "graphs": self.graphs, "weights": self.client.get_weights(),
            "training_state": self.client.get_training_state(),
            "discriminator": self.discriminator,
        }

    def site_bundle(self):
        return {
            "opt": vars(self.opt), "client_id": 0,
            "data": {
                task: {split: {"x": self.inputs, "y": self.labels} for split in ("train", "test")}
                for task in range(3)
            },
        }

    def fl_context(self, app_dir, site="site-1"):
        context = Mock()
        context.get_job_id.return_value = "prepared-job"
        context.get_identity_name.return_value = site
        context.get_workspace.return_value.get_app_dir.return_value = str(app_dir)
        return context

    def successful_reply(self, request, payload=None):
        return Shareable({
            key: request[key] for key in ("request_id", "client_id", "operation", "task")
        } | {"payload": {"ok": True} if payload is None else payload})

    def deliver(self, rpc_task, reply, site="site-1"):
        client_task = SimpleNamespace(client=SimpleNamespace(name=site), result=reply)
        rpc_task.result_received_cb(client_task, Mock())

    def test_fobs_roundtrip_preserves_graphs_weights_and_populated_adam_state(self):
        result = execute_client("train", self.client, 2, self.graphs, self.loader, seed=117)
        noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).T.requires_grad_()
        original = {"graphs": self.graphs, "result": result, "view": noncontiguous, "tuple": (1, None)}
        wire = adapter.cpu_tree(original)
        self.assertTrue(wire["view"].is_contiguous())
        self.assertFalse(wire["view"].requires_grad)
        self.assertTrue(wire["result"]["training_state"]["optimizer"]["state"])

        restored = fobs.loads(fobs.dumps(wire))

        self.assert_nested_equal(restored, wire)
        with torch.no_grad():
            noncontiguous.fill_(-1)
        self.assertFalse(torch.equal(wire["view"], noncontiguous))
        self.graphs[0].fill(-1)
        self.assertTrue(np.all(wire["graphs"][0] >= 0))

    def test_proxy_registers_exact_replay_metadata_without_reading_samples(self):
        proxy = self.proxy()
        for task in range(3):
            proxy.register_task(task, UnreadableLoader())
        state = proxy.get_training_state()

        self.assertEqual(set(state["task_label_counts"]), {0, 1, 2})
        for task in range(3):
            torch.testing.assert_close(state["task_label_counts"][task], torch.tensor([2, 2]))
            self.assertEqual(state["task_batch_sizes"][task], [4])
        state["task_label_counts"][0].fill_(0)
        torch.testing.assert_close(proxy.get_training_state()["task_label_counts"][0], torch.tensor([2, 2]))
        before = proxy.get_training_state()
        proxy.register_task(0, UnreadableLoader())
        self.assert_nested_equal(proxy.get_training_state(), before)

    def test_proxy_rejects_inconsistent_metadata(self):
        bundle = copy.deepcopy(self.proxy_bundle)
        bundle["task_metadata"][0]["label_counts"] = torch.tensor([2, 1])
        proxy = self.proxy(bundle=bundle)
        with self.assertRaisesRegex(ValueError, "replay metadata"):
            proxy.register_task(0, UnreadableLoader())
        self.assertNotIn(0, proxy.get_training_state()["task_label_counts"])

    def test_executor_matches_shared_runtime_and_reads_only_its_site_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            app_dir = Path(directory)
            (app_dir / "data").mkdir()
            bundle_path = app_dir / "data" / "site.pt"
            torch.save(self.site_bundle(), bundle_path)
            context = self.fl_context(app_dir)
            executor = adapter.BreastGFCLExecutor()
            with patch.object(adapter.torch, "load", wraps=torch.load) as load:
                for operation in ("encode", "train", "test"):
                    with self.subTest(operation=operation):
                        request = self.request(operation)
                        local = ModifiedClient(0, self.opt)
                        local.set_weights(request["weights"])
                        local.set_training_state(request["training_state"])
                        local.set_server_discriminator(self.discriminator)
                        loader = self.loader if operation != "test" else DataLoader(
                            TensorDataset(self.inputs, self.labels), batch_size=4, shuffle=False,
                        )
                        expected = execute_client(operation, local, 2, self.graphs, loader, seed=request["seed"])
                        with patch.object(adapter, "execute_client", wraps=execute_client) as shared_runtime:
                            reply = executor.execute(adapter.TASK_NAME, Shareable({"payload": request}), context, Signal())
                        self.assertEqual(reply.get_return_code(), ReturnCode.OK, reply.get("error"))
                        shared_runtime.assert_called_once()
                        self.assert_nested_equal(reply["payload"], expected)
                load.assert_called_once()
                self.assertEqual(Path(load.call_args.args[0]), bundle_path)
            context.get_workspace.return_value.get_app_dir.assert_called_once_with("prepared-job")

    def test_executor_rejects_other_sites_and_mismatched_client_ids(self):
        for site, client_id in (("site-2", 0), ("site-1", 1)):
            with self.subTest(site=site, client_id=client_id):
                executor = adapter.BreastGFCLExecutor()
                request = self.request()
                request["client_id"] = client_id
                with patch.object(adapter, "_load_bundle", return_value=self.site_bundle()), patch.object(executor, "log_exception"):
                    reply = executor.execute(adapter.TASK_NAME, Shareable({"payload": request}), self.fl_context("unused", site), Signal())
                self.assertEqual(reply.get_return_code(), ReturnCode.EXECUTION_EXCEPTION)
                self.assertTrue(reply["error"])

    def test_executor_abort_does_not_load_site_data(self):
        signal = Signal()
        signal.trigger(True)
        executor = adapter.BreastGFCLExecutor()
        with patch.object(adapter, "_load_bundle") as load:
            reply = executor.execute(adapter.TASK_NAME, Shareable({"payload": self.request()}), Mock(), signal)
        self.assertEqual(reply.get_return_code(), ReturnCode.TASK_ABORTED)
        load.assert_not_called()

    def test_transport_addresses_one_site_and_preserves_payload(self):
        transport, controller, _signal = self.transport()
        proxy = self.proxy(transport)
        transport.set_round(2, 3)
        future = transport.generate_encodings(proxy, 2, self.graphs, UnreadableLoader())
        dispatch = controller.requests[0]
        self.assertEqual(dispatch["targets"], ["site-1"])
        request = dispatch["task"].data["payload"]
        self.assertEqual(request["seed"], adapter.operation_seed(self.opt.seed, 2, 3, 0, "encode"))
        self.assertNotIn("data", request)
        self.assertNotIn("dataloader", request)
        payload = {"encodings": [torch.ones(2, 16)], "graph_embeddings": [torch.ones(2, 4)]}
        self.deliver(dispatch["task"], self.successful_reply(request, payload))

        self.assert_nested_equal(transport.gather([future]), [payload])

    def test_transport_rejects_wrong_site_identity_error_and_missing_payload(self):
        cases = ("site", "client_id", "request_id", "operation", "task", "error", "missing_payload", "non_shareable")
        for case in cases:
            with self.subTest(case=case):
                transport, controller, _signal = self.transport()
                future = transport.generate_encodings(self.proxy(), 2, self.graphs, None)
                rpc_task = controller.requests[0]["task"]
                reply = self.successful_reply(rpc_task.data["payload"])
                site = "site-1"
                if case == "site":
                    site = "site-2"
                elif case in ("client_id", "task"):
                    reply[case] += 1
                elif case in ("request_id", "operation"):
                    reply[case] = "wrong"
                elif case == "error":
                    reply = make_reply(ReturnCode.EXECUTION_EXCEPTION)
                    reply["error"] = "Failed client training"
                elif case == "missing_payload":
                    del reply["payload"]
                elif case == "non_shareable":
                    reply = {"payload": "invalid"}
                self.deliver(rpc_task, reply, site)
                with self.assertRaises((RuntimeError, KeyError)):
                    transport.gather([future])
                self.assertEqual(controller.cancelled, [rpc_task])

    def test_transport_timeout_abort_and_completion_without_result_fail(self):
        for failure in ("timeout", "abort", "missing_result"):
            with self.subTest(failure=failure):
                transport, controller, signal = self.transport()
                with patch.object(adapter.time, "monotonic", return_value=0.0):
                    future = transport.generate_encodings(self.proxy(), 2, self.graphs, None)
                rpc_task = controller.requests[0]["task"]
                if failure == "abort":
                    signal.trigger(True)
                elif failure == "missing_result":
                    rpc_task.task_done_cb(rpc_task, Mock())
                expected = TimeoutError if failure == "timeout" else RuntimeError
                with self.assertRaises(expected):
                    transport.gather([future])
                self.assertEqual(controller.cancelled, [rpc_task])
                self.assertTrue(future.done())


if __name__ == "__main__":
    unittest.main()
