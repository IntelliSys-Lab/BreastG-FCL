import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import nvflare_job
from model.server import Server


class NVFlareJobTest(unittest.TestCase):
    def setUp(self):
        self.opt = SimpleNamespace(
            num_clients=2, num_task=3, num_classes=2, batch_size=2, device="cpu",
            client_spatial_features=[np.ones((2, 3), dtype=np.float32) for _ in range(3)],
            client_temporal_features=[np.ones((2, 2), dtype=np.float32) for _ in range(3)],
            nvflare_workspace=None, nvflare_threads=None, nvflare_gpu=None,
        )
        self.loaders = {}
        for client in range(2):
            self.loaders[client] = {}
            for task in range(3):
                features = torch.arange(24, dtype=torch.float32).reshape(6, 4) + 100 * client + 10 * task
                labels = (torch.arange(6) + client + task) % 2
                dataset = TensorDataset(features, labels)
                self.loaders[client][task] = {
                    "train": DataLoader(Subset(dataset, [4, 1, 3]), batch_size=2, shuffle=True),
                    "test": DataLoader(Subset(dataset, [5, 0, 2]), batch_size=2),
                }
        clients = []
        for client in range(2):
            proxy = Mock()
            proxy.get_weights.return_value = {
                key: {"weight": torch.full((2, 2), float(client))}
                for key in ("encoder", "predictor", "generator")
            }
            proxy.get_training_state.return_value = {
                "optimizer": {"state": {}}, "scheduler": {},
                "task_label_counts": {}, "task_batch_sizes": {},
            }
            clients.append(proxy)
        server = Mock()
        server.get_discriminator.return_value = {"weight": torch.ones(2, 2)}
        attention = Mock()
        attention.state_dict.return_value = {"spatial_attention.weight": torch.ones(2, 2)}
        self.workflow = SimpleNamespace(
            clients=clients, dataloaders=self.loaders, server=server, dygat=attention,
            device=torch.device("cpu"),
        )

    def prepare(self):
        with patch.object(nvflare_job, "ParallelServerGFedCL", return_value=self.workflow):
            return nvflare_job.prepare_bundles(self.opt)

    def assert_no_raw_data(self, value):
        if isinstance(value, dict):
            for key, child in value.items():
                self.assertNotIn(key, ("data", "x", "y", "dataset", "dataloader"))
                self.assert_no_raw_data(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self.assert_no_raw_data(child)

    def assert_split_matches(self, exported, dataset):
        expected_features = torch.stack([dataset[index][0] for index in range(len(dataset))])
        expected_labels = torch.tensor([dataset[index][1] for index in range(len(dataset))])
        torch.testing.assert_close(exported["x"], expected_features, rtol=0, atol=0)
        torch.testing.assert_close(exported["y"], expected_labels, rtol=0, atol=0)

    def test_preparation_preserves_each_existing_split_and_subset_order(self):
        workflow, server, sites = self.prepare()

        self.assertIs(workflow, self.workflow)
        self.assertEqual(len(sites), 2)
        self.assert_no_raw_data(server)
        for client_id, site in enumerate(sites):
            self.assertEqual(site["client_id"], client_id)
            self.assertNotIn("client_spatial_features", site["opt"])
            self.assertNotIn("client_temporal_features", site["opt"])
            for task in range(3):
                for split in ("train", "test"):
                    self.assert_split_matches(site["data"][task][split], self.loaders[client_id][task][split].dataset)
                labels = site["data"][task]["train"]["y"]
                metadata = server["clients"][client_id]["task_metadata"][task]
                torch.testing.assert_close(metadata["label_counts"], torch.bincount(labels, minlength=2))
                self.assertEqual(metadata["batch_sizes"], [2, 1])

    def test_export_keeps_raw_data_in_separate_site_apps_and_copies_only_code(self):
        _workflow, server, sites = self.prepare()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "source"
            project.mkdir()
            for package in ("model", "utils", "configs", "federated"):
                (project / package).mkdir()
                (project / package / "__init__.py").write_text("# code fixture\n")
                (project / package / "private.bin").write_bytes(b"must not be copied")
            (project / "breastgfcl.py").write_text("# coordinator fixture\n")
            (project / "data").mkdir()
            (project / "data" / "raw.pt").write_bytes(b"private source data")
            job_dir = root / "exported job"
            with patch.object(nvflare_job, "PROJECT", project):
                nvflare_job.export_job(job_dir, server, sites)

            meta = json.loads((job_dir / "meta.json").read_text())
            self.assertEqual(meta["deploy_map"], {
                "app_server": ["server"], "app_site_1": ["site-1"], "app_site_2": ["site-2"],
            })
            self.assertEqual(meta["mandatory_clients"], ["site-1", "site-2"])
            server_files = list((job_dir / "app_server" / "data").iterdir())
            self.assertEqual([path.name for path in server_files], ["server.pt"])
            self.assert_no_raw_data(torch.load(server_files[0], weights_only=False))
            for client_id in range(2):
                app = job_dir / f"app_site_{client_id + 1}"
                self.assertEqual([path.name for path in (app / "data").iterdir()], ["site.pt"])
                local = torch.load(app / "data" / "site.pt", weights_only=False)
                self.assertEqual(local["client_id"], client_id)
                for task in range(3):
                    for split in ("train", "test"):
                        self.assert_split_matches(local["data"][task][split], self.loaders[client_id][task][split].dataset)
            for app_name in meta["deploy_map"]:
                custom_files = [path for path in (job_dir / app_name / "custom").rglob("*") if path.is_file()]
                self.assertTrue(custom_files)
                self.assertTrue(all(path.suffix == ".py" for path in custom_files))
                self.assertFalse((job_dir / app_name / "custom" / "data").exists())

    def test_simulator_runs_in_a_fresh_interpreter_with_explicit_site_roster(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            job_dir = output / "job with spaces"
            self.opt.nvflare_threads = 1

            def child_process(command, **kwargs):
                self.assertEqual(command[:3], [sys.executable, "-m", "nvflare.private.fed.app.simulator.simulator"])
                self.assertEqual(command[3], str(job_dir))
                self.assertEqual(command[command.index("--clients") + 1], "site-1,site-2")
                self.assertEqual(command[command.index("--n_clients") + 1], "2")
                self.assertEqual(command[command.index("--threads") + 1], "1")
                self.assertNotIn("--gpu", command)
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                self.assertFalse(kwargs.get("shell", False))
                kwargs["stdout"].write("child simulator completed\n")
                torch.save({"completed": True}, output / "final_state.pt")
                return SimpleNamespace(returncode=0)

            with patch.object(nvflare_job.subprocess, "run", side_effect=child_process) as run:
                nvflare_job.run_simulator(self.opt, job_dir, output)
            run.assert_called_once()
            self.assertIn("child simulator completed", (output / "simulator.log").read_text())

    def test_simulator_passes_explicit_workspace_and_gpu_to_child(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.opt.nvflare_workspace = str(output / "separate workspace")
            self.opt.nvflare_gpu = "1"
            torch.save({}, output / "final_state.pt")
            with patch.object(nvflare_job.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                nvflare_job.run_simulator(self.opt, output / "job", output)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--workspace") + 1], self.opt.nvflare_workspace)
            self.assertEqual(command[command.index("--gpu") + 1], "1")

    def test_simulator_requires_successful_exit_and_final_state_artifact(self):
        for return_code, artifact in ((1, True), (0, False)):
            with self.subTest(return_code=return_code, artifact=artifact):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    if artifact:
                        torch.save({}, output / "final_state.pt")
                    with patch.object(nvflare_job.subprocess, "run", return_value=SimpleNamespace(returncode=return_code)):
                        with self.assertRaisesRegex(RuntimeError, "NVFlare failed or produced no final state"):
                            nvflare_job.run_simulator(self.opt, output / "job", output)

    def test_final_state_captures_discriminator_adam_schedule_and_detached_copies(self):
        server = Server(SimpleNamespace(device="cpu", nh=8, nt=2, lr_d=0.001, beta1=0.9))
        latent = torch.randn(4, 8)
        graph_rows = torch.rand(4, 2)
        server.train_discriminator([latent], [graph_rows])
        server.update_learning_rate()
        self.workflow.server = server
        source_weight = torch.randn(2, 2, requires_grad=True)
        self.workflow.clients[0].get_weights.return_value["encoder"]["weight"] = source_weight

        snapshot = nvflare_job.final_state(self.workflow, {"accuracy": [0.5]}, [0])

        optimizer = snapshot["discriminator_optimizer"]
        self.assertTrue(optimizer["state"])
        self.assertEqual(snapshot["discriminator_scheduler"]["last_epoch"], 1)
        self.assertEqual(optimizer["param_groups"][0]["lr"], server.optimizer_D.param_groups[0]["lr"])
        for parameter_state in optimizer["state"].values():
            self.assertEqual(parameter_state["step"].item(), 1)
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(parameter_state[key].device.type, "cpu")
                self.assertFalse(parameter_state[key].requires_grad)
        exported_weight = snapshot["weights"]["encoder"]["weight"]
        self.assertEqual(exported_weight.device.type, "cpu")
        self.assertFalse(exported_weight.requires_grad)
        torch.testing.assert_close(exported_weight, source_weight.detach())
        with torch.no_grad():
            source_weight.add_(1)
        server.train_discriminator([latent], [graph_rows])
        self.assertFalse(torch.equal(exported_weight, source_weight.detach()))
        self.assertTrue(all(state["step"].item() == 1 for state in optimizer["state"].values()))


if __name__ == "__main__":
    unittest.main()
