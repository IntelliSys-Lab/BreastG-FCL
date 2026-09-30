"""Prepare the shared TCGA split and run it through NVFlare's simulator."""

import copy
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from breastgfcl import ParallelServerGFedCL
from configs.TCGA_BRCA import build_parser, finalize_opt
from federated.runtime import restore_rng, snapshot_rng


PROJECT = Path(__file__).resolve().parent


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def smoke_loaders(opt):
    """Small deterministic fixture; never used by the real TCGA run."""
    rng = np.random.default_rng(opt.seed)
    opt.client_spatial_features = [rng.normal(size=(opt.num_clients, 3)).astype(np.float32)
                                   for _ in range(opt.num_task)]
    opt.client_temporal_features = [rng.normal(size=(opt.num_clients, 2)).astype(np.float32)
                                    for _ in range(opt.num_task)]
    loaders = {}
    for client_id in range(opt.num_clients):
        loaders[client_id] = {}
        for task in range(opt.num_task):
            loaders[client_id][task] = {}
            for split in ("train", "test"):
                x = torch.from_numpy(rng.normal(size=(8, opt.input_dim)).astype(np.float32))
                y = torch.arange(8) % opt.num_classes
                loaders[client_id][task][split] = DataLoader(
                    TensorDataset(x, y), batch_size=opt.batch_size,
                    shuffle=opt.shuffle and split == "train",
                )
    return loaders


def dataset_tensors(dataset):
    if not len(dataset):
        raise ValueError("Cannot package an empty client/task dataset")
    # Read by index to preserve the existing split/order; no second partition.
    samples = [dataset[index] for index in range(len(dataset))]
    return {"x": torch.stack([x for x, _ in samples]).cpu(),
            "y": torch.as_tensor([int(y) for _, y in samples], dtype=torch.long)}


def prepare_bundles(opt, smoke=False):
    loaders = smoke_loaders(opt) if smoke else None
    # Only initialization/data preparation: this sentinel never executes tasks.
    workflow = ParallelServerGFedCL(opt, transport=object(), dataloaders=loaders)
    sites, client_states = [], []
    for client_id, client in enumerate(workflow.clients):
        data, metadata = {}, {}
        for task in range(opt.num_task):
            data[task] = {split: dataset_tensors(workflow.dataloaders[client_id][task][split].dataset)
                          for split in ("train", "test")}
            labels = data[task]["train"]["y"]
            metadata[task] = {
                "label_counts": torch.bincount(labels, minlength=opt.num_classes),
                "batch_sizes": [min(opt.batch_size, len(labels) - start)
                                for start in range(0, len(labels), opt.batch_size)],
            }
        site_opt = {key: value for key, value in vars(opt).items()
                    if not key.startswith("client_spatial") and not key.startswith("client_temporal")}
        sites.append({"opt": cpu_tree(site_opt), "client_id": client_id, "data": data})
        client_states.append({"weights": cpu_tree(client.get_weights()),
                              "training_state": cpu_tree(client.get_training_state()),
                              "task_metadata": metadata})
    server_bundle = {
        "opt": cpu_tree(vars(opt)),
        "discriminator": cpu_tree(workflow.server.get_discriminator()),
        "graph_state": cpu_tree(workflow.dygat.state_dict()),
        "clients": client_states,
        "rng_state": snapshot_rng(workflow.device),
    }
    return workflow, server_bundle, sites


def export_job(job_dir, server_bundle, site_bundles):
    """Export traditional NVFlare apps; each site alone receives its raw split."""
    job_dir = Path(job_dir)
    if job_dir.exists():
        raise FileExistsError(f"Job directory already exists: {job_dir}")
    job_dir.mkdir(parents=True)
    deploy_map = {"app_server": ["server"]}
    applications = [("app_server", "server.pt", server_bundle, True)]
    for index, bundle in enumerate(site_bundles):
        app_name = f"app_site_{index + 1}"
        deploy_map[app_name] = [f"site-{index + 1}"]
        applications.append((app_name, "site.pt", bundle, False))
    for app_name, bundle_name, bundle, is_server in applications:
        app = job_dir / app_name
        (app / "config").mkdir(parents=True)
        (app / "data").mkdir()
        (app / "custom").mkdir()
        # Copy code only: never bundle source data/dump directories on the server.
        for package in ("model", "utils", "configs", "federated"):
            target = app / "custom" / package
            target.mkdir()
            for source in (PROJECT / package).glob("*.py"):
                shutil.copy2(source, target / source.name)
        shutil.copy2(PROJECT / "breastgfcl.py", app / "custom" / "breastgfcl.py")
        torch.save(bundle, app / "data" / bundle_name)
        component = {
            "path": "federated.nvflare_adapter.BreastGFCLController" if is_server else
                    "federated.nvflare_adapter.BreastGFCLExecutor",
            "args": {"bundle_name": bundle_name},
        }
        config = {"format_version": 2}
        if is_server:
            config["workflows"] = [{"id": "breastg_fcl", **component}]
        else:
            config["executors"] = [{"tasks": ["breastgfcl_rpc"], "executor": component}]
        config_name = "config_fed_server.json" if is_server else "config_fed_client.json"
        (app / "config" / config_name).write_text(json.dumps(config, indent=2) + "\n")
    (job_dir / "meta.json").write_text(json.dumps({
        "name": "breastg_fcl", "resource_spec": {}, "deploy_map": deploy_map,
        "min_clients": len(site_bundles),
        "mandatory_clients": [f"site-{i + 1}" for i in range(len(site_bundles))],
    }, indent=2) + "\n")
    return job_dir


def final_state(workflow, metrics, rounds):
    return cpu_tree({"weights": workflow.clients[0].get_weights(),
                     "discriminator": workflow.server.get_discriminator(),
                     "discriminator_optimizer": workflow.server.optimizer_D.state_dict(),
                     "discriminator_scheduler": workflow.server.lr_scheduler_D.state_dict(),
                     "graph_state": workflow.dygat.state_dict(),
                     "training_states": [c.get_training_state() for c in workflow.clients],
                     "metrics": metrics, "rounds": rounds})


def compare_states(expected, actual, path="state"):
    """Require exact CPU equality, including optimizer state and all metrics."""
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual) or expected.dtype != actual.dtype or not torch.equal(expected, actual):
            raise AssertionError(f"Ray/NVFlare tensor mismatch at {path}")
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(expected, actual, err_msg=path)
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"Ray/NVFlare keys mismatch at {path}")
        for key in expected:
            compare_states(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise AssertionError(f"Ray/NVFlare length mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare_states(left, right, f"{path}[{index}]")
    elif expected != actual:
        raise AssertionError(f"Ray/NVFlare value mismatch at {path}: {expected} != {actual}")


def run_simulator(opt, job_dir, output):
    # NVFlare forks its server. A fresh interpreter keeps the preparer's CUDA
    # context and the Ray reference's autograd threads out of that fork.
    workspace = str(Path(opt.nvflare_workspace or output / "nvflare_workspace").resolve())
    command = [sys.executable, "-m", "nvflare.private.fed.app.simulator.simulator",
               str(job_dir), "--workspace", workspace,
               "--clients", ",".join(f"site-{i + 1}" for i in range(opt.num_clients)),
               "--n_clients", str(opt.num_clients),
               "--threads", str(opt.nvflare_threads or opt.num_clients)]
    gpu = opt.nvflare_gpu or ("0" if opt.device == "cuda" else None)
    if gpu is not None:
        command.extend(["--gpu", gpu])
    log_path = output / "simulator.log"
    print(f"Running NVFlare; simulator log: {log_path}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    # The SDK may return zero after a failed controller, so require its result.
    if result.returncode != 0 or not (output / "final_state.pt").exists():
        raise RuntimeError(f"NVFlare failed or produced no final state (exit code {result.returncode}); "
                           f"inspect {log_path} and {workspace}")


def main(args=None):
    parser = build_parser()
    parser.description = "BreastG-FCL on NVFlare, sharing the Ray training workflow"
    parser.add_argument("--smoke", action="store_true", help="Use a small synthetic validation fixture")
    parser.add_argument("--verify-ray", action="store_true", help="Compare complete CPU state against Ray")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--nvflare-workspace", default=None)
    parser.add_argument("--nvflare-threads", type=int, default=None)
    parser.add_argument("--nvflare-gpu", default=None)
    parser.add_argument("--nvflare-timeout", type=int, default=300)
    opt = finalize_opt(parser.parse_args(args))
    if opt.smoke:
        opt.input_dim, opt.nh, opt.ni, opt.noise_dim = 8, 16, 16, 5
        opt.batch_size, opt.num_local_epochs, opt.num_rounds = 4, 1, 2
        opt.gat_hidden_dim, opt.gat_embedding_dim, opt.gat_heads = 32, 16, 2
        opt.no_bn, opt.p = False, 0.1
    if opt.verify_ray and opt.device != "cpu":
        parser.error("--verify-ray uses exact CPU comparison; add --device cpu")
    if opt.verify_ray and opt.export_only:
        parser.error("--verify-ray requires execution, so cannot be combined with --export-only")
    if opt.nvflare_timeout <= 0:
        parser.error("--nvflare-timeout must be positive")
    if opt.nvflare_threads is not None and opt.nvflare_threads <= 0:
        parser.error("--nvflare-threads must be positive")
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    output = Path(opt.output_dir).resolve()
    opt.output_dir = str(output)
    workflow, bundle, sites = prepare_bundles(opt, opt.smoke)
    job_dir = export_job(output / "nvflare_job", bundle, sites)
    print(f"Exported NVFlare job: {job_dir}", flush=True)
    if opt.export_only:
        return

    expected = None
    if opt.verify_ray:
        import ray
        from federated.ray_transport import RayTransport
        ray.init(num_cpus=min(opt.num_clients, 4), include_dashboard=False, log_to_driver=False,
                 runtime_env={"env_vars": {"PYTHONPATH": str(PROJECT), "OMP_NUM_THREADS": "1",
                                            "MKL_NUM_THREADS": "1"}})
        workflow.transport = RayTransport(opt)
        original_output = opt.output_dir
        opt.output_dir = str(output / "ray_reference")
        Path(opt.output_dir).mkdir()
        restore_rng(bundle["rng_state"])
        try:
            metrics, rounds, _ = workflow.train_GFedCL()
            expected = final_state(workflow, metrics, rounds)
            torch.save(expected, output / "ray_reference" / "final_state.pt")
        finally:
            ray.shutdown()
            opt.output_dir = original_output

    # Prepared models/data are no longer needed by the launcher; the exported
    # apps own their copies. Release GPU memory before starting simulator sites.
    del workflow, bundle, sites
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_simulator(opt, job_dir, output)
    if expected is not None:
        actual = torch.load(output / "final_state.pt", map_location="cpu", weights_only=False)
        compare_states(expected, actual)
        report = {"equal": True, "comparison": "exact CPU tensor and metric equality",
                  "clients": opt.num_clients, "tasks": opt.num_task, "rounds_per_task": opt.num_rounds,
                  "fixture": "synthetic" if opt.smoke else "TCGA-BRCA",
                  "seed": opt.seed, "replay": opt.replay,
                  "checked": ["E", "F", "G", "D", "attention", "client_Adam", "client_schedulers",
                              "D_Adam", "D_scheduler", "replay_metadata", "metrics"]}
        (output / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PASS: Ray and NVFlare have identical parameters, optimizer state, graphs and metrics", flush=True)
    print(f"NVFlare results: {output}", flush=True)


if __name__ == "__main__":
    main()
