# BreastG-FCL with NVIDIA FLARE

This branch implements the BreastG-FCL research workflow with the
[NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) Collab API. The server calls
published client methods like remote Python functions and uses their return
values directly; FLARE handles task dispatch, transport, and result collection.

The current version uses deterministic, non-IID synthetic data so the complete
federated continual-learning workflow can be run without access to controlled
medical datasets. It is an initial research prototype intended to evolve into
an example contribution to the NVIDIA FLARE repository.

## What is implemented

- A synchronous Collab API server/client workflow.
- Multiple clients and sequential continual-learning tasks.
- Spatial-temporal client relationship graph construction.
- Global graph-discriminator training.
- Local replay and weighted model aggregation.
- Evaluation over the current and previous tasks.
- Optional client-side Laplace perturbation of graph summaries and encodings.

The federated control flow is defined in
[`server.py`](research/breastg-fcl/server.py), while site-local operations are
published from [`client.py`](research/breastg-fcl/client.py).

## Project layout

```text
research/breastg-fcl/
├── job.py                 # CollabRecipe and simulator entry point
├── server.py              # @collab.main federated workflow
├── client.py              # Site-local @collab.publish methods
├── aggregation.py         # Result validation and aggregation
├── graph.py               # Spatial-temporal relationship graph
├── model.py               # PyTorch model and discriminator
├── data.py                # Synthetic continual-learning data
├── config.py              # Experiment configuration
├── tests/                 # Focused unit tests
└── requirements.txt
```

See the [research example README](research/breastg-fcl/README.md) for the
workflow design, privacy notes, and current limitations.

## Install

Run all commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r research/breastg-fcl/requirements.txt
```

The prototype requires Python, PyTorch, and `nvflare>=2.9.0rc3,<3.0`. Install a
CUDA-compatible PyTorch build separately if GPU execution is required.

## Run a smoke experiment

```bash
python research/breastg-fcl/job.py \
  --num-clients 2 \
  --num-tasks 2 \
  --rounds-per-task 1 \
  --local-epochs 1
```

By default, the simulator writes its workspace and generated FLARE job under:

```text
/tmp/nvflare/simulation/breastg_fcl_collab/
```

Use `python research/breastg-fcl/job.py --help` to see all experiment options.
For example, GPU mappings can be supplied with `--gpu "[0],[1]"`.

## Run tests

```bash
PYTHONPATH=research/breastg-fcl \
  python -m unittest discover -s research/breastg-fcl/tests -v
```

The tests cover graph normalization and temporal history, weighted aggregation,
failure handling, deterministic site data, and model tensor shapes.

## Current scope

This version does not include a TCGA/TCIA data adapter, production provisioning,
automatic checkpoint recovery, experiment figures, or an exact reproduction of
the paper. Real clinical data must remain site-local and can be integrated later
through a data-provider adapter without changing the Collab API control flow.

## License

The project is licensed under the [Apache License 2.0](LICENSE). Medical
datasets, pretrained models, and third-party assets are not redistributed.
