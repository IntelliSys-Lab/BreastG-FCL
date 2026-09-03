# BreastG-FCL with the NVFlare Collab API

This research example expresses a graph-based federated continual-learning
workflow as ordinary Python calls with the NVFlare Collab API. The server uses
the return values of remotely published client methods directly; NVFlare hides
the task dispatch, transport, and result collection.

The initial version is deliberately self-contained. It uses deterministic,
non-IID synthetic data so that reviewers can run the complete workflow without
access to controlled medical data. A real TCGA/TCIA site-local data adapter can
replace the synthetic provider without changing the federated control flow.

## Workflow

For each continual task, the server performs the following operations:

1. Call `get_graph_summary()` on every site.
2. Construct the spatial-temporal client relationship graph.
3. Call `collect_encodings()` and train the global graph discriminator.
4. Call `train()` with the global model, discriminator, and relationship graph.
5. Aggregate client models by the number of locally processed examples.
6. Call `evaluate()` for the current and previous tasks.

The control flow is implemented in `server.py` with `@collab.main`.
Site-local operations are implemented in `client.py` with `@collab.init` and
`@collab.publish`. All modules live directly in this research project directory
to match the compact layout used by the Collab examples.

## Project layout

```text
research/breastg-fcl/
├── job.py                 # CollabRecipe and simulator entry point
├── server.py              # @collab.main federated workflow
├── client.py              # Site-local @collab.publish methods
├── aggregation.py         # Result validation and weighted aggregation
├── graph.py               # Spatial-temporal relationship graph
├── model.py               # PyTorch model and discriminator
├── data.py                # Synthetic site-local continual data
├── config.py              # Shared experiment configuration
├── tests/                 # Focused unit tests
└── requirements.txt
```

## Install

The Collab API is available in the NVFlare 2.9 release line. This initial
version accepts the current 2.9 release candidate so it can be tested before
the final package is published. When working from the NVFlare `main` branch,
install the repository in editable mode and then install the example
requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r research/breastg-fcl/requirements.txt
```

## Run a smoke experiment

From this directory:

```bash
python job.py \
  --num-clients 2 \
  --num-tasks 2 \
  --rounds-per-task 1 \
  --local-epochs 1
```

The simulator writes server and site logs and the generated NVFlare job under:

```text
/tmp/nvflare/simulation/breastg_fcl_collab/
```

The `@collab.main` return value contains the final model and discriminator
states, graph artifacts, and metric history. Explicit checkpoint export is
planned for the next iteration.

Run with `--help` for the available experiment controls. GPU assignment can be
provided through NVFlare's simulator mapping syntax, for example
`--gpu "[0],[1]"`.

## Tests

```bash
cd research/breastg-fcl
python -m unittest discover -s tests -v
```

The tests cover graph normalization and temporal history, weighted aggregation,
failure handling, deterministic site data, and model tensor shapes.

## Privacy note

`encoding_noise_scale` and `summary_noise_scale` apply Laplace perturbation at
the client before values are returned to the server. These controls are useful
for testing where privacy transformations belong in a Collab workflow, but they
do not by themselves constitute a formal differential-privacy guarantee. A
production study must define clipping, sensitivity, accounting, and an explicit
privacy budget.

The Collab API does not use NVFlare's standard task/result filter pipeline, so
privacy transformations required by this example are performed explicitly in
the published client methods.

## Current scope

- Included: synchronous Collab workflow, synthetic site data, continual tasks,
  spatial-temporal graph construction, discriminator training, local replay,
  weighted model aggregation, and cross-task evaluation.
- Not yet included: TCGA/TCIA data adapter, production provisioning, automatic
  retry/checkpoint resume, experiment figures, or exact paper reproduction.
- The fixed distance-based attention function is a transparent reproduction
  assumption until the original learned attention architecture is available.

## License

This contribution is intended for the NVIDIA FLARE repository and follows its
Apache License 2.0 contribution requirements. Medical datasets, pretrained
models, and third-party assets are not redistributed by this example.
