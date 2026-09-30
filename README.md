# BreastG-FCL

This branch runs BreastG-FCL with **NVFlare 2.7.1** by default. NVFlare and
the Ray verification backend use the same coordinator, client operations,
models, losses, replay logic, and aggregation.

Run every command from the repository root.

## 1. Install the environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate gfedcl
```

Or using pip:

```bash
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build separately when GPU execution is
required.

## 2. Prepare TCGA-BRCA expression data

Download the GDC manifest, clinical metadata, and STAR-Counts files:

```bash
python TCGA-BRCA/scripts/download_tcga_brca.py
```

To inspect the manifest before downloading the expression files:

```bash
python TCGA-BRCA/scripts/download_tcga_brca.py --metadata-only
```

The training workflow expects:

```text
TCGA-BRCA/metadata/gdc_files_manifest.tsv
TCGA-BRCA/metadata/gdc_clinical_cases.tsv
TCGA-BRCA/data/raw/<file_id>/*.rna_seq.augmented_star_gene_counts.tsv
```

Interrupted downloads can be resumed by running the same command again.

## 3. Prepare TCIA graph features

Download and checksum the official TCGA-Breast-Radiogenomics artifacts, then
build the spatial and temporal patient tables:

```bash
python TCGA-BRCA/scripts/download_tcia_official_radiogenomics.py
```

The graph workflow expects:

```text
TCGA-BRCA/data/tcia_official_radiogenomics/official_spatial_patient_features.csv
TCGA-BRCA/data/tcia_official_radiogenomics/official_temporal_patient_features.csv
TCGA-BRCA/data/tcia_official_radiogenomics/official_source_manifest.json
```

Verify the public source artifacts:

```bash
python TCGA-BRCA/scripts/audit_healthcom26_reproduction.py
```

## 4. Run with NVFlare

After preparing the real TCGA/TCIA data, run the default experiment:

```bash
python TCGA-BRCA/main.py \
  --seed 42 \
  --output-dir TCGA-BRCA/dump/breastgfcl_seed42
```

`TCGA-BRCA/main.py` delegates to `nvflare_job.main`, which exports an NVFlare
job and runs the NVFlare simulator in a fresh Python process. The real-data defaults are
4 clients, 3 continual tasks, 10 communication rounds per task, 20 local
epochs per round, 800-dimensional latents, 100-dimensional generator noise,
replay enabled, and a temporal window of 2 tasks. The existing Laplace-noise
steps remain in the coordinator. Defaults are defined in
`TCGA-BRCA/configs/TCGA_BRCA.py`.

Choose a fresh output directory for each run: the exporter refuses to replace
an existing `nvflare_job/` directory. To export without starting the simulator,
add `--export-only`. `--nvflare-timeout` controls the client-operation timeout
(default: 300 seconds); increase it for longer local training. The simulator
also accepts `--nvflare-workspace`, `--nvflare-threads`, and `--nvflare-gpu`.

Data preparation calls the existing TCGA loader once and preserves its
client/task assignments. Each exported `app_site_<n>/data/site.pt` contains
only that site's assigned train/test data. `app_server/data/server.pt`
contains summaries, model state, label counts, and related metadata, without
raw expression inputs. The `custom/` directories contain Python code only.

Graph conditioning rows and discriminator outputs have one dimension per client
(4 by default). Their dimensions automatically follow `--num-clients`;
the compatibility options `--nt` and `--nd-out` do not override this setting.

Spatial attention uses a GFedCL-inspired summary encoder (`128 -> 64`) and
four additive GAT heads with 32 dimensions per head. Each head applies
LeakyReLU and row-wise softmax; the four attention matrices are averaged.
LayerNorm replaces the reference encoder's BatchNorm to support small client
counts. Temporal attention projects the concatenated DCE summaries from the
latest `m` tasks into 64-dimensional queries and keys, then applies scaled
dot-product attention. Early windows are left-padded with zeros so the same
projection layers apply to every task. Spatial and temporal attention are
multiplied and row-normalized.

The attention networks have differentiable `forward` methods but are not
trained separately by default. The seed fixes their initialization, and
`learn()` is a compatibility inference entry point: `--gat-epochs` does not
train these networks. Inference temporarily disables dropout. Per-task
checkpoints record the attention parameters, DCE history, and network
configuration. These reference networks do not recover the authors'
unpublished attention weights or training procedure.

The encoder `E`, predictor `F`, and replay generator `G` train jointly on
clients; the discriminator `D` trains on the server. `E` takes expression data
and a graph row, without labels. `G` takes Gaussian noise, labels, and a graph
row and produces an 800-dimensional latent by default. Its default noise
width of 100 (`--noise-dim`) is an implementation choice, not a reported
author setting.

With `--replay true` (the default), local training combines current real data
with synthetic latents for every task up to and including the current task.
Historical labels are sampled from saved per-task label counts; replay does
not read historical raw inputs. With `--replay false`, `G` still trains on
synthetic latents for the current task. Current and replay losses run in the
same client operation, and all resulting `E/F/G` updates enter equal-weight
aggregation. Each client's optimizer, scheduler, and task metadata return to
the coordinator for the next round. `D` takes one optimizer step per round.
NVFlare changes task delivery, not these training steps or the positions of
the graph/latent Laplace-noise operations. Each client operation has a seed
derived from the experiment, task, round, client, and operation; its random
number state is restored afterward so execution does not advance the
server's random number generators.

The server trains `D` to reconstruct graph rows by minimizing MSE. A client's
copy of `D` has frozen weights, but its input gradients remain enabled so the
adversarial loss updates `E/G`. Only the graph-construction attention networks
remain without a training objective. Old encoder/generator checkpoints
cannot be loaded directly into these new `E/G` architectures.

For a CPU simulator test and exact comparison with Ray, use a fresh directory:

```bash
python TCGA-BRCA/nvflare_job.py \
  --device cpu \
  --smoke \
  --verify-ray \
  --output-dir /tmp/unique_directory
```

`--smoke` uses a **synthetic fixture**, with the default 4 clients and 3 tasks,
2 rounds per task, 1 local epoch, 8 input features, 16 latent dimensions,
5 noise dimensions, and batch size 4. It requires no TCGA/TCIA downloads and
is only a transport/training check; its metrics are not paper results.

`--verify-ray` runs both backends from the same prepared partition and initial
state. It requires CPU execution and exact equality of model tensors,
attention state, client and discriminator optimizer/scheduler state, replay metadata, and
metrics. A successful comparison writes `parity_report.json`; a mismatch
raises an error. It cannot be combined with `--export-only`.

## 5. Validate the implementation

```bash
python -m compileall -q TCGA-BRCA
python -m unittest discover -s TCGA-BRCA/tests -v
```

The regression tests cover graph attention, temporal history, E/F/G/D updates,
replay, RNG isolation, and transport contracts. Use the simulator command
above to verify complete Ray/NVFlare CPU parity.

## 6. Inspect outputs

Each output directory contains:

```text
final_state.pt
simulator.log
nvflare_job/
nvflare_workspace/
round_accuracy.csv
all_tasks_accuracy.csv
relational_graphs/task_<k>_spatial.npy
relational_graphs/task_<k>_temporal.npy
relational_graphs/task_<k>_temporal_window.npy
relational_graphs/task_<k>_fused.npy
relational_graphs/task_<k>_attention.pt
```

`final_state.pt` records E/F/G/D parameters, attention state, client and
discriminator optimizer/scheduler state, replay metadata, and metrics.
Simulator console output is in `simulator.log`; site/server logs are in its workspace
(or the directory selected by `--nvflare-workspace`). With `--verify-ray`,
`ray_reference/` contains the reference run and `parity_report.json` records
a successful comparison. An export-only run creates the job but does not
produce training results.

Check exact-author reproducibility separately:

```bash
python TCGA-BRCA/scripts/audit_healthcom26_reproduction.py --require-exact
```

This command intentionally returns exit status 2 while the original attention
parameterization, trained weights and optimization procedure, four-region
preprocessing, cohort mapping, and experimental partition details remain
unresolved.
