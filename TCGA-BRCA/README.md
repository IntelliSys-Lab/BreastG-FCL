# TCGA-BRCA BreastG-FCL

This implementation combines disease-aware TCIA spatial/temporal graph
construction with the GFedCL encoder, predictor, latent replay generator,
and server discriminator roles. Graph attention runs forward without a
separate training objective; `E/F/G/D` are trained. The TCGA prediction
cohort, labels, and task/client partition remain unchanged. This branch uses
NVFlare 2.7.1 by default and retains Ray for backend-equivalence checks.

## Data roles

| Source | Workflow role |
| --- | --- |
| Existing TCGA-BRCA GFedCL loaders | Prediction samples, labels, clients, and tasks (unchanged) |
| TCIA lesion morphology radiomics | Spatial client/task summary `s_i^k` |
| TCIA DCE kinetic radiomics | Temporal client/task summary `r_i^k` |

TCIA features are aggregated from the training patients already assigned to
each client/task and are used only to generate the graph. They are not
concatenated to the TCGA input and do not replace the existing GFedCL training
cohort, labels, or partition.

The public TCIA analysis result provides 91 lesion-radiomics rows, of which 84
match its PAM50 and clinical workbooks. The downloader separates 25 morphology
fields and 11 kinetic fields and records source checksums:

```bash
python TCGA-BRCA/scripts/download_tcia_official_radiogenomics.py
```

## Learning and replay

| Component | Input and output | Update |
| --- | --- | --- |
| Encoder `E` | RNA-seq vector + graph row → latent | Trained locally on current real samples; no label input |
| Generator `G` | Gaussian noise + label + graph row → synthetic latent | Trained locally on current and enabled replay tasks |
| Predictor `F` | Real or synthetic latent → class prediction | Trained locally together with `E/G` |
| Discriminator `D` | Real or synthetic latent → graph row | Trained on the server by minimizing graph-row MSE |
| Graph attention | TCIA spatial/DCE summaries → relational graph | Forward computation only; no attention optimizer |

Both `E` and `G` output `nh` dimensions (800 by default). The graph row and
`D` output have one dimension per client (4 by default). `G` uses a label
embedding and graph projection, concatenates them with noise, and maps the
result through an MLP to the latent space. Its default noise dimension of
100 (`--noise-dim`) is a public implementation choice; the papers and local
GFedCL reference do not provide an original author value.

For task `k`, clients upload current real latents and synthetic latents for
tasks `j <= k` when replay is enabled. The server updates `D` on those
latents and their matching graph rows, then sends `D` back to clients.
Local training minimizes prediction NLL minus `lambda_gan` times graph-row
MSE, using current real latents from `E` and the sum of synthetic losses
from `G`. The local copy of `D` stays in evaluation mode with frozen
parameters; autograd through its input remains enabled, so its adversarial
loss reaches both `E` and `G`.

`--replay true` covers every prior task as well as the current task.
`--replay false` removes historical synthetic losses but still trains `G`
on the current task. Clients retain per-task label counts and batch sizes;
historical labels are sampled from those counts and combined with fresh
Gaussian noise and the corresponding task's graph row. Replay does not
read historical raw expression samples. Current-task labels can condition
`G`, but never enter `E`, including during evaluation.

Each client training operation computes current and replay updates on the
same client instance. The coordinator averages all clients' `E/F/G` weights
with equal weight and redistributes them, retaining each client's returned
Adam state, scheduler state, and task metadata for subsequent rounds. The
server takes one `D` optimizer step per communication round. The joint Adam optimizer
has separate parameter groups controlled by `--lr-e`, `--lr-f`, and
`--lr-g`; the server uses `--lr-d`. All four default to `1e-4`.

The new encoder has no label embedding, and `GNet` now generates latents
instead of graph embeddings. Old `E/G` checkpoints therefore cannot be
loaded directly into these architectures; start a new run or provide an
explicit migration. Attention checkpoints described below have a separate
format and purpose.

## Relational graph

For task `k`, the graph generator standardizes each summary feature across
clients (as in the previous scorer), then computes

```text
alpha_ij^k = mean_h softmax_j a_s,h(s_i^k, s_j^k)
R_i^k      = r_i^{max(1,k-m+1)} || ... || r_i^k
q_i^k      = W_Q pad_left(R_i^k)
key_j^k    = W_K pad_left(R_j^k)
beta_ij^k  = softmax_j ((q_i^k)^T key_j^k / sqrt(64))
G_ij^k     = alpha_ij^k beta_ij^k /
             (sum_l alpha_il^k beta_il^k + epsilon)
```

The temporal window and multiplicative fusion follow the paper directly.
Spatial attention uses the local GFedCL reference network structure: a
client-summary encoder with hidden/output dimensions `128 -> 64`, followed
by four additive GAT heads, each with a 32-dimensional projection. A head
scores a client pair with `LeakyReLU(a^T [W h_i || W h_j])` and applies
row-wise softmax; the four attention matrices are averaged. The encoder uses
LayerNorm in place of the reference's BatchNorm to support small client
counts. Configurable dropout defaults to `0.2`.

Temporal attention uses trainable query/key projections and the paper's scaled
dot-product score. It concatenates the latest `m` DCE summaries, including the
current task, and left-pads early windows with zeros to a fixed width of
`m * temporal_feature_dimension`. The same projection layers produce
64-dimensional queries and keys for every task. The attention temperature
divides the spatial and temporal scores before their respective softmaxes;
the equations above show the default temperature of `1.0`.

Each task writes its spatial attention, temporal attention, temporal-window
input, and fused graph to `OUTPUT_DIR/relational_graphs/*.npy` for auditing.
It also saves `task_<k>_attention.pt` in that directory, containing the
attention `state_dict` (including DCE history in extra state) and
`network_config` (dimensions, window, temperature, epsilon, and related
settings).

To restore a checkpoint on CPU, construct `BreastGraphGenerator` with
`SimpleNamespace(**checkpoint["network_config"])` and load
`checkpoint["state_dict"]`. The stored DCE history allows graph generation
to continue with the next task.

Both networks expose a differentiable `forward` path. By default, no separate
attention training objective or optimizer is applied: the experiment seed
fixes initialization, and `learn()` is a compatibility inference entry point
whose `epochs` argument is unused. It temporarily switches to evaluation
mode to disable dropout and restores the previous mode afterward.
Consequently, a saved checkpoint records the reference network state, not
evidence of attention training. The authors' exact BreastG-FCL network
parameterization, trained weights, supervision, and optimization remain
unpublished; this implementation does not claim to recover them.

## NVFlare execution and data packaging

Install the repository environment, which pins `nvflare==2.7.1`. Ray remains
installed for optional parity verification. `python TCGA-BRCA/main.py` calls
`nvflare_job.main`; it exports a traditional NVFlare job and launches the
simulator in a fresh Python process, avoiding inherited PyTorch autograd threads
and CUDA state when the simulator forks. The server Controller and site Executors dispatch the shared
`federated.runtime.execute_client` operations. The existing `breastgfcl`
coordinator owns task/round order, graph construction, discriminator updates,
and aggregation for both transports.

Preparation calls `setup_tcga_brca_loaders` once for a real-data run. It
packages those existing splits by dataset index, without repartitioning:

| Exported artifact | Contents |
| --- | --- |
| `nvflare_job/app_site_<n>/data/site.pt` | Only that site's assigned train/test tensors and configuration |
| `nvflare_job/app_server/data/server.pt` | Client summaries, initial model state, task label counts/batch sizes, configuration, and RNG state; no raw expression inputs |
| Each application's `custom/` directory | Shared Python code, without the source `data/` or `dump/` directories |

The server reconstructs its coordination state from the server bundle and
sends operations to sites; it does not reload or repartition TCGA data.
NVFlare and Ray use the same model, loss, replay, optimizer, equal-weight
E/F/G aggregation, and once-per-round D update. Graph and latent Laplace noise
remain at the same points in that shared coordination loop.

Each operation receives a seed derived from the experiment seed, task,
round, client ID, and operation. The runtime saves and restores Python,
NumPy, Torch CPU, and the client's CUDA RNG state, so the transport does not
advance server randomness. Importing the coordinator does not start Ray.

## Run and verify

Run all commands from the repository root. Follow the root README's
[data preparation](../README.md#data-preparation) first, including resetting
the checked-in download ledger on a fresh clone without raw data. Then run:

```bash
python TCGA-BRCA/main.py --seed 42 --output-dir TCGA-BRCA/dump/nvflare_seed42
```

The root README also documents the [selected development configuration and
results](../README.md#selected-development-configuration). Its overrides do
not change the defaults below; the reported three-seed scores were measured
on Ray with a fixed partition, not by sweeping the NVFlare CLI seed.

The real-data defaults are 4 clients, 3 tasks, 10 rounds per task, 20 local
epochs per round, latent width 800, noise width 100, and replay enabled.
Choose a fresh output directory: an existing `nvflare_job/` is never replaced
by the exporter.

To exercise both transports without downloading data:

```bash
python TCGA-BRCA/nvflare_job.py \
  --device cpu \
  --smoke \
  --verify-ray \
  --output-dir /tmp/unique_directory
```

This uses a deterministic **synthetic fixture**, not the TCGA cohort. Its
default topology is 4 clients and 3 tasks, with 2 rounds per task and 1 local
epoch. Input, latent, and noise widths are 8, 16, and 5; batch size is 4.
The fixture also reduces attention dimensions for a short integration run.
Its metrics are not comparable with the paper's reported results.

`--verify-ray` compares both complete runs from the same data and model
initialization. CPU tensor values must match exactly, with no numeric
tolerance; client and discriminator Adam/scheduler state, replay metadata, attention state,
and metrics are also compared. A successful run writes `parity_report.json`;
a mismatch raises an error. Run this command to establish parity for your
environment. `--verify-ray` requires `--device cpu`; GPU numerical equivalence
needs separate validation.

| Option | Behavior |
| --- | --- |
| `--export-only` | Write the NVFlare job without starting training; incompatible with `--verify-ray` |
| `--nvflare-timeout 300` | Client-operation timeout in seconds; increase for longer training calls |
| `--nvflare-workspace PATH` | Simulator workspace; defaults to `OUTPUT_DIR/nvflare_workspace` |
| `--nvflare-threads N` | Simulator worker threads; defaults to the client count |
| `--nvflare-gpu IDS` | GPU assignment passed to the simulator; CPU runs need no GPU setting |

Graph controls remain available through the same default entry point:

```bash
python TCGA-BRCA/main.py \
  --temporal-window 2 \
  --attention-temperature 1.0 \
  --gat-dropout 0.2 \
  --graph-epsilon 1e-8
python -m unittest discover -s TCGA-BRCA/tests -v
```

After training, the output directory contains `final_state.pt`,
`round_accuracy.csv`, `all_tasks_accuracy.csv`, `relational_graphs/`,
`nvflare_job/`, and the default `nvflare_workspace/`. The final-state artifact
records E/F/G/D parameters, attention state, client and discriminator
optimizer/scheduler state, replay metadata, and
metrics. `--verify-ray` additionally writes the reference run under
`ray_reference/` and a `parity_report.json` on success. Simulator logs are in
the selected workspace, with console output in `simulator.log`. Export-only runs produce the job without training
artifacts.

Run the strict-reproduction audit with:

```bash
python TCGA-BRCA/scripts/audit_healthcom26_reproduction.py --require-exact
```

The audit succeeds for the public TCIA source artifacts but intentionally
returns a non-zero status for exact-author reproduction while the original
attention parameterization, weights and training procedure, four-region
preprocessing pipeline, cohort mapping, and experiment configuration remain
unpublished.
