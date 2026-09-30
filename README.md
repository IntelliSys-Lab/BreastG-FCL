# BreastG-FCL

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

## 4. Run the reproduction experiment

Run the default TCGA-BRCA reproduction experiment:

```bash
python TCGA-BRCA/main.py \
  --seed 42 \
  --output-dir TCGA-BRCA/dump/breastgfcl_seed42
```

The default configuration uses 4 clients, 3 continual tasks, 10 communication
rounds, 20 local epochs, a temporal window of 2 tasks, and differential
privacy. Configuration defaults are defined in
`TCGA-BRCA/configs/TCGA_BRCA.py`.

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
same Ray worker, and all resulting `E/F/G` updates enter aggregation. Each
client's optimizer, scheduler, and task metadata return to the driver for
the next round.

The server trains `D` to reconstruct graph rows by minimizing MSE. A client's
copy of `D` has frozen weights, but its input gradients remain enabled so the
adversarial loss updates `E/G`. Only the graph-construction attention networks
remain without a training objective. Old encoder/generator checkpoints
cannot be loaded directly into these new `E/G` architectures.

For a CPU connectivity test before the full run:

```bash
python TCGA-BRCA/main.py \
  --device cpu \
  --max-genes 16 \
  --num-local-epochs 1 \
  --num-rounds 1 \
  --output-dir TCGA-BRCA/dump/smoke_test
```

The smoke-test metrics are not paper-comparable results.

## 5. Validate the implementation

```bash
python -m compileall -q TCGA-BRCA
python -m unittest discover -s TCGA-BRCA/tests -v
```

The regression tests validate row-normalized attention, the temporal sliding
window, multiplicative spatial-temporal fusion, and strict input validation.

## 6. Inspect outputs

Each output directory contains:

```text
run.log
round_accuracy.csv
all_tasks_accuracy.csv
plots/
relational_graphs/task_<k>_spatial.npy
relational_graphs/task_<k>_temporal.npy
relational_graphs/task_<k>_temporal_window.npy
relational_graphs/task_<k>_fused.npy
relational_graphs/task_<k>_attention.pt
```

Check exact-author reproducibility separately:

```bash
python TCGA-BRCA/scripts/audit_healthcom26_reproduction.py --require-exact
```

This command intentionally returns exit status 2 while the original attention
parameterization, trained weights and optimization procedure, four-region
preprocessing, cohort mapping, and experimental partition details remain
unresolved.
