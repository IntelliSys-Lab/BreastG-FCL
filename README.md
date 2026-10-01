# BreastG-FCL: Graph-Conditioned Federated Continual Learning for Breast Cancer Radiogenomics

This repository contains the research implementation for our IEEE HealthCom 2026 paper:<br>
**BreastG-FCL: Graph-Conditioned Federated Continual Learning for Breast Cancer Radiogenomics**<br>
Qingyang Yu, Jingyi Wang, Xinyue Zhang, Miao Pan, Ziyue Xu, Hao Wang<br>
*Accepted at IEEE HealthCom, 2026.*

## Abstract

BreastG-FCL addresses learning across institutions whose data evolve over
successive tasks. It builds client relations from breast morphology and
DCE-MRI summaries, then uses the resulting graphs to condition representation
learning and generative replay. This implementation applies the method to
Normal versus Tumor classification on TCGA-BRCA RNA-seq using NVIDIA FLARE.
TCIA features supply graph context. A shared training workflow supports
NVFlare execution and comparison with the Ray reference backend.

## Papers and Links

[[Announcement]](https://intellisys.haow.us/news/) [[Ray Code]](https://github.com/IntelliSys-Lab/BreastG-FCL/tree/main) [[NVFlare Code]](https://github.com/IntelliSys-Lab/BreastG-FCL/tree/nvflare)

The BreastG-FCL title and citation follow the supplied camera-ready
manuscript. The announcement is not a paper download, and a public
paper/preprint URL has not been verified.

- GFedCL: [paper](https://ryougish1k1.github.io/assets/pdf/gfedcl.pdf) and
  [upstream implementation](https://github.com/IntelliSys-Lab/GFedCL), the
  reference for the component roles and spatial attention structure.
- [NVIDIA FLARE](https://github.com/NVIDIA/NVFlare).

## Objective

Demonstrate how an NVFlare server Controller and client Executors can run
graph-conditioned federated continual learning while preserving the
reference training operations. Readers can prepare the public data, run a
four-client experiment, inspect graph and model checkpoints, and check exact
CPU agreement between NVFlare and Ray.

## Method Summary

This is a horizontal federated learning simulation: clients have different
records with the same expression features and two-class target. Clinical
stage defines the task sequence, while client heterogeneity arises from
the assigned records and their TCIA summaries. TCIA features are used only
for the graph, not concatenated to the prediction input.

| Component | Operation | Training |
| --- | --- | --- |
| Encoder `E` | Expression vector + client graph row → latent | Client |
| Predictor `F` | Real or generated latent → class prediction | Client |
| Generator `G` | Gaussian noise + label + graph row → replay latent | Client |
| Discriminator `D` | Latent → reconstructed graph row | Server |
| Spatial/temporal attention | Morphology/DCE summaries → client relations | Forward computation only; no attention optimizer |

For each task, spatial multi-head additive attention and temporal
query/key attention over the latest two tasks are multiplied and normalized
by row. Each communication round uploads real and synthetic latents, updates
`D` once on the server, trains `E/F/G` on clients, and averages their weights
with equal client weights. Client optimizer and scheduler states persist
between rounds.

`G` generates latent features, not expression records or images. With replay
enabled, synthetic training covers all tasks up to and including the current
task, using saved label counts and graph rows without reading historical raw
inputs. `E` never receives labels. During client training, `D` has frozen
parameters but passes gradients to `E/G`. The local objective combines
prediction NLL with a negative, weighted graph-reconstruction MSE; the server
minimizes that MSE. Graph and latent Laplace-noise operations remain in the
shared workflow.

See the [implementation notes](TCGA-BRCA/README.md) for attention equations,
network dimensions, replay details, and transport contracts.

## Repository Layout

```text
TCGA-BRCA/
├── main.py                 # NVFlare entry point
├── nvflare_job.py          # Data packaging, job export, simulator, parity check
├── breastgfcl.py           # Shared task/round coordinator
├── configs/                # Defaults and paper-reproduction protocol
├── federated/              # Shared client operations and transport adapters
├── model/                  # E, F, G, D and attention networks
├── utils/                  # Data loading, partitioning and evaluation
├── scripts/                # Data downloads and source audit
├── tests/                  # Training and transport regression tests
└── metadata/               # GDC manifest and clinical metadata snapshot
```

## Setup

Use a Linux environment with Python 3.10 and **NVFlare 2.7.1**. The commands
below create an isolated Conda environment. CPU execution is supported for
the synthetic smoke test and exact backend comparison. CUDA training needs
a compatible NVIDIA driver and PyTorch build.

The local Ray development runs used one NVIDIA GeForce RTX 3090 (24 GB)
per run. Real-data execution also needs the downloaded expression files,
clinical metadata, and TCIA feature tables described below.

```bash
git clone --branch nvflare https://github.com/IntelliSys-Lab/BreastG-FCL.git
cd BreastG-FCL
conda env create -f environment.yml
conda activate gfedcl
```

Alternatively, in an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA environment when using a
GPU. Run all subsequent commands from the repository root.

## Data Preparation

### TCGA-BRCA expression and clinical data

Source: the NCI GDC [TCGA-BRCA project](https://portal.gdc.cancer.gov/projects/TCGA-BRCA).
The downloader selects only open-access STAR-Counts expression files;
[GDC open data access](https://gdc.cancer.gov/access-data/data-access-processes-and-tools)
does not require authentication. Data use remains subject to
[GDC policies](https://gdc.cancer.gov/about-gdc/gdc-policies), including the
source acknowledgement requirements.

The repository includes a GDC clinical metadata snapshot. The downloader
refreshes the expression manifest and downloads STAR-Counts files; it does
not refresh clinical metadata. Expression files are downloaded separately
and are not included in this repository.

**On a fresh clone without raw data**, first move aside the checked-in
download ledger. It records a previous download and would otherwise cause
the script to skip files that are absent locally:

```bash
mv -n TCGA-BRCA/metadata/downloaded_files.json \
  TCGA-BRCA/metadata/downloaded_files.json.bak
python TCGA-BRCA/scripts/download_tcga_brca.py
```

For subsequent downloads, run only the Python command to resume using the
local ledger. If raw files have been removed, their ledger entries must also
be reset. `--metadata-only` refreshes the manifest without downloading
expression files. Preserve the manifest, clinical snapshot, and raw data
used by an experiment: a later GDC query can return a different cohort.

Expected inputs:

```text
TCGA-BRCA/metadata/gdc_files_manifest.tsv
TCGA-BRCA/metadata/gdc_clinical_cases.tsv
TCGA-BRCA/data/raw/<file_id>/*.rna_seq.augmented_star_gene_counts.tsv
```

### TCIA spatial and temporal features

Source: the TCIA [TCGA-Breast-Radiogenomics analysis result](https://www.cancerimagingarchive.net/analysis-result/tcga-breast-radiogenomics/),
data DOI [10.7937/K9/TCIA.2014.8SIPIY6G](https://doi.org/10.7937/K9/TCIA.2014.8SIPIY6G).
The five source artifacts used by this downloader are listed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Follow the dataset's
citation instructions and [TCIA usage policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).
This path uses the public analysis files and does not require downloading
the original DICOM collection.

Download those artifacts with checksum verification, then inspect the
source/reproduction audit:

```bash
python TCGA-BRCA/scripts/download_tcia_official_radiogenomics.py
python TCGA-BRCA/scripts/audit_healthcom26_reproduction.py
```

The downloader produces `official_spatial_patient_features.csv`,
`official_temporal_patient_features.csv`, and `official_source_manifest.json`
in `TCGA-BRCA/data/tcia_official_radiogenomics/`. The public tables contain
84 matched patients, with 25 morphology fields and 11 kinetic fields.
Client/task graph summaries aggregate matching training patients only.

### Continual tasks and client partition

The default `clinical_stage` strategy defines three successive tasks. Each
has the same two labels: Normal (`0`) and Tumor (`1`).

| Task | Tumor population | Controls |
| --- | --- | --- |
| 0: early | AJCC Stage 0/I | Assigned Normal records |
| 1: intermediate | AJCC Stage II | Assigned Normal records |
| 2: advanced | AJCC Stage III/IV or metastatic | Assigned Normal records |

Normal records are distributed across tasks. Within each task and label,
records are split approximately 80/20 into train/test, then distributed
across four clients. Unknown-stage tumors are excluded by default. These
are stage-defined tasks, not longitudinal visits of the same patient.
The loader saves the assignments in
`OUTPUT_DIR/tcga_brca_partitioning_seed<seed>.pkl`.

The expression input defaults to 4,096 `tpm_unstranded` gene values with
`log1p` and standardization. Partitioning is at expression-file level, not
patient level; see the evaluation limitations below.

## Run Instructions

### Default NVFlare simulation

After preparing the data:

```bash
python TCGA-BRCA/main.py \
  --seed 42 \
  --output-dir TCGA-BRCA/dump/nvflare_default_seed42
```

The launcher exports an NVFlare job, then starts the simulator in a fresh
Python process. Choose a fresh output directory for each run: an existing
`nvflare_job/` will not be overwritten.

### Selected development configuration

The following applies the configuration selected during the local parameter
search. It is an explicit override of the repository defaults:

```bash
python TCGA-BRCA/main.py \
  --seed 42 --num-clients 4 --num-task 3 \
  --num-rounds 10 --num-local-epochs 3 \
  --nh 800 --noise-dim 100 --batch-size 32 \
  --replay true --lambda-gan 0.1 \
  --lr-e 1e-4 --lr-f 3e-4 --lr-g 1e-4 --lr-d 1e-3 \
  --p 0.1 --no-bn true --shuffle false \
  --sensitivity 1.0 --epsilon 1.0 \
  --output-dir TCGA-BRCA/dump/nvflare_selected_seed42
```

| Setting | Repository default | Selected configuration |
| --- | ---: | ---: |
| Clients / tasks | 4 / 3 | 4 / 3 |
| Rounds per task | 10 | 10 |
| Local epochs per round | 20 | 3 |
| Latent / noise dimension | 800 / 100 | 800 / 100 |
| Graph row / discriminator output dimension | 4 (client count) | 4 |
| Batch size | 32 | 32 |
| Learning rates `E / F / G / D` | `1e-4 / 1e-4 / 1e-4 / 1e-4` | `1e-4 / 3e-4 / 1e-4 / 1e-3` |
| `lambda_gan` | 0.5 | 0.1 |
| Dropout `p` | 0.2 | 0.1 |
| `no_bn` | `true` | `true` |
| Training DataLoader shuffle | `true` | `false` |
| Replay | Enabled | Enabled |

Despite its name, `--no-bn true` disables the encoder's **LayerNorm**.
`--shuffle false` changes training batch order, not the data partition.
Graph dimensions follow `--num-clients`; `--nt` and `--nd-out` are
compatibility options. All defaults are in
[TCGA_BRCA.py](TCGA-BRCA/configs/TCGA_BRCA.py).

### Simulator options

| Option | Purpose |
| --- | --- |
| `--device cpu` | Run without CUDA; CUDA is selected by default when available |
| `--nvflare-gpu 0` | Simulator GPU selection; defaults to `0` for CUDA runs |
| `--nvflare-threads 4` | Simulator concurrency; defaults to the client count |
| `--nvflare-timeout 300` | Client-operation timeout in seconds; increase for longer local training |
| `--nvflare-workspace PATH` | Override the simulator workspace |
| `--export-only` | Export the job without training |

Each exported `app_site_<n>/data/site.pt` contains that site's train/test
partition. `app_server/data/server.pt` holds coordination state and summaries,
without raw expression inputs. Application `custom/` directories contain
code only.

### Outputs and verification

A completed NVFlare run writes:

```text
OUTPUT_DIR/
├── tcga_brca_partitioning_seed<seed>.pkl  # Real-data runs
├── final_state.pt
├── round_accuracy.csv
├── all_tasks_accuracy.csv
├── relational_graphs/
│   ├── task_<k>_spatial.npy
│   ├── task_<k>_temporal.npy
│   ├── task_<k>_temporal_window.npy
│   ├── task_<k>_fused.npy
│   └── task_<k>_attention.pt
├── simulator.log
├── nvflare_job/
└── nvflare_workspace/
```

`final_state.pt` includes model and attention state, client/server optimizer
and scheduler state, replay metadata, and metrics. Simulator/site logs live
in the workspace; `simulator.log` captures the simulator console. Export-only
runs produce the job without training results.

#### Backend verification

To check the complete NVFlare/Ray workflow without downloading data:

```bash
python TCGA-BRCA/nvflare_job.py \
  --device cpu --smoke --verify-ray \
  --output-dir TCGA-BRCA/dump/smoke_parity
```

The synthetic fixture uses four clients, three tasks, two rounds per task,
and one local epoch, with reduced model dimensions. Its scores are not
TCGA results. `--verify-ray` requires exact CPU equality of model tensors,
attention, optimizer/scheduler state, replay metadata, and metrics. A passing
run writes `parity_report.json` and `ray_reference/`; a mismatch raises an
error. Omit `--smoke` to compare the real-data workflow on CPU.

Run the regression suite with:

```bash
python -m unittest discover -s TCGA-BRCA/tests -v
```

## Expected Results

A successful real-data simulation produces the checkpoints, graph matrices,
and metric files listed above. A successful CPU parity check writes
`parity_report.json` with `"equal": true` and checks the complete saved state.
Smoke-test accuracy is only a diagnostic of the synthetic workflow, not an
expected TCGA score.

### Recorded development benchmark

The selected configuration was evaluated with the **Ray backend on `main`**
at commit `fe97e2347fa2c6ebac408868cc11d3d18c1df296`. All three runs used the
same frozen seed-42 data partition; only the training seed changed. Each
completed 3 tasks × 10 rounds, followed by one final evaluation pass.

| Training seed | Accuracy (%) | Balanced accuracy (%) | Normal recall (%) |
| --- | ---: | ---: | ---: |
| 42 | 93.30 | 81.57 | 69.57 |
| 43 | 95.68 | 82.99 | 69.57 |
| 44 | 95.41 | 82.84 | 69.57 |
| Mean ± sample standard deviation | 94.80 ± 1.30 | 82.47 ± 0.78 | 69.57 ± 0.00 |

Accuracy and balanced accuracy are unweighted means over the 12 client/task
cells. Normal recall pools predictions across cells: all three runs identify
16 of 23 Normal records. The benchmark contains 896 training records and
223 evaluation records (23 Normal, 200 Tumor), after excluding 112
unknown-stage tumors from the 1,231-record expression cohort.

**These are development-benchmark results.** The former test split was used
to select among 16 full-training configurations, so these scores are not
independent test estimates or reproduction of the paper's reported score.
93.30% is the seed-42 result, not a stable three-seed average. The NVFlare
command above applies the selected hyperparameters; it does not establish
these scores on NVFlare. Changing its `--seed` also rebuilds the partition,
so a plain CLI seed sweep differs from the fixed-partition replications.

The saved experiment bundles, checkpoints, and search scripts are local
artifacts under the ignored `TCGA-BRCA/dump/` directory and are not shipped
with this repository. Exact replay of the table requires those frozen
artifacts in addition to the training source.

The inherited data protocol also limits interpretation: expression
standardization uses the full cohort, TCIA normalization uses the full
84-patient table, and the recorded train/test split shares 41 patient IDs
and 2 sample IDs. The selected run's discriminator loss increases markedly
late in training; completing all rounds with finite tensors does not
establish stable adversarial convergence.

### Paper-reproduction scope

This implementation follows the paper's graph fusion and temporal-window
formulation, with GFedCL-inspired attention networks and public TCIA
features. It does not recover the original unpublished attention weights,
attention optimization, complete four-region MRI preprocessing, cohort
mapping, or experimental partition. Attention is differentiable but runs
without training; `--gat-epochs` does not optimize it. The generator noise
width of 100 is an implementation choice.

The public-source audit checks the available artifacts. The stricter command
below intentionally exits with status `2` while exact-author reproduction
remains unresolved:

```bash
python TCGA-BRCA/scripts/audit_healthcom26_reproduction.py --require-exact
```

See the [reproduction protocol](TCGA-BRCA/configs/healthcom26_reproduction_protocol.json)
for the recorded assumptions and missing author details.

## License

This repository is released under the [MIT License](LICENSE).

External code, data, and dependencies retain their own terms:

| Source | License or data terms |
| --- | --- |
| GFedCL reference implementation | [MIT](https://github.com/IntelliSys-Lab/GFedCL/blob/main/LICENSE); retain upstream notices when reusing its code |
| NVIDIA FLARE | [Apache-2.0](https://github.com/NVIDIA/NVFlare/blob/main/LICENSE) |
| TCGA-BRCA expression and clinical data | [GDC policies](https://gdc.cancer.gov/about-gdc/gdc-policies) |
| TCIA radiogenomic analysis artifacts | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), with the dataset's citation requirements |

The project's MIT license does not relicense these datasets or dependencies.
Data source links and access instructions are in [Data Preparation](#data-preparation).
No pretrained author checkpoints are distributed with this implementation.

## Requirements

The project dependency specifications are [requirements.txt](requirements.txt)
and [environment.yml](environment.yml). NVFlare is pinned to **2.7.1**;
Ray remains a dependency for the reference transport and parity check. No
unreleased NVFlare features or source installation are required.

The locally verified environment used:

| Dependency | Verified version |
| --- | --- |
| Python | 3.10.12 |
| NVFlare | 2.7.1 |
| PyTorch / torchvision | 2.2.1+cu118 / 0.17.1+cu118 |
| Ray | 2.55.1 |
| NumPy | 1.26.4 |

The requirements files allow broader versions for several packages; they
are not an exact lockfile for this environment. Spreadsheet readers
`openpyxl==3.1.5` and `xlrd==2.0.2` support the TCIA source workbooks.

## Citation

```bibtex
@inproceedings{yu2026breastgfcl,
  title     = {{BreastG-FCL}: Graph-Conditioned Federated Continual Learning for Breast Cancer Radiogenomics},
  author    = {Yu, Qingyang and Wang, Jingyi and Zhang, Xinyue and Pan, Miao and Xu, Ziyue and Wang, Hao},
  booktitle = {IEEE HealthCom},
  year      = {2026}
}

@inproceedings{yu2026gfedcl,
  title     = {{GFedCL}: Graph-Based Federated Continual Learning with Spatial and Temporal Awareness},
  author    = {Yu, Qingyang and Hua, Yang and Zhang, Qizhen and Wang, Hao},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
