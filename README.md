# BreastG-FCL TCGA-BRCA Reproduction Instructions

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

The default configuration uses 10 clients, 3 continual tasks, 10 communication
rounds, 20 local epochs, a temporal window of 2 tasks, and differential
privacy. Configuration defaults are defined in
`TCGA-BRCA/configs/TCGA_BRCA.py`.

For a CPU connectivity test before the full run:

```bash
python TCGA-BRCA/main.py \
  --device cpu \
  --max-genes 16 \
  --num-local-epochs 1 \
  --num-rounds 1 \
  --gat-epochs 1 \
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
```

Check exact-author reproducibility separately:

```bash
python TCGA-BRCA/scripts/audit_healthcom26_reproduction.py --require-exact
```

This command intentionally returns exit status 2 while unpublished attention
architectures, four-region preprocessing, cohort mapping, and experimental
partition details remain unresolved.
