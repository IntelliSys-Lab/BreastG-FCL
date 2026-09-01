# TCGA-BRCA BreastG-FCL

This implementation keeps the existing GFedCL federated continual-learning,
replay, discriminator, and aggregation workflow unchanged. BreastG-FCL is
implemented only at the relational-graph boundary: the generic
model-update-derived graph is replaced by disease-aware TCIA spatial and
temporal attention.

## Data roles

| Source | Workflow role |
| --- | --- |
| Existing TCGA-BRCA GFedCL loaders | Prediction samples, labels, clients, and tasks (unchanged) |
| TCIA lesion morphology radiomics | Spatial client/task summary `s_i^k` |
| TCIA DCE kinetic radiomics | Temporal client/task summary `r_i^k` |

TCIA features are aggregated from the training patients already assigned to
each client/task and are used only to generate the graph. They are not
concatenated to the TCGA input and do not replace the existing GFedCL training
cohort, labels, partition, replay, discriminator, or aggregation logic.

The public TCIA analysis result provides 91 lesion-radiomics rows, of which 84
match its PAM50 and clinical workbooks. The downloader separates 25 morphology
fields and 11 kinetic fields and records source checksums:

```bash
python TCGA-BRCA/scripts/download_tcia_official_radiogenomics.py
```

## Relational graph

For task `k`, the graph generator computes

```text
alpha_ij^k = softmax_j a_s(s_i^k, s_j^k)
R_i^k      = r_i^{max(1,k-m+1)} || ... || r_i^k
beta_ij^k  = softmax_j b_t(R_i^k, R_j^k)
G_ij^k     = alpha_ij^k beta_ij^k /
             (sum_l alpha_il^k beta_il^k + epsilon)
```

The temporal window and multiplicative fusion follow the paper directly; the
former weighted-addition fusion has been removed.
Each task writes its spatial attention, temporal attention, temporal-window
input, and fused graph to `OUTPUT_DIR/relational_graphs/*.npy` for auditing.

The paper does not specify the architectures, learned parameters, supervision,
or optimization of `a_s` and `b_t`. To keep the public implementation
deterministic and auditable, both use a fixed negative squared-distance score
on standardized client summaries before the published row-wise softmax. This
is a declared reproduction assumption, not an exact recovery of the authors'
unpublished attention networks.

## Run and verify

```bash
python TCGA-BRCA/main.py
python -m unittest discover -s TCGA-BRCA/tests -v
```

Graph-related controls are:

```bash
python TCGA-BRCA/main.py \
  --temporal-window 2 \
  --attention-temperature 1.0 \
  --graph-epsilon 1e-8
```

Run the strict-reproduction audit with:

```bash
python TCGA-BRCA/scripts/audit_healthcom26_reproduction.py --require-exact
```

The audit succeeds for the public TCIA source artifacts but intentionally
returns a non-zero status for exact-author reproduction while the attention
networks, four-region preprocessing pipeline, cohort mapping, and experiment
configuration remain unpublished.
