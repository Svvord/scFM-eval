# scFoundry

**scFoundry** is a unified, reproducible framework for **deploying, running and evaluating single-cell foundation models (scFMs)**. One command gives you zero-shot embeddings, label transfer on frozen embeddings, supervised fine-tuning, the benchmark metrics of our paper, or representation-geometry probes — for any supported model, in its original container, with its authors' recipe. No per-model environment setup, no code.

It is built on **Nextflow**: every model runs in a pinned container, every launch is recorded, and the same commands work on a workstation with Docker or on an HPC cluster with Apptainer/Singularity. This repository is the home of scFoundry and of the benchmark described in the accompanying paper.

📖 **Documentation: <https://svvord.github.io/scFM-eval-docs/>** — installation, tutorials, the reference for every task and method, and the benchmark results with an interactive explorer.

**[2026.08]** scFoundry 0.2: `pip`-installable `scfoundry` command with workspaces; `embed` covers the scFMs, PCA, scVI and the batch-integration methods; new `transfer` task (prototype / kNN / logistic regression / MLP on frozen embeddings); `finetune` restricted to methods that update parameters; new `benchmark` and `geometry` tasks (the manuscript's metrics and representation-geometry probes); revised CellFM, Geneformer, scFoundation, SCimilarity and scPRINT implementations.
**[2026.03.04]** Fine-tuning implementation released.
**[2026.01.13]** Few-shot learning implementation released; scPRINT deployment fixes.

---

## Requirements

| Requirement | Notes |
|---|---|
| Linux `x86_64` | |
| **Python ≥ 3.7** | for the `scfoundry` command only; standard library |
| **Nextflow ≥ 24.10** | https://www.nextflow.io/docs/latest/install.html (needs Java 17+) |
| **Apptainer**, **Singularity** or **Docker** | models run in pinned containers |
| NVIDIA driver ≥ 525 | for tasks that run a model on the GPU |

## Installation

```bash
pip install scfoundry                                       # release (PyPI)
pip install git+https://github.com/Svvord/scFM-eval.git     # latest from GitHub
```

Install into the environment that provides `nextflow`. Full instructions, including the developer install: [Installation](https://svvord.github.io/scFM-eval-docs/getting-started/installation.html).

## Quick start

```bash
scfoundry init my_project && cd my_project    # workspace: nextflow.config, data/, cache/, results/, runs/
scfoundry download --method scgpt             # official checkpoint -> data/model_weights/
scfoundry embed --method scgpt --data cells.h5ad
# -> results/embeddings/scgpt/cells.h5ad   (embedding in adata.X, original obs kept)
```

Input is an AnnData `.h5ad` with **raw counts over the full transcriptome**; see [Input data format](https://svvord.github.io/scFM-eval-docs/data/input-format.html). Four demo files under [`data/demo/`](data/demo) satisfy the contract; the [Quickstart](https://svvord.github.io/scFM-eval-docs/getting-started/quickstart.html) embeds, scores and annotates them end to end.

## Tasks

| Task | What it does | Guide |
|---|---|---|
| `download` | fetch a model's official checkpoint | [Model weights](https://svvord.github.io/scFM-eval-docs/getting-started/model-weights.html) |
| `embed` | cell embeddings: zero-shot scFMs, PCA / scVI references, batch-integration methods | [Embed](https://svvord.github.io/scFM-eval-docs/tasks/embed.html) |
| `transfer` | label a query from a labelled reference on frozen embeddings (logreg / prototype / kNN / MLP) | [Transfer](https://svvord.github.io/scFM-eval-docs/tasks/transfer.html) |
| `finetune` | update a model's parameters with its authors' recipe, then predict | [Fine-tune](https://svvord.github.io/scFM-eval-docs/tasks/finetune.html) |
| `benchmark` | the paper's 13 metrics: biological conservation and batch mixing | [Benchmark](https://svvord.github.io/scFM-eval-docs/tasks/benchmark.html) |
| `geometry` | representation-geometry probes: effective dimension, anisotropy, R_NX, intrinsic dimension, partial η² | [Geometry](https://svvord.github.io/scFM-eval-docs/tasks/geometry.html) |

`scfoundry list methods` prints the method × task matrix; `scfoundry <task> --help` the options of a task.

## Supported methods

Zero-shot scFMs: **Cell2Sentence, CELLama, CellFM, CellPLM, Geneformer, GenePT, LangCell, scBERT, scCello, scFoundation, scGPT, SCimilarity, scPRINT, UCE**, and **Novae** for spatial data. References: **PCA**, **scVI** (CELLxGENE Census). Integration: **scGPT (integrated), scVI (de novo), Harmony, Seurat CCA, Seurat RPCA**.

Containers, pinned versions, default checkpoints and per-method parameters: [Supported methods](https://svvord.github.io/scFM-eval-docs/reference/methods.html).

## Benchmark results

Every method over 26 Tabula Sapiens v2 tissues (548,977 cells), with default settings throughout: [Benchmark results](https://svvord.github.io/scFM-eval-docs/results/) and the [interactive explorer](https://svvord.github.io/scFM-eval-docs/results/explorer.html). The tables are what `scfoundry benchmark` and `scfoundry geometry` compute, so your own numbers are directly comparable — see [Reproducing the paper](https://svvord.github.io/scFM-eval-docs/reproducing/).

## Citation

If scFoundry or the benchmark is useful for your research, please cite:

> Siyu Hou, Penghui Yang, Wenjing Ma, Jade Xiaoqing Wang and Xiang Zhou (2026).
> A unified framework enables accessible deployment and comprehensive benchmarking
> of single-cell foundation models.

```
@article{hou2026unified,
  title = {A unified framework enables accessible deployment and comprehensive benchmarking of single-cell foundation models},
  author = {Hou, Siyu and Yang, Penghui and Ma, Wenjing and Wang, Jade Xiaoqing and Zhou, Xiang},
  year = {2026},
  publisher = {Cold Spring Harbor Laboratory},
  journal = {bioRxiv}
}
```

Please also cite the upstream paper of every model whose results you report — links on the [citation page](https://svvord.github.io/scFM-eval-docs/about/citation.html).
