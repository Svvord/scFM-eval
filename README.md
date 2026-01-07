
# scFM-eval

**scFM-eval** is a unified, reproducible computational framework for deploying and evaluating **single-cell foundation models (scFMs)**.  
It is built on **Nextflow DSL2** and provides standardized execution, containerized environments, and automated embedding inference across multiple scFM methods.

---

## System Requirements

- **OS**: Linux (`linux/amd64`)
- **GPU**: NVIDIA GPU required  
  - NVIDIA driver ≥ **525**
- **Container runtime**:
  - `Docker` **or**
  - `Apptainer` (formerly Singularity)
- **Nextflow**:
  - Tested with `Nextflow ≥ 25.10.0`
  - Any version supporting **DSL2** should work

---

## Installation

### 1. Install Nextflow

Please follow the official instructions:  
👉 https://github.com/nextflow-io/nextflow

After installation, verify:

```bash
nextflow -v
```

### 2. Download scFM-eval
```bash
git clone https://github.com/Svvord/scFM-eval.git
```

---

## First-Time Setup (Required Once)

### Step 1. Choose Your Container Backend

Open `nextflow.config` and select **one** container runtime:

* **Docker**

```groovy
docker {
    enabled = true
}
apptainer {
    enabled = false
}
```

* **Apptainer (Singularity)**
  (Default; no changes needed unless you modified it before)

> ⚠️ This only needs to be done **once**.
> Subsequent runs require no further configuration.

---

### Step 2. Download Model Checkpoints

Pretrained model weights must be downloaded **once** before first use.

We provide a helper script `download_model_weights.nf` to fetch official checkpoints and place them in the correct directory structure.

#### Example: Download weights for **scGPT**

```bash
nextflow download_model_weights.nf --method scgpt
```

📌 **Important notes**:

* You only need to download model weights **once**
* Downloaded weights are cached locally and reused automatically
* You may also manually place weights if you follow the same directory structure

#### Directory Structure Example (scGPT)

```text
data/
└── model_weights/
    └── scGPT/
        └── scGPT_human/
```

* The default scGPT version is `scGPT_human`
* To specify this version explicitly in later runs:

```text
--model "scGPT/scGPT_human"
```

* If no version is specified, the framework will use the **default pretrained model**

---

## First Run Notes (Important)

* On the **first execution of a method**, Nextflow will automatically:

  * Pull the corresponding container image
  * Cache the image and model weights locally
* This initial run may take longer
* **No additional setup is needed** once caching is complete

---

## Embedding Inference (Zero-shot)

Embedding inference can be performed with **a single command**.

We provide a small demo dataset:

```text
data/demo/colon_1000.h5ad
```

### Example Command

```bash
nextflow embed_by_scfm.nf \
  --method scgpt \
  --data data/demo/colon_1000.h5ad
```

Required arguments:

* `--method`: scFM method name (e.g. `scgpt`)
* `--data`: input dataset in `.h5ad` format

---

## Output

Results are written to:

```text
results/embedding/<method_name>/
```

* Embeddings are stored as `.h5ad` files
* The embedding matrix can be accessed via:

```python
adata = sc.read_h5ad("results/embedding/scgpt/colon_1000.h5ad")
embeddings = adata.X
```

---

## Supported Methods & Environments

| Method | Container | Model Version | Notes |
| ------ | --------- | ------------- | ----- |
| CELLama | [housy17/cellama:latest](https://hub.docker.com/repository/docker/housy17/cellama/general) | [v0.1.0](https://github.com/portrai-io/CELLama) |       |
| CellFM | [housy17/cellfm:latest](https://hub.docker.com/repository/docker/housy17/cellfm/general) | [5054a2a](https://github.com/biomed-AI/CellFM-torch) |       |
| CellPLM | [housy17/cellplm:latest](https://hub.docker.com/repository/docker/housy17/cellplm/general) | [v0.1.0](https://github.com/OmicsML/CellPLM) |       |
| Geneformer | [housy17/geneformer:latest](https://hub.docker.com/repository/docker/housy17/geneformer/general) | [v0.1.0](https://huggingface.co/ctheodoris/Geneformer) |       |
| GenePT | [housy17/genept:latest](https://hub.docker.com/repository/docker/housy17/genept/general) | [3602699](https://github.com/yiqunchen/GenePT) |       |
| LangCell | [housy17/langcell:latest](https://hub.docker.com/repository/docker/housy17/langcell) | [69e41ef](https://github.com/PharMolix/LangCell) |       |
| scBERT | [housy17/scbert:latest](https://hub.docker.com/repository/docker/housy17/scbert/general) | [v1.0.0](https://github.com/TencentAILabHealthcare/scBERT) |       |
| scCello | [housy17/sccello:latest](https://hub.docker.com/repository/docker/housy17/sccello/general) | [767585b](https://github.com/DeepGraphLearning/scCello) |       |
| scFoundation | [housy17/scfoundation:latest](https://hub.docker.com/repository/docker/housy17/scfoundation/general) | [397631c](https://github.com/biomap-research/scFoundation) |       |
| scGPT | [housy17/scgpt:latest](https://hub.docker.com/repository/docker/housy17/scgpt/general) | [v0.2.4](https://github.com/bowang-lab/scGPT) |       |
| SCimilarity | [housy17/scsimilarity:latest](https://hub.docker.com/repository/docker/housy17/scsimilarity/general) | [v0.4.1](https://genentech.github.io/scimilarity/index.html) |       |
| scPRINT | [housy17/scprint:latest](https://hub.docker.com/repository/docker/housy17/scprint/general) | [v2.3.5](https://github.com/cantinilab/scPRINT) |       |
| UCE | [housy17/uce:latest](https://hub.docker.com/repository/docker/housy17/uce) | [8227a65](https://github.com/snap-stanford/UCE) |       |




📌 *This table will be expanded as more models and configurations are added.*

---

## Tutorials & Documentation

A detailed tutorial covering:

* Advanced parameters
* Batch size and resource control
* Few-shot workflows
* Fine-tuning workflows
* Benchmark evaluation

👉 **Tutorial link: (coming soon)**

---

## Citation

If this framework or any of the tools provided here are useful for your research, **please cite our work** — it helps us a lot.
> Siyu Hou, Penghui Yang, Wenjing Ma, Jade Xiaoqing Wang and Xiang Zhou (2026). 
> A unified framework enables accessible deployment and comprehensive benchmarking
> of single-cell foundation models.

> **BioRxiv preprint:** *Coming soon*

```

```
