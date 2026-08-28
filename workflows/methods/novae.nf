params.novae_radius = null
params.novae_technology = "visium"
params.novae_coord_type = null
params.novae_n_neighs = null
params.novae_delaunay = null
params.model = "Novae/novae-human-0"
params.novae_prototypes = 32
params.emb_results_dir = "results"

process embed_by_novae {

    tag "${id}"

    label "gpu_task"

    container "housy17/novae:1.0.0"

    publishDir "${params.emb_results_dir}/embeddings/novae", mode: 'copy',
               saveAs: { filename -> "${id}.h5ad" },
               enabled: params.emb_results_dir as boolean

    input:
    tuple val(id), path(raw_h5ad)

    output:
    tuple val(id), val("Novae"), path("*_embeddings.h5ad")

    script:
    """
    #!/usr/bin/env python
    import ast
    import os
    from pathlib import Path

    import novae
    import numpy as np
    import pandas as pd
    import scanpy as sc

    adata = sc.read_h5ad("${raw_h5ad}")
    original_obs = adata.obs.copy()

    max_expression = float(adata.X.max())
    if max_expression >= 10 and int(max_expression) != max_expression:
        print(
            "Preprocessing high-valued non-integer adata.X with "
            "sc.pp.normalize_total and sc.pp.log1p"
        )
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    radius_text = "${params.novae_radius}".strip()
    radius = None if radius_text.lower() in {"", "none", "null"} else ast.literal_eval(radius_text)
    technology_text = "${params.novae_technology}".strip()
    technology = None if technology_text.lower() in {"", "none", "null"} else technology_text
    coord_type_text = "${params.novae_coord_type}".strip()
    coord_type = None if coord_type_text.lower() in {"", "none", "null"} else coord_type_text
    n_neighs_text = "${params.novae_n_neighs}".strip()
    n_neighs = None if n_neighs_text.lower() in {"", "none", "null"} else int(n_neighs_text)
    delaunay_text = "${params.novae_delaunay}".strip().lower()
    if delaunay_text in {"", "none", "null"}:
        delaunay = None
    elif delaunay_text in {"true", "1", "yes"}:
        delaunay = True
    elif delaunay_text in {"false", "0", "no"}:
        delaunay = False
    else:
        raise ValueError(f"Invalid novae_delaunay value: {delaunay_text!r}")

    if n_neighs is not None and n_neighs <= 0:
        raise ValueError(f"novae_n_neighs must be positive; received {n_neighs}")

    num_prototypes = ${params.novae_prototypes}
    if not 0 < num_prototypes < adata.n_obs:
        raise ValueError(
            f"novae_prototypes must be positive and smaller than the number of spots/cells "
            f"({adata.n_obs}); received {num_prototypes}"
        )

    novae.spatial_neighbors(
        adata,
        radius=radius,
        technology=technology,
        coord_type=coord_type,
        n_neighs=n_neighs,
        delaunay=delaunay,
    )

    if "spatial" not in adata.obsm:
        raise KeyError("Novae spatial coordinates not found after spatial neighbor construction")
    output_spatial = adata.obsm["spatial"].copy()

    model_dir = Path("/data/model_weights") / "${params.model}"
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Missing Novae model directory: {model_dir}. "
            "Run the download_novae_checkpoints process first."
        )

    model = novae.Novae.from_pretrained(
        os.fspath(model_dir),
        local_files_only=True,
    )
    model.swav_head.num_prototypes = num_prototypes
    model.compute_representations(adata, zero_shot=True, accelerator="cuda")

    if "novae_latent" not in adata.obsm:
        raise KeyError("Novae latent representations not found in AnnData")

    embedding = np.asarray(adata.obsm["novae_latent"], dtype=np.float32)
    adata_embedding = sc.AnnData(
        X=embedding,
        obs=original_obs,
        var=pd.DataFrame(index=[f"V{i+1}" for i in range(embedding.shape[1])]),
    )
    adata_embedding.obsm["spatial"] = output_spatial
    adata_embedding.write_h5ad("novae_embeddings.h5ad")
    """
}
