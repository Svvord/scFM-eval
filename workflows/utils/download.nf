process download_scgpt_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p scGPT
    cd scGPT
    gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=drive_link
    echo "scGPT_human checkpoints downloaded!"
    """
}

process download_cellama_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p CELLama
    cd CELLama
    python - << 'EOF'
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir="all-MiniLM-L6-v2",
        local_dir_use_symlinks=False  # 可选，禁用符号链接确保真正拷贝
    )
    EOF
    echo "CELLama checkpoints downloaded!"
    """

}


process download_cellfm_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p CellFM
    cd CellFM
    hf download ShangguanNingyuan/CellFM "CellFM_80M_weight.ckpt" --local-dir ./
    echo "CellFM checkpoints downloaded!"
    """

}

process download_cellplm_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    set -euo pipefail
    cd "${projectDir}/data/model_weights"
    mkdir -p CellPLM
    cd CellPLM
    # Official CellPLM checkpoints are distributed only through the authors' Dropbox
    # shared folder (see https://github.com/OmicsML/CellPLM). Dropbox serves the whole
    # folder as one zip (~2.3 GB: two checkpoints + demo data); we keep only the
    # 20231027_85M checkpoint and its config, which is what the framework uses.
    if [ -s 20231027_85M.best.ckpt ] && [ -s 20231027_85M.config.json ]; then
        echo "CellPLM checkpoints already exist!"
    else
        curl -L --fail --retry 3 --retry-delay 10 \\
            -o CellPLM.zip.tmp \\
            "https://www.dropbox.com/scl/fo/i5rmxgtqzg7iykt2e9uqm/h?rlkey=o8hi0xads9ol07o48jdityzv1&dl=1"
        unzip -o -j CellPLM.zip.tmp \\
            "ckpt/20231027_85M.best.ckpt" "ckpt/20231027_85M.config.json" -d ./
        rm -f CellPLM.zip.tmp
    fi
    sha256sum -c <<'EOF'
a5f0bc6f18a34c5ae6cc69a4d726870e7c7684242c7001c7f69c3cb81d6816df  20231027_85M.best.ckpt
c9827f8cc2a8eda8e3e0726e1f21e2cbfd3a9582ee211d45f3855bfa81a2a686  20231027_85M.config.json
EOF
    echo "CellPLM checkpoints downloaded!"
    """
}

process download_geneformer_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p Geneformer
    cd Geneformer
    hf download ctheodoris/Geneformer "Geneformer-V2-316M/config.json" --local-dir ./
    hf download ctheodoris/Geneformer "Geneformer-V2-316M/generation_config.json" --local-dir ./
    hf download ctheodoris/Geneformer "Geneformer-V2-316M/training_args.bin" --local-dir ./
    hf download ctheodoris/Geneformer "Geneformer-V2-316M/model.safetensors" --local-dir ./
    echo "Geneformer checkpoints downloaded!"
    """

}

process download_genept_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p GenePT
    cd GenePT
    curl -L -o GenePT_emebdding_v2.zip "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1"
    unzip GenePT_emebdding_v2.zip
    rm -rf GenePT_emebdding_v2.zip
    mv GenePT_*_v2/* ./
    rm -rf GenePT_*_v2/
    mv GenePT_gene_protein_embedding_model_3_text.pickle* GenePT_gene_protein_embedding_model_3_text.pickle
    echo "GenePT checkpoints downloaded!"
    """

}

process download_langcell_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p LangCell
    cd LangCell
    gdown --remaining-ok --folder https://drive.google.com/drive/folders/1Su6PtuFahlVMWEgD1i-wahx4Gu3a0oCR?usp=drive_link
    echo "LangCell checkpoints downloaded!"
    """

}

process download_scbert_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p scBERT
    cd scBERT
    gdown https://drive.google.com/uc?id=1_Pgk_o8AtQtoXr_ZLQx0eJYoSWzjxC8f
    echo "scBERT checkpoints downloaded!"
    """

}

process download_sccello_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p scCello
    cd scCello
    python - << 'EOF'
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="katarinayuan/scCello-zeroshot",
        local_dir="scCello-zeroshot",
        local_dir_use_symlinks=False
    )
    EOF
    echo "scCello checkpoints downloaded!"
    """

}

process download_scfoundation_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p scFoundation
    cd scFoundation
    hf download genbio-ai/scFoundation "models.ckpt" --local-dir ./
    echo "scFoundation checkpoints downloaded!"
    """

}

process download_scimilarity_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p SCimilarity
    cd SCimilarity
    curl -L -o model_v1.1.tar.gz https://zenodo.org/records/10685499/files/model_v1.1.tar.gz?download=1
    tar -zxvf model_v1.1.tar.gz
    rm -rf model_v1.1.tar.gz
    echo "SCimilarity checkpoints downloaded!"
    """

}

process download_scprint_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    set -euo pipefail
    cd "${projectDir}/data/model_weights"
    mkdir -p scPRINT
    cd scPRINT
    # Checkpoint naming history (upstream change, not ours): the checkpoint benchmarked
    # in scFM-eval was originally released on Hugging Face (jkobject/scPRINT) as
    # `v2-medium.ckpt`. In 2026 the authors reorganised the model repo for scPRINT 2.0:
    # the very same file was renamed to `medium-v1.5.ckpt` (content-preserving rename,
    # commit fed0f6bb), and the file under that name was later replaced with different
    # content. We therefore keep the official file name but pin the download to the
    # rename commit, so `medium-v1.5.ckpt` is byte-identical to the original
    # `v2-medium.ckpt` (sha256 a4cf0753...). Do not drop --revision.
    if [ -s medium-v1.5.ckpt ]; then
        echo "scPRINT checkpoints already exist!"
    else
        hf download jkobject/scPRINT medium-v1.5.ckpt \\
            --revision fed0f6bbc9aba1c648df7442e93fe41ded0175bb \\
            --local-dir ./
    fi
    sha256sum -c <<'EOF'
a4cf0753270d4ff451a5dbddadd4c59a53e10e4c3e96a8af0db407ad893c36c5  medium-v1.5.ckpt
EOF
    echo "scPRINT checkpoints downloaded!"
    """
}

process download_uce_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p UCE
    cd UCE
    curl -L https://ndownloader.figshare.com/files/43423236 -o 33l_8ep_1024t_1280.torch
    curl -L https://ndownloader.figshare.com/files/42706555 -o species_offsets.pkl
    curl -L https://ndownloader.figshare.com/files/42706558 -o species_chrom.csv
    curl -L https://ndownloader.figshare.com/files/42706576 -o 4layer_model.torch
    curl -L https://ndownloader.figshare.com/files/42706585 -o all_tokens.torch
    curl -L https://ndownloader.figshare.com/files/42715213 -o protein_embeddings.tar.gz
    tar -zxvf protein_embeddings.tar.gz
    echo "UCE checkpoints downloaded!"
    """

}

process download_c2s_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    cd "${projectDir}/data/model_weights"
    mkdir -p C2S
    cd C2S
    hf download vandijklab/C2S-Pythia-410m-cell-type-prediction --local-dir ./C2S-Pythia-410m-cell-type-prediction
    echo "C2S checkpoints downloaded!"
    """
}

process download_scvi_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    """
    set -euo pipefail
    cd "${projectDir}/data/model_weights"
    mkdir -p scVI/Census2024-07-01-HomoSapiens
    cd scVI/Census2024-07-01-HomoSapiens
    if [ -s model.pt ]; then
        echo "scVI Census checkpoints already exist!"
    else
        curl -L --fail --retry 3 --retry-delay 10 \\
            -o model.pt.tmp \\
            https://cellxgene-contrib-public.s3.amazonaws.com/models/scvi/2024-07-01/homo_sapiens/model.pt
        mv model.pt.tmp model.pt
        echo "scVI Census checkpoints downloaded!"
    fi
    """
}

process download_novae_checkpoints {

    container "housy17/scfm_download:latest"

    output:
    stdout

    script:
    def model_path = params.containsKey('model') ? params.get('model') : "Novae/novae-human-0"
    def repo_by_model = [
        "Novae/novae-human-0": "prism-oncology/novae-human-0",
        "Novae/novae-mouse-0": "prism-oncology/novae-mouse-0",
        "Novae/novae-brain-0": "prism-oncology/novae-brain-0",
    ]
    def repo_id = repo_by_model[model_path]
    if( !repo_id )
        throw new IllegalArgumentException("Unsupported Novae model '${model_path}'. Allowed: ${repo_by_model.keySet().join(', ')}")

    """
    set -euo pipefail
    cd "${projectDir}/data/model_weights"
    mkdir -p "${model_path}"
    hf download ${repo_id} \\
        --local-dir "./${model_path}"
    echo "Novae checkpoints downloaded: ${model_path}"
    """
}
