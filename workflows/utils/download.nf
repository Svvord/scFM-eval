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
    cd "${projectDir}/data/model_weights"
    mkdir -p CellPLM
    cd CellPLM
    gdown --fuzzy https://drive.google.com/file/d/1r-5f_mprimHJbR1GvIHvuBIErwMc5hrJ/view?usp=drive_link
    gdown --fuzzy https://drive.google.com/file/d/1KqV8Sd1qqiDP35AM8dX0deZnxzXgCCrH/view?usp=drive_link
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
    gdown --fuzzy https://drive.google.com/file/d/1lnj6OcLCQUQmh8mNEyHdHA6i4oHOCHAu/view?usp=drive_link
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
    cd "${projectDir}/data/model_weights"
    mkdir -p scPRINT
    cd scPRINT
    gdown --fuzzy https://drive.google.com/file/d/1B6GKVfd7SNSliwYF6Z3Wfyli63CMx1_z/view?usp=drive_link
    echo "scPRINT checkpoints downloaded!"
    """
    // Recently (Jan 2026), scPRINT was updated and all the previous checkpoints were removed,
    // but I have a backup that I downloaded earlier.

    // """
    // cd "${projectDir}/data/model_weights"
    // mkdir -p scPRINT
    // cd scPRINT
    // hf download jkobject/scPRINT "v2-medium.ckpt" --local-dir ./
    // echo "scPRINT checkpoints downloaded!"
    // """

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