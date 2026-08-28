/*
 * Task: download -- fetch pretrained checkpoints into params.model_weights_dir
 * (bind-mounted into every container as /data/model_weights).
 */

params.method = null

include { download_scgpt_checkpoints } from '../utils/download'
include { download_cellama_checkpoints } from '../utils/download'
include { download_cellfm_checkpoints } from '../utils/download'
include { download_cellplm_checkpoints } from '../utils/download'
include { download_geneformer_checkpoints } from '../utils/download'
include { download_genept_checkpoints } from '../utils/download'
include { download_langcell_checkpoints } from '../utils/download'
include { download_scbert_checkpoints } from '../utils/download'
include { download_sccello_checkpoints } from '../utils/download'
include { download_scfoundation_checkpoints } from '../utils/download'
include { download_scimilarity_checkpoints } from '../utils/download'
include { download_scprint_checkpoints } from '../utils/download'
include { download_uce_checkpoints } from '../utils/download'
include { download_c2s_checkpoints } from '../utils/download'
include { download_scvi_checkpoints } from '../utils/download'
include { download_novae_checkpoints } from '../utils/download'

workflow DOWNLOAD {
    main:
    def runners = [
        'scgpt':        { download_scgpt_checkpoints() },
        'cellama':      { download_cellama_checkpoints() },
        'cellfm':       { download_cellfm_checkpoints() },
        'cellplm':      { download_cellplm_checkpoints() },
        'geneformer':   { download_geneformer_checkpoints() },
        'genept':       { download_genept_checkpoints() },
        'langcell':     { download_langcell_checkpoints() },
        'scbert':       { download_scbert_checkpoints() },
        'sccello':      { download_sccello_checkpoints() },
        'scfoundation': { download_scfoundation_checkpoints() },
        'scimilarity':  { download_scimilarity_checkpoints() },
        'scprint':      { download_scprint_checkpoints() },
        'uce':          { download_uce_checkpoints() },
        'c2s':          { download_c2s_checkpoints() },
        'scvi':         { download_scvi_checkpoints() },
        'novae':        { download_novae_checkpoints() },
    ]

    if( !params.method )
        error "Usage: scfoundry download --method <method>   (or: nextflow run main.nf --task download --method <method>)\nAvailable: ${runners.keySet().join(', ')}"

    def selected = params.method.toString().trim().toLowerCase()
    def run = runners[selected]
    if( !run )
        error "Unknown method '${params.method}'. Available: ${runners.keySet().join(', ')}"

    run().view()
}
