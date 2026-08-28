/*
 * Task: embed -- zero-shot cell embeddings.
 *
 * Input : one .h5ad (raw counts in X, gene symbols in var, see README)
 * Output: ${params.emb_results_dir}/embeddings/<method>/<id>.h5ad with the embedding in X
 */

params.method = null
params.data   = null

include { embed_by_c2s } from '../methods/c2s'
include { embed_by_cellama } from '../methods/cellama'
include { embed_by_cellfm } from '../methods/cellfm'
include { embed_by_cellplm } from '../methods/cellplm'
include { embed_by_geneformer } from '../methods/geneformer'
include { embed_by_genept_w } from '../methods/genept'
include { embed_by_langcell } from '../methods/langcell'
include { embed_by_novae } from '../methods/novae'
include { embed_by_pca } from '../methods/pca'
include { embed_by_scbert } from '../methods/scbert'
include { embed_by_sccello } from '../methods/sccello'
include { embed_by_scfoundation } from '../methods/scfoundation'
include { embed_by_scgpt } from '../methods/scgpt'
include { embed_by_scimilarity } from '../methods/scimilarity'
include { embed_by_scprint } from '../methods/scprint'
include { embed_by_scvi } from '../methods/scvi'
include { embed_by_uce } from '../methods/uce'

workflow EMBED {
    main:
    def runners = [
        'c2s':          { ch -> embed_by_c2s(ch) },
        'cellama':      { ch -> embed_by_cellama(ch) },
        'cellfm':       { ch -> embed_by_cellfm(ch) },
        'cellplm':      { ch -> embed_by_cellplm(ch) },
        'geneformer':   { ch -> embed_by_geneformer(ch) },
        'genept':       { ch -> embed_by_genept_w(ch) },
        'langcell':     { ch -> embed_by_langcell(ch) },
        'novae':        { ch -> embed_by_novae(ch) },
        'pca':          { ch -> embed_by_pca(ch) },
        'scbert':       { ch -> embed_by_scbert(ch) },
        'sccello':      { ch -> embed_by_sccello(ch) },
        'scfoundation': { ch -> embed_by_scfoundation(ch) },
        'scgpt':        { ch -> embed_by_scgpt(ch) },
        'scimilarity':  { ch -> embed_by_scimilarity(ch) },
        'scprint':      { ch -> embed_by_scprint(ch) },
        'scvi':         { ch -> embed_by_scvi(ch) },
        'uce':          { ch -> embed_by_uce(ch) },
    ]

    if( !params.method || !params.data )
        error "Usage: scfoundry embed --method <method> --data <file.h5ad>   (or: nextflow run main.nf --task embed --method <method> --data <file.h5ad>)\nAvailable: ${runners.keySet().join(', ')}"

    def selected = params.method.toString().trim().toLowerCase()
    def run = runners[selected]
    if( !run )
        error "Unknown method '${params.method}'. Available: ${runners.keySet().join(', ')}"

    data_ch = Channel.fromPath(params.data, checkIfExists: true)
        .map { file -> tuple("${file.baseName}", file) }
    run(data_ch)
}
