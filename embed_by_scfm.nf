nextflow.enable.dsl=2

params.method = null
params.data   = null

include { embed_by_c2s } from './workflows/methods/c2s'
include { embed_by_cellama } from './workflows/methods/cellama'
include { embed_by_cellfm } from './workflows/methods/cellfm'
include { embed_by_cellplm } from './workflows/methods/cellplm'
include { embed_by_geneformer } from './workflows/methods/geneformer'
include { embed_by_genept_w } from './workflows/methods/genept.nf'
include { embed_by_langcell } from './workflows/methods/langcell'
include { embed_by_scbert } from './workflows/methods/scbert'
include { embed_by_sccello } from './workflows/methods/sccello'
include { embed_by_scfoundation } from './workflows/methods/scfoundation'
include { embed_by_scgpt } from './workflows/methods/scgpt'
include { embed_by_scimilarity } from './workflows/methods/scimilarity'
include { embed_by_scprint } from './workflows/methods/scprint'
include { embed_by_uce } from './workflows/methods/uce'


if( !params.method || !params.data )
    exit 1, "Usage: nextflow embed_by_scfm.nf --method <A|B|C> --data <path>"

data_ch = Channel.fromPath(params.data)

def parseMethods(String methodsStr) {
    def s = methodsStr.trim().toLowerCase()
    return s
}

workflow {

    def selected = parseMethods(params.method)

    def runners = [
        'scgpt': { ch -> embed_by_scgpt(ch) },
        'cellama': { ch -> embed_by_cellama(ch) },
        'cellfm': { ch -> embed_by_cellfm(ch) },
        'cellplm': { ch -> embed_by_cellplm(ch) },
        'geneformer': { ch -> embed_by_geneformer(ch) },
        'genept': { ch -> embed_by_genept_w(ch) },
        'langcell': { ch -> embed_by_langcell(ch) },
        'scbert': { ch -> embed_by_scbert(ch) },
        'sccello': { ch -> embed_by_sccello(ch) },
        'scfoundation': { ch -> embed_by_scfoundation(ch) },
        'scimilarity': { ch -> embed_by_scimilarity(ch) },
        'scprint': { ch -> embed_by_scprint(ch) },
        'uce': { ch -> embed_by_uce(ch) },
        'c2s': { ch -> embed_by_c2s(ch) },
    ]

    def run = runners[ selected ]
    if( !run )
        exit 1, "Unknown method '${params.method}'. Allowed: ${runners.keySet().join(', ')}"

    data_ch = data_ch.map{ file -> tuple("${file.baseName}", file) }
    run(data_ch)

}
