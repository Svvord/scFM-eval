nextflow.enable.dsl=2

include { download_scgpt_checkpoints } from './workflows/utils/download'
include { download_cellama_checkpoints } from './workflows/utils/download'
include { download_cellfm_checkpoints } from './workflows/utils/download'
include { download_cellplm_checkpoints } from './workflows/utils/download'
include { download_geneformer_checkpoints } from './workflows/utils/download'
include { download_genept_checkpoints } from './workflows/utils/download'
include { download_langcell_checkpoints } from './workflows/utils/download'
include { download_scbert_checkpoints } from './workflows/utils/download'
include { download_sccello_checkpoints } from './workflows/utils/download'
include { download_scfoundation_checkpoints } from './workflows/utils/download'
include { download_scimilarity_checkpoints } from './workflows/utils/download'
include { download_scprint_checkpoints } from './workflows/utils/download'
include { download_uce_checkpoints } from './workflows/utils/download'
include { download_c2s_checkpoints } from './workflows/utils/download'




params.method = null
if( !params.method )
    exit 1, "Usage: nextflow download_model_weights.nf --method <A|B|C>"

def parseMethods(String methodsStr) {
    def s = methodsStr.trim().toLowerCase()
    return s
}

workflow {

    def selected = parseMethods(params.method)

    def runners = [
        'scgpt': { ch -> download_scgpt_checkpoints() },
        'cellama': { ch -> download_cellama_checkpoints() },
        'cellfm': { ch -> download_cellfm_checkpoints() },
        'cellplm': { ch -> download_cellplm_checkpoints() },
        'geneformer': { ch -> download_geneformer_checkpoints() },
        'genept': { ch -> download_genept_checkpoints() },
        'langcell': { ch -> download_langcell_checkpoints() },
        'scbert': { ch -> download_scbert_checkpoints() },
        'sccello': { ch -> download_sccello_checkpoints() },
        'scfoundation': { ch -> download_scfoundation_checkpoints() },
        'scimilarity': { ch -> download_scimilarity_checkpoints() },
        'scprint': { ch -> download_scprint_checkpoints() },
        'uce': { ch -> download_uce_checkpoints() },
        'c2s': { ch -> download_c2s_checkpoints() },
    ]

    def run = runners[ selected ]
    if( !run )
        exit 1, "Unknown method '${params.method}'. Allowed: ${runners.keySet().join(', ')}"

    run().view()

}