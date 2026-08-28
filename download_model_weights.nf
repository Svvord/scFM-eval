nextflow.enable.dsl=2

// DEPRECATED entry point, kept so that the commands published with the first
// version of scFM-eval keep working. Equivalent new commands:
//     scfoundry download --method <method>
//     nextflow run main.nf --task download --method <method>

include { DOWNLOAD } from './workflows/tasks/download'

workflow {
    log.warn "download_model_weights.nf is deprecated; use 'scfoundry download --method <method>' (or 'nextflow run main.nf --task download')."
    DOWNLOAD()
}
