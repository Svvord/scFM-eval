nextflow.enable.dsl=2

// DEPRECATED entry point, kept so that the commands published with the first
// version of scFM-eval keep working. Equivalent new commands:
//     scfoundry embed --method <method> --data <file.h5ad>
//     nextflow run main.nf --task embed --method <method> --data <file.h5ad>

include { EMBED } from './workflows/tasks/embed'

workflow {
    log.warn "embed_by_scfm.nf is deprecated; use 'scfoundry embed --method <method> --data <file.h5ad>' (or 'nextflow run main.nf --task embed')."
    EMBED()
}
