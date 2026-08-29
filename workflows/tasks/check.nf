/*
 * Task: check -- advisory check of input .h5ad files against the input contract.
 *
 * Reports, per file, whether X holds raw counts over the full transcriptome, whether the
 * gene and cell identifiers the models read are present and unique, and whether the
 * label / batch / spatial columns a task or method needs are there. Nothing is modified
 * and the run always succeeds: the report is printed and published under
 * ${params.results_dir}/check/<file>_check.txt.
 */

params.data        = null
params.method      = null
params.label_key   = "cell_type"
params.batch_key   = "batch_id"
params.role        = ""           // reference | query | ""
params.results_dir = "results"


process check_h5ad {

    tag "${label}"

    label 'cpu_task'

    debug true

    container "housy17/scllms:latest"

    publishDir "${params.results_dir}/check", mode: 'copy',
               saveAs: { filename -> filename.replaceFirst(/^out\//, '') },
               enabled: params.results_dir as boolean

    input:
    tuple val(label), path(files, stageAs: 'data/*')

    output:
    path "out/**"

    script:
    """
    python /code/check/validate_h5ad.py \\
        --data-dir data \\
        --pattern '*.h5ad' \\
        --method '${params.method ?: ""}' \\
        --label-key '${params.label_key}' \\
        --batch-key '${params.batch_key}' \\
        --role '${params.role ?: ""}' \\
        --output-dir out
    """
}


workflow CHECK {
    main:
    if( !params.data )
        error "Usage: scfoundry check --data <file.h5ad | directory | glob> [--method NAME] [--label-key cell_type] [--batch-key batch_id] [--role reference|query]"

    def spec = params.data.toString()
    def isGlob = (spec =~ /[*?\[]/).find()          // file(glob) would return a list
    def pattern = isGlob ? spec : (file(spec).isDirectory() ? "${spec}/*.h5ad" : spec)
    def files = Channel.fromPath(pattern, checkIfExists: true).collect()
    def inputs = files.map { list -> tuple(list.size() == 1 ? file(list[0]).baseName : "${list.size()} files", list) }
    check_h5ad(inputs)
}
