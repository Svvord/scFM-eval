/*
 * Task: geometry -- representation-geometry probes of embedding files.
 *
 * Each embedding file (output of `embed`) is paired with the raw-count input it was
 * computed from (same file name) and scored with: participation ratio (effective
 * dimension), spectral and cell-pair anisotropy (mixed / within / between cell types),
 * within-batch expression-neighbourhood preservation (R_NX against an analytic
 * Pearson-residual reference, k = 15/30/50), TwoNN intrinsic dimension, and cell-type /
 * batch partial eta-squared. One container task per call, one method label.
 * Output: ${params.results_dir}/geometry/<method>/<sample>_<method>.csv (+ per-batch
 * R_NX tables) and ${params.results_dir}/<method>_geometry.csv.
 */

params.embedding   = null
params.data        = null
params.method      = null
params.label_key   = "cell_type"
params.batch_key   = "batch_id"
params.max_cells   = 20000
params.seed        = 0
params.results_dir = "results"


process geometry_probes {

    tag "${method}"

    label 'cpu_task'

    debug true

    container "housy17/biometrics:cu12"

    publishDir "${params.results_dir}", mode: 'copy',
               saveAs: { filename -> filename.replaceFirst(/^out\//, '') },
               enabled: params.results_dir as boolean

    input:
    tuple val(method), path(embeddings, stageAs: 'embeddings/*'), path(data, stageAs: 'data/*')

    output:
    path "out/**"

    script:
    """
    python /code/geometry/embedding_geometry.py \\
        --embedding-dir embeddings \\
        --data-dir data \\
        --pattern '*.h5ad' \\
        --method '${method}' \\
        --label-key '${params.label_key}' \\
        --batch-key '${params.batch_key}' \\
        --max-cells ${params.max_cells} \\
        --seed ${params.seed} \\
        --output-dir out
    """
}


def resolve_h5ad(String spec) {
    def isGlob = (spec =~ /[*?\[]/).find()
    return isGlob ? spec : (file(spec).isDirectory() ? "${spec}/*.h5ad" : spec)
}


workflow GEOMETRY {
    main:
    if( !params.embedding || !params.data )
        error "Usage: scfoundry geometry --embedding <file.h5ad | directory | glob> --data <input file.h5ad | directory> [--method NAME] [--label-key cell_type] [--batch-key batch_id]"

    def emb_files = Channel.fromPath(resolve_h5ad(params.embedding.toString()), checkIfExists: true).collect()
    def data_files = Channel.fromPath(resolve_h5ad(params.data.toString()), checkIfExists: true).collect()
    def inputs = emb_files.combine(data_files).map { emb, data ->
        def label = params.method ?: file(emb[0]).parent.name
        tuple(label.toString(), emb, data)
    }
    geometry_probes(inputs)
}
