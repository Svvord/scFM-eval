/*
 * Task: benchmark -- score embedding files (output of `embed`).
 *
 * Biological conservation on obs[label_key]: NMI, HOM, COM, FMI, ARI (Leiden with the
 * resolution matched to the number of cell types, or KMeans), ASW, cLISI, Acc@kNN, GC.
 * Batch mixing on obs[batch_key]: kBET, BRAS, iLISI, CiLISI.
 *
 * All embedding files of one call are scored in one container task and reported
 * under one method label (default: the name of the directory holding the files).
 * Output: ${params.results_dir}/benchmark/ with the per-sample tables, cluster labels
 * and the combined <method>_*_metrics_{long,wide}.csv tables.
 */

params.embedding       = null
params.method          = null
params.label_key       = "cell_type"
params.batch_key       = "batch_id"
params.metrics         = "bio"        // bio | batch | all
params.clustering      = "leiden"     // leiden | kmeans
params.batch_max_cells = 0            // 0 = exact full-data batch metrics
params.results_dir     = "results"


process benchmark_embeddings {

    tag "${method}"

    label 'cpu_task'

    debug true

    container "housy17/biometrics:cu12"

    publishDir "${params.results_dir}/benchmark", mode: 'copy',
               saveAs: { filename -> filename.replaceFirst(/^out\//, '') },
               enabled: params.results_dir as boolean

    input:
    tuple val(method), path(embeddings, stageAs: 'embeddings/*')

    output:
    path "out/**"

    script:
    """
    python /code/benchmark/embedding_metrics.py \\
        --embedding-dir embeddings \\
        --pattern '*.h5ad' \\
        --method '${method}' \\
        --label-key '${params.label_key}' \\
        --batch-key '${params.batch_key}' \\
        --metric-set ${params.metrics} \\
        --clustering ${params.clustering} \\
        --batch-max-cells ${params.batch_max_cells} \\
        --output-dir out
    """
}


workflow BENCHMARK {
    main:
    if( !params.embedding )
        error "Usage: scfoundry benchmark --embedding <file.h5ad | directory | glob> [--method NAME] [--label-key cell_type] [--batch-key batch_id] [--metrics bio|batch|all] [--clustering leiden|kmeans]"
    if( !(params.metrics in ['bio', 'batch', 'all']) )
        error "--metrics must be bio, batch or all (got '${params.metrics}')"
    if( !(params.clustering in ['leiden', 'kmeans']) )
        error "--clustering must be leiden or kmeans (got '${params.clustering}')"

    def spec = params.embedding.toString()
    def isGlob = (spec =~ /[*?\[]/).find()          // file(glob) would return a list
    def pattern = isGlob ? spec : (file(spec).isDirectory() ? "${spec}/*.h5ad" : spec)
    def files = Channel.fromPath(pattern, checkIfExists: true).collect()
    def inputs = files.map { list ->
        def label = params.method ?: file(list[0]).parent.name
        tuple(label.toString(), list)
    }
    benchmark_embeddings(inputs)
}
