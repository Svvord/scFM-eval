/*
 * Task: transfer -- frozen-embedding label transfer.
 *
 * The reference and query files are embedded with the chosen (frozen) method, a
 * lightweight classifier is fitted on the reference embedding using obs[label_key]
 * (prototype | knn | logreg | mlp), and the query cells are labelled with it. No model
 * parameters are updated; with a tiny reference this is the few-shot setting.
 *
 * Modes (inferred from the inputs):
 *   --reference R --query Q      fit on R, predict Q
 *   --reference R                fit only   -> transfer/models/<method>/<classifier>/<R-id>/
 *   --query Q --fitted DIR       predict only with a previously fitted model directory
 * Outputs: transfer/predictions/<method>/<classifier>/<Q-id>_predicted_{probs,labels}.tsv
 */

params.method     = null
params.reference  = null
params.query      = null
params.fitted     = null
params.classifier = "logreg"
params.label_key  = "cell_type"
params.knn_k      = 15
params.transfer_results_dir = "results"

include { embed_by_c2s } from '../methods/c2s'
include { embed_by_cellama } from '../methods/cellama'
include { embed_by_cellfm } from '../methods/cellfm'
include { embed_by_cellplm } from '../methods/cellplm'
include { embed_by_geneformer } from '../methods/geneformer'
include { embed_by_genept_w } from '../methods/genept'
include { embed_by_langcell } from '../methods/langcell'
include { embed_by_scbert } from '../methods/scbert'
include { embed_by_sccello } from '../methods/sccello'
include { embed_by_scfoundation } from '../methods/scfoundation'
include { embed_by_scgpt } from '../methods/scgpt'
include { embed_by_scimilarity } from '../methods/scimilarity'
include { embed_by_scprint } from '../methods/scprint'
include { embed_by_scvi } from '../methods/scvi'
include { embed_by_uce } from '../methods/uce'


process transfer_fit {

    tag "${id}:${params.classifier}"

    label 'cpu_task'

    debug true   // surface the classifier's summary and warnings (e.g. k capped for a tiny reference)

    container "housy17/scllms:latest"

    publishDir "${params.transfer_results_dir}/transfer/models/${method}/${params.classifier}", mode: 'copy',
               enabled: params.transfer_results_dir as boolean

    input:
    tuple val(id), val(method), path(embedding_h5ad)

    output:
    tuple val(id), val(method), path("${id}")

    script:
    """
    python /code/transfer/classifier.py fit \\
        --embedding ${embedding_h5ad} \\
        --label-key '${params.label_key}' \\
        --classifier ${params.classifier} \\
        --method ${method} \\
        --knn-k ${params.knn_k} \\
        --out ${id}
    """
}


process transfer_predict {

    tag "${id}:${params.classifier}"

    label 'cpu_task'

    debug true   // surface the classifier's summary and warnings (e.g. k capped for a tiny reference)

    container "housy17/scllms:latest"

    publishDir "${params.transfer_results_dir}/transfer/predictions/${method}/${params.classifier}", mode: 'copy',
               pattern: "*_predicted_*.tsv", enabled: params.transfer_results_dir as boolean

    input:
    tuple val(id), val(method), path(embedding_h5ad), path(model_dir)

    output:
    tuple val(id), val(method), path("${id}_predicted_probs.tsv"), path("${id}_predicted_labels.tsv")

    script:
    """
    python /code/transfer/classifier.py predict \\
        --embedding ${embedding_h5ad} \\
        --model ${model_dir} \\
        --out-prefix ${id}
    """
}


workflow TRANSFER {
    main:
    def embedders = [
        'c2s':          { ch -> embed_by_c2s(ch) },
        'cellama':      { ch -> embed_by_cellama(ch) },
        'cellfm':       { ch -> embed_by_cellfm(ch) },
        'cellplm':      { ch -> embed_by_cellplm(ch) },
        'geneformer':   { ch -> embed_by_geneformer(ch) },
        'genept':       { ch -> embed_by_genept_w(ch) },
        'langcell':     { ch -> embed_by_langcell(ch) },
        'scbert':       { ch -> embed_by_scbert(ch) },
        'sccello':      { ch -> embed_by_sccello(ch) },
        'scfoundation': { ch -> embed_by_scfoundation(ch) },
        'scgpt':        { ch -> embed_by_scgpt(ch) },
        'scimilarity':  { ch -> embed_by_scimilarity(ch) },
        'scprint':      { ch -> embed_by_scprint(ch) },
        'scvi':         { ch -> embed_by_scvi(ch) },
        'uce':          { ch -> embed_by_uce(ch) },
    ]
    def classifiers = ['prototype', 'knn', 'logreg', 'mlp']

    if( !params.method )
        error "Usage: scfoundry transfer --method <method> --reference <ref.h5ad> [--query <query.h5ad>] [--classifier ${classifiers.join('|')}]\n       scfoundry transfer --method <method> --query <query.h5ad> --fitted <model_dir>\nAvailable methods: ${embedders.keySet().join(', ')}"
    def selected = params.method.toString().trim().toLowerCase()
    def embed = embedders[selected]
    if( !embed )
        error "Unknown method '${params.method}'. Available: ${embedders.keySet().join(', ')}"
    if( !(params.classifier in classifiers) )
        error "Unknown classifier '${params.classifier}'. Available: ${classifiers.join(', ')}"

    def doFit     = params.reference as boolean
    def doPredict = params.query as boolean
    if( !doFit && !doPredict )
        error "Provide --reference (fit), --reference and --query (fit + predict), or --query and --fitted (predict)."
    if( doPredict && !doFit && !params.fitted )
        error "--query without --reference needs --fitted <model_dir> (a previously fitted transfer model)."
    if( doFit && params.fitted )
        error "--fitted cannot be combined with --reference (fit a new model or predict with an existing one, not both)."

    def refFile   = doFit ? file(params.reference, checkIfExists: true) : null
    def queryFile = doPredict ? file(params.query, checkIfExists: true) : null
    if( refFile && queryFile && refFile.baseName == queryFile.baseName )
        error "Reference and query files must have different base names (both are '${refFile.baseName}'); outputs are named by them."

    // Embed reference and query in ONE invocation of the method workflow, then split by role.
    def raw_ch  = Channel.empty()
    def role_ch = Channel.empty()
    if( doFit ) {
        raw_ch  = raw_ch.mix(Channel.of(tuple(refFile.baseName, refFile)))
        role_ch = role_ch.mix(Channel.of(tuple(refFile.baseName, 'reference')))
    }
    if( doPredict ) {
        raw_ch  = raw_ch.mix(Channel.of(tuple(queryFile.baseName, queryFile)))
        role_ch = role_ch.mix(Channel.of(tuple(queryFile.baseName, 'query')))
    }
    embedded = embed(raw_ch)                                   // tuple(id, display_name, embedding_h5ad)
        .combine(role_ch, by: 0)
        .map { id, display, h5ad, role -> tuple(id, selected, h5ad, role) }

    def model_ch
    if( doFit ) {
        ref_emb = embedded.filter { it[3] == 'reference' }.map { id, m, h5ad, role -> tuple(id, m, h5ad) }
        transfer_fit(ref_emb)
        model_ch = transfer_fit.out.map { id, m, dir -> dir }.first()
    } else {
        model_ch = Channel.value(file(params.fitted, checkIfExists: true))
    }
    if( doPredict ) {
        query_emb = embedded.filter { it[3] == 'query' }.map { id, m, h5ad, role -> tuple(id, m, h5ad) }
        transfer_predict(query_emb.combine(model_ch))
    }
}
