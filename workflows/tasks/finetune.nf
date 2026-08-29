/*
 * Task: finetune -- supervised fine-tuning with parameter updates.
 *
 * Each method follows its authors' recipe (see workflows/methods/<method>.nf for the
 * defaults of finetune_epoch / finetune_batch_size / finetune_eval_size). Methods whose
 * official adaptation keeps the backbone frozen and trains a probe are served by
 * `transfer --classifier mlp` instead; scCello is kept here because its official recipe
 * is exactly this linear probing.
 *
 * Modes (inferred from the inputs):
 *   --reference R --query Q      fine-tune on R, predict Q
 *   --reference R                fine-tune only -> finetune/finetuned_models/<method>/<R-id>/
 *   --query Q --fitted DIR       predict with a fine-tuned model directory
 * Outputs: finetune/prediction/<method>/<Q-id>_predicted_{probs,labels}.tsv
 */

params.method    = null
params.reference = null
params.query     = null
params.fitted    = null
params.finetune_label_key   = "cell_type"
params.finetune_results_dir = "results"

include { finetune_by_cellama;      predict_by_cellama      } from './fine-tune'
include { finetune_by_cellfm;       predict_by_cellfm       } from './fine-tune'
include { finetune_by_cellplm;      predict_by_cellplm      } from './fine-tune'
include { finetune_by_geneformer;   predict_by_geneformer   } from './fine-tune'
include { finetune_by_langcell;     predict_by_langcell     } from './fine-tune'
include { finetune_by_scbert;       predict_by_scbert       } from './fine-tune'
include { finetune_by_sccello;      predict_by_sccello      } from './fine-tune'
include { finetune_by_scfoundation; predict_by_scfoundation } from './fine-tune'
include { finetune_by_scgpt;        predict_by_scgpt        } from './fine-tune'

workflow FINETUNE {
    main:
    def fitters = [
        'cellama':      { ch -> finetune_by_cellama(ch) },
        'cellfm':       { ch -> finetune_by_cellfm(ch) },
        'cellplm':      { ch -> finetune_by_cellplm(ch) },
        'geneformer':   { ch -> finetune_by_geneformer(ch) },
        'langcell':     { ch -> finetune_by_langcell(ch) },
        'scbert':       { ch -> finetune_by_scbert(ch) },
        'sccello':      { ch -> finetune_by_sccello(ch) },
        'scfoundation': { ch -> finetune_by_scfoundation(ch) },
        'scgpt':        { ch -> finetune_by_scgpt(ch) },
    ]
    def predictors = [
        'cellama':      { ch -> predict_by_cellama(ch) },
        'cellfm':       { ch -> predict_by_cellfm(ch) },
        'cellplm':      { ch -> predict_by_cellplm(ch) },
        'geneformer':   { ch -> predict_by_geneformer(ch) },
        'langcell':     { ch -> predict_by_langcell(ch) },
        'scbert':       { ch -> predict_by_scbert(ch) },
        'sccello':      { ch -> predict_by_sccello(ch) },
        'scfoundation': { ch -> predict_by_scfoundation(ch) },
        'scgpt':        { ch -> predict_by_scgpt(ch) },
    ]

    if( !params.method )
        error "Usage: scfoundry finetune --method <method> --reference <ref.h5ad> [--query <query.h5ad>]\n       scfoundry finetune --method <method> --query <query.h5ad> --fitted <model_dir>\nAvailable methods: ${fitters.keySet().join(', ')}"
    def selected = params.method.toString().trim().toLowerCase()
    if( !fitters[selected] )
        error "Unknown method '${params.method}'. Available: ${fitters.keySet().join(', ')}"

    def doFit     = params.reference as boolean
    def doPredict = params.query as boolean
    if( !doFit && !doPredict )
        error "Provide --reference (fine-tune), --reference and --query (fine-tune + predict), or --query and --fitted (predict)."
    if( doPredict && !doFit && !params.fitted )
        error "--query without --reference needs --fitted <model_dir> (a previously fine-tuned model directory)."
    if( doFit && params.fitted )
        error "--fitted cannot be combined with --reference."

    def refFile   = doFit ? file(params.reference, checkIfExists: true) : null
    def queryFile = doPredict ? file(params.query, checkIfExists: true) : null
    if( refFile && queryFile && refFile.baseName == queryFile.baseName )
        error "Reference and query files must have different base names (both are '${refFile.baseName}'); outputs are named by them."

    def model_ch
    if( doFit ) {
        ref_ch = Channel.of(tuple(refFile.baseName, refFile))
        model_ch = fitters[selected](ref_ch)                     // tuple(id, model_weights)
    }
    if( doPredict ) {
        if( doFit ) {
            pred_ch = Channel.of(queryFile).combine(model_ch)
                .map { q, ref_id, weights -> tuple(q.baseName, q, weights) }
        } else {
            pred_ch = Channel.of(tuple(queryFile.baseName, queryFile, file(params.fitted, checkIfExists: true)))
        }
        predictors[selected](pred_ch)
    }
}
