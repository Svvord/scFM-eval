nextflow.enable.dsl=2

params.finetune_label_key = "cell_type"
params.method  = null
params.mode    = "fit_pred"
params.test    = null
params.train   = null
params.fitted  = null

include { finetune_by_cellama } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_scimilarity } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_genept_w } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_cellfm } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_cellplm } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_sccello } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_scbert } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_scfoundation } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_scprint } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_scgpt } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_langcell } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_geneformer } from "./workflows/tasks/fine-tune.nf"
include { finetune_by_uce } from "./workflows/tasks/fine-tune.nf"

include { predict_by_cellama } from "./workflows/tasks/fine-tune.nf"
include { predict_by_scimilarity } from "./workflows/tasks/fine-tune.nf"
include { predict_by_genept_w } from "./workflows/tasks/fine-tune.nf"
include { predict_by_cellfm } from "./workflows/tasks/fine-tune.nf"
include { predict_by_cellplm } from "./workflows/tasks/fine-tune.nf"
include { predict_by_sccello } from "./workflows/tasks/fine-tune.nf"
include { predict_by_scbert } from "./workflows/tasks/fine-tune.nf"
include { predict_by_scfoundation } from "./workflows/tasks/fine-tune.nf"
include { predict_by_scprint } from "./workflows/tasks/fine-tune.nf"
include { predict_by_scgpt } from "./workflows/tasks/fine-tune.nf"
include { predict_by_langcell } from "./workflows/tasks/fine-tune.nf"
include { predict_by_geneformer } from "./workflows/tasks/fine-tune.nf"
include { predict_by_uce } from "./workflows/tasks/fine-tune.nf"


if( !params.method ) {
    exit 1, """Please provide method name using --method parameter."""
}

if (!['fit', 'pred', 'fit_pred'].contains(params.mode)) {
    exit 1, """Invalid mode: ${params.mode}. Must be one of: fit_pred, fit, pred. Default: fit_pred
    - 'fit_pred': Provide train and test data, finetune model then immediately predict
    - 'fit': Provide train data only, finetune model
    - 'pred': Provide finetuned model and test data, return predictions"""
}

if (['fit', 'fit_pred'].contains(params.mode) && !params.train) {
    exit 1, """Train data is required for mode '${params.mode}'.
    Please provide train data using --train parameter."""
}
train_ch = ['fit', 'fit_pred'].contains(params.mode) ? Channel.fromPath(params.train) : null
fit_input = train_ch ? train_ch.map{ file -> tuple("${file.baseName}", file) } : null

if (['pred', 'fit_pred'].contains(params.mode) && !params.test) {
    exit 1, """Test data is required for mode '${params.mode}'.
    Please provide test data using --test parameter."""
}
if (params.mode=='pred' && !params.fitted) {
    exit 1, """Finetuned model is required for mode '${params.mode}'.
    Please provide finetuned model using --fitted parameter."""
}
test_ch = ['pred', 'fit_pred'].contains(params.mode) ? Channel.fromPath(params.test) : null
model_weights_ch = params.mode=='pred' ? Channel.fromPath(params.fitted) : null
pred_input = model_weights_ch ? test_ch.combine(model_weights_ch).map{ test, model_weights -> tuple("${test.baseName}", test, model_weights) } : null

def parseMethods(String methodsStr) {
    def s = methodsStr.trim().toLowerCase()
    return s
}

def selected = parseMethods(params.method)

def runners = [
    'cellama': [
        'fit': { ch -> finetune_by_cellama(ch) },
        'pred': { ch -> predict_by_cellama(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'scimilarity': [
        'fit': { ch -> finetune_by_scimilarity(ch) },
        'pred': { ch -> predict_by_scimilarity(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'genept': [
        'fit': { ch -> finetune_by_genept_w(ch) },
        'pred': { ch -> predict_by_genept_w(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'cellfm': [
        'fit': { ch -> finetune_by_cellfm(ch) },
        'pred': { ch -> predict_by_cellfm(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'cellplm': [
        'fit': { ch -> finetune_by_cellplm(ch) },
        'pred': { ch -> predict_by_cellplm(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'sccello': [
        'fit': { ch -> finetune_by_sccello(ch) },
        'pred': { ch -> predict_by_sccello(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'scbert': [
        'fit': { ch -> finetune_by_scbert(ch) },
        'pred': { ch -> predict_by_scbert(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'scfoundation': [
        'fit': { ch -> finetune_by_scfoundation(ch) },
        'pred': { ch -> predict_by_scfoundation(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'scprint': [
        'fit': { ch -> finetune_by_scprint(ch) },
        'pred': { ch -> predict_by_scprint(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'scgpt': [
        'fit': { ch -> finetune_by_scgpt(ch) },
        'pred': { ch -> predict_by_scgpt(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'langcell': [
        'fit': { ch -> finetune_by_langcell(ch) },
        'pred': { ch -> predict_by_langcell(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'geneformer': [
        'fit': { ch -> finetune_by_geneformer(ch) },
        'pred': { ch -> predict_by_geneformer(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
    'uce': [
        'fit': { ch -> finetune_by_uce(ch) },
        'pred': { ch -> predict_by_uce(ch) },
        'fit_pred': { ch -> finetune_and_predict() }
    ],
]

workflow finetune_and_predict {
    ModelWeights = runners[ selected ][ 'fit' ](fit_input)
    InferChannel = test_ch.combine(ModelWeights).map{ test, foo, model_weights -> tuple("${test.baseName}", test, model_weights) }
    runners[ selected ][ 'pred' ](InferChannel)
}

workflow {

    log.warn "finetune_by_scfm.nf is deprecated; use 'scfoundry finetune --method <method> --reference <train.h5ad> --query <test.h5ad>' (or 'nextflow run main.nf --task finetune'). This entry point will be removed in a future release."

    if( !runners[ selected ] )
        exit 1, "Unknown method '${params.method}'. Allowed: ${runners.keySet().join(', ')}"

    def run = runners[ selected ][ params.mode ]
    
    def inputs = [
        'fit': fit_input,
        'pred': pred_input,
        'fit_pred': null
    ]

    run(inputs[params.mode])

}