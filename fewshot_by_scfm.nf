nextflow.enable.dsl=2

params.label_key = "cell_type"
params.method  = null
params.mode    = "fit_infer"
params.query   = null
params.support = null
params.fitted  = null

include { fewshot_by_c2s } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_cellama } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_cellfm } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_cellplm } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_geneformer } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_genept_w } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_langcell } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_scbert } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_sccello } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_scfoundation } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_scgpt } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_scimilarity } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_scprint } from "./workflows/tasks/few-shot.nf"
include { fewshot_by_uce } from "./workflows/tasks/few-shot.nf"

include { inference_by_c2s } from "./workflows/tasks/few-shot.nf"
include { inference_by_cellama } from "./workflows/tasks/few-shot.nf"
include { inference_by_cellfm } from "./workflows/tasks/few-shot.nf"
include { inference_by_cellplm } from "./workflows/tasks/few-shot.nf"
include { inference_by_geneformer } from "./workflows/tasks/few-shot.nf"
include { inference_by_genept_w } from "./workflows/tasks/few-shot.nf"
include { inference_by_langcell } from "./workflows/tasks/few-shot.nf"
include { inference_by_scbert } from "./workflows/tasks/few-shot.nf"
include { inference_by_sccello } from "./workflows/tasks/few-shot.nf"
include { inference_by_scfoundation } from "./workflows/tasks/few-shot.nf"
include { inference_by_scgpt } from "./workflows/tasks/few-shot.nf"
include { inference_by_scimilarity } from "./workflows/tasks/few-shot.nf"
include { inference_by_scprint } from "./workflows/tasks/few-shot.nf"
include { inference_by_uce } from "./workflows/tasks/few-shot.nf"

if( !params.method ) {
    exit 1, """Please provide method name using --method parameter."""
}

if (!['fit', 'infer', 'fit_infer'].contains(params.mode)) {
    exit 1, """Invalid mode: ${params.mode}. Must be one of: fit_infer, fit, infer. Default: fit_infer
    - 'fit_infer': Provide support and query data, fit prototypes then immediately infer
    - 'fit': Provide support data only, fit prototypes for each class
    - 'infer': Provide fitted prototypes and query data, return inference results"""
}

if (['fit', 'fit_infer'].contains(params.mode) && !params.support) {
    exit 1, """Support data is required for mode '${params.mode}'.
    Please provide support data using --support parameter."""
}
support_ch = ['fit', 'fit_infer'].contains(params.mode) ? Channel.fromPath(params.support) : null
fit_input = support_ch ? support_ch.map{ file -> tuple("${file.baseName}", file) } : null

if (['infer', 'fit_infer'].contains(params.mode) && !params.query) {
    exit 1, """Query data is required for mode '${params.mode}'.
    Please provide query data using --query parameter."""
}
if (params.mode=='infer' && !params.fitted) {
    exit 1, """Fitted model is required for mode '${params.mode}'.
    Please provide fitted model using --fitted parameter."""
}
query_ch = ['infer', 'fit_infer'].contains(params.mode) ? Channel.fromPath(params.query) : null
prototype_ch = params.mode=='infer' ? Channel.fromPath(params.fitted) : null
infer_input = prototype_ch ? query_ch.combine(prototype_ch).map{ query, prototype -> tuple("${query.baseName}", query, prototype) } : null


def parseMethods(String methodsStr) {
    def s = methodsStr.trim().toLowerCase()
    return s
}

def selected = parseMethods(params.method)

def runners = [
    'scgpt': [
        'fit': { ch -> fewshot_by_scgpt(ch) },
        'infer': { ch -> inference_by_scgpt(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'cellama': [
        'fit': { ch -> fewshot_by_cellama(ch) },
        'infer': { ch -> inference_by_cellama(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'cellfm': [
        'fit': { ch -> fewshot_by_cellfm(ch) },
        'infer': { ch -> inference_by_cellfm(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'cellplm': [
        'fit': { ch -> fewshot_by_cellplm(ch) },
        'infer': { ch -> inference_by_cellplm(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'geneformer': [
        'fit': { ch -> fewshot_by_geneformer(ch) },
        'infer': { ch -> inference_by_geneformer(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'genept': [
        'fit': { ch -> fewshot_by_genept_w(ch) },
        'infer': { ch -> inference_by_genept_w(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'langcell': [
        'fit': { ch -> fewshot_by_langcell(ch) },
        'infer': { ch -> inference_by_langcell(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'scbert': [
        'fit': { ch -> fewshot_by_scbert(ch) },
        'infer': { ch -> inference_by_scbert(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'sccello': [
        'fit': { ch -> fewshot_by_sccello(ch) },
        'infer': { ch -> inference_by_sccello(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'scfoundation': [
        'fit': { ch -> fewshot_by_scfoundation(ch) },
        'infer': { ch -> inference_by_scfoundation(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'scimilarity': [
        'fit': { ch -> fewshot_by_scimilarity(ch) },
        'infer': { ch -> inference_by_scimilarity(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'scprint': [
        'fit': { ch -> fewshot_by_scprint(ch) },
        'infer': { ch -> inference_by_scprint(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'uce': [
        'fit': { ch -> fewshot_by_uce(ch) },
        'infer': { ch -> inference_by_uce(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
    'c2s': [
        'fit': { ch -> fewshot_by_c2s(ch) },
        'infer': { ch -> inference_by_c2s(ch) },
        'fit_infer': { ch -> fit_and_infer() }
    ],
]


workflow fit_and_infer {
    Prototypes = runners[ selected ][ 'fit' ](fit_input)
    InferChannel = query_ch.combine(Prototypes).map{ query, foo, prototype -> tuple("${query.baseName}", query, prototype) }
    runners[ selected ][ 'infer' ](InferChannel)
}

workflow {

    if( !runners[ selected ] )
        exit 1, "Unknown method '${params.method}'. Allowed: ${runners.keySet().join(', ')}"

    def run = runners[ selected ][ params.mode ]
    
    def inputs = [
        'fit': fit_input,
        'infer': infer_input,
        'fit_infer': null
    ]

    run(inputs[params.mode])

}
