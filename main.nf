#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*
 * scFoundry -- single Nextflow entry point.
 *
 * Usage (preferred):   scfoundry <task> --method <method> [options]
 * Usage (direct):      nextflow run main.nf --task <task> --method <method> [options]
 *
 * Each task lives in workflows/tasks/<task>.nf and dispatches to the per-method
 * modules in workflows/methods/. Method modules keep their own parameter defaults:
 * a `params.x = ...` assignment inside an included module is scoped to that module,
 * while `--x` on the command line overrides all of them.
 */

params.task = null

include { DOWNLOAD } from './workflows/tasks/download'
include { EMBED } from './workflows/tasks/embed'
include { TRANSFER } from './workflows/tasks/transfer'
include { FINETUNE } from './workflows/tasks/finetune'
include { BENCHMARK } from './workflows/tasks/benchmark'

workflow {
    def tasks = [
        'download': { DOWNLOAD() },
        'embed':    { EMBED() },
        'transfer': { TRANSFER() },
        'finetune': { FINETUNE() },
        'benchmark': { BENCHMARK() },
    ]

    def selected = params.task ? params.task.toString().trim().toLowerCase() : null
    if( !selected )
        error "No task given. Use: scfoundry <task> ...  or  nextflow run main.nf --task <task> ...\nAvailable tasks: ${tasks.keySet().join(', ')}"

    def run = tasks[selected]
    if( !run )
        error "Unknown task '${params.task}'. Available tasks: ${tasks.keySet().join(', ')}"

    run()
}
