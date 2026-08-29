"""Command-line launcher for scFoundry.

scFoundry is a Nextflow DSL2 framework for deploying, running and evaluating
single-cell foundation models. This module is a thin, dependency-free front end:
it locates the bundled pipeline and the user's workspace, validates the request,
creates an isolated run directory and invokes

    nextflow run <pipeline>/main.nf --task <task> ...

Nextflow stays the execution engine (containers, caching, -resume, HPC executors).
Only the Python standard library is used (Python >= 3.6).

Layout
------
pipeline   main.nf, workflows/, bin/, conf/, nextflow.example.config. Bundled inside
           the installed package (scfoundry/pipeline) or, for a development checkout,
           the repository root.
workspace  A directory created by `scfoundry init` holding nextflow.config, model
           weights, container cache, results and run records. Commands find it like
           git finds a repository: --workspace, $SCFOUNDRY_WORKSPACE, or by walking up
           from the current directory to a `.scfoundry.json` marker.
"""
import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict

from . import __version__ as VERSION

PROG = "scfoundry"
MARKER = ".scfoundry.json"
CONFIG_NAME = "nextflow.config"
TEMPLATE_NAME = "nextflow.example.config"
CONTAINER_RUNTIMES = ("apptainer", "singularity", "docker")

# Task catalogue. "implemented" tasks get a sub-command; the others are shown by
# `scfoundry list tasks` as planned and are added one release at a time.
TASKS = OrderedDict([
    ("download",  dict(help="Download pretrained model checkpoints", implemented=True)),
    ("embed",     dict(help="Zero-shot cell embeddings", implemented=True)),
    ("transfer",  dict(help="Label transfer with frozen embeddings (prototype / knn / logreg / mlp)", implemented=True)),
    ("finetune",  dict(help="Supervised fine-tuning (parameter updates) / prediction", implemented=True)),
    ("benchmark", dict(help="Score embeddings (biological conservation, batch mixing)", implemented=True)),
    ("geometry",  dict(help="Representation-geometry probes of embeddings", implemented=True)),
])

# Nextflow parameters that receive --outdir, per task (all set to the same path).
OUTDIR_PARAMS = {
    "download": [],
    "embed": ["emb_results_dir"],
    "transfer": ["emb_results_dir", "transfer_results_dir"],
    "finetune": ["emb_results_dir", "finetune_results_dir"],
    "benchmark": ["results_dir"],
    "geometry": ["results_dir"],
}


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
def info(msg):
    print("[{}] {}".format(PROG, msg))


def warn(msg):
    print("[{}] warning: {}".format(PROG, msg), file=sys.stderr)


def die(msg, code=2):
    print("[{}] error: {}".format(PROG, msg), file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------- #
# Pipeline location
# --------------------------------------------------------------------------- #
PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def pipeline_dir():
    """Directory containing main.nf, workflows/, bin/, conf/."""
    env = os.environ.get("SCFOUNDRY_PIPELINE")
    candidates = [os.path.realpath(env)] if env else [
        os.path.join(PKG_DIR, "pipeline"),           # installed wheel
        os.path.dirname(PKG_DIR),                    # development checkout
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "main.nf")):
            return cand
    die("cannot locate the scFoundry pipeline (main.nf). Looked in: {}".format(", ".join(candidates)))


def load_registry(pipe):
    path = os.path.join(pipe, "conf", "methods.json")
    try:
        with open(path) as fh:
            return json.load(fh, object_pairs_hook=OrderedDict)
    except (OSError, ValueError) as exc:
        die("cannot read method registry {}: {}".format(path, exc))


def methods_for_task(registry, task):
    return [m for m, spec in registry.items() if task in spec.get("tasks", [])]


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #
class Workspace(object):
    def __init__(self, root):
        self.root = os.path.realpath(root)

    @property
    def marker(self):
        return os.path.join(self.root, MARKER)

    @property
    def config(self):
        return os.path.join(self.root, CONFIG_NAME)

    @property
    def runs_dir(self):
        return os.path.join(self.root, "runs")

    @property
    def results_dir(self):
        return os.path.join(self.root, "results")

    @property
    def weights_dir(self):
        return os.path.join(self.root, "data", "model_weights")

    @property
    def cache_dir(self):
        return os.path.join(self.root, "cache")

    def exists(self):
        return os.path.isfile(self.marker)


def find_workspace(explicit=None):
    if explicit:
        ws = Workspace(explicit)
        if not ws.exists():
            die("{} is not a scFoundry workspace (no {}); run '{} init {}' first".format(
                ws.root, MARKER, PROG, explicit))
        return ws
    env = os.environ.get("SCFOUNDRY_WORKSPACE")
    if env:
        ws = Workspace(env)
        if not ws.exists():
            die("$SCFOUNDRY_WORKSPACE={} is not a scFoundry workspace (no {})".format(env, MARKER))
        return ws
    cur = os.getcwd()
    while True:
        ws = Workspace(cur)
        if ws.exists():
            return ws
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    die("not inside a scFoundry workspace. Run '{} init' in the directory you want to work in "
        "(or '{} init DIR'), or pass --workspace DIR.".format(PROG, PROG))


# --------------------------------------------------------------------------- #
# Tool discovery
# --------------------------------------------------------------------------- #
def find_nextflow(explicit=None, required=True):
    cand = explicit or os.environ.get("SCFOUNDRY_NEXTFLOW") or "nextflow"
    path = shutil.which(cand) if os.path.sep not in cand else (cand if os.access(cand, os.X_OK) else None)
    if path is None and required:
        die("'{}' not found on PATH. scFoundry needs Nextflow (https://www.nextflow.io/docs/latest/install.html); "
            "activate the environment that provides it, or set SCFOUNDRY_NEXTFLOW / --nextflow.".format(cand))
    return path or cand


def nextflow_version(nf):
    try:
        out = subprocess.run([nf, "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, timeout=120).stdout
        m = re.search(r"version\s+([\w.\-]+)", out)
        return m.group(1) if m else out.strip()
    except Exception as exc:  # noqa: BLE001
        return "unknown ({})".format(exc)


def detect_runtime():
    for rt in CONTAINER_RUNTIMES:
        if shutil.which(rt):
            return rt
    return None


# --------------------------------------------------------------------------- #
# Parameter helpers
# --------------------------------------------------------------------------- #
def coerce(value):
    """Mimic Nextflow's command-line parameter coercion for passthrough options."""
    if value is True:
        return True
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?", value):
        return float(value)
    return value


def passthrough_to_params(tokens):
    """Turn leftover `--key value` / `--flag` / `--key=value` tokens into params."""
    params = OrderedDict()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--") or tok == "--":
            die("unrecognised argument '{}'".format(tok))
        key, eq, val = tok[2:].partition("=")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            die("'{}' is not a valid parameter name".format(tok))
        if eq:
            params[key] = coerce(val)
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):  # allows negative numbers
            params[key] = coerce(tokens[i + 1])
            i += 2
        else:
            params[key] = True
            i += 1
    return params


def safe_id(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")
    return text or "run"


def input_id(path):
    base = os.path.basename(path.rstrip("/"))
    return safe_id(re.sub(r"\.h5ad$", "", base))


def run_signature(method, inp):
    return "_".join(x for x in (safe_id(method), inp) if x)


def shell_quote(s):
    return s if re.fullmatch(r"[A-Za-z0-9_./:=@%+,-]+", s) else "'" + s.replace("'", "'\"'\"'") + "'"


# --------------------------------------------------------------------------- #
# Configuration template handling (init)
# --------------------------------------------------------------------------- #
def set_runtime(config_text, runtime):
    """Enable exactly one of docker/singularity/apptainer in a config text."""
    if runtime not in CONTAINER_RUNTIMES:
        die("unknown container runtime '{}'".format(runtime))
    out, block = [], None
    for line in config_text.splitlines(True):
        m = re.match(r"^(\w+)\s*\{\s*$", line)
        if m:
            block = m.group(1)
        elif re.match(r"^\}\s*$", line):
            block = None
        elif block in CONTAINER_RUNTIMES and re.match(r"^\s*enabled\s*=", line):
            line = re.sub(r"=\s*(true|false)", "= {}".format("true" if block == runtime else "false"), line)
        out.append(line)
    return "".join(out)


def set_gpu_id(config_text, gpu_id):
    value = "null" if gpu_id in (None, "") else json.dumps(str(gpu_id))
    text, n = re.subn(r"^(\s*gpu_id\s*=\s*).*$", lambda m: m.group(1) + value, config_text, count=1, flags=re.M)
    if n == 0:
        die("{} has no 'gpu_id' parameter; cannot set --gpu-id".format(TEMPLATE_NAME))
    return text


def cmd_init(args):
    pipe = pipeline_dir()
    template = os.path.join(pipe, TEMPLATE_NAME)
    if not os.path.isfile(template):
        die("template {} is missing from the pipeline".format(template))

    ws = Workspace(args.directory or os.getcwd())
    if ws.exists() and not args.force:
        die("{} is already a scFoundry workspace; use --force to rewrite its {}".format(ws.root, CONFIG_NAME))
    if os.path.isfile(ws.config) and not ws.exists() and not args.force:
        die("{} already exists in {}; use --force to overwrite it".format(CONFIG_NAME, ws.root))

    runtime = args.runtime or detect_runtime()
    if runtime is None:
        warn("no container runtime (apptainer/singularity/docker) found on PATH; defaulting to apptainer")
        runtime = "apptainer"

    for d in (ws.root, ws.weights_dir, os.path.join(ws.cache_dir, ".home"), os.path.join(ws.cache_dir, ".shared"),
              ws.results_dir, ws.runs_dir):
        os.makedirs(d, exist_ok=True)
    with open(template) as fh:
        text = fh.read()
    text = set_runtime(text, runtime)
    if args.gpu_id is not None:
        text = set_gpu_id(text, args.gpu_id)
    with open(ws.config, "w") as fh:
        fh.write(text)
    meta = OrderedDict([
        ("scfoundry_version", VERSION),
        ("created", _dt.datetime.now().isoformat(timespec="seconds")),
        ("pipeline", pipe),
        ("container_runtime", runtime),
    ])
    with open(ws.marker, "w") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")

    info("workspace: {}".format(ws.root))
    info("wrote {} (container runtime: {}{})".format(
        os.path.relpath(ws.config, os.getcwd()), runtime,
        ", gpu_id: {}".format(args.gpu_id) if args.gpu_id is not None else ""))
    info("pipeline:  {}".format(pipe))

    # ---- environment report (never fatal) ---------------------------------- #
    rows = []
    nf = find_nextflow(args.nextflow, required=False)
    nf_path = shutil.which(nf) if os.path.sep not in nf else (nf if os.access(nf, os.X_OK) else None)
    rows.append(("nextflow", "{} ({})".format(nextflow_version(nf_path), nf_path) if nf_path
                 else "NOT FOUND -- install Nextflow or activate the environment that has it"))
    rt_path = shutil.which(runtime)
    rows.append((runtime, rt_path or "NOT FOUND on PATH"))
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "-L"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 universal_newlines=True, timeout=60).stdout.strip()
            gpus = [l for l in out.splitlines() if l.startswith("GPU")]
            rows.append(("GPU", "{} visible".format(len(gpus)) if gpus
                         else "nvidia-smi found but no GPU visible (fine on a login node)"))
        except Exception as exc:  # noqa: BLE001
            rows.append(("GPU", "nvidia-smi failed: {}".format(exc)))
    else:
        rows.append(("GPU", "nvidia-smi not found (fine on a login node; model tasks need a GPU node)"))
    rows.append(("python", "{} ({})".format(sys.version.split()[0], sys.executable)))
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print("  {:<{w}}  {}".format(k, v, w=width))
    rel = os.path.relpath(ws.root, os.getcwd())
    cd = "" if rel == "." else "cd {}\n  ".format(shell_quote(rel))
    print("\nNext steps:\n  {cd}{p} download --method scgpt\n  {p} list methods".format(cd=cd, p=PROG))
    return 0


# --------------------------------------------------------------------------- #
# Run directories and Nextflow invocation
# --------------------------------------------------------------------------- #
def find_resumable(ws, task, method, inp_id, name):
    """Run directory to resume: an explicit name, or the newest run of this task with
    the same method/input (from run.json; falls back to the directory-name signature)."""
    task_dir = os.path.join(ws.runs_dir, task)
    if name is not True:  # explicit run name
        cand = os.path.join(task_dir, name)
        if not os.path.isdir(cand):
            die("run '{}' not found under {}".format(name, task_dir))
        return cand
    if not os.path.isdir(task_dir):
        die("nothing to resume: no previous '{}' runs in {}".format(task, ws.root))
    signature = run_signature(method, inp_id)
    matches = []
    for d in os.listdir(task_dir):
        path = os.path.join(task_dir, d)
        if not os.path.isdir(path):
            continue
        meta_path = os.path.join(path, "run.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                continue
            if meta.get("method") == method and (meta.get("input") or "") == (inp_id or ""):
                matches.append((meta.get("started") or "", d, path))
        elif d.endswith("_" + signature):
            matches.append(("", d, path))
    if not matches:
        die("nothing to resume: no previous '{}' run for '{}'".format(task, signature))
    matches.sort()
    return matches[-1][2]


def make_run_dir(ws, task, method, inp_id, run_name, resume, create=True):
    if resume:
        return find_resumable(ws, task, method, inp_id, resume), True
    signature = run_signature(method, inp_id)
    name = safe_id(run_name) if run_name else "{}_{}".format(_dt.datetime.now().strftime("%Y%m%d-%H%M%S"), signature)
    path = os.path.join(ws.runs_dir, task, name)
    if os.path.exists(path):
        die("run directory already exists: {} (use --resume {} or another --run-name)".format(path, name))
    if create:
        os.makedirs(path)
    return path, False


def launch(task, method, inp_id, params, args, extra_nf_args):
    """Common driver: resolve workspace/pipeline, build params, invoke Nextflow."""
    pipe = pipeline_dir()
    ws = find_workspace(args.workspace)
    if not os.path.isfile(ws.config):
        die("{} not found in workspace {}; run '{} init --force {}'".format(CONFIG_NAME, ws.root, PROG, ws.root))
    nf = find_nextflow(args.nextflow, required=not args.dry_run)

    run_dir, resumed = make_run_dir(ws, task, method, inp_id, args.run_name, args.resume, create=not args.dry_run)

    full = OrderedDict([("task", task), ("method", method)])
    full.update(params)
    outdir_params = OUTDIR_PARAMS.get(task, [])
    if outdir_params:
        outdir = os.path.realpath(args.outdir) if args.outdir else ws.results_dir
        for key in outdir_params:
            full[key] = outdir
    elif getattr(args, "outdir", None):
        warn("--outdir is not used by task '{}'".format(task))
    if args.gpu is not None:
        full["gpu_id"] = str(args.gpu)
    if args.weights_dir:
        full["model_weights_dir"] = os.path.realpath(args.weights_dir)
    if args.cache_dir:
        full["cache_dir"] = os.path.realpath(args.cache_dir)
    full.update(args.passthrough)

    cmd = [nf]
    if args.quiet:
        cmd.append("-quiet")
    cmd += ["-log", "nextflow.log", "run", os.path.join(pipe, "main.nf"),
            "-params-file", "params.json", "-work-dir", "work"]
    if not sys.stdout.isatty() or args.quiet:
        cmd += ["-ansi-log", "false"]
    if resumed:
        cmd.append("-resume")
    # Nextflow auto-loads <pipeline>/nextflow.config; in a development checkout that IS
    # the workspace config, so only pass -c when the workspace lives elsewhere.
    if ws.root != os.path.realpath(pipe):
        cmd += ["-c", ws.config]
    if args.profile:
        cmd += ["-profile", args.profile]
    if args.config:
        cmd += ["-c", os.path.realpath(args.config)]
    cmd += list(extra_nf_args)

    env = dict(os.environ)
    env["SCFOUNDRY_WORKSPACE"] = ws.root
    env["SCFOUNDRY_PIPELINE"] = pipe
    # The pipeline modules use the classic DSL2 style (top-level params defaults,
    # `exit` statements). Nextflow >= 26 parses the strict syntax by default and
    # rejects that style, so select the legacy parser unless the user chose one.
    env.setdefault("NXF_SYNTAX_PARSER", "v1")

    if args.dry_run:
        info("dry run -- workspace {}".format(ws.root))
        info("would create {}".format(run_dir if not resumed else run_dir + " (resume)"))
        print("params.json:")
        print(json.dumps(full, indent=2))
        print("command (cwd = run directory; env SCFOUNDRY_WORKSPACE={} NXF_SYNTAX_PARSER={}):".format(ws.root, env["NXF_SYNTAX_PARSER"]))
        print("  " + " ".join(shell_quote(c) for c in cmd))
        return 0

    with open(os.path.join(run_dir, "params.json"), "w") as fh:
        json.dump(full, fh, indent=2)
        fh.write("\n")
    with open(os.path.join(run_dir, "command.sh"), "a") as fh:
        fh.write("#!/usr/bin/env bash\n# generated by {} {} on {}\nexport SCFOUNDRY_WORKSPACE={}\nexport SCFOUNDRY_PIPELINE={}\nexport NXF_SYNTAX_PARSER={}\ncd {}\n{}\n".format(
            PROG, VERSION, _dt.datetime.now().isoformat(timespec="seconds"),
            shell_quote(ws.root), shell_quote(pipe), shell_quote(env["NXF_SYNTAX_PARSER"]), shell_quote(run_dir),
            " ".join(shell_quote(c) for c in cmd)))
    meta = OrderedDict([
        ("task", task), ("method", method), ("input", inp_id or None),
        ("workspace", ws.root), ("pipeline", pipe), ("run_dir", run_dir), ("resumed", resumed),
        ("started", _dt.datetime.now().isoformat(timespec="seconds")),
        ("finished", None), ("exit_code", None), ("status", "running"),
        ("scfoundry_version", VERSION), ("nextflow", nf),
    ])
    meta_path = os.path.join(run_dir, "run.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    info("task={} method={}{}".format(task, method, " input={}".format(inp_id) if inp_id else ""))
    info("workspace: {}".format(ws.root))
    info("run directory: {}".format(os.path.relpath(run_dir, os.getcwd())))
    if not args.quiet:
        info("$ " + " ".join(shell_quote(c) for c in cmd))
    try:
        proc = subprocess.run(cmd, cwd=run_dir, env=env)
        code = proc.returncode
    except KeyboardInterrupt:
        code = 130
    meta["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    meta["exit_code"] = code
    meta["status"] = "ok" if code == 0 else ("interrupted" if code == 130 else "failed")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    if code == 0:
        info("done ({}).".format(meta["status"]))
    else:
        warn("nextflow exited with code {}; see {}".format(code, os.path.join(run_dir, "nextflow.log")))
    return code


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #
def require_method(registry, task, method):
    method = method.strip().lower()
    allowed = methods_for_task(registry, task)
    if method not in allowed:
        die("method '{}' does not support task '{}'. Available: {}".format(method, task, ", ".join(allowed)))
    return method


def cmd_download(args, registry, extra):
    method = require_method(registry, "download", args.method)
    params = OrderedDict()
    if args.model:
        params["model"] = args.model
    return launch("download", method, "", params, args, extra)


def cmd_embed(args, registry, extra):
    method = require_method(registry, "embed", args.method)
    data = os.path.realpath(args.data)
    if not os.path.isfile(data):
        die("input file not found: {}".format(args.data))
    params = OrderedDict([("data", data)])
    if args.model:
        params["model"] = args.model
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.batch_key:
        if registry[method].get("category") != "integration":
            warn("--batch-key is only used by integration methods; '{}' ignores it".format(method))
        params["batch_key"] = args.batch_key
    return launch("embed", method, input_id(data), params, args, extra)


TRANSFER_CLASSIFIERS = ("prototype", "knn", "logreg", "mlp")


def cmd_transfer(args, registry, extra):
    method = require_method(registry, "transfer", args.method)
    if not args.reference and not args.query:
        die("provide --reference (fit), --reference and --query (fit + predict), or --query with --fitted (predict)")
    if args.query and not args.reference and not args.fitted:
        die("--query without --reference needs --fitted <model_dir>")
    if args.reference and args.fitted:
        die("--fitted cannot be combined with --reference")
    params = OrderedDict()
    for key in ("reference", "query"):
        val = getattr(args, key)
        if val:
            path = os.path.realpath(val)
            if not os.path.isfile(path):
                die("{} file not found: {}".format(key, val))
            params[key] = path
    if args.fitted:
        fitted = os.path.realpath(args.fitted)
        if not os.path.isfile(os.path.join(fitted, "meta.json")):
            die("--fitted must be a transfer model directory containing meta.json: {}".format(args.fitted))
        params["fitted"] = fitted
    params["classifier"] = args.classifier
    if args.label_key:
        params["label_key"] = args.label_key
    if args.model:
        params["model"] = args.model
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.knn_k is not None:
        params["knn_k"] = args.knn_k
    inp = input_id(args.query or args.reference)
    return launch("transfer", method, inp, params, args, extra)


def cmd_finetune(args, registry, extra):
    method = require_method(registry, "finetune", args.method)
    if not args.reference and not args.query:
        die("provide --reference (fine-tune), --reference and --query (fine-tune + predict), or --query with --fitted (predict)")
    if args.query and not args.reference and not args.fitted:
        die("--query without --reference needs --fitted <model_dir>")
    if args.reference and args.fitted:
        die("--fitted cannot be combined with --reference")
    params = OrderedDict()
    for key in ("reference", "query"):
        val = getattr(args, key)
        if val:
            path = os.path.realpath(val)
            if not os.path.isfile(path):
                die("{} file not found: {}".format(key, val))
            params[key] = path
    if args.fitted:
        fitted = os.path.realpath(args.fitted)
        if not os.path.isdir(fitted):
            die("--fitted must be a fine-tuned model directory: {}".format(args.fitted))
        params["fitted"] = fitted
    params["finetune_label_key"] = args.label_key or "cell_type"
    if args.epochs is not None:
        params["finetune_epoch"] = args.epochs
    if args.batch_size is not None:
        params["finetune_batch_size"] = args.batch_size
    if args.model:
        params["model"] = args.model
    inp = input_id(args.query or args.reference)
    return launch("finetune", method, inp, params, args, extra)


def cmd_benchmark(args, extra):
    spec = args.embedding
    path = os.path.realpath(spec)
    if os.path.isdir(path):
        label = args.method or os.path.basename(path.rstrip("/"))
        inp = os.path.basename(path.rstrip("/"))
    elif os.path.isfile(path):
        label = args.method or os.path.basename(os.path.dirname(path))
        inp = input_id(path)
    elif any(ch in spec for ch in "*?["):
        label = args.method or os.path.basename(os.path.dirname(path))
        inp = safe_id(os.path.basename(os.path.dirname(path)) or "glob")
    else:
        die("embedding path not found: {}".format(spec))
    metrics = args.metrics or ("all" if args.batch_key else "bio")
    params = OrderedDict([
        ("embedding", path), ("label_key", args.label_key), ("metrics", metrics), ("clustering", args.clustering),
    ])
    if args.batch_key:
        params["batch_key"] = args.batch_key
    if args.batch_max_cells is not None:
        params["batch_max_cells"] = args.batch_max_cells
    return launch("benchmark", safe_id(label), inp, params, args, extra)


def resolve_embedding_spec(spec, method_override):
    """Return (absolute spec, method label, input id) for a file / directory / glob spec."""
    path = os.path.realpath(spec)
    if os.path.isdir(path):
        return path, method_override or os.path.basename(path.rstrip("/")), os.path.basename(path.rstrip("/"))
    if os.path.isfile(path):
        return path, method_override or os.path.basename(os.path.dirname(path)), input_id(path)
    if any(ch in spec for ch in "*?["):
        return path, method_override or os.path.basename(os.path.dirname(path)), safe_id(os.path.basename(os.path.dirname(path)) or "glob")
    die("path not found: {}".format(spec))


def cmd_geometry(args, extra):
    emb, label, inp = resolve_embedding_spec(args.embedding, args.method)
    data = os.path.realpath(args.data)
    if not (os.path.isfile(data) or os.path.isdir(data) or any(ch in args.data for ch in "*?[")):
        die("input data path not found: {}".format(args.data))
    params = OrderedDict([("embedding", emb), ("data", data if not any(ch in args.data for ch in "*?[") else args.data),
                          ("label_key", args.label_key), ("batch_key", args.batch_key)])
    if args.max_cells is not None:
        params["max_cells"] = args.max_cells
    if args.seed is not None:
        params["seed"] = args.seed
    return launch("geometry", safe_id(label), inp, params, args, extra)


def cmd_list(args, registry):
    if args.what == "tasks":
        print("{:<11} {:<12} {}".format("task", "status", "description"))
        for tid, spec in TASKS.items():
            print("{:<11} {:<12} {}".format(tid, "available" if spec["implemented"] else "planned", spec["help"]))
        return 0
    items = list(registry.items())
    if args.task:
        if args.task not in TASKS:
            die("unknown task '{}'".format(args.task))
        items = [(m, s) for m, s in items if args.task in s.get("tasks", [])]
    print("{:<18} {:<20} {:<12} {:<4} {:<34} {}".format("method", "name", "category", "gpu", "tasks", "container"))
    for mid, spec in items:
        tasks = [t for t in TASKS if t in spec.get("tasks", [])]
        print("{:<18} {:<20} {:<12} {:<4} {:<34} {}".format(
            mid, spec.get("name", mid), spec.get("category", ""), "yes" if spec.get("gpu", True) else "no",
            ",".join(tasks), spec.get("container", "")))
    return 0


def cmd_runs(args):
    ws = find_workspace(args.workspace)
    rows = []
    if os.path.isdir(ws.runs_dir):
        for task in sorted(os.listdir(ws.runs_dir)):
            if args.task and task != args.task:
                continue
            tdir = os.path.join(ws.runs_dir, task)
            for name in os.listdir(tdir):
                meta_path = os.path.join(tdir, name, "run.json")
                if not os.path.isfile(meta_path):
                    continue
                try:
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                except (OSError, ValueError):
                    continue
                rows.append((meta.get("started", ""), task, meta.get("method", ""), meta.get("status", "?"),
                             str(meta.get("exit_code", "")), os.path.relpath(os.path.join(tdir, name), os.getcwd())))
    rows.sort(reverse=True)
    if not rows:
        info("no runs recorded in workspace {}".format(ws.root))
        return 0
    print("{:<20} {:<10} {:<14} {:<12} {:<5} {}".format("started", "task", "method", "status", "exit", "run directory"))
    for r in rows[: args.limit]:
        print("{:<20} {:<10} {:<14} {:<12} {:<5} {}".format(*r))
    return 0


def cmd_info(args):
    pipe = pipeline_dir()
    print("scfoundry  {}".format(VERSION))
    print("pipeline   {}".format(pipe))
    nf = find_nextflow(args.nextflow, required=False)
    nf_path = shutil.which(nf) if os.path.sep not in nf else nf
    print("nextflow   {}".format("{} ({})".format(nextflow_version(nf_path), nf_path) if nf_path else "not found"))
    try:
        ws = find_workspace(args.workspace)
        print("workspace  {}".format(ws.root))
        print("config     {}{}".format(ws.config, "" if os.path.isfile(ws.config) else "  (MISSING -- run init --force)"))
    except SystemExit:
        print("workspace  none (run '{} init')".format(PROG))
    return 0


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def add_workspace_option(p):
    p.add_argument("--workspace", metavar="DIR", help="workspace directory (default: $SCFOUNDRY_WORKSPACE or the "
                                                     "nearest parent of the current directory containing " + MARKER + ")")


def add_task_options(p):
    g = p.add_argument_group("run options")
    add_workspace_option(g)
    g.add_argument("--outdir", help="root directory for published results (default: <workspace>/results)")
    g.add_argument("--run-name", help="name of the run directory under <workspace>/runs/<task>/")
    g.add_argument("--resume", nargs="?", const=True, metavar="RUN",
                   help="resume the newest run with the same method/input (or the named run)")
    g.add_argument("--gpu", help="GPU index (or comma list) exposed to the container, e.g. 0")
    g.add_argument("--weights-dir", help="override the model weight directory (params.model_weights_dir)")
    g.add_argument("--cache-dir", help="override the container/cache directory (params.cache_dir)")
    g.add_argument("--config", help="additional Nextflow config file (-c)")
    g.add_argument("--profile", help="Nextflow -profile to activate")
    g.add_argument("--nextflow", help="path to the nextflow executable (or set SCFOUNDRY_NEXTFLOW)")
    g.add_argument("--dry-run", action="store_true", help="print params and the nextflow command; run nothing")
    g.add_argument("--quiet", action="store_true", help="less console output")
    p.epilog = ("Unknown --options are forwarded to Nextflow as pipeline parameters "
                "(e.g. --batch_size 32). Arguments after '--' are passed verbatim to 'nextflow run'.")


def build_parser():
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="scFoundry: deploy, run and evaluate single-cell foundation models (Nextflow-based).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version="{} {}".format(PROG, VERSION))
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    p = sub.add_parser("init", help="create a workspace (nextflow.config, data/, cache/, results/, runs/) and check the environment")
    p.add_argument("directory", nargs="?", help="workspace directory (default: current directory)")
    p.add_argument("--runtime", choices=CONTAINER_RUNTIMES, help="container runtime to enable (default: auto-detect)")
    p.add_argument("--gpu-id", help="pin the GPU index used by containers (params.gpu_id)")
    p.add_argument("--force", action="store_true", help="rewrite nextflow.config of an existing workspace")
    p.add_argument("--nextflow", help="path to the nextflow executable")

    p = sub.add_parser("download", help=TASKS["download"]["help"])
    p.add_argument("--method", required=True, help="method whose checkpoints to download (see 'list methods')")
    p.add_argument("--model", help="model variant, e.g. Novae/novae-mouse-0")
    add_task_options(p)

    p = sub.add_parser("embed", help=TASKS["embed"]["help"],
                       description="Compute cell embeddings: zero-shot with a pretrained scFM, a reference method "
                                   "(pca, scvi), or an integration method that trains on your data using batch labels "
                                   "(scgpt_integrated, scvi_denovo, harmony, seurat_cca, seurat_rpca). Output: "
                                   "<outdir>/embeddings/<method>/<input-id>.h5ad (embedding matrix in .X, original obs kept).")
    p.add_argument("--method", required=True, help="method to run (see 'list methods --task embed')")
    p.add_argument("--data", required=True, metavar="H5AD", help="input AnnData: raw counts in X, gene symbols in var")
    p.add_argument("--model", help="model variant under data/model_weights, e.g. scGPT/scGPT_human (default: the method's default)")
    p.add_argument("--batch-size", type=int, help="inference batch size (default: the method's default)")
    p.add_argument("--batch-key", metavar="OBS_COLUMN",
                   help="obs column holding batch labels, used by the integration methods only "
                        "(default: batch_id; a missing column is treated as a single batch)")
    add_task_options(p)

    p = sub.add_parser("transfer", help=TASKS["transfer"]["help"],
                       description="Label transfer with frozen embeddings: embed the reference and query with a "
                                   "pretrained model, fit a lightweight classifier on the reference labels and "
                                   "predict the query. No model parameters are updated (few-shot when the "
                                   "reference is tiny). Outputs: <outdir>/transfer/models/<method>/<classifier>/"
                                   "<ref-id>/ and <outdir>/transfer/predictions/<method>/<classifier>/<query-id>_predicted_*.tsv")
    p.add_argument("--method", required=True, help="embedding method (see 'list methods --task transfer')")
    p.add_argument("--reference", metavar="H5AD", help="labelled reference cells (fit)")
    p.add_argument("--query", metavar="H5AD", help="cells to label (predict)")
    p.add_argument("--fitted", metavar="DIR", help="previously fitted model directory (predict without --reference)")
    p.add_argument("--classifier", choices=TRANSFER_CLASSIFIERS, default="logreg",
                   help="classifier on the frozen embeddings (default: logreg)")
    p.add_argument("--label-key", default="cell_type", help="obs column with reference labels (default: cell_type)")
    p.add_argument("--knn-k", type=int, help="neighbours for --classifier knn (default: 15)")
    p.add_argument("--model", help="model variant under data/model_weights (default: the method's default)")
    p.add_argument("--batch-size", type=int, help="embedding batch size (default: the method's default)")
    add_task_options(p)

    p = sub.add_parser("finetune", help=TASKS["finetune"]["help"],
                       description="Fine-tune a model on labelled reference cells following its authors' recipe "
                                   "(model parameters are updated), then predict the query. Methods whose official "
                                   "adaptation keeps the backbone frozen are available as 'transfer --classifier mlp'. "
                                   "Outputs: <outdir>/finetune/finetuned_models/<method>/<ref-id>/ and "
                                   "<outdir>/finetune/prediction/<method>/<query-id>_predicted_*.tsv")
    p.add_argument("--method", required=True, help="method to fine-tune (see 'list methods --task finetune')")
    p.add_argument("--reference", metavar="H5AD", help="labelled reference cells (fine-tune)")
    p.add_argument("--query", metavar="H5AD", help="cells to label (predict)")
    p.add_argument("--fitted", metavar="DIR", help="previously fine-tuned model directory (predict without --reference)")
    p.add_argument("--label-key", default="cell_type", help="obs column with reference labels (default: cell_type)")
    p.add_argument("--epochs", type=int, help="training epochs (default: the method's recipe)")
    p.add_argument("--batch-size", type=int, help="training batch size (default: the method's recipe)")
    p.add_argument("--model", help="pretrained model variant under data/model_weights (default: the method's default)")
    add_task_options(p)

    p = sub.add_parser("benchmark", help=TASKS["benchmark"]["help"],
                       description="Score embedding files produced by `embed`: biological conservation on the "
                                   "cell-type labels (NMI, HOM, COM, FMI, ARI on Leiden/KMeans clusters; ASW, cLISI, "
                                   "Acc@kNN, graph connectivity) and, with --batch-key, batch mixing (kBET, BRAS, "
                                   "iLISI, CiLISI). Output: <outdir>/benchmark/ (per-sample tables, cluster labels, "
                                   "<method>_*_metrics_{long,wide}.csv).")
    p.add_argument("--embedding", required=True, metavar="PATH", help="embedding .h5ad, a directory of them, or a glob")
    p.add_argument("--method", help="method label written into the tables (default: the directory name)")
    p.add_argument("--label-key", default="cell_type", help="obs column with cell-type labels (default: cell_type)")
    p.add_argument("--batch-key", metavar="OBS_COLUMN", help="obs column with batch labels; enables the batch metrics")
    p.add_argument("--metrics", choices=("bio", "batch", "all"), help="metric set (default: all if --batch-key is given, else bio)")
    p.add_argument("--clustering", choices=("leiden", "kmeans"), default="leiden", help="clustering for the label-agreement metrics (default: leiden)")
    p.add_argument("--batch-max-cells", type=int, help="stratified subsample size for the batch metrics (default 0 = all cells)")
    add_task_options(p)

    p = sub.add_parser("geometry", help=TASKS["geometry"]["help"],
                       description="Representation-geometry probes for embedding files produced by `embed`, each paired "
                                   "with the raw-count input it came from: participation ratio, spectral and cell-pair "
                                   "anisotropy, within-batch expression-neighbourhood preservation (R_NX), TwoNN "
                                   "intrinsic dimension and cell-type / batch partial eta-squared. Output: "
                                   "<outdir>/geometry/<method>/ and <outdir>/<method>_geometry.csv.")
    p.add_argument("--embedding", required=True, metavar="PATH", help="embedding .h5ad, a directory of them, or a glob")
    p.add_argument("--data", required=True, metavar="PATH", help="the raw-count input .h5ad (or a directory; matched to embeddings by file name)")
    p.add_argument("--method", help="method label written into the tables (default: the embedding directory name)")
    p.add_argument("--label-key", default="cell_type", help="obs column with cell-type labels (default: cell_type)")
    p.add_argument("--batch-key", default="batch_id", help="obs column with batch labels (default: batch_id; a missing column means one batch)")
    p.add_argument("--max-cells", type=int, help="subsample datasets larger than this (default 20000)")
    p.add_argument("--seed", type=int, help="seed for subsampling and pair sampling (default 0)")
    add_task_options(p)

    p = sub.add_parser("list", help="list methods or tasks")
    p.add_argument("what", choices=["methods", "tasks"])
    p.add_argument("--task", help="only methods supporting this task")

    p = sub.add_parser("runs", help="list recorded runs of the workspace")
    add_workspace_option(p)
    p.add_argument("--task", help="only this task")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("info", help="show version, pipeline and workspace locations")
    add_workspace_option(p)
    p.add_argument("--nextflow", help="path to the nextflow executable")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra_nf_args = []
    if "--" in argv:
        idx = argv.index("--")
        argv, extra_nf_args = argv[:idx], argv[idx + 1:]

    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    args.passthrough = OrderedDict()
    if unknown:
        if args.command in ("init", "list", "runs", "info"):
            parser.error("unrecognised arguments: {}".format(" ".join(unknown)))
        args.passthrough = passthrough_to_params(unknown)
        warn("forwarding unrecognised option(s) to Nextflow as parameters: {}".format(
            ", ".join("--{}={}".format(k, v) for k, v in args.passthrough.items())))

    if args.command == "init":
        return cmd_init(args)
    if args.command == "info":
        return cmd_info(args)
    if args.command == "runs":
        return cmd_runs(args)
    registry = load_registry(pipeline_dir())
    if args.command == "list":
        return cmd_list(args, registry)
    if args.command == "download":
        return cmd_download(args, registry, extra_nf_args)
    if args.command == "embed":
        return cmd_embed(args, registry, extra_nf_args)
    if args.command == "transfer":
        return cmd_transfer(args, registry, extra_nf_args)
    if args.command == "finetune":
        return cmd_finetune(args, registry, extra_nf_args)
    if args.command == "benchmark":
        return cmd_benchmark(args, extra_nf_args)
    if args.command == "geometry":
        return cmd_geometry(args, extra_nf_args)
    parser.error("unknown command {}".format(args.command))


if __name__ == "__main__":
    sys.exit(main())
