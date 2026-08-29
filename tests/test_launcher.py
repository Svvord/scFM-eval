"""Unit tests for the `scfoundry` launcher (no Nextflow needed).

Run from the repository root:  python3 -m unittest discover -s tests -v
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scfoundry import cli  # noqa: E402


class Sandbox:
    """A throw-away pipeline directory (only the files the launcher reads) plus a
    scratch area for workspaces. Commands run `python -m scfoundry` as a subprocess."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="scfoundry-test-")
        self.pipeline = os.path.join(self.dir, "pipeline")
        os.makedirs(os.path.join(self.pipeline, "conf"))
        shutil.copy(os.path.join(REPO, "nextflow.example.config"), self.pipeline)
        shutil.copy(os.path.join(REPO, "conf", "methods.json"), os.path.join(self.pipeline, "conf"))
        shutil.copy(os.path.join(REPO, "main.nf"), self.pipeline)
        self.fake_nextflow = os.path.join(self.dir, "fake-nextflow")
        with open(self.fake_nextflow, "w") as fh:
            fh.write("#!/usr/bin/env bash\n"
                     "{ printf '%s\\n' \"$PWD\" \"$@\"; printf 'ENV:%s\\n' \"$SCFOUNDRY_WORKSPACE\" \"$SCFOUNDRY_PIPELINE\" \"$NXF_SYNTAX_PARSER\"; } > nf-call.txt\n"
                     "exit ${FAKE_NF_EXIT:-0}\n")
        os.chmod(self.fake_nextflow, 0o755)

    def run(self, *args, cwd=None, env=None):
        e = dict(os.environ, SCFOUNDRY_PIPELINE=self.pipeline, SCFOUNDRY_NEXTFLOW=self.fake_nextflow)
        e.pop("SCFOUNDRY_WORKSPACE", None)
        e["PYTHONPATH"] = REPO + os.pathsep + e.get("PYTHONPATH", "")
        if env:
            e.update(env)
        return subprocess.run([sys.executable, "-m", "scfoundry"] + list(args), cwd=cwd or self.dir, env=e,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    def cleanup(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class PureFunctionTests(unittest.TestCase):
    def test_coerce_matches_nextflow_cli_semantics(self):
        c = cli.coerce
        self.assertEqual(c("64"), 64)
        self.assertEqual(c("0.2"), 0.2)
        self.assertIs(c("true"), True)
        self.assertIs(c("False"), False)
        self.assertEqual(c("scGPT/scGPT_human"), "scGPT/scGPT_human")
        self.assertEqual(c("1e5"), 100000.0)

    def test_passthrough_pairs_flags_equals_and_negative_numbers(self):
        p = cli.passthrough_to_params(["--batch_size", "32", "--flag", "--pool=max", "--x", "-1"])
        self.assertEqual(dict(p), {"batch_size": 32, "flag": True, "pool": "max", "x": -1})
        with self.assertRaises(SystemExit):
            cli.passthrough_to_params(["stray"])

    def test_ids_and_signatures(self):
        self.assertEqual(cli.input_id("/a/b/colon 1000.h5ad"), "colon-1000")
        self.assertEqual(cli.run_signature("scgpt", "colon_1000"), "scgpt_colon_1000")
        self.assertEqual(cli.run_signature("scgpt", ""), "scgpt")

    def test_set_runtime_enables_exactly_one_block(self):
        with open(os.path.join(REPO, "nextflow.example.config")) as fh:
            tpl = fh.read()
        for rt in ("docker", "singularity", "apptainer"):
            text = cli.set_runtime(tpl, rt)
            enabled, block = {}, None
            for line in text.splitlines():
                if line.rstrip().endswith("{") and not line.startswith(" "):
                    block = line.split("{")[0].strip()
                elif line.startswith("}"):
                    block = None
                elif block in ("docker", "singularity", "apptainer") and "enabled" in line:
                    enabled[block] = "true" in line
            self.assertEqual(enabled, {"docker": rt == "docker", "singularity": rt == "singularity",
                                       "apptainer": rt == "apptainer"}, rt)
            self.assertIn("conda {", text)

    def test_set_gpu_id(self):
        self.assertIn('gpu_id = "1"', cli.set_gpu_id("params {\n    gpu_id = null\n}\n", "1"))


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()

    def tearDown(self):
        self.sb.cleanup()

    def test_version_help_list(self):
        self.assertIn("scfoundry", self.sb.run("--version").stdout)
        r = self.sb.run("download", "--help")
        self.assertEqual(r.returncode, 0)
        self.assertIn("--workspace", r.stdout)
        r = self.sb.run("list", "methods", "--task", "download")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("scgpt", r.stdout)
        self.assertIn("novae", r.stdout)
        r = self.sb.run("list", "tasks")
        self.assertIn("download", r.stdout)

    def test_init_creates_workspace_layout(self):
        ws = os.path.join(self.sb.dir, "ws")
        r = self.sb.run("init", ws, "--runtime", "docker", "--gpu-id", "0")
        self.assertEqual(r.returncode, 0, r.stderr)
        for rel in ("nextflow.config", ".scfoundry.json", "data/model_weights", "cache/.home", "results", "runs"):
            self.assertTrue(os.path.exists(os.path.join(ws, rel)), rel)
        with open(os.path.join(ws, "nextflow.config")) as fh:
            text = fh.read()
        self.assertIn('gpu_id             = "0"', text)
        self.assertRegex(text, r"docker \{\n\s+enabled = true")
        self.assertRegex(text, r"apptainer \{\n\s+enabled = false")
        with open(os.path.join(ws, ".scfoundry.json")) as fh:
            self.assertEqual(json.load(fh)["container_runtime"], "docker")
        # refuses to re-init without --force
        r = self.sb.run("init", ws, "--runtime", "docker")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--force", r.stderr)
        self.assertEqual(self.sb.run("init", ws, "--runtime", "apptainer", "--force").returncode, 0)

    def test_init_defaults_to_cwd(self):
        ws = os.path.join(self.sb.dir, "here")
        os.makedirs(ws)
        r = self.sb.run("init", "--runtime", "apptainer", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(os.path.isfile(os.path.join(ws, ".scfoundry.json")))

    def test_task_refuses_outside_workspace(self):
        r = self.sb.run("download", "--method", "scgpt")
        self.assertEqual(r.returncode, 2)
        self.assertIn("init", r.stderr)

    def test_workspace_discovery_walks_up_and_env_and_flag(self):
        ws = os.path.join(self.sb.dir, "ws")
        self.sb.run("init", ws, "--runtime", "apptainer")
        sub = os.path.join(ws, "some", "deep", "dir")
        os.makedirs(sub)
        r = self.sb.run("download", "--method", "scgpt", "--dry-run", cwd=sub)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("workspace " + os.path.realpath(ws), r.stdout)
        r = self.sb.run("download", "--method", "scgpt", "--dry-run", env={"SCFOUNDRY_WORKSPACE": ws})
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self.sb.run("download", "--method", "scgpt", "--dry-run", "--workspace", ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self.sb.run("download", "--method", "scgpt", "--dry-run", "--workspace", self.sb.dir)
        self.assertEqual(r.returncode, 2)

    def test_unknown_method_for_task(self):
        ws = os.path.join(self.sb.dir, "ws")
        self.sb.run("init", ws, "--runtime", "apptainer")
        r = self.sb.run("download", "--method", "nosuchmodel", "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 2)
        self.assertIn("does not support task", r.stderr)

    def test_dry_run_builds_params_and_command(self):
        ws = os.path.join(self.sb.dir, "ws")
        self.sb.run("init", ws, "--runtime", "apptainer")
        r = self.sb.run("download", "--method", "SCGPT", "--dry-run", "--gpu", "1", "--weights-dir", "w",
                        "--cache-dir", "c", "--model", "scGPT/scGPT_human", "--foo_bar", "7", "--", "-with-trace",
                        cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        params = json.loads(r.stdout.split("params.json:")[1].split("command (")[0])
        self.assertEqual(params["task"], "download")
        self.assertEqual(params["method"], "scgpt")
        self.assertEqual(params["model"], "scGPT/scGPT_human")
        self.assertEqual(params["gpu_id"], "1")
        self.assertTrue(params["model_weights_dir"].endswith("/w"))
        self.assertTrue(params["cache_dir"].endswith("/c"))
        self.assertEqual(params["foo_bar"], 7)
        self.assertIn("forwarding unrecognised option", r.stderr)
        cmd = r.stdout.split("command (")[1]
        self.assertIn("-params-file params.json", cmd)
        self.assertIn("-work-dir work", cmd)
        self.assertIn("-c " + os.path.join(os.path.realpath(ws), "nextflow.config"), cmd)  # workspace != pipeline
        self.assertIn("-with-trace", cmd)
        self.assertIn(os.path.join(self.sb.pipeline, "main.nf"), cmd)
        self.assertFalse(os.listdir(os.path.join(ws, "runs")))  # dry run leaves nothing behind

    def test_embed_batch_key_only_for_integration_methods(self):
        ws = os.path.join(self.sb.dir, "ws")
        self.sb.run("init", ws, "--runtime", "apptainer")
        data = os.path.join(ws, "x.h5ad")
        open(data, "w").close()
        r = self.sb.run("embed", "--method", "harmony", "--data", data, "--batch-key", "donor", "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        params = json.loads(r.stdout.split("params.json:")[1].split("command (")[0])
        self.assertEqual(params["batch_key"], "donor")
        self.assertEqual(params["emb_results_dir"], os.path.join(os.path.realpath(ws), "results"))
        self.assertNotIn("--batch-key is only used", r.stderr)
        r = self.sb.run("embed", "--method", "scgpt", "--data", data, "--batch-key", "donor", "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--batch-key is only used", r.stderr)
        r = self.sb.run("embed", "--method", "scgpt", "--data", os.path.join(ws, "missing.h5ad"), "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 2)

    def test_transfer_modes(self):
        ws = os.path.join(self.sb.dir, "ws")
        self.sb.run("init", ws, "--runtime", "apptainer")
        ref, q = os.path.join(ws, "ref.h5ad"), os.path.join(ws, "q.h5ad")
        for f in (ref, q):
            open(f, "w").close()
        r = self.sb.run("transfer", "--method", "scgpt", "--reference", ref, "--query", q, "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        params = json.loads(r.stdout.split("params.json:")[1].split("command (")[0])
        self.assertEqual(params["classifier"], "logreg")
        self.assertEqual((params["reference"], params["query"]), (os.path.realpath(ref), os.path.realpath(q)))
        self.assertTrue(params["transfer_results_dir"].endswith("/results"))
        self.assertIn("_scgpt_q", r.stdout)                       # run named after the query
        r = self.sb.run("transfer", "--method", "scgpt", "--reference", ref, "--classifier", "knn", "--knn-k", "5", "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('"knn_k": 5', r.stdout)
        self.assertEqual(self.sb.run("transfer", "--method", "scgpt", "--query", q, "--dry-run", cwd=ws).returncode, 2)
        model = os.path.join(ws, "model"); os.makedirs(model); open(os.path.join(model, "meta.json"), "w").write("{}")
        r = self.sb.run("transfer", "--method", "scgpt", "--query", q, "--fitted", model, "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.sb.run("transfer", "--method", "scgpt", "--reference", ref, "--fitted", model, "--dry-run", cwd=ws).returncode, 2)
        self.assertEqual(self.sb.run("transfer", "--method", "pca", "--reference", ref, "--dry-run", cwd=ws).returncode, 2)

    def test_finetune_modes(self):
        ws = os.path.join(self.sb.dir, "ws")
        self.sb.run("init", ws, "--runtime", "apptainer")
        ref, q = os.path.join(ws, "train.h5ad"), os.path.join(ws, "test.h5ad")
        for f in (ref, q):
            open(f, "w").close()
        r = self.sb.run("finetune", "--method", "scgpt", "--reference", ref, "--query", q, "--epochs", "3", "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        params = json.loads(r.stdout.split("params.json:")[1].split("command (")[0])
        self.assertEqual(params["finetune_label_key"], "cell_type")
        self.assertEqual(params["finetune_epoch"], 3)
        self.assertTrue(params["finetune_results_dir"].endswith("/results"))
        self.assertEqual(self.sb.run("finetune", "--method", "scgpt", "--query", q, "--dry-run", cwd=ws).returncode, 2)
        model = os.path.join(ws, "model"); os.makedirs(model)
        self.assertEqual(self.sb.run("finetune", "--method", "scgpt", "--query", q, "--fitted", model, "--dry-run", cwd=ws).returncode, 0)
        # frozen-backbone methods are not fine-tuning methods
        r = self.sb.run("finetune", "--method", "uce", "--reference", ref, "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 2)
        self.assertIn("does not support task", r.stderr)

    def test_benchmark_inputs(self):
        ws = os.path.join(self.sb.dir, "ws")
        self.sb.run("init", ws, "--runtime", "apptainer")
        emb_dir = os.path.join(ws, "results", "embeddings", "scgpt"); os.makedirs(emb_dir)
        emb = os.path.join(emb_dir, "colon.h5ad"); open(emb, "w").close()
        r = self.sb.run("benchmark", "--embedding", emb, "--dry-run", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        params = json.loads(r.stdout.split("params.json:")[1].split("command (")[0])
        self.assertEqual((params["method"], params["metrics"], params["clustering"]), ("scgpt", "bio", "leiden"))
        self.assertTrue(params["results_dir"].endswith("/results"))
        r = self.sb.run("benchmark", "--embedding", emb_dir, "--batch-key", "donor", "--clustering", "kmeans", "--dry-run", cwd=ws)
        params = json.loads(r.stdout.split("params.json:")[1].split("command (")[0])
        self.assertEqual((params["method"], params["metrics"], params["batch_key"], params["clustering"]), ("scgpt", "all", "donor", "kmeans"))
        self.assertEqual(self.sb.run("benchmark", "--embedding", os.path.join(ws, "nope.h5ad"), "--dry-run", cwd=ws).returncode, 2)

    def test_dev_checkout_mode_omits_duplicate_config(self):
        # workspace == pipeline dir: Nextflow auto-loads <pipeline>/nextflow.config, so no -c
        r = self.sb.run("init", self.sb.pipeline, "--runtime", "apptainer")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self.sb.run("download", "--method", "scgpt", "--dry-run", cwd=self.sb.pipeline)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn(" -c ", r.stdout.split("command (")[1])

    def test_real_run_records_metadata_and_resume(self):
        ws = os.path.join(self.sb.dir, "ws")
        self.sb.run("init", ws, "--runtime", "apptainer")
        r = self.sb.run("download", "--method", "scgpt", "--run-name", "myrun", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        run_dir = os.path.join(ws, "runs", "download", "myrun")
        with open(os.path.join(run_dir, "nf-call.txt")) as fh:
            lines = fh.read().splitlines()
        self.assertEqual(os.path.realpath(lines[0]), os.path.realpath(run_dir))     # launched inside the run dir
        self.assertNotIn("-resume", lines)
        self.assertIn("ENV:" + os.path.realpath(ws), lines)                          # workspace exported to Nextflow
        self.assertIn("ENV:" + self.sb.pipeline, lines)
        self.assertIn("ENV:v1", lines)                                               # legacy parser for Nextflow >= 26
        with open(os.path.join(run_dir, "params.json")) as fh:
            self.assertEqual(json.load(fh)["method"], "scgpt")
        with open(os.path.join(run_dir, "run.json")) as fh:
            meta = json.load(fh)
        self.assertEqual((meta["status"], meta["exit_code"]), ("ok", 0))
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "command.sh")))
        # same name refused; --resume (by metadata) reuses it and adds -resume
        self.assertEqual(self.sb.run("download", "--method", "scgpt", "--run-name", "myrun", cwd=ws).returncode, 2)
        r = self.sb.run("download", "--method", "scgpt", "--resume", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(run_dir, "nf-call.txt")) as fh:
            self.assertIn("-resume", fh.read().split())
        self.assertIn("myrun", self.sb.run("runs", cwd=ws).stdout)
        # a failing nextflow is reported and recorded
        r = self.sb.run("download", "--method", "uce", cwd=ws, env={"FAKE_NF_EXIT": "3"})
        self.assertEqual(r.returncode, 3)
        self.assertIn("failed", self.sb.run("runs", "--task", "download", cwd=ws).stdout)

    def test_info(self):
        ws = os.path.join(self.sb.dir, "ws")
        self.sb.run("init", ws, "--runtime", "apptainer")
        r = self.sb.run("info", cwd=ws)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("pipeline   " + self.sb.pipeline, r.stdout)
        self.assertIn("workspace  " + os.path.realpath(ws), r.stdout)


if __name__ == "__main__":
    unittest.main()
