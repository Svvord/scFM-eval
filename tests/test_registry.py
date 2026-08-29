"""conf/methods.json must agree with the `runners` maps in the task workflows.

Run:  python3 -m unittest discover -s tests
"""
import json
import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_FILES = {
    "download": ["workflows/tasks/download.nf"],
    "embed": ["workflows/tasks/embed.nf", "embed_by_scfm.nf"],
    "transfer": ["workflows/tasks/transfer.nf"],
    "finetune": ["workflows/tasks/finetune.nf"],
}


def runner_keys(path):
    """Return the top-level method keys of the first `runners = [ ... ]` map in a .nf file.

    Nested maps (e.g. the per-mode {'fit': ..., 'infer': ...} maps of the legacy
    few-shot/fine-tune entry points) are skipped by tracking bracket depth.
    """
    with open(path) as fh:
        text = fh.read()
    m = re.search(r"runners\s*=\s*\[", text)
    if not m:
        return None
    keys, depth, i = set(), 0, m.end() - 1
    key_re = re.compile(r"'([a-z0-9_]+)'\s*:")
    while i < len(text):
        ch = text[i]
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
            if depth == 0:
                break
        elif ch == "'" and depth == 1:
            km = key_re.match(text, i)
            if km:
                keys.add(km.group(1))
                i = km.end()
                continue
        i += 1
    return keys


class RegistryConsistencyTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(REPO, "conf", "methods.json")) as fh:
            self.registry = json.load(fh)

    def test_every_task_map_matches_registry(self):
        checked = 0
        for task, candidates in TASK_FILES.items():
            keys = None
            for rel in candidates:
                path = os.path.join(REPO, rel)
                if os.path.isfile(path):
                    keys = runner_keys(path)
                    if keys:
                        break
            if not keys:
                continue  # task not (yet) present in this checkout
            expected = {m for m, spec in self.registry.items() if task in spec.get("tasks", [])}
            self.assertEqual(keys, expected, "task '{}' ({}): runners map vs conf/methods.json".format(task, rel))
            checked += 1
        self.assertGreaterEqual(checked, 1)

    def test_registry_entries_are_well_formed(self):
        for mid, spec in self.registry.items():
            self.assertRegex(mid, r"^[a-z0-9_]+$")
            for key in ("name", "category", "container", "tasks"):
                self.assertIn(key, spec, mid)
            self.assertIn(spec["category"], ("zero-shot", "reference", "integration"), mid)
            self.assertTrue(spec["tasks"], mid)


if __name__ == "__main__":
    unittest.main()
