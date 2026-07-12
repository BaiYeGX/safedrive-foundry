import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".collab" / "config.json"
TEMPLATE = ROOT / ".collab" / "task-template.json"
SCRIPT_DIR = ROOT / "scripts" / "collab"


class CollaborationControlTests(unittest.TestCase):
    def test_config_and_template_are_valid_json(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], 1)
        self.assertEqual(template["schema_version"], 1)
        self.assertRegex(template["task_id"], r"^G[1-8]-\d{2}$")
        self.assertGreaterEqual(template["max_turns"], 1)
        self.assertTrue(config["grok"]["disable_subagents"])
        self.assertTrue(config["grok"]["disable_memory"])
        self.assertIn("tasks/G0/", config["protected_paths"])

    def test_expected_scripts_exist_and_use_strict_mode(self):
        expected = {
            "Common.ps1",
            "Test-CollabEnvironment.ps1",
            "New-GxWorktree.ps1",
            "Invoke-GrokTask.ps1",
            "Invoke-IndependentVerification.ps1",
        }
        self.assertEqual({path.name for path in SCRIPT_DIR.glob("*.ps1")}, expected)
        common = (SCRIPT_DIR / "Common.ps1").read_text(encoding="utf-8")
        self.assertIn("Set-StrictMode -Version Latest", common)

    def test_grok_runner_has_required_noninteractive_guards(self):
        text = (SCRIPT_DIR / "Invoke-GrokTask.ps1").read_text(encoding="utf-8")
        for token in ("--prompt-file", "--output-format", "--no-subagents", "--no-memory", "--max-turns"):
            self.assertIn(token, text)
        self.assertRegex(text, re.compile(r"main.*master", re.DOTALL))

    def test_verifier_checks_scope_diff_and_both_environments(self):
        text = (SCRIPT_DIR / "Invoke-IndependentVerification.ps1").read_text(encoding="utf-8")
        for token in ("allowed_paths", "protected_paths", "diff --check", "windows", "wsl"):
            self.assertIn(token, text)

    def test_runtime_artifacts_are_gitignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".collab/runs/", ignore)
        self.assertIn(".collab/local.json", ignore)


if __name__ == "__main__":
    unittest.main()
