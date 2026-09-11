"""Exercise production sampling/checkpoint recovery with a small CPU model."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.hybrid.test_c3_sft import _sample, c3_sft as c3

try:
    import torch
except ImportError:
    torch = None


class Interrupted(Exception):
    pass


@unittest.skipIf(torch is None, "torch unavailable")
class C3RecoveryTests(unittest.TestCase):
    def run_training(self, interrupted_after: int | None = None):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = torch.nn.Module()
                self.language_model.lora_A = torch.nn.Parameter(torch.randn(2, 2) * .05)
                self.adaptors = torch.nn.Module()
                self.adaptors.driving = torch.nn.Module()
                self.adaptors.driving.route_head = torch.nn.Linear(2, 40)
                self.adaptors.driving.speed_wps_head = torch.nn.Linear(2, 20)
                self.vision_model = torch.nn.Linear(2, 2)
                self.wp_encoder = torch.nn.Linear(2, 2)

        def example(runtime, model, sample, device):
            return SimpleNamespace(driving_label=SimpleNamespace(
                path=torch.tensor([sample.route_target], dtype=torch.float32),
                waypoints=torch.tensor([sample.speed_target], dtype=torch.float32),
                eval_infos={"route_mask": torch.tensor([sample.route_mask]), "speed_mask": torch.tensor([sample.speed_mask])},
            ))

        def forward(model, item, train):
            hidden = torch.tensor([[1., .5]]) @ model.language_model.lora_A
            return {"route": model.adaptors.driving.route_head(hidden).reshape(1, 20, 2),
                    "speed_wps": model.adaptors.driving.speed_wps_head(hidden).reshape(1, 10, 2)}, hidden, None

        samples = tuple([_sample(f"train-{i}", "train") for i in range(158)] + [_sample(f"val-{i}") for i in range(53)])
        manifest = {"manifest_sha256": "fixture-manifest"}
        with tempfile.TemporaryDirectory(prefix="c3-recovery-test-") as temporary:
            args = c3._parser().parse_args(["train", "--run-id", "c3-repair-fixture", "--output-root", temporary, "--device", "cpu"])
            run = c3._ensure_run_dir(args)
            config = c3._config_payload(args)
            for name, payload in (("audit.json", {"status": "AUDIT_PASSED"}), ("smoke.json", {"status": "SMOKE_PASSED"}), ("m0_predictions.json", {})):
                c3._write_json(run / name, payload)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch("torch.cuda.is_available", return_value=False))
                stack.enter_context(patch.object(c3, "_load_manifest", return_value=(samples, manifest)))
                stack.enter_context(patch.object(c3, "_validate_run_config", return_value=config))
                stack.enter_context(patch.object(c3, "_smoke_contract_errors", return_value=[]))
                stack.enter_context(patch.object(c3, "_make_runtime", side_effect=lambda args: (None, Tiny(), "cpu", SimpleNamespace(fixture=True))))
                stack.enter_context(patch.object(c3, "_make_example", side_effect=example))
                stack.enter_context(patch.object(c3, "_forward_driving", side_effect=forward))
                stack.enter_context(patch.object(c3, "_update_resource_ledger", return_value={}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if interrupted_after is not None:
                    original = c3._set_schedule

                    def interrupt(optimizer, update, **kwargs):
                        if update == interrupted_after + 1:
                            raise Interrupted()
                        return original(optimizer, update, **kwargs)

                    with patch.object(c3, "_set_schedule", side_effect=interrupt):
                        with self.assertRaises(Interrupted):
                            c3.cmd_train(args)
                    checkpoint = c3._checkpoint_metadata(run / "checkpoint_latest.pt")
                    old_log = (run / "training.jsonl").read_bytes()
                    # Also cover a partially written next log record.
                    with (run / "training.jsonl").open("ab") as handle:
                        handle.write(b'{"update":')
                    args.resume = True
                    c3.cmd_train(args)
                    events = list((run / "recovery").glob("*.json"))
                    self.assertEqual(len(events), 1)
                    event = c3._read_json(events[0])
                    self.assertGreater(event["uncommitted_wall_upper_bound_s"], 0.0)
                    archived = run / "recovery" / f"{event['source_log_sha256']}.jsonl"
                    self.assertEqual(archived.read_bytes(), old_log + b'{"update":')
                else:
                    c3.cmd_train(args)
                summary = c3._read_json(run / "m1_training_summary.json")
                self.assertEqual(summary["actual_updates"], 200)
                self.assertEqual(summary["samples_seen"], 790)
                self.assertEqual(set(summary["root_exposures"].values()), {5})
                self.assertEqual(summary["window_size_histogram"], {"2": 5, "4": 195})
                self.assertTrue(summary["frozen_parameters_unchanged"])
                return summary["final_trainable_fingerprints"]["model_digest"], [r["root_ids"] for r in summary["logs"]]

    def test_recovery_across_warmup_unfreeze_and_epoch_tail(self):
        continuous = self.run_training()
        for step in (1, 9, 10, 11, 39, 40, 41):
            with self.subTest(crash_after_update=step):
                self.assertEqual(self.run_training(step), continuous)

    def test_committed_log_corruption_is_rejected_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            path = run / "training.jsonl"
            path.write_text('{"update": 999}\n', encoding="utf-8")
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "committed_log_digest"):
                c3._reconcile_training_log(run, {"log_count": 1, "training_log_sha256": "wrong"})
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
