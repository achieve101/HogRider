import json
import random
import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from phase3g_longtrain_final_evaluation import (
    consume_final_evaluation_once,
    strengthened_final_acceptance,
)
from phase3g_model import GenerativeInnovationFIRController
from run_phase3g_longtrain import (
    exploration_acceptance,
    select_locked_candidate,
    should_extend_staircase,
    validate_protocol,
    verify_protected_files,
)
from train_phase3g import (
    _validate_resume_manifest,
    capture_rng_state,
    require_resume_compatibility,
    restore_rng_state,
    save_checkpoint,
    optimizer_to,
)


ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=json.loads(
    (ROOT/"artifacts/phase3g_longtrain_protocol.json").read_text(encoding="utf-8")
)
BASELINE=json.loads(
    (ROOT/"artifacts/phase3g_epoch15_baseline.json").read_text(encoding="utf-8")
)


class Phase3GLongtrainTests(unittest.TestCase):
    def test_formal_baseline_hashes_are_immutable(self):
        results=verify_protected_files(BASELINE, root=ROOT)
        self.assertTrue(all(value["passed"] for value in results))

    def test_protocol_keeps_final_paths_sealed_and_omits_lopo(self):
        validate_protocol(PROTOCOL)
        sealed=PROTOCOL["sealed_path_policy"]
        self.assertEqual(sealed["development_paths_one_based"], list(range(1,9)))
        self.assertEqual(sealed["final_paths_one_based"], [9,10])
        self.assertFalse(sealed["lopo_rerun"])

    def test_staircase_extends_only_when_best_is_in_last_three_epochs(self):
        base={"stop_reason":"epoch_budget_exhausted", "last_completed_epoch":30}
        self.assertTrue(should_extend_staircase({**base,"best_epoch":28}, 30))
        self.assertTrue(should_extend_staircase({**base,"best_epoch":30}, 30))
        self.assertFalse(should_extend_staircase({**base,"best_epoch":27}, 30))
        self.assertFalse(should_extend_staircase({**base,"best_epoch":30,"stop_reason":"patience_exhausted"}, 30))
        self.assertFalse(should_extend_staircase({**base,"best_epoch":30,"last_completed_epoch":29}, 30))

    def test_exploration_balanced_gate(self):
        summary={
            "formal_candidate_eligible":True, "final_paths_touched":False,
            "acceptance":{"passed":True},
            "development_metrics":{
                "phase3_selection_score":20.30,
                "rebound_score_db":0.50,
                "worst_path_primary_db":4.00,
            },
        }
        self.assertTrue(exploration_acceptance(summary, PROTOCOL)["passed"])
        summary["development_metrics"]["worst_path_primary_db"]=3.98
        self.assertFalse(exploration_acceptance(summary, PROTOCOL)["passed"])

    def test_candidate_lock_requires_all_three_safe_and_selects_highest_d(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); values=[]
            for seed,score in ((2026,20.1),(2027,20.4),(2028,20.2)):
                checkpoint=root/f"seed_{seed}.pt"; checkpoint.write_bytes(str(seed).encode())
                summary={
                    "acceptance":{"passed":True}, "formal_candidate_eligible":True,
                    "final_paths_touched":False,
                    "development_metrics":{"phase3_selection_score":score},
                    "selected_checkpoint":str(checkpoint.resolve()),
                }
                values.append({
                    "seed":seed, "frozen_budget":30,
                    "summary_path":str(root/f"seed_{seed}.json"), "summary":summary,
                })
            lock=select_locked_candidate(values, "protocol-hash")
            self.assertEqual(lock["selected_seed"], 2027)
            self.assertFalse(lock["final_evaluation_consumed"])
            self.assertFalse(lock["lopo_rerun"])
            values[0]["summary"]["acceptance"]["passed"]=False
            with self.assertRaises(RuntimeError):
                select_locked_candidate(values, "protocol-hash")

    def test_exact_rng_resume_matches_uninterrupted_streams(self):
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        loader=torch.Generator().manual_seed(9)

        def draw():
            return (
                random.random(), float(np.random.random()), float(torch.rand(())),
                torch.randperm(11, generator=loader).tolist(),
            )

        draw()
        state=capture_rng_state(loader)
        expected=[draw(),draw()]
        random.seed(77); np.random.seed(77); torch.manual_seed(77); loader.manual_seed(77)
        restore_rng_state(state, loader)
        actual=[draw(),draw()]
        self.assertEqual(actual, expected)

    def test_model_and_adam_state_resume_matches_uninterrupted_updates(self):
        def make():
            model=torch.nn.Linear(3,2)
            optimizer=optim.Adam(model.parameters(), lr=1e-2, amsgrad=True)
            return model,optimizer

        torch.manual_seed(11)
        uninterrupted,uninterrupted_optimizer=make()
        interrupted=copy.deepcopy(uninterrupted)
        interrupted_optimizer=optim.Adam(interrupted.parameters(), lr=1e-2, amsgrad=True)
        inputs=[torch.randn(4,3) for _ in range(5)]
        targets=[torch.randn(4,2) for _ in range(5)]

        def update(model, optimizer, index):
            optimizer.zero_grad(set_to_none=True)
            loss=(model(inputs[index])-targets[index]).square().mean()
            loss.backward(); optimizer.step()

        for index in range(5):
            update(uninterrupted, uninterrupted_optimizer, index)
        for index in range(2):
            update(interrupted, interrupted_optimizer, index)
        payload={
            "model":copy.deepcopy(interrupted.state_dict()),
            "optimizer":copy.deepcopy(interrupted_optimizer.state_dict()),
        }
        resumed,resumed_optimizer=make()
        resumed.load_state_dict(payload["model"])
        resumed_optimizer.load_state_dict(payload["optimizer"])
        optimizer_to(resumed_optimizer, torch.device("cpu"))
        for index in range(2,5):
            update(resumed, resumed_optimizer, index)
        for first,second in zip(uninterrupted.parameters(), resumed.parameters()):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        first_state=uninterrupted_optimizer.state_dict()
        second_state=resumed_optimizer.state_dict()
        self.assertEqual(first_state["param_groups"], second_state["param_groups"])
        self.assertEqual(set(first_state["state"]), set(second_state["state"]))
        for parameter in first_state["state"]:
            self.assertEqual(
                set(first_state["state"][parameter]),
                set(second_state["state"][parameter]),
            )
            for key,value in first_state["state"][parameter].items():
                other=second_state["state"][parameter][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, other, rtol=0, atol=0)
                else:
                    self.assertEqual(value, other)

    def test_old_checkpoint_requires_explicit_legacy_permission(self):
        old={"epoch":15, "model_state_dict":{}, "optimizer_state_dict":{}}
        with self.assertRaises(ValueError):
            require_resume_compatibility(old, False)
        self.assertTrue(require_resume_compatibility(old, True))
        current={
            "checkpoint_format_version":2, "rng_state":{}, "training_state":{},
        }
        self.assertFalse(require_resume_compatibility(current, False))

    def test_exact_resume_rejects_changed_data_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); checkpoints=root/"checkpoints"; checkpoints.mkdir()
            checkpoint=checkpoints/"latest.pt"; checkpoint.write_bytes(b"placeholder")
            original={
                "path_indices_zero_based":list(range(8)),
                "input_sha256":{"dataset/sh.npy":"abc"},
                "sealed_paths_touched":False,
            }
            (root/"synthesis_manifest.json").write_text(
                json.dumps(original), encoding="utf-8",
            )
            _validate_resume_manifest(checkpoint, "synthesis_manifest.json", original)
            changed={**original, "input_sha256":{"dataset/sh.npy":"changed"}}
            with self.assertRaises(ValueError):
                _validate_resume_manifest(checkpoint, "synthesis_manifest.json", changed)

    def test_v2_checkpoint_persists_optimizer_rng_and_selection_state(self):
        model=GenerativeInnovationFIRController(
            num_experts=2, fir_length=8, path_length=3, n_fft=8,
            block_size=2, hidden_size=4, latent_size=2,
        )
        optimizer=optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
        generator=torch.Generator().manual_seed(123)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"checkpoint.pt"
            save_checkpoint(
                path, model, optimizer, 7, {"stage":"generalize"},
                {"development":{"phase3_selection_score":9.0}},
                loader_generator=generator, best_score=9.0,
                best_epoch=6, stale=1,
            )
            value=torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(value["checkpoint_format_version"], 2)
        self.assertIn("optimizer_state_dict", value)
        self.assertIn("rng_state", value)
        self.assertEqual(value["training_state"]["completed_epoch"], 7)
        self.assertEqual(value["training_state"]["best_epoch"], 6)
        self.assertEqual(value["training_state"]["stale"], 1)

    def test_final_receipt_is_exclusive_and_cannot_be_reused(self):
        lock={
            "selected_seed":2027, "selected_checkpoint":"candidate.pt",
            "selected_checkpoint_sha256":"abc",
        }
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); lock_path=root/"candidate_lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            receipt=root/"final_evaluation_receipt.json"
            consume_final_evaluation_once(
                receipt, lock_path=lock_path, lock=lock,
            )
            self.assertTrue(receipt.is_file())
            with self.assertRaises(FileExistsError):
                consume_final_evaluation_once(
                    receipt, lock_path=lock_path, lock=lock,
                )

    def test_strengthened_final_gate_compares_against_current_formal_model(self):
        original={"passed":True}
        development={
            "primary_score_db":18.0, "rebound_score_db":1.6,
            "worst_path_primary_db":4.0,
        }
        final={
            "primary_score_db":18.0, "rebound_score_db":1.6,
            "worst_path_primary_db":4.0,
        }
        result=strengthened_final_acceptance(original, development, final, PROTOCOL)
        self.assertTrue(result["passed"])
        final["worst_path_primary_db"]=3.0
        self.assertFalse(strengthened_final_acceptance(original, development, final, PROTOCOL)["passed"])


if __name__ == "__main__":
    unittest.main()
