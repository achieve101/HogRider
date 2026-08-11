"""Run the bounded Phase-3G E0/E1/E2/E3 decision tree."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


E0_DIAGNOSTICS={
    "analytic_interpolation_median_gain_db":{
        "time_domain_affine":-11.5717, "frequency_domain_affine":-13.2123, "kernel_ridge":-5.5972,
    },
    "oracle_subspace_projection":{
        "rank_7_median_gain_db":1.0468, "rank_7_non_degrading_folds":5,
        "conclusion":"pure expert-span interpolation is rejected",
    },
}


def _run_training(output: Path, args: argparse.Namespace, stage: str, checkpoint: str | None, seed: int,
                  hidden: int=32, latent: int=16) -> dict:
    command=[
        sys.executable, "-u", "train_phase3g.py", "--stage", stage,
        "--dataset-dir", args.dataset_dir, "--output-dir", str(output), "--device", args.device,
        "--seed", str(seed), "--hidden-size", str(hidden), "--latent-size", str(latent),
        "--stress-cases", str(args.stress_cases), "--batch-size", str(args.batch_size),
        "--gradient-accumulation", str(args.gradient_accumulation),
    ]
    if checkpoint: command.extend(["--checkpoint", checkpoint])
    if args.max_train_batches: command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.smoke:
        command.extend(["--epochs", "1", "--samples-per-epoch", "1", "--calibration-batches", "1"])
    subprocess.run(command, check=True)
    return json.loads((output/"summary.json").read_text(encoding="utf-8"))


def _run_lopo(output: Path, args: argparse.Namespace, hidden: int, latent: int) -> dict:
    command=[
        sys.executable, "-u", "phase3g_lopo.py", "--dataset-dir", args.dataset_dir,
        "--output-root", str(output), "--device", args.device,
        "--hidden-size", str(hidden), "--latent-size", str(latent),
        "--batch-size", str(args.batch_size), "--gradient-accumulation", str(args.gradient_accumulation),
    ]
    if args.smoke: command.extend(["--warmup-epochs", "1", "--generalize-epochs", "1", "--samples-per-epoch", "1", "--max-train-batches", "1"])
    subprocess.run(command, check=True)
    return json.loads((output/"lopo_summary.json").read_text(encoding="utf-8"))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stress-cases", type=int, default=48)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--run-lopo", action="store_true")
    parser.add_argument("--run-e3", action="store_true")
    parser.add_argument("--run-three-seeds", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args=parser.parse_args()
    root=Path(args.output_root or f"runs/phase3g_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    root.mkdir(parents=True, exist_ok=False)
    selected_hidden=32; selected_latent=16
    (root/"P3G-E0.json").write_text(json.dumps(E0_DIAGNOSTICS, ensure_ascii=False, indent=2), encoding="utf-8")
    e1=_run_training(root/"P3G-E1", args, "warmup", None, args.seed)
    p3r_path=Path("runs/phase3r_suite_seed2026/P3R-E1c/summary.json")
    p3r=json.loads(p3r_path.read_text(encoding="utf-8"))["metrics"]
    e1_primary=e1["development_metrics"]["primary_score_db"]
    warmup_pass=e1_primary >= p3r["primary_score_db"]-0.25
    result={
        "phase":"3G", "P3G-E0":E0_DIAGNOSTICS,
        "P3G-E1":{"summary":str(root/"P3G-E1"/"summary.json"), "warmup_passed":warmup_pass},
        "formal_model":"P1-E2", "final_paths_touched":False,
    }
    if not warmup_pass:
        result["stop_reason"]="P3G-E1 primary score dropped more than 0.25 dB from P3R-E1c"
        (root/"suite_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    e2=_run_training(root/"P3G-E2", args, "generalize", e1["selected_checkpoint"], args.seed)
    result["P3G-E2"]={"summary":str(root/"P3G-E2"/"summary.json"), "development_passed":e2["acceptance"]["passed"]}
    result["selected_development_checkpoint"]=e2["selected_checkpoint"]
    if not e2["acceptance"]["passed"]:
        result["stop_reason"]="P3G-E2 development or continuous-path stress gate failed"
    elif args.run_lopo:
        lopo=_run_lopo(root/"P3G-LOPO", args, 32, 16)
        result["LOPO"]={"summary":str(root/"P3G-LOPO"/"lopo_summary.json"), "passed":lopo["acceptance"]["passed"]}
        if lopo["acceptance"]["passed"]:
            result["ready_for_three_seed_retraining"]=True
        elif args.run_e3:
            correction=lopo["recommended_single_correction"]
            hidden=int(correction["hidden_size"]); latent=int(correction["latent_size"])
            selected_hidden=hidden; selected_latent=latent
            e3_warm=_run_training(root/"P3G-E3-warmup", args, "warmup", None, args.seed, hidden, latent)
            e3=_run_training(root/"P3G-E3", args, "generalize", e3_warm["selected_checkpoint"], args.seed, hidden, latent)
            result["P3G-E3"]={"capacity":correction, "summary":str(root/"P3G-E3"/"summary.json"),
                               "development_passed":e3["acceptance"]["passed"]}
            if e3["acceptance"]["passed"]:
                e3_lopo=_run_lopo(root/"P3G-E3-LOPO", args, hidden, latent)
                result["P3G-E3-LOPO"]={"summary":str(root/"P3G-E3-LOPO"/"lopo_summary.json"),
                                        "passed":e3_lopo["acceptance"]["passed"]}
                result["ready_for_three_seed_retraining"]=e3_lopo["acceptance"]["passed"]
    if result.get("ready_for_three_seed_retraining") and args.run_three_seeds:
        seed_results=[]
        for seed in (2026, 2027, 2028):
            warm=_run_training(root/f"seed_{seed}_warmup", args, "warmup", None, seed,
                               selected_hidden, selected_latent)
            trained=_run_training(root/f"seed_{seed}_generalize", args, "generalize",
                                  warm["selected_checkpoint"], seed, selected_hidden, selected_latent)
            seed_results.append({
                "seed":seed, "summary":str(root/f"seed_{seed}_generalize"/"summary.json"),
                "checkpoint":trained["selected_checkpoint"], "development_passed":trained["acceptance"]["passed"],
            })
        result["three_seed_results"]=seed_results
        result["ready_for_final_paths"]=all(value["development_passed"] for value in seed_results)
    (root/"suite_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
