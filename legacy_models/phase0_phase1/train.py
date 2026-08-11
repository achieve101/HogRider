import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PreconvolutedANCDataset, apply_dynamic_path
from legacy_models.phase0_phase1.model import TimeDomainANC


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Phase-0 DEEPANC strong baseline.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-duration", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    return parser.parse_args()


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by CUDA >= 10.2 for deterministic CuBLAS operations.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms make comparisons between experiments trustworthy.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_arg!r} was requested but CUDA is unavailable.")
    return device


def scan_noise_files(noise_dir):
    noise_dir = Path(noise_dir)
    noise_files = sorted(
        [path.name for path in noise_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"],
        key=str.casefold,
    )
    if not noise_files:
        raise FileNotFoundError(f"No WAV files found in {noise_dir}.")
    return noise_files


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path, value):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def save_checkpoint(path, model, optimizer, epoch, config, data_split, metrics):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "data_split": data_split,
        "metrics": metrics,
    }
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def evaluate_model(model, test_loader, device, sr=48000, scenario_title="Test",
                   save_prefix=None, output_dir=None):
    """Evaluate legacy whole-segment NR and optionally create diagnostic plots."""
    model.eval()
    nr_list = []
    all_d_np = []
    all_e_np = []
    make_plots = save_prefix is not None and output_dir is not None

    with torch.no_grad():
        for x_t, sh, d_t in test_loader:
            x_t = x_t.to(device, non_blocking=True)
            sh = sh.to(device, non_blocking=True)
            d_t = d_t.to(device, non_blocking=True)

            y_t = model(x_t)
            a_t = apply_dynamic_path(y_t, sh)
            e_t = d_t - a_t

            d_np = d_t[0].cpu().numpy()
            e_np = e_t[0].cpu().numpy()
            energy_d = float(np.sum(d_np ** 2))
            energy_e = float(np.sum(e_np ** 2))
            nr_list.append(float(10 * np.log10(energy_d / (energy_e + 1e-12))))

            if make_plots:
                all_d_np.append(d_np)
                all_e_np.append(e_np)

    metrics = {
        "scenario_title": scenario_title,
        "nr_db_per_path": nr_list,
        "average_nr_db": float(np.mean(nr_list)),
        "worst_nr_db": float(np.min(nr_list)),
        "best_nr_db": float(np.max(nr_list)),
    }

    print(f"\n>> [{scenario_title}] Independent Scene NR: "
          f"{[f'{value:.2f}dB' for value in nr_list]}")
    print(f">> [{scenario_title}] Average NR: {metrics['average_nr_db']:.2f} dB, "
          f"Worst NR: {metrics['worst_nr_db']:.2f} dB")

    if make_plots:
        plot_evaluation(
            all_d_np, all_e_np, nr_list, metrics["average_nr_db"], sr,
            scenario_title, save_prefix, Path(output_dir),
        )
    return metrics


def plot_evaluation(all_d_np, all_e_np, nr_list, avg_nr, sr, scenario_title,
                    save_prefix, output_dir):
    num_scenarios = len(all_d_np)
    cols = 2
    rows = max(1, (num_scenarios + 1) // 2)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_time, axes_time = plt.subplots(rows, cols, figsize=(10, 2.5 * rows), squeeze=False)
    axes_time = axes_time.flatten()
    plot_duration = 0.9
    plot_samples = min(int(plot_duration * sr), len(all_d_np[0]))
    time_axis = np.arange(plot_samples) * 1000 / sr

    for index in range(num_scenarios):
        axes_time[index].plot(
            time_axis, all_d_np[index][:plot_samples],
            label="Primary Noise $d(t)$", color="blue", alpha=0.6,
        )
        axes_time[index].plot(
            time_axis, all_e_np[index][:plot_samples],
            label="Residual Noise $e(t)$", color="red", alpha=0.8,
        )
        axes_time[index].set_title(
            f"Scenario {index + 1} Time Domain (First {plot_duration:.1f}s) | "
            f"NR: {nr_list[index]:.2f} dB"
        )
        axes_time[index].set_xlabel("Time (ms)")
        axes_time[index].set_ylabel("Amplitude")
        axes_time[index].legend(loc="upper right")
        axes_time[index].grid(True)
    for index in range(num_scenarios, len(axes_time)):
        axes_time[index].axis("off")

    fig_time.suptitle(
        f"[{scenario_title}] Time Domain Noise Cancellation | Average NR: {avg_nr:.2f} dB",
        fontsize=16,
    )
    fig_time.tight_layout(rect=[0, 0.03, 1, 0.95])
    time_path = output_dir / f"anc_{save_prefix}_time_result.png"
    fig_time.savefig(time_path, dpi=300)
    plt.close(fig_time)
    print(f">> Time-domain results saved to {time_path}")

    fig_freq, axes_freq = plt.subplots(rows, cols, figsize=(10, 2.5 * rows), squeeze=False)
    axes_freq = axes_freq.flatten()
    for index in range(num_scenarios):
        axes_freq[index].psd(
            all_d_np[index], NFFT=1024, Fs=sr,
            label="Primary Noise", color="blue", alpha=0.6,
        )
        axes_freq[index].psd(
            all_e_np[index], NFFT=1024, Fs=sr,
            label="Residual Noise", color="red", alpha=0.8,
        )
        axes_freq[index].set_title(f"Scenario {index + 1} Frequency Spectrum")
        axes_freq[index].set_xlabel("Frequency (Hz)")
        axes_freq[index].set_ylabel("Power/Frequency (dB/Hz)")
        axes_freq[index].legend()
        axes_freq[index].grid(True, which="both", ls="-", alpha=0.5)
        axes_freq[index].set_xscale("log")
        axes_freq[index].set_xlim(left=20, right=sr / 2)
    for index in range(num_scenarios, len(axes_freq)):
        axes_freq[index].axis("off")

    fig_freq.suptitle(
        f"[{scenario_title}] Frequency Domain PSD (0 to {sr // 2}Hz)", fontsize=16,
    )
    fig_freq.tight_layout(rect=[0, 0.03, 1, 0.95])
    frequency_path = output_dir / f"anc_{save_prefix}_freq_result.png"
    fig_freq.savefig(frequency_path, dpi=300)
    plt.close(fig_freq)
    print(f">> Frequency-domain results saved to {frequency_path}")


def build_loaders(args, device, train_noises, test_noises, train_paths, test_paths):
    train_dataset = PreconvolutedANCDataset(
        args.dataset_dir, train_noises, train_paths,
        segment_duration=args.segment_duration, sr=48000, is_train=True,
        samples_per_epoch=args.samples_per_epoch,
    )
    test_seen_dataset = PreconvolutedANCDataset(
        args.dataset_dir, test_noises, train_paths,
        segment_duration=args.segment_duration, sr=48000, is_train=False,
    )
    test_unseen_dataset = PreconvolutedANCDataset(
        args.dataset_dir, test_noises, test_paths,
        segment_duration=args.segment_duration, sr=48000, is_train=False,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common_loader_args = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if args.num_workers > 0:
        common_loader_args["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=generator, **common_loader_args,
    )
    test_seen_loader = DataLoader(
        test_seen_dataset, batch_size=1, shuffle=False, **common_loader_args,
    )
    test_unseen_loader = DataLoader(
        test_unseen_dataset, batch_size=1, shuffle=False, **common_loader_args,
    )
    return train_loader, test_seen_loader, test_unseen_loader


def main():
    args = parse_args()
    if args.epochs <= 0 or args.samples_per_epoch <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, samples-per-epoch, and batch-size must be positive.")
    if args.validation_interval <= 0:
        raise ValueError("validation-interval must be positive.")

    set_global_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"runs/phase0_{timestamp}")
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True)

    dataset_dir = Path(args.dataset_dir)
    all_noise_files = scan_noise_files(dataset_dir / "NOISE")
    if len(all_noise_files) < 3:
        train_noises = all_noise_files
        test_noises = all_noise_files
        print("Warning: fewer than three noises; testing reuses training noises.")
    else:
        train_noises = all_noise_files[:-2]
        test_noises = all_noise_files[-2:]

    sh_paths = np.load(dataset_dir / "sh.npy").T
    num_paths = len(sh_paths)
    if num_paths < 2:
        raise ValueError("At least two secondary paths are required for a held-out split.")
    train_count = max(1, int(num_paths * 0.8))
    train_count = min(train_count, num_paths - 1)
    train_paths = list(range(train_count))
    test_paths = list(range(train_count, num_paths))

    data_split = {
        "all_noise_files": all_noise_files,
        "train_noise_files": train_noises,
        "test_noise_files": test_noises,
        "train_path_indices_zero_based": train_paths,
        "test_path_indices_zero_based": test_paths,
        "secondary_path_shape_after_transpose": list(sh_paths.shape),
    }
    config = vars(args).copy()
    config.update({
        "resolved_device": str(device),
        "output_dir": str(output_dir),
        "model": {
            "name": "TimeDomainANC",
            "in_channels": 1,
            "out_channels": 1,
            "hidden_channels": 32,
            "num_layers": 10,
        },
    })
    save_json(output_dir / "config.json", config)
    save_json(output_dir / "data_split.json", data_split)

    print(f"Detected noise files ({len(all_noise_files)}): {all_noise_files}")
    print(f"Training noises ({len(train_noises)}): {train_noises}")
    print(f"Held-out noises ({len(test_noises)}): {test_noises}")
    print(f"Training paths: {[index + 1 for index in train_paths]}")
    print(f"Held-out paths: {[index + 1 for index in test_paths]}")
    print(f"Samples per epoch: {args.samples_per_epoch}; "
          f"optimizer steps per epoch: {(args.samples_per_epoch + args.batch_size - 1) // args.batch_size}")
    print(f"Run artifacts: {output_dir}")

    train_loader, test_seen_loader, test_unseen_loader = build_loaders(
        args, device, train_noises, test_noises, train_paths, test_paths,
    )

    model = TimeDomainANC(
        in_channels=1, out_channels=1, hidden_channels=32, num_layers=10,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}")

    history_path = output_dir / "history.jsonl"
    best_mean_nr = -float("inf")
    best_robust_nr = -float("inf")
    best_mean_epoch = None
    best_robust_epoch = None
    latest_metrics = None

    print("\n=== Phase-0 strong baseline training ===")
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum = 0.0
        sample_count = 0

        for x_t, sh, d_t in train_loader:
            x_t = x_t.to(device, non_blocking=True)
            sh = sh.to(device, non_blocking=True)
            d_t = d_t.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            y_t = model(x_t)
            a_t = apply_dynamic_path(y_t, sh)
            e_t = d_t - a_t
            loss = torch.mean(e_t ** 2)
            loss.backward()
            optimizer.step()

            batch_size = x_t.shape[0]
            loss_sum += loss.item() * batch_size
            sample_count += batch_size

        average_loss = loss_sum / max(1, sample_count)
        epoch_record = {
            "epoch": epoch,
            "train_loss": average_loss,
            "samples": sample_count,
            "duration_seconds": time.perf_counter() - epoch_start,
        }
        print(f"Epoch [{epoch}/{args.epochs}] Avg Loss: {average_loss:.8f} "
              f"({sample_count} samples, {epoch_record['duration_seconds']:.2f}s)")

        should_validate = (
            epoch == 1 or epoch == args.epochs or epoch % args.validation_interval == 0
        )
        if should_validate:
            print(f"\n--- Validation at epoch {epoch} ---")
            seen_metrics = evaluate_model(
                model, test_seen_loader, device,
                scenario_title="Unseen Noise, Seen Paths",
            )
            unseen_metrics = evaluate_model(
                model, test_unseen_loader, device,
                scenario_title="Unseen Noise, Unseen Paths",
            )
            all_nr = seen_metrics["nr_db_per_path"] + unseen_metrics["nr_db_per_path"]
            latest_metrics = {
                "seen_paths": seen_metrics,
                "unseen_paths": unseen_metrics,
                "combined_mean_nr_db": float(np.mean(all_nr)),
                "combined_worst_nr_db": float(np.min(all_nr)),
            }
            epoch_record["validation"] = latest_metrics

            if latest_metrics["combined_mean_nr_db"] > best_mean_nr:
                best_mean_nr = latest_metrics["combined_mean_nr_db"]
                best_mean_epoch = epoch
                save_checkpoint(
                    checkpoint_dir / "best_mean_nr.pt", model, optimizer, epoch,
                    config, data_split, latest_metrics,
                )
                print(f">> New best mean-NR checkpoint: {best_mean_nr:.2f} dB")

            if latest_metrics["combined_worst_nr_db"] > best_robust_nr:
                best_robust_nr = latest_metrics["combined_worst_nr_db"]
                best_robust_epoch = epoch
                save_checkpoint(
                    checkpoint_dir / "best_robust_nr.pt", model, optimizer, epoch,
                    config, data_split, latest_metrics,
                )
                print(f">> New best robust checkpoint: worst path {best_robust_nr:.2f} dB")

        save_checkpoint(
            checkpoint_dir / "latest.pt", model, optimizer, epoch,
            config, data_split, latest_metrics,
        )
        append_jsonl(history_path, epoch_record)

    training_seconds = time.perf_counter() - training_start

    # Final diagnostics use the best average-NR checkpoint, not an arbitrary last epoch.
    best_checkpoint_path = checkpoint_dir / "best_mean_nr.pt"
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    print(f"\n=== Final evaluation using best mean-NR checkpoint (epoch {best_checkpoint['epoch']}) ===")
    final_seen = evaluate_model(
        model, test_seen_loader, device,
        scenario_title="Unseen Noise, Seen Paths",
        save_prefix="seen_paths", output_dir=output_dir,
    )
    final_unseen = evaluate_model(
        model, test_unseen_loader, device,
        scenario_title="Unseen Noise, Unseen Paths",
        save_prefix="unseen_paths", output_dir=output_dir,
    )
    final_nr = final_seen["nr_db_per_path"] + final_unseen["nr_db_per_path"]
    summary = {
        "training_seconds": training_seconds,
        "parameter_count": parameter_count,
        "best_mean_epoch": best_mean_epoch,
        "best_mean_nr_db": best_mean_nr,
        "best_robust_epoch": best_robust_epoch,
        "best_robust_worst_path_nr_db": best_robust_nr,
        "final_checkpoint": str(best_checkpoint_path),
        "final_seen_paths": final_seen,
        "final_unseen_paths": final_unseen,
        "final_combined_mean_nr_db": float(np.mean(final_nr)),
        "final_combined_worst_nr_db": float(np.min(final_nr)),
    }
    save_json(output_dir / "summary.json", summary)

    print(f"\nTraining completed in {training_seconds:.2f}s")
    print(f"Best mean-NR epoch: {best_mean_epoch}, score: {best_mean_nr:.2f} dB")
    print(f"Best robust epoch: {best_robust_epoch}, worst path: {best_robust_nr:.2f} dB")
    print(f"Artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
