"""HABS entry point: audit the available data, then run the arousal experiment."""

from pathlib import Path


def process(data_dir: Path, out_dir: Path) -> None:
    from arousal.pipeline import audit, run_experiment

    report = audit(data_dir, out_dir)
    if report["status"] != "ready":
        print("Data audit written. Need labels and emotion-video BDFs to train.")
        return
    # CUDA is optional; use it automatically on the HABS worker.
    import torch
    run_experiment(data_dir, out_dir, device="cuda" if torch.cuda.is_available() else "cpu")
