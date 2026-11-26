"""Exercise one crossed fold end to end using explicitly synthetic fixtures."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arousal import MODALITIES
from arousal.data import Trial
from arousal.evaluation import crossed_splits, train_validation_split
from arousal.pipeline import run_experiment
from arousal.quality import FAULTS, inject_fault, quality_features
from arousal.runtime import TrialStream
from arousal.signals import RestState
from test_causality import signals


def test_one_fold_training_calibration_reports_and_matched_groups(tmp_path, monkeypatch):
    import arousal.pipeline as pipeline
    torch.set_num_threads(1)
    trials = [Trial(f"p{p}-v{v}", f"P{p:03}", 1, f"v{v}.mp4",
                    "men" if p % 3 == 0 else "women", v % 3,
                    Path("synthetic-video"), Path("synthetic-rest"), Path("synthetic-meta"), 5., 10.)
              for p in range(18) for v in range(18)]
    table = pd.DataFrame([asdict(t) for t in trials])
    # One real crossed assignment; leave the entire split/disjointness code active.
    chosen = None
    for fold, train, test in crossed_splits(table, 3):
        outer = table.iloc[train].reset_index(drop=True)
        tr, va = train_validation_split(outer)
        if {"men", "women"} <= set(outer.iloc[tr].sex) & set(outer.iloc[va].sex):
            chosen = fold, train, test
            break
    assert chosen is not None
    monkeypatch.setattr(pipeline, "audit", lambda *args: {"status": "ready"})
    monkeypatch.setattr(pipeline, "find_trials", lambda *args: (trials, []))
    monkeypatch.setattr(pipeline, "crossed_splits", lambda *args: iter([chosen]))
    raw = signals()
    rest = RestState.fit(signals(12, seed=8), 1000)
    windows = [w for w in TrialStream(1000, rest).push(raw) if w["time"] in (5., 7., 10.)]
    q = np.stack([quality_features(signals(1)[k], 1000) for k in MODALITIES])
    for window in windows:
        window["quality"][:] = q

    class SyntheticCache:
        def __init__(self, out):
            pass
        def get(self, trial, fault=False):
            if not fault:
                return windows
            changed = []
            for window in windows:
                copy = dict(window, quality=window["quality"].copy())
                copy["quality"][-1, 0, 7] = 1.
                changed.append(copy)
            return changed
    monkeypatch.setattr(pipeline, "WindowCache", SyntheticCache)
    rows, rng = [], np.random.default_rng(7)
    for pid in table.participant.unique():
        for repeat in range(3):
            raw = signals(1, seed=repeat)
            for kind in MODALITIES:
                for fault in ("clean", *FAULTS):
                    x = raw[kind] if fault == "clean" else inject_fault(raw[kind], fault, 1000, rng)
                    rows.append(dict(participant=pid, modality=kind, type=fault, source="synthetic",
                                     target=int(fault != "clean"), features=quality_features(x, 1000)))
    monkeypatch.setattr(pipeline, "calibration_rows", lambda *args: rows)
    monkeypatch.setattr(pipeline, "quality_delay_tests", lambda *args: [])
    predictions = run_experiment(tmp_path, tmp_path, folds=3, epochs=1)
    assert {"global", "global_all", "sex_specific"} == set(predictions.variant)
    counts = pd.read_csv(tmp_path / "training_counts.csv")
    assert set(counts.status) == {"trained"}
    for condition in ("clean", "injected"):
        for gating in (True, False):
            rows = predictions[(predictions.condition == condition) & (predictions.gating == gating)]
            assert set(rows[rows.variant == "global"].trial) == set(rows[rows.variant == "sex_specific"].trial)
    assert len(list(tmp_path.glob("0*.png"))) == 4
    assert (tmp_path / "metrics.csv").exists()
    assert len(list((tmp_path / "models").glob("*/*/model.pt"))) == 3
