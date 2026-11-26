"""Offline experiments; deployment uses the same TrialStream and Predictor."""

from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from . import MODALITIES
from .alarm import AlarmConfig, alarm_metrics, choose_alarm
from .data import EEG_NAMES, Recording, find_trials, marker_lookup
from .evaluation import assert_disjoint, crossed_splits, train_validation_split, trial_metrics, write_reports
from .quality import FAULTS, QualityGate, calibration_rows, inject_fault, quality_features
from .runtime import Predictor, TrialStream, mask_window
from .signals import RestState


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
                               allow_nan=False) + "\n")


def audit(root, out):
    """A sample-only checkout gets an honest data-availability report."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"curated_bdfs": len(list(root.glob("CURATED/P*/*.bdf"))),
              "video_bdfs": len(list(root.glob("CURATED/P*/*_task-emotion-video_*.bdf"))),
              "class_order": ["low", "mid", "high"], "eeg_channels": 62,
              "normalisation": "rest-only exponential statistics, frozen during videos",
              "status": "not_ready"}
    try:
        trials, skipped = find_trials(root)
        report.update(trials=len(trials), participants=len({t.participant for t in trials}),
                      videos=len({t.video for t in trials}), skipped=skipped)
        if trials:
            report["status"] = "ready"
    except FileNotFoundError as error:
        report["reason"] = str(error)
    write_json(out / "data_audit.json", report)
    return report


@lru_cache(maxsize=8)
def resting_state(path):
    with Recording(path) as recording:
        return RestState.fit(recording.read(0, recording.seconds), recording.fs)


def prepare_trial(trial, fault=False):
    rest = resting_state(trial.rest)
    with Recording(trial.bdf) as recording:
        fs = recording.fs
        raw = recording.read(trial.onset - 5, trial.onset + min(10., trial.duration))
    if fault:
        # Fixed per-trial assignment balances five fault types over four modalities.
        number = int(hashlib.sha256(trial.id.encode()).hexdigest()[:8], 16)
        modality, kind = MODALITIES[number % 4], FAULTS[(number // 4) % len(FAULTS)]
        start, stop = 8 * fs, 9 * fs  # response [3,4) s, never baseline/rating data
        if raw[modality].shape[1] >= stop:
            raw[modality][:, start:stop] = inject_fault(raw[modality][:, start:stop], kind, fs,
                                                      np.random.default_rng(number))
    return TrialStream(fs, rest, duration=trial.duration).push(raw)


class WindowCache:
    """One short trial at a time on disk; avoid loading whole video recordings."""
    def __init__(self, out):
        self.path = out / "cache"
        self.path.mkdir(exist_ok=True)

    def get(self, trial, fault=False):
        # Include source mtimes to invalidate modified datasets.
        signature = str([(str(p), p.stat().st_mtime_ns) for p in (trial.bdf, trial.rest, trial.metadata)])
        digest = hashlib.sha256(("v6" + signature).encode()).hexdigest()[:12]
        path = self.path / f"{trial.id}-{digest}-{'fault' if fault else 'clean'}.pkl"
        if not path.exists():
            windows = prepare_trial(trial, fault)
            with path.open("wb") as handle:
                pickle.dump(windows, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return windows
        with path.open("rb") as handle:
            return pickle.load(handle)


def training_arrays(trials, cache, gate, t_start):
    eeg, physio, masks, times, targets, ids = [], [], [], [], [], []
    for trial in trials:
        for window in cache.get(trial):
            x, p, mask = mask_window(window, gate)
            if window["time"] < t_start or mask.sum() < gate.min_clean:
                continue
            eeg.append(x); physio.append(p); masks.append(mask)
            times.append(window["time"]); targets.append(trial.target); ids.append(trial.id)
    if not eeg:
        raise ValueError("No reliable training/validation windows; inspect quality labels and coverage")
    return {"eeg": np.stack(eeg), "physio": np.stack(physio), "mask": np.stack(masks),
            "time": np.asarray(times, np.float32), "target": np.asarray(targets), "trial": np.asarray(ids)}


def predict_trials(predictor, trials, cache, variant, condition="clean", gating=True):
    rows = []
    for trial in trials:
        from .alarm import Alarm
        predictor.alarm = Alarm(predictor.alarm_config)
        for window in cache.get(trial, fault=condition == "injected"):
            result = predictor.predict_window(window, gating)
            result.update(trial=trial.id, participant=trial.participant, video=trial.video,
                          sex=trial.sex, target=trial.target, variant=variant,
                          condition=condition, gating=gating)
            rows.append(result)
    return pd.DataFrame(rows)


def save_predictor(predictor, directory, details):
    import joblib
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {k: v.detach().cpu() for k, v in predictor.model.state_dict().items()},
                "mean": torch.as_tensor(predictor.scaler[0]), "scale": torch.as_tensor(predictor.scaler[1]),
                "temperature": predictor.temperature, "t_start": predictor.t_start,
                "alarm": asdict(predictor.alarm_config)}, directory / "model.pt")
    joblib.dump(predictor.gate, directory / "quality.joblib")
    write_json(directory / "details.json", details)


def load_predictor(directory, device="cpu"):
    """Load only bundles generated by this project; joblib is not an untrusted format."""
    import torch
    # Tiny EEGNet windows need one CPU thread; GPU kernels still use device.
    # Initialize before LightGBM to avoid macOS native thread-pool deadlocks.
    torch.set_num_threads(1)
    import joblib
    directory = Path(directory)
    gate = joblib.load(directory / "quality.joblib")
    from .model import FusionModel
    saved = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model = FusionModel().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return Predictor(model, gate,
                     (saved["mean"].cpu().numpy(), saved["scale"].cpu().numpy()),
                     saved["temperature"], AlarmConfig(**saved["alarm"]), saved["t_start"])


def tune_variant(train, validation, cache, gate, epochs, seed, device):
    from .model import fit_temperature, train_model
    # Choose start time with validation trial F1; the test split is untouched.
    best = None
    trials_by_id = {t.id: t for t in validation}
    for t_start in (0., 2., 3., 4., 5.):
        training = training_arrays(train, cache, gate, t_start)
        validating = training_arrays(validation, cache, gate, t_start)
        model, scaler, history = train_model(training, validating, epochs, seed, device)
        predictor = Predictor(model, gate, scaler, 1., AlarmConfig(), t_start)
        rows = predict_trials(predictor, validation, cache, "validation")
        metrics = trial_metrics(rows)
        score = metrics.get("macro_f1", -1)
        entry = {"t_start": t_start, "validation": metrics, "history": history}
        print(f"    start={t_start:g}s validation macro-F1={score:.3f}", flush=True)
        if best is None or score > best[0]:
            best = score, predictor, entry
    _, predictor, details = best
    validating = training_arrays(list(trials_by_id.values()), cache, gate, predictor.t_start)
    predictor.temperature = fit_temperature(predictor.model, validating, predictor.scaler)
    rows = predict_trials(predictor, validation, cache, "validation")
    return predictor, details, rows


def run_experiment(root, out, folds=3, epochs=20, seed=7, device="cpu", target_fah=1.):
    """All fitted transforms, quality labels and hyperparameters exclude test data."""
    root, out = Path(root), Path(out)
    report = audit(root, out)
    if report["status"] != "ready":
        raise ValueError("Training unavailable; see data_audit.json. Full labels and video BDFs are required.")
    trials, _ = find_trials(root)
    table = pd.DataFrame([asdict(t) for t in trials])
    table.to_csv(out / "trials.csv", index=False)
    cache = WindowCache(out)
    quality_rows = calibration_rows(root, table.metadata)
    if not quality_rows:
        raise ValueError("No verified artefact/rest calibration windows")
    pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in quality_rows]).to_csv(
        out / "quality_label_provenance.csv", index=False)
    predictions, qualities, splits, counts, selected_alarms, delays = [], [], [], [], [], []
    for fold, outer_train, test_ids in crossed_splits(table, folds, seed):
        outer = table.iloc[outer_train].reset_index(drop=True)
        tr, va = train_validation_split(outer, seed)
        train_ids, val_ids = outer.iloc[tr].id, outer.iloc[va].id
        train = [t for t in trials if t.id in set(train_ids)]
        validation = [t for t in trials if t.id in set(val_ids)]
        test = [trials[i] for i in test_ids]
        train_table, val_table, test_table = table[table.id.isin(train_ids)], table[table.id.isin(val_ids)], table.iloc[test_ids]
        assert_disjoint(train_table, val_table, test_table)
        split = {"fold": fold, "train": train_table.id.tolist(), "validation": val_table.id.tolist(),
                 "test": test_table.id.tolist(), "discarded_mixed_rectangles": len(table)-len(train)-len(validation)-len(test)}
        splits.append(split)
        print(f"{fold}: {len(train)} train / {len(validation)} validation / {len(test)} test trials", flush=True)
        gate = QualityGate.fit(quality_rows, set(t.participant for t in train))
        qualities.extend({"fold": fold, **r} for r in gate.evaluate(quality_rows, set(t.participant for t in test)))
        predictors = {}
        for variant, sex in (("global", None), ("women", "women"), ("men", "men")):
            training = train if sex is None else [t for t in train if t.sex == sex]
            validating = validation if sex is None else [t for t in validation if t.sex == sex]
            count = {"fold": fold, "variant": variant, "train_trials": len(training),
                     "train_participants": len({t.participant for t in training}),
                     "validation_trials": len(validating), "training_men_fraction":
                     sum(t.sex == "men" for t in training) / len(training) if training else 0.}
            counts.append(count)
            if not training or set(t.target for t in training) != {0, 1, 2} or set(t.target for t in validating) != {0, 1, 2}:
                count["status"] = "insufficient classes/groups"
                if variant == "global":
                    raise ValueError(f"{fold}: insufficient global classes after split")
                continue
            print(f"  {variant}: {len(training)} training trials", flush=True)
            # Shared train-only artefact models keep gating identical for A vs B.
            predictor, details, val_predictions = tune_variant(training, validating, cache, gate, epochs, seed, device)
            predictor.alarm_config, val_alarm = choose_alarm(val_predictions, target_fah)
            details.update(fold=fold, variant=variant, counts=count,
                           target_false_alarms_per_hour=target_fah, validation_alarm=val_alarm,
                           temperature=predictor.temperature,
                           note="Observed target on validation only; no guarantee on test/live data")
            save_predictor(predictor, out / "models" / fold / variant, details)
            predictors[variant] = predictor
            count["status"] = "trained"
            selected = test if sex is None else [t for t in test if t.sex == sex]
            if not selected:
                continue
            for condition in ("clean", "injected"):
                for gating in (True, False):
                    rows = predict_trials(predictor, selected, cache,
                                          "global_all" if sex is None else "sex_specific", condition, gating)
                    rows["fold"] = fold
                    predictions.append(rows)
                    selected_alarms.append({"fold": fold, "variant": variant, "condition": condition,
                                            "gating": gating, "threshold": predictor.alarm_config.threshold,
                                            **alarm_metrics(rows, predictor.alarm_config)})
            # Use stream-level partial-window fault tests for detection delay.
            if variant == "global":
                delays.extend(quality_delay_tests(root, test, gate, seed, fold))
        matched = [t for t in test if t.sex in predictors]
        if matched and "global" in predictors:
            for condition in ("clean", "injected"):
                for gating in (True, False):
                    rows = predict_trials(predictors["global"], matched, cache, "global", condition, gating)
                    rows["fold"] = fold
                    predictions.append(rows)
        write_json(out / "splits.json", splits)
        pd.DataFrame(counts).to_csv(out / "training_counts.csv", index=False)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(out / "window_predictions.csv", index=False)
    pd.DataFrame(qualities).to_csv(out / "quality_metrics.csv", index=False)
    pd.DataFrame(delays).to_csv(out / "quality_detection_delay.csv", index=False)
    pd.DataFrame(selected_alarms).to_csv(out / "selected_alarm_test_metrics.csv", index=False)
    write_reports(all_predictions, out)
    latency = all_predictions.latency_ms
    write_json(out / "latency.json", {"scope": "cached feature -> quality gate -> model -> alarm; use replay for full step",
                                      "median_ms": latency.median(), "p95_ms": latency.quantile(.95),
                                      "max_ms": latency.max(), "p95_under_500ms": bool(latency.quantile(.95) < 500)})
    packages = ("numpy", "scipy", "pandas", "torch", "braindecode", "lightgbm", "scikit-learn")
    write_json(out / "run.json", {"folds": folds, "epochs": epochs, "seed": seed, "device": device,
                                  "target_fah": target_fah, "versions": {p: importlib.metadata.version(p) for p in packages},
                                  "men_fraction_participants": table.drop_duplicates("participant").sex.eq("men").mean(),
                                  "note": "gender proxy; men's smaller training counts are in training_counts.csv"})
    return all_predictions


def quality_delay_tests(root, trials, gate, seed, fold):
    """Partial 1 s faults at known onset; evaluate every .5 s, including misses.

    Uses held-out rest only as an evaluation reference, never to fit the gate.
    Reports individual events so delay distributions and miss rates are auditable.
    """
    rng, results = np.random.default_rng(seed), []
    for restpath in sorted({t.rest for t in trials}):
        good = resting_state(restpath).good
        with Recording(restpath) as recording:
            if recording.seconds < 12:
                continue
            raw = recording.read(5, 9)
            fs = recording.fs
        for index, modality in enumerate(MODALITIES):
            for kind in FAULTS:
                modified = {k: v.copy() for k, v in raw.items()}
                modified[modality][:, 2*fs:3*fs] = inject_fault(
                    modified[modality][:, 2*fs:3*fs], kind, fs, rng)
                delay = None
                false_episodes, was_bad = 0, False
                for end in np.arange(1., 4.01, .5):
                    a, b = round((end - 1) * fs), round(end * fs)
                    q = np.stack([quality_features(modified[k][good, a:b] if k == "eeg"
                                                   else modified[k][:, a:b], fs,
                                                   tuple(np.asarray(EEG_NAMES)[good]) if k == "eeg" else None)
                                  for k in MODALITIES])
                    bad = not gate.clean(q)[index]
                    if end <= 2:
                        false_episodes += int(bad and not was_bad)
                        was_bad = bad
                    if 2 < end <= 3.5 and bad and delay is None:
                        delay = float(end - 2)
                results.append({"fold": fold, "rest": str(restpath), "modality": modality,
                                "type": kind, "detected": delay is not None, "delay_s": delay,
                                "clean_false_alarm_episodes_per_hour": false_episodes / (1.5 / 3600),
                                "reference": "injected into held-out rest; weak clean reference"})
    # Measure instructed EEG events too, without treating an already-bad
    # pre-event window as a successful new-event detection.
    lookup = marker_lookup(root)
    for trial in {t.metadata: t for t in trials}.values():
        meta = json.loads(trial.metadata.read_text())
        seg = meta["dataset_export"]["segments"].get("artefact")
        if not seg or not (trial.metadata.parent / seg["file"]).exists():
            continue
        good = resting_state(trial.rest).good
        with Recording(trial.metadata.parent / seg["file"]) as recording:
            for event in meta["events"]:
                marker = int(event["marker_id"])
                if marker not in range(10006, 10020, 2):
                    continue
                if marker not in lookup or lookup[marker][0] != marker + 1:
                    raise ValueError(f"Unverified artefact marker {marker}")
                onset = event["onset_s"] - seg["metadata_time_origin_s"]
                horizon = min(float(event["duration_s"]), 2.)
                if onset < 1 or onset + horizon > recording.seconds:
                    continue
                raw = recording.read(onset - 1, onset + horizon)
                fs, delay, baseline_bad = recording.fs, None, False
                for elapsed in np.arange(0., horizon + .001, .5):
                    a, b = round(elapsed * fs), round((1 + elapsed) * fs)
                    q = np.stack([quality_features(raw[k][good, a:b] if k == "eeg"
                                                   else raw[k][:, a:b], fs,
                                                   tuple(np.asarray(EEG_NAMES)[good]) if k == "eeg" else None)
                                  for k in MODALITIES])
                    bad = not gate.clean(q)[0]
                    if elapsed == 0:
                        baseline_bad = bad
                    elif bad and not baseline_bad and delay is None:
                        delay = float(elapsed)
                results.append({"fold": fold, "rest": str(trial.rest), "modality": "eeg",
                                "type": lookup[marker][1], "detected": delay is not None,
                                "delay_s": delay, "baseline_already_flagged": baseline_bad,
                                "reference": "instructed held-out EEG event; 2 s detection horizon"})
    return results
