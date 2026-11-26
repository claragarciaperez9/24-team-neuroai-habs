"""Small independent quality classifiers with explicit weak-label provenance."""

from dataclasses import dataclass
from functools import lru_cache
import json

import numpy as np
import pandas as pd
from scipy.signal import welch

from . import MODALITIES
from .data import EEG_NAMES, Recording, marker_lookup
from .signals import RestState


FAULTS = ("flatline", "saturation", "noise_burst", "electrode_pop", "dropout")
QUALITY_NAMES = ("log_variance", "line_ratio", "high_ratio", "correlation", "flatness",
                 "log_range", "log_jump", "nonfinite", "clipping")


@lru_cache(maxsize=64)
def nearest_channels(names):
    """Three nearest available electrodes in the standard 10-05 montage."""
    import mne
    positions = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    xyz = np.stack([positions[name] for name in names])
    distance = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    np.fill_diagonal(distance, np.inf)
    return distance.argsort(axis=1)[:, :min(3, len(names) - 1)]


def quality_features(x, fs, channel_names=None):
    """Use raw 1 s data, before band-pass/notch can hide line noise or flatlines."""
    if x.shape[0] == 0:
        # No usable EEG channels in rest: force this modality to be unavailable.
        return np.asarray([0, 0, 0, 0, 1, 0, 0, 1, 1], np.float32)
    missing = 1 - np.isfinite(x).mean()
    x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    variance = np.var(x, axis=1)
    f, psd = welch(x, fs=fs, nperseg=min(fs, x.shape[1]), axis=1)
    power = psd.sum(axis=1) + 1e-12
    line = psd[:, (f >= 48) & (f <= 52)].sum(axis=1) / power
    high = psd[:, (f > 30) & (f < min(100, fs / 2))].sum(axis=1) / power
    if len(x) > 1:
        centred = x - x.mean(axis=1, keepdims=True)
        normalized = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
        if channel_names is not None:
            if len(channel_names) != len(x):
                raise ValueError("Channel positions and quality data disagree")
            neighbours = nearest_channels(tuple(channel_names))
            correlations = normalized @ normalized.T
            corr = np.median(correlations[np.arange(len(x))[:, None], neighbours].mean(axis=1))
        else:
            # Generic synthetic/reference arrays can omit electrode names.
            reference = np.median(x, axis=0)
            ref = reference - reference.mean()
            corr = np.median(normalized @ ref / max(np.linalg.norm(ref), 1e-12))
    else:
        corr = 1.
    differences = np.abs(np.diff(x, axis=1))
    flat = np.max(np.mean(differences < 1e-8, axis=1))
    span = np.ptp(x, axis=1)
    extrema = ((x == x.max(axis=1, keepdims=True)) |
               (x == x.min(axis=1, keepdims=True))).mean(axis=1).max()
    return np.asarray([np.log1p(np.median(variance)), line.max(), high.max(), corr, flat,
                       np.log1p(span.max()), np.log1p(differences.max()), missing, extrema], np.float32)


def rule_label(x, fs, modality):
    """Conservative weak labels, not expert annotations or an ICLabel claim."""
    q = quality_features(x, fs)
    if q[7] > 0:
        return "dropout"
    if q[4] > .98:
        return "flatline"
    if q[8] > .15:
        return "saturation"
    if modality == "eeg":
        if q[1] > .3:
            return "line_noise"
        if q[2] > .6:
            return "muscle"
        if np.ptp(x[:2], axis=1).max() > 250:
            return "blink"
    return "clean"


def inject_fault(x, kind, fs, rng):
    """Deterministic via caller RNG; corrupt a full one-second region."""
    y = x.copy()
    scale = max(float(np.nanstd(x)), .1)
    if kind == "flatline":
        y[:] = np.nanmean(x)
    elif kind == "saturation":
        y = np.clip(y, np.nanmean(x) - .15 * scale, np.nanmean(x) + .15 * scale)
    elif kind == "noise_burst":
        y += rng.normal(0, 12 * scale, y.shape)
    elif kind == "electrode_pop":
        y += 20 * scale * np.exp(-np.arange(y.shape[1]) / (.08 * fs))[None, :]
    elif kind == "dropout":
        y[:] = np.nan
    else:
        raise ValueError(kind)
    return y


def calibration_rows(root, metadata_paths, seed=7):
    """Verify instructed markers against the release lookup, using local onsets."""
    lookup, rng, records = marker_lookup(root), np.random.default_rng(seed), []
    for path in sorted(set(metadata_paths)):
        meta = json.loads(path.read_text())
        export = meta["dataset_export"]
        pid = export["participant_id"]
        restseg = export["segments"].get("resting-state")
        if restseg is None or not (path.parent / restseg["file"]).exists():
            continue
        with Recording(path.parent / restseg["file"]) as recording:
            good = RestState.fit(recording.read(0, recording.seconds), recording.fs).good
        for task in ("resting-state", "artefact"):
            seg = export["segments"].get(task)
            if not seg or not (path.parent / seg["file"]).exists():
                continue
            with Recording(path.parent / seg["file"]) as recording:
                for event in meta["events"]:
                    marker = int(event["marker_id"])
                    if marker not in range(10000, 10020, 2):
                        continue
                    if marker not in lookup or lookup[marker][0] != marker + 1:
                        raise ValueError(f"Unverified artefact marker {marker}")
                    onset = event["onset_s"] - seg["metadata_time_origin_s"]
                    # Skip boundaries/startup; all labels cover full 1 s windows.
                    a = max(2., onset + 1.)
                    b = min(recording.seconds, onset + event["duration_s"] - .1)
                    if onset < -.002 or b <= a:
                        continue
                    for start in np.arange(a, b - 1., 2.):
                        block = recording.read(start, start + 1)
                        for modality in MODALITIES:
                            values = block[modality]
                            if modality == "eeg":
                                values = values[good]
                            label = rule_label(values, recording.fs, modality)
                            source = "rule"
                            if task == "artefact" and marker >= 10006 and modality == "eeg":
                                label, source = lookup[marker][1], "instructed"
                            def add(v, label, source):
                                records.append({"participant": pid, "modality": modality,
                                                "type": label, "target": int(label != "clean"),
                                                "source": source, "metadata": str(path),
                                                "start": start, "features": quality_features(v, recording.fs,
                                                    tuple(np.asarray(EEG_NAMES)[good]) if modality == "eeg" else None)})
                            add(values, label, source)
                            # Inject only into weak-clean reference windows.
                            if label == "clean":
                                for fault in FAULTS:
                                    add(inject_fault(values, fault, recording.fs, rng), fault, "injected")
    return records


@dataclass
class QualityGate:
    classifiers: dict
    threshold: float = .5
    min_clean: int = 2

    @classmethod
    def fit(cls, rows, participants):
        from lightgbm import LGBMClassifier
        models = {}
        for modality in MODALITIES:
            subset = [r for r in rows if r["participant"] in participants and r["modality"] == modality]
            y = np.asarray([r["target"] for r in subset])
            if len(np.unique(y)) != 2:
                raise ValueError(f"Quality training needs clean and artefact {modality} examples")
            models[modality] = LGBMClassifier(n_estimators=60, num_leaves=7, max_depth=3,
                                            min_child_samples=10, class_weight="balanced",
                                            verbosity=-1, n_jobs=1, random_state=7).fit(
                pd.DataFrame(np.stack([r["features"] for r in subset]), columns=QUALITY_NAMES), y)
        return cls(models)

    def clean(self, quality):
        return self.clean_many(quality[None])[0]

    def clean_many(self, quality):
        """Batch the four small trees to keep per-step inference inexpensive."""
        return np.stack([(self.classifiers[k].predict_proba(
            pd.DataFrame(quality[:, i], columns=QUALITY_NAMES))[:, 1] < self.threshold)
                         & (quality[:, i, 7] == 0) & (quality[:, i, 4] < .98)
                         for i, k in enumerate(MODALITIES)], axis=1)

    def evaluate(self, rows, participants):
        from sklearn.metrics import precision_score, recall_score
        results = []
        for modality in MODALITIES:
            subset = [r for r in rows if r["participant"] in participants and r["modality"] == modality]
            if not subset:
                continue
            x = np.stack([r["features"] for r in subset])
            predictions = self.classifiers[modality].predict_proba(pd.DataFrame(x, columns=QUALITY_NAMES))[:, 1] >= self.threshold
            predictions |= (x[:, 7] > 0) | (x[:, 4] >= .98)
            labels = np.array([r["target"] for r in subset])
            types = np.array([r["type"] for r in subset])
            # Precision against a clean comparison set for each artefact type.
            for kind in sorted(set(types) - {"clean"}):
                select = (types == kind) | (types == "clean")
                results.append({"modality": modality, "type": kind,
                                "precision": precision_score(labels[select], predictions[select], zero_division=0)
                                if (types == "clean").any() else None,
                                "recall": recall_score(labels[select], predictions[select], zero_division=0),
                                "n_artifact": int(np.sum(types == kind)),
                                "clean_reference_windows": int(np.sum(types == "clean")),
                                "label_source": "weak/injected"})
            clean = labels == 0
            results.append({"modality": modality, "type": "clean_false_alarms",
                            "false_positive_windows_per_hour": float(predictions[clean].mean() * 3600) if clean.any() else None,
                            "label_source": "weak/injected"})
        return results
