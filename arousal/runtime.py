"""Shared streaming window builder and model inference for a single new trial."""

from time import perf_counter

import numpy as np

from . import MODALITIES
from .alarm import Alarm
from .data import EEG_NAMES
from .quality import quality_features
from .signals import FEATURE_SLICES, Preprocessor, physiology_features


class TrialStream:
    """Input starts exactly at -5 s. No access to earlier ratings or later samples.

    Keep at most 15 s. The raw 1 s quality checks precede model prediction.
    A physiology modality is masked if its baseline or response contains a fault.
    EEG needs only its last 2 s to be clean; bad rest channels stay zero.
    """

    def __init__(self, fs, rest, gate=None, duration=10., t_start=0.):
        self.fs, self.rest, self.gate = fs, rest, gate
        self.duration, self.t_start = min(10., duration), t_start
        self.preprocessor = Preprocessor(fs, rest)
        self.raw = {k: np.empty((len(rest.good) if k == "eeg" else 1, 0)) for k in MODALITIES}
        self.processed = {}
        self.received, self.next_boundary = 0, fs // 2
        self.quality_history = []
        self.quality_times = []

    def push(self, block):
        """Accept arbitrary-size chunks, emitting every prediction due in the chunk."""
        n = block["eeg"].shape[1]
        if any(block[k].shape[1] != n for k in MODALITIES):
            raise ValueError("Inconsistent chunk lengths")
        if self.received + n > round((5 + self.duration) * self.fs):
            raise ValueError("Chunk extends beyond video offset")
        outputs, offset = [], 0
        while offset < n:
            count = min(n - offset, self.next_boundary - self.received)
            piece = {k: block[k][:, offset:offset + count] for k in MODALITIES}
            for k in MODALITIES:
                self.raw[k] = np.concatenate([self.raw[k], piece[k]], axis=1)
            processed = self.preprocessor.push(piece)
            for k, values in processed.items():
                self.processed[k] = np.concatenate([self.processed.get(k, values[:, :0]), values], axis=1)
            self.received += count
            offset += count
            if self.received == self.next_boundary:
                t = self.received / self.fs - 5
                if self.received >= self.fs:
                    quality = np.stack([quality_features(
                        self.raw[k][self.rest.good, -self.fs:] if k == "eeg" else self.raw[k][:, -self.fs:], self.fs,
                        tuple(np.asarray(EEG_NAMES)[self.rest.good]) if k == "eeg" else None)
                                        for k in MODALITIES])
                    self.quality_history.append(quality)
                    self.quality_times.append(t)
                if 1. <= t <= self.duration:
                    quality = np.stack(self.quality_history)
                    times = np.asarray(self.quality_times)
                    mask = np.ones(4, dtype=bool)
                    if self.gate is not None:
                        clean = self.gate.clean_many(quality)
                        mask = clean.all(axis=0)
                        # Three overlapping 1 s checks cover exactly [t-2,t).
                        mask[0] = clean[times >= t - 1., 0].all()
                    end = self.processed["eeg"].shape[1]
                    outputs.append({"time": t, "eeg": self.processed["eeg"][:, end-500:end].astype(np.float32),
                                    "physio": physiology_features(self.processed, end, self.rest),
                                    "mask": mask, "quality": quality, "quality_times": times})
                self.next_boundary += self.fs // 2
        return outputs


def mask_window(window, gate, gating=True):
    """Apply the fold's train-only quality models to cached quality features."""
    mask = np.ones(4, dtype=bool)
    if gating:
        clean = gate.clean_many(window["quality"])
        mask = clean.all(axis=0)
        mask[0] = clean[window["quality_times"] >= window["time"] - 1., 0].all()
    eeg, physio = window["eeg"].copy(), window["physio"].copy()
    if not mask[0]:
        eeg[:] = 0
    for i, kind in enumerate(MODALITIES[1:], 1):
        if not mask[i]:
            physio[FEATURE_SLICES[kind]] = 0
    return eeg, physio, mask


class Predictor:
    def __init__(self, model, gate, scaler, temperature, alarm_config, t_start=0.):
        self.model, self.gate, self.scaler = model, gate, scaler
        self.temperature, self.alarm_config, self.t_start = temperature, alarm_config, t_start

    def start(self, fs, rest, duration=10.):
        self.stream = TrialStream(fs, rest, self.gate, duration, self.t_start)
        self.alarm = Alarm(self.alarm_config)

    def predict_window(self, window, gating=True):
        import torch
        started = perf_counter()
        eeg, physio, mask = mask_window(window, self.gate, gating)
        reliable = mask.sum() >= self.gate.min_clean and window["time"] >= self.t_start
        result = {"time": window["time"], "reliable": bool(reliable),
                  "status": "ok" if reliable else "unreliable", "mask": mask.tolist(),
                  "p_low": None, "p_mid": None, "p_high": None}
        if reliable:
            # Scale then mask again: a missing modality must be exactly zero.
            physio = (physio - self.scaler[0]) / self.scaler[1]
            for i, kind in enumerate(MODALITIES[1:], 1):
                if not mask[i]:
                    physio[FEATURE_SLICES[kind]] = 0
            device = next(self.model.parameters()).device
            with torch.inference_mode():
                logits = self.model(torch.as_tensor(eeg[None], device=device),
                                    torch.as_tensor(physio[None], dtype=torch.float32, device=device),
                                    torch.tensor([[window["time"] / 10]], device=device),
                                    torch.as_tensor(mask[None], dtype=torch.float32, device=device))
                probabilities = (logits / self.temperature).softmax(-1)[0].double().cpu().numpy()
                probabilities = (probabilities / probabilities.sum()).tolist()
            result.update(zip(("p_low", "p_mid", "p_high"), probabilities))
        active, onset = self.alarm.update(result["p_high"] or 0., reliable)
        result.update(alarm=active, alarm_onset=onset, latency_ms=1000 * (perf_counter() - started))
        return result

    def push(self, block):
        started = perf_counter()
        windows = self.stream.push(block)
        preprocess_ms = 1000 * (perf_counter() - started)
        results = [self.predict_window(w) for w in windows]
        for result in results:
            result["latency_ms"] += preprocess_ms / max(len(results), 1)
        return results
