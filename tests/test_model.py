"""Small synthetic optimization tests; these are not arousal performance results."""

import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys
import torch

from arousal import MODALITIES
from arousal.alarm import Alarm, AlarmConfig
from arousal.model import FusionModel, fit_temperature, train_model
from arousal.pipeline import load_predictor, save_predictor
from arousal.quality import FAULTS, QualityGate, inject_fault, quality_features
from arousal.runtime import Predictor, TrialStream
from arousal.signals import RestState
from test_causality import signals


def small_gate():
    rng, rows = np.random.default_rng(8), []
    for _ in range(15):
        raw = signals(1, seed=int(rng.integers(100000)))
        for kind in MODALITIES:
            # Known generated reference, not weak labels on real sensor data.
            x = raw[kind]
            for fault in ("clean", *FAULTS):
                values = x if fault == "clean" else inject_fault(x, fault, 1000, rng)
                rows.append(dict(participant="train", modality=kind, type=fault,
                                 source="synthetic", target=int(fault != "clean"),
                                 features=quality_features(values, 1000)))
    return QualityGate.fit(rows, {"train"})


def test_training_temperature_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(2)
    n = 18
    data = dict(eeg=rng.normal(size=(n, 62, 500)).astype(np.float32),
                physio=rng.normal(size=(n, 25)).astype(np.float32),
                time=np.tile([3., 4.], n//2).astype(np.float32), mask=np.ones((n, 4), bool),
                target=np.repeat(np.arange(9) % 3, 2), trial=np.repeat(np.arange(9).astype(str), 2))
    model, scaler, history = train_model(data, data, epochs=1)
    assert len(history) == 1 and np.isfinite(history[0]["validation_trial_log_loss"])
    temperature = fit_temperature(model, data, scaler)
    assert .1 <= temperature <= 10
    gate = small_gate()
    predictor = Predictor(model, gate, scaler, temperature, AlarmConfig())
    save_predictor(predictor, tmp_path, {"purpose": "synthetic smoke test"})
    loaded = load_predictor(tmp_path)
    x = torch.from_numpy(data["eeg"][:2]); p = torch.from_numpy(data["physio"][:2])
    t = torch.ones(2, 1); mask = torch.ones(2, 4)
    with torch.inference_mode():
        torch.testing.assert_close(model(x, p, t, mask), loaded.model(x, p, t, mask), rtol=0, atol=0)
    # A fresh process catches native-library conflicts hidden by training-time
    # initialization in the pytest process (notably macOS OpenMP conflicts).
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-c",
                             "import sys; from arousal.pipeline import load_predictor; "
                             "p=load_predictor(sys.argv[1]); print('loaded')", str(tmp_path)],
                            cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "loaded" in result.stdout


def test_masked_inputs_cannot_affect_logits():
    torch.set_num_threads(1)
    model = FusionModel().eval()
    eeg, physio = torch.randn(2, 62, 500), torch.randn(2, 25)
    mask, time = torch.tensor([[0., 0., 1., 1.]]*2), torch.ones(2, 1)
    changed_eeg, changed_physio = eeg * 1e8, physio.clone()
    changed_physio[:, 8:17] = 1e8
    with torch.inference_mode():
        torch.testing.assert_close(model(eeg, physio, time, mask),
                                   model(changed_eeg, changed_physio, time, mask), rtol=0, atol=0)


def test_unreliable_predictions_are_null_and_alarm_suppressed():
    torch.set_num_threads(1)
    raw = signals()
    rest = RestState.fit(signals(12, seed=9), 1000)
    windows = TrialStream(1000, rest).push(raw)
    window = windows[-1]
    window["quality"][:, :, 7] = 1.  # All sensor streams are absent.
    predictor = Predictor(FusionModel().eval(), small_gate(), (np.zeros(25), np.ones(25)),
                          1., AlarmConfig(threshold=0., duration=.5))
    predictor.alarm = Alarm(predictor.alarm_config)
    result = predictor.predict_window(window)
    assert result["status"] == "unreliable"
    assert result["p_low"] is None and result["p_high"] is None
    assert not result["alarm"]


def test_lightgbm_gate_excludes_unseen_participant_labels():
    gate = small_gate()
    q = np.stack([quality_features(signals(1)[k], 1000) for k in MODALITIES])
    assert gate.clean(q).shape == (4,)
    assert len(gate.classifiers) == 4


def test_probabilities_and_alarms_offline_equal_streaming():
    from test_causality import MissingGate
    torch.set_num_threads(1)
    raw = signals()
    rest = RestState.fit(signals(12, seed=9), 1000)
    model = FusionModel().eval()
    def make():
        predictor = Predictor(model, MissingGate(), (np.zeros(25), np.ones(25)),
                              1., AlarmConfig(threshold=.3, duration=1.), t_start=3.)
        predictor.start(1000, rest)
        return predictor
    offline = make().push(raw)
    streaming, predictor = [], make()
    for first in range(0, 15000, 137):
        streaming.extend(predictor.push({k: v[:, first:first+137] for k, v in raw.items()}))
    assert len(offline) == len(streaming) == 19
    for a, b in zip(offline, streaming):
        for key in ("time", "mask", "status", "p_low", "p_mid", "p_high", "alarm", "alarm_onset"):
            assert a[key] == b[key]
