"""Tests target future leakage, arbitrary chunking and artifact masking."""

import numpy as np
import pytest

from arousal import MODALITIES
from arousal.alarm import Alarm, AlarmConfig
from arousal.runtime import TrialStream, mask_window
from arousal.signals import Filters, Preprocessor, RestState


def signals(seconds=15, fs=1000, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * fs)) / fs
    eeg = 10 * np.sin(2*np.pi*10*t)[None] + rng.normal(0, 2, (62, len(t)))
    return {"eeg": eeg, "ecg": (100*np.sin(2*np.pi*1.2*t))[None],
            "eda": (2 + .1*np.sin(2*np.pi*.1*t))[None],
            "resp": (np.sin(2*np.pi*.25*t))[None]}


@pytest.fixture(scope="module")
def rest():
    return RestState.fit(signals(12, seed=8), 1000)


def test_offline_equals_arbitrary_streaming_chunks(rest):
    raw = signals()
    offline = Preprocessor(1000, rest).push(raw)
    streaming = Preprocessor(1000, rest)
    pieces = {k: [] for k in offline}
    edges = [0, 1, 129, 503, 1002, 2345, 4500, 11111, 15000]
    for a, b in zip(edges[:-1], edges[1:]):
        for k, values in streaming.push({k: v[:, a:b] for k, v in raw.items()}).items():
            pieces[k].append(values)
    for k in offline:
        np.testing.assert_array_equal(offline[k], np.concatenate(pieces[k], axis=1))


def test_window_features_offline_equals_streaming_and_shapes(rest):
    raw = signals()
    offline = TrialStream(1000, rest).push(raw)
    live = TrialStream(1000, rest)
    streaming = []
    for a in range(0, 15000, 137):
        streaming.extend(live.push({k: v[:, a:a+137] for k, v in raw.items()}))
    assert [w["time"] for w in offline] == list(np.arange(1, 10.01, .5))
    assert len(offline) == len(streaming) == 19
    for a, b in zip(offline, streaming):
        assert a["eeg"].shape == (62, 500)
        assert a["physio"].shape == (25,)
        for k in ("eeg", "physio", "quality"):
            np.testing.assert_array_equal(a[k], b[k])


def test_future_cannot_change_past(rest):
    raw, changed = signals(), signals()
    for k in MODALITIES:
        changed[k][:, 8500:] = 1e6
    a = TrialStream(1000, rest).push(raw)
    b = TrialStream(1000, rest).push(changed)
    for x, y in zip(a, b):
        if x["time"] <= 3.5:
            for key in ("eeg", "physio", "quality"):
                np.testing.assert_array_equal(x[key], y[key])


def test_reject_samples_after_video_offset(rest):
    with pytest.raises(ValueError, match="offset"):
        TrialStream(1000, rest, duration=1).push(signals(7))


class MissingGate:
    min_clean = 2
    def clean_many(self, q):
        return q[:, :, 7] == 0


def test_faults_mask_entire_physiological_context_and_eeg_recovers(rest):
    raw = signals()
    raw["eeg"][:, 8000:9000] = np.nan
    raw["ecg"][:, 8000:9000] = np.nan
    windows = TrialStream(1000, rest).push(raw)
    at4 = next(w for w in windows if w["time"] == 4)
    eeg, physio, mask = mask_window(at4, MissingGate())
    assert mask.tolist() == [False, False, True, True]
    assert np.all(eeg == 0) and np.all(physio[8:17] == 0)
    at8 = next(w for w in windows if w["time"] == 8)
    _, _, mask = mask_window(at8, MissingGate())
    assert mask.tolist() == [True, False, True, True]


def test_bad_rest_channels_excluded_from_common_average():
    x = signals(1)
    x["eeg"][0] = 1e9
    good = np.ones(62, bool); good[0] = False
    output = Filters(1000, good).push(x)["eeg"]
    assert np.all(output[0] == 0)
    np.testing.assert_allclose(output[good].mean(0), 0, atol=1e-12)


def test_constant_eda_has_no_filter_startup_scr():
    raw = signals(6)
    raw["eda"][:] = 2.
    processed = Filters(1000, np.ones(62, bool)).push(raw)
    np.testing.assert_allclose(processed["eda"], 2., atol=1e-12)
    np.testing.assert_allclose(processed["phasic"], 0., atol=1e-9)


def test_unusable_rest_eeg_is_masked_without_disabling_good_physiology():
    baseline = signals(12)
    baseline["eeg"][:] = np.nan
    rest = RestState.fit(baseline, 1000)
    assert not rest.good.any()
    window = TrialStream(1000, rest).push(signals())[-1]
    eeg, _, mask = mask_window(window, MissingGate())
    assert np.all(eeg == 0)
    assert mask.tolist() == [False, True, True, True]


def test_unreliable_resets_alarm_and_hysteresis():
    alarm = Alarm(AlarmConfig(threshold=.6, duration=1., tau=.01))
    assert alarm.update(.9) == (False, False)
    assert alarm.update(.9) == (True, True)
    assert alarm.update(.55) == (True, False)
    assert alarm.update(.9, reliable=False) == (False, False)
    assert alarm.update(.9) == (False, False)
    assert alarm.update(.1) == (False, False)
