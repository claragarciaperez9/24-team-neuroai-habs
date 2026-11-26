"""Stateful SOS filters. Offline and live paths call exactly the same code."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from . import MODALITIES


class SOS:
    def __init__(self, coefficients, channels):
        self.sos = np.asarray(coefficients)
        self.state = np.zeros((len(self.sos), channels, 2))
        self.initialized = False

    def push(self, x):
        if x.shape[1] == 0:
            return x.copy()
        if not self.initialized:
            # The first available sample is causal and avoids a fictitious EDA SCR
            # caused by ramping a constant baseline up from zero filter state.
            self.state = sosfilt_zi(self.sos)[:, None, :] * x[:, 0][None, :, None]
            self.initialized = True
        y, self.state = sosfilt(self.sos, x, axis=1, zi=self.state)
        return y


class Filters:
    """Filter before integer decimation; preserve phase across arbitrary chunks."""

    def __init__(self, fs, good):
        if fs < 250 or fs % 250:
            raise ValueError("Input rate must be an integer multiple of 250 Hz")
        self.factor, self.position = fs // 250, 0
        self.good = np.asarray(good, bool)
        notch = tf2sos(*iirnotch(50, 30, fs=fs))
        self.filters = {
            "eeg": SOS(np.vstack([butter(4, [.5, 45], btype="bandpass", fs=fs, output="sos"), notch]), len(good)),
            "ecg": SOS(butter(4, [5, 35], btype="bandpass", fs=fs, output="sos"), 1),
            "eda": SOS(butter(2, 3, fs=fs, output="sos"), 1),
            "resp": SOS(butter(2, [.08, .7], btype="bandpass", fs=fs, output="sos"), 1),
        }
        self.tonic = SOS(butter(2, .05, fs=250, output="sos"), 1)

    def push(self, raw):
        n = raw["eeg"].shape[1]
        if any(raw[k].shape[1] != n for k in MODALITIES):
            raise ValueError("Modalities must share sample timestamps")
        choose = (np.arange(n) + self.position) % self.factor == 0
        self.position += n
        out = {k: filt.push(np.nan_to_num(raw[k], nan=0., posinf=0., neginf=0.))[:, choose]
               for k, filt in self.filters.items()}
        eeg = out["eeg"]
        if self.good.any():
            eeg -= eeg[self.good].mean(axis=0, keepdims=True)
        eeg[~self.good] = 0
        out["tonic"] = self.tonic.push(out["eda"])
        out["phasic"] = out["eda"] - out["tonic"]
        return out


def exponential_stats(x, alpha=.001):
    """Estimate EMA mean/variance on rest; freeze both before the video starts."""
    mean, variance = x[:, 0].copy(), np.ones(x.shape[0])
    for sample in x.T:
        delta = sample - mean
        mean += alpha * delta
        variance = (1 - alpha) * (variance + alpha * delta * delta)
    return mean[:, None], np.sqrt(np.maximum(variance, 1e-8))[:, None]


@dataclass
class RestState:
    good: np.ndarray
    mean: dict
    scale: dict

    @classmethod
    def fit(cls, raw, fs):
        # Rest-only bad-channel selection, in microvolts. No video information.
        eeg = raw["eeg"]
        finite = np.isfinite(eeg).all(axis=1)
        filled = np.nan_to_num(eeg, nan=0., posinf=0., neginf=0.)
        spread = np.std(filled, axis=1)
        diff = np.std(np.diff(filled, axis=1), axis=1)
        median = max(float(np.median(diff[finite])) if finite.any() else 0., 1e-6)
        good = (np.isfinite(eeg).all(axis=1) & (spread > .1) &
                (diff > .01) & (diff < max(5 * median, 20)))
        if good.sum() < len(good) // 2:
            # EEG is unavailable, but valid physiology may still support inference.
            good[:] = False
        processed = Filters(fs, good).push(raw)
        # Exclude filter startup; estimate the exponential standardisation on rest.
        mean, scale = {}, {}
        for kind, values in processed.items():
            mean[kind], scale[kind] = exponential_stats(values[:, min(1250, values.shape[1] // 2):])
        return cls(good, mean, scale)


class Preprocessor:
    def __init__(self, fs, rest):
        self.rest = rest
        self.filters = Filters(fs, rest.good)

    def push(self, raw):
        physical = self.filters.push(raw)
        eeg = (physical["eeg"] - self.rest.mean["eeg"]) / self.rest.scale["eeg"]
        eeg[~self.rest.good] = 0
        # Physiology stays in physical units for HR and SCR thresholds.
        physical["eeg"] = eeg
        return physical


def causal_peaks(x, threshold, refractory, fs=250):
    """Confirm a local maximum only after its next sample arrives (4 ms delay)."""
    peaks, previous = [], -10**9
    for i in range(1, len(x) - 1):
        if x[i] > threshold and x[i] > x[i - 1] and x[i] >= x[i + 1] and i - previous >= refractory * fs:
            peaks.append(i)
            previous = i
    return np.asarray(peaks, dtype=int)


def summaries(x):
    return [float(np.mean(x)), float(np.std(x)), float(np.ptp(x)),
            float(np.mean(np.abs(np.diff(x)))) if len(x) > 1 else 0.]


def physiology_features(processed, end, rest):
    """25 features. Baseline [-5,0), response [0,t); no post-t samples.

    HR/RMSSD on short clips can be unavailable: emit zero plus availability bits.
    The ECG detector threshold/polarity is fixed from rest, not a future maximum.
    """
    fs, zero = 250, 1250
    def ranges(kind):
        return processed[kind][0, :zero], processed[kind][0, zero:end]

    base, response = ranges("eda")
    bton, tonic = ranges("tonic")
    _, phasic = ranges("phasic")
    # A SCR is a positive phasic peak above .05 uS, with a 1 s refractory period.
    scr = causal_peaks(phasic, .05, 1.)
    eda = [np.mean(response) - np.mean(base), np.mean(tonic) - np.mean(bton),
           np.std(response), np.max(phasic), np.mean(np.maximum(phasic, 0)),
           len(scr) / (len(response) / fs), float(np.mean(phasic[scr])) if len(scr) else 0.,
           (tonic[-1] - tonic[0]) / max(len(tonic) / fs, 1 / fs)]
    # Keep peak detection in uS, then scale amplitude features by patient rest.
    for index in (0, 1, 2, 3, 4, 6, 7):
        eda[index] /= max(float(rest.scale["eda"][0, 0]), .01)

    base, response = ranges("ecg")
    # Rectification handles inverted lead polarity. Threshold learned on rest.
    threshold = max(2. * float(rest.scale["ecg"][0, 0]), .1)
    ecg_past = processed["ecg"][0, :end]
    peaks = causal_peaks(np.abs(ecg_past), threshold, .3)
    rr = np.diff(peaks) / fs
    # Reject ectopic/out-of-range intervals; never bridge an invalid interval for RMSSD.
    valid = (rr >= .3) & (rr <= 2.)
    baseline = (peaks[1:] < zero) & valid
    current = (peaks[:-1] >= zero) & valid
    brr, rrr = rr[baseline], rr[current]
    hr = 60 / rrr if len(rrr) else np.array([])
    bhr = 60 / brr if len(brr) else np.array([])
    adjacent = current[1:] & current[:-1]
    drr = np.diff(rr)[adjacent]
    ecg = [float(hr.mean() - bhr.mean()) if len(hr) and len(bhr) else 0.,
           float(hr.mean()) if len(hr) else 0., float(hr.std()) if len(hr) else 0.,
           float(np.sqrt(np.mean(drr ** 2)) * 1000) if len(drr) else 0.,
           float(np.std(rrr) * 1000) if len(rrr) > 1 else 0.,
           float(np.sum(peaks >= zero) / (len(response) / fs)),
           float(len(hr) > 0), float(len(bhr) > 0), float(len(drr) > 0)]

    base, response = ranges("resp")
    # Breath peaks use a rest-derived threshold, with no look-ahead beyond t.
    threshold = .2 * float(rest.scale["resp"][0, 0])
    def rate(x):
        peaks = causal_peaks(x, threshold, 1.5)
        return float(60 / np.mean(np.diff(peaks) / fs)) if len(peaks) >= 2 else 0.
    brate, rrate = rate(base), rate(response)
    base = (base - rest.mean["resp"][0, 0]) / rest.scale["resp"][0, 0]
    response = (response - rest.mean["resp"][0, 0]) / rest.scale["resp"][0, 0]
    resp = [rrate - brate, rrate, np.std(response) - np.std(base),
            np.ptp(response) - np.ptp(base), *summaries(response)]
    return np.nan_to_num(np.asarray([*eda, *ecg, *resp], np.float32))


FEATURE_SLICES = {"eda": slice(0, 8), "ecg": slice(8, 17), "resp": slice(17, 25)}
