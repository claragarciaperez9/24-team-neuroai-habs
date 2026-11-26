"""Read only CURATED channels and short trial slices, never whole video files."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib


# The release has 64 electrodes. Fix the same 62 inputs for every patient.
# F9/F10 are the two most lateral frontal electrodes; this is a design choice.
EEG_NAMES = tuple("Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T9 T7 C3 C4 T8 T10 "
                  "CP5 CP1 CP2 CP6 P9 P7 P3 Pz P4 P8 P10 O1 O2 AF7 AF3 AF4 AF8 "
                  "F5 F1 F2 F6 FC3 FCz FC4 C5 C1 C2 C6 CP3 CP4 P5 P1 P2 P6 "
                  "PO5 PO3 PO4 PO6 FT7 FT8 TP7 TP8 PO7 PO8 POz".split())
VOLT_TO_UV = {"V": 1e6, "mV": 1e3, "uV": 1., "µV": 1., "μV": 1., "nV": .001}


def sex_group(value):
    """Use the recorded gender field as a proxy; do not infer biological sex."""
    return {"f": "women", "female": "women", "woman": "women", "women": "women", "femme": "women",
            "m": "men", "male": "men", "man": "men", "men": "men", "homme": "men"}.get(
                str(value).strip().lower(), "unknown")


def read_labels(root):
    root = Path(root)
    path = next((root / n for n in ("labels.csv", "labels_train.csv")
                 if (root / n).exists()), None)
    if path is None:
        raise FileNotFoundError("Need labels.csv or labels_train.csv at the dataset root.")
    labels = pd.read_csv(path, dtype=str)
    # Exclude the entire row, including sentinels outside the target column.
    labels = labels[~labels.eq("TESTSUBJECT").any(axis=1)].copy()
    score = pd.to_numeric(labels.rating_arousal, errors="coerce")
    flag = labels.flagged_outliers.str.strip().str.lower()
    labels = labels[flag.isin(["0", "0.0", "false"]) & score.between(1, 9)].copy()
    score = pd.to_numeric(labels.rating_arousal)
    labels["target"] = np.where(score < 3.5, 0, np.where(score > 6.5, 2, 1))
    labels["day"] = labels.day.str.extract(r"(\d+)")[0].astype(int)
    labels["start_marker"] = pd.to_numeric(labels.start_marker).astype(int)
    labels["end_marker"] = pd.to_numeric(labels.end_marker).astype(int)
    if "gender" not in labels:
        labels["gender"] = "unknown"
    labels["sex"] = labels.gender.map(sex_group)
    labels["video"] = labels.video.map(lambda v: str(v).replace("\\", "/").split("/")[-1])
    return labels


def marker_lookup(root):
    path = Path(root) / "markers.csv"
    if not path.exists():
        # The bundled lookup is the release's codebook, not event timestamps.
        path = Path(__file__).parents[1] / "starting_notebooks" / "markers.csv"
    table = pd.read_csv(path)
    return {int(r["Start Marker"]): (int(r["End Marker"]), r["Emotion or task"])
            for _, r in table.iterrows()}


@dataclass
class Trial:
    id: str
    participant: str
    day: int
    video: str
    sex: str
    target: int
    bdf: Path
    rest: Path
    metadata: Path
    onset: float
    duration: float


def find_trials(root):
    """Join participant + day + marker + video; reject ambiguous joins."""
    root = Path(root)
    labels, lookup = read_labels(root), marker_lookup(root)
    trials, skipped = [], []
    for path in sorted(root.glob("CURATED/P*/*_metadata.json")):
        meta = json.loads(path.read_text())
        export = meta["dataset_export"]
        pid = export["participant_id"]
        day = int("".join(c for c in str(export["day"]) if c.isdigit()))
        segments = export["segments"]
        if "emotion-video" not in segments or "resting-state" not in segments:
            continue
        seg, rest = segments["emotion-video"], segments["resting-state"]
        bdf, restpath = path.parent / seg["file"], path.parent / rest["file"]
        if not bdf.exists() or not restpath.exists():
            skipped.append({"session": path.stem, "reason": "missing video/rest BDF"})
            continue
        if rest["metadata_time_origin_s"] + rest["duration_s"] > min(
                e["onset_s"] for e in meta["events"] if e.get("media_file")):
            raise ValueError(f"Rest is not pre-video: {path}")
        rows = labels[(labels.participant_id == pid) & (labels.day == day)]
        for i, event in enumerate(meta["events"]):
            if event.get("trial_type") != "emotion_video":
                continue
            marker = int(event["marker_id"])
            video = Path(str(event.get("media_file"))).name
            matches = rows[(rows.start_marker == marker) & (rows.video == video)]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous repeated trial: {pid}, day {day}, marker {marker}")
            if matches.empty:
                continue
            row = matches.iloc[0]
            if marker not in lookup or lookup[marker][0] != row.end_marker:
                raise ValueError(f"Marker pair disagrees with markers.csv: {marker}")
            onset = float(event["onset_s"]) - float(seg["metadata_time_origin_s"])
            duration = float(event["duration_s"])
            # A segment can omit the first Prepare screen. Never pad it with ratings.
            if onset < 5 or onset + min(10., duration) > float(seg["duration_s"]) + .002:
                skipped.append({"session": path.stem, "event": i,
                                "reason": "baseline/response outside segment"})
                continue
            trials.append(Trial(f"{pid}-d{day}-e{i}", pid, day, video, row.sex,
                                int(row.target), bdf, restpath, path, onset, duration))
    return trials, skipped


class Recording:
    """Native-rate selected signals in physical units; record_map is mandatory."""

    def __init__(self, path):
        self.path = Path(path)
        mapping = json.loads(self.path.with_suffix(".record_map").read_text())
        self.reader = pyedflib.EdfReader(str(self.path))
        try:
            labels = self.reader.getSignalLabels()
            groups = {}
            for stream in mapping["stream_map"].values():
                for spec in stream.values():
                    correction = spec.get("correction", {})
                    if "Channels" not in correction:
                        continue
                    ids = spec.get("channel_indices")
                    if ids is None:
                        a, b = spec["channel_ranges"]
                        ids = list(range(a, b + 1))
                    for index, detail in zip(ids, correction["Channels"], strict=True):
                        name = detail["name"]
                        if not (labels[index] == name or
                                (len(labels[index]) == 16 and name.startswith(labels[index]))):
                            raise ValueError(f"record_map mismatch: {name}")
                        kind = correction.get("RecordType", "").upper()
                        groups.setdefault(kind, {})[name] = index
            self.indices = {"eeg": [groups.get("EEG", {}).get(n) for n in EEG_NAMES]}
            for key, kind in (("ecg", "ECG"), ("eda", "GSR"), ("resp", "RESPIRATION")):
                # Some maps call GSR SKIN_CONDUCTANCE.
                candidates = groups.get(kind, groups.get("SKIN_CONDUCTANCE", {}) if key == "eda" else {})
                self.indices[key] = list(candidates.values())[:1]
            chosen = [i for i in sum(self.indices.values(), []) if i is not None]
            if not chosen:
                raise ValueError("No supported modalities in record_map")
            rates = {self.reader.getSampleFrequency(i) for i in chosen}
            if len(rates) != 1 or next(iter(rates)) % 250:
                raise ValueError(f"Need a shared integer multiple of 250 Hz, got {rates}")
            self.fs = int(next(iter(rates)))
            self.seconds = min(self.reader.getNSamples()[i] / self.fs for i in chosen)
            # BDF pads to whole records. Padding is not measured physiology.
            self.seconds = min(self.seconds, int(mapping["TotalSamples"]) / self.fs)
        except Exception:
            self.reader.close()
            raise

    def read(self, start, end):
        if start < 0 or end > self.seconds + 1 / self.fs:
            raise ValueError("Requested interval outside the recording")
        first, stop = round(start * self.fs), round(end * self.fs)
        result = {}
        for kind, indices in self.indices.items():
            values = []
            for index in indices:
                if index is None:
                    values.append(np.full(stop - first, np.nan))
                    continue
                x = self.reader.readSignal(index, first, stop - first)
                unit = self.reader.getPhysicalDimension(index).strip()
                if kind in ("eeg", "ecg"):
                    x = x * VOLT_TO_UV[unit]
                elif kind == "eda" and unit in VOLT_TO_UV:
                    volts = x * VOLT_TO_UV[unit] / 1e6
                    resistance = np.interp(volts, [-2.14, -1.99755, -1.59528, .0167, 1.70074, 2.00199],
                                           [1000, 500, 250, 125, 50, 25])
                    x = 1000 / resistance
                elif kind == "eda" and unit not in ("uS", "µS", "μS"):
                    raise ValueError(f"Unknown EDA unit: {unit}")
                values.append(x)
            result[kind] = np.vstack(values) if values else np.full((1, stop - first), np.nan)
        return result

    def close(self):
        self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
