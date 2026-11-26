"""Replay one trial at wall-clock speed, optionally through a real LSL loopback."""

import json
from pathlib import Path
from time import perf_counter, sleep
import uuid

import numpy as np

from . import MODALITIES
from .data import EEG_NAMES, Recording
from .pipeline import load_predictor, write_json
from .signals import RestState


def replay(model_dir, bdf, rest_bdf, onset, out, duration=10., lsl=False, speed=1.):
    if onset < 5:
        raise ValueError("Replay needs the complete [-5,0) Prepare screen")
    if speed <= 0 or duration < 1:
        raise ValueError("Need speed > 0 and duration >= 1 s")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor(model_dir)
    with Recording(rest_bdf) as rest_recording:
        rest = RestState.fit(rest_recording.read(0, rest_recording.seconds), rest_recording.fs)
    rows, step_latencies, source, inlet, probabilities = [], [], None, None, None
    transport_failed = False
    with Recording(bdf) as recording:
        fs = recording.fs
        predictor.start(fs, rest, duration)
        if lsl:
            from pylsl import StreamInfo, StreamInlet, StreamOutlet, local_clock
            identifier = str(uuid.uuid4())
            info = StreamInfo("NeuroAI-Replay", "EEG", 65, fs, "double64", identifier)
            channels = info.desc().append_child("channels")
            for name in (*EEG_NAMES, "ECG", "EDA", "RESP"):
                channels.append_child("channel").append_child_value("label", name)
            info.desc().append_child_value("time_origin", "first sample is -5 s relative to video onset")
            info.desc().append_child_value("eeg_ecg_unit", "uV")
            info.desc().append_child_value("eda_unit", "uS")
            source = StreamOutlet(info, chunk_size=fs//2)
            # A real inlet consumes samples from the outlet; predictions use its data.
            inlet = StreamInlet(info, max_buflen=30, max_chunklen=fs//2)
            inlet.open_stream(timeout=10.)
            if not source.wait_for_consumers(5.):
                raise RuntimeError("No LSL replay consumer connected")
            probabilities = StreamOutlet(StreamInfo("NeuroAI-Arousal", "Arousal", 1, 2., "string", identifier + "-arousal"))
        started = perf_counter()
        with (out / "replay.jsonl").open("w") as handle:
            count = round((5 + min(10., duration)) * fs)
            for first in range(0, count, fs // 2):
                stop = min(first + fs // 2, count)
                # Time I/O, signal processing, quality, model and alarm together.
                step_started = perf_counter()
                block = recording.read(onset - 5 + first / fs, onset - 5 + stop / fs)
                if lsl:
                    values = np.concatenate([block[k] for k in MODALITIES], axis=0).T
                    received = []
                    if not transport_failed:
                        source.push_chunk(values.tolist(), local_clock())
                        deadline = perf_counter() + .35
                        while len(received) < len(values) and perf_counter() < deadline:
                            samples, _ = inlet.pull_chunk(timeout=.05, max_samples=len(values)-len(received))
                            received.extend(samples)
                        transport_failed = len(received) != len(values)
                    if transport_failed:
                        # Do not stitch later samples over a gap. Abstain for the
                        # remainder of the trial, preserving the scheduled output.
                        block = {k: np.full_like(v, np.nan) for k, v in block.items()}
                    else:
                        received = np.asarray(received).T
                        block = {"eeg": received[:62], "ecg": received[62:63],
                                 "eda": received[63:64], "resp": received[64:65]}
                results = predictor.push(block)
                elapsed = (perf_counter() - step_started) * 1000
                step_latencies.append(elapsed)
                for result in results:
                    result["step_latency_ms"] = elapsed
                    if transport_failed:
                        result["reason"] = "transport_dropout"
                    text = json.dumps(result)
                    handle.write(text + "\n")
                    handle.flush()
                    print(text, flush=True)
                    if probabilities:
                        probabilities.push_sample([text])
                    rows.append(result)
                remaining = (stop / fs) / speed - (perf_counter() - started)
                if remaining > 0:
                    sleep(min(remaining, .5 / speed))
        if inlet is not None:
            inlet.close_stream()
    write_json(out / "replay_latency.json", {"steps": len(step_latencies), "predictions": len(rows),
                                            "includes": "BDF I/O, LSL if enabled, preprocessing, quality, model, alarm",
                                            "median_ms": np.median(step_latencies),
                                            "p95_ms": np.quantile(step_latencies, .95),
                                            "max_ms": max(step_latencies),
                                            "all_steps_under_500ms": bool(max(step_latencies) < 500)})
    return rows
