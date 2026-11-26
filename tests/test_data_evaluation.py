import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arousal.alarm import AlarmConfig, alarm_metrics, choose_alarm
from arousal.data import find_trials, read_labels
from arousal.evaluation import assert_disjoint, crossed_splits, trial_metrics


def test_label_boundaries_and_exclusions(tmp_path):
    rows = []
    for i, rating in enumerate([1, 3.49, 3.5, 6.5, 6.51, 9, "TESTSUBJECT", 5]):
        rows.append(dict(participant_id=f"P{i}", day=1, video="a.mp4", gender="F",
                         start_marker=10100, end_marker=10101, rating_arousal=rating,
                         flagged_outliers=1 if i == 7 else 0))
    pd.DataFrame(rows).to_csv(tmp_path / "labels.csv", index=False)
    labels = read_labels(tmp_path)
    assert labels.target.tolist() == [0, 0, 1, 1, 2, 2]
    assert set(labels.sex) == {"women"}


def test_alignment_and_ambiguous_markers_fail(tmp_path):
    folder = tmp_path / "CURATED" / "P001"
    folder.mkdir(parents=True)
    for name in ("video.bdf", "rest.bdf"):
        (folder / name).touch()
    event = dict(marker_id=10100, onset_s=120., duration_s=10.,
                 trial_type="emotion_video", media_file="a.mp4")
    meta = {"events": [event], "dataset_export": {"participant_id": "P001", "day": "day1",
            "segments": {"resting-state": {"file": "rest.bdf", "metadata_time_origin_s": 0., "duration_s": 30.},
                         "emotion-video": {"file": "video.bdf", "metadata_time_origin_s": 100., "duration_s": 40.}}}}
    (folder / "P001_day-1_metadata.json").write_text(json.dumps(meta))
    labels = pd.DataFrame([dict(participant_id="P001", day=1, video="a.mp4", gender="M",
                               start_marker=10100, end_marker=10101, rating_arousal=7, flagged_outliers=0)])
    labels.to_csv(tmp_path / "labels.csv", index=False)
    trials, _ = find_trials(tmp_path)
    assert len(trials) == 1 and trials[0].onset == 20.
    pd.concat([labels, labels]).to_csv(tmp_path / "labels.csv", index=False)
    with pytest.raises(ValueError, match="Ambiguous"):
        find_trials(tmp_path)


def test_both_people_and_videos_unseen_and_each_trial_tested_once():
    table = pd.DataFrame([dict(id=f"{p}-{v}", participant=p, video=v)
                          for p in range(9) for v in range(9)])
    seen = []
    for _, train, test in crossed_splits(table, 3):
        assert not set(table.iloc[train].participant) & set(table.iloc[test].participant)
        assert not set(table.iloc[train].video) & set(table.iloc[test].video)
        seen.extend(test)
    assert sorted(seen) == list(range(len(table)))


def test_trial_metrics_aggregate_windows_and_expose_abstentions():
    rows = pd.DataFrame([dict(trial="a", target=0, reliable=True, p_low=.8, p_mid=.1, p_high=.1)]*19 +
                        [dict(trial="b", target=2, reliable=True, p_low=.1, p_mid=.1, p_high=.8),
                         dict(trial="c", target=1, reliable=False, p_low=None, p_mid=None, p_high=None)])
    metrics = trial_metrics(rows)
    assert metrics["trials"] == 3 and metrics["predicted_trials"] == 2
    assert metrics["trial_coverage"] == 2/3
    assert metrics["log_loss"] == pytest.approx(-np.log(.8))


def test_alarm_threshold_uses_validation_and_episode_exposure():
    rows = pd.DataFrame([dict(trial=str(target), target=target, time=t, reliable=True,
                             p_high=.95 if target == 2 else .05)
                         for target in (0, 2) for t in np.arange(1, 10.01, .5)])
    config, metrics = choose_alarm(rows, target_fah=0)
    assert metrics["false_alarms_per_hour"] == 0
    assert metrics["sensitivity"] == 1
    result = alarm_metrics(rows, AlarmConfig(threshold=.01, duration=.5, tau=.01))
    assert result["false_alarm_episodes"] == 1
    assert result["negative_exposure_s"] == 9.5


def test_real_sample_record_map_and_artifact_marker_alignment():
    from arousal.data import Recording
    from arousal.quality import calibration_rows
    root = Path(__file__).parents[1] / "sample"
    path = root / "CURATED/P001/P001_day-1_task-resting-state_multimodal.bdf"
    with Recording(path) as recording:
        block = recording.read(0, 1.)
        assert block["eeg"].shape == (62, 1000)
        assert block["eda"].min() > 0
        assert recording.seconds == pytest.approx(95.08)  # Excludes BDF padding.
    rows = calibration_rows(root, [path.parent / "P001_day-1_metadata.json"])
    instructed = [r for r in rows if r["source"] == "instructed"]
    assert {r["type"] for r in instructed} == {
        "Blink repeatedly", "Move eyes from right to left without moving the head",
        "Move eyes up and down without moving the head", "Turn head from left to right",
        "Tilt head up and down", "Clench jaw", "Swallow repeatedly"}
    # The first instructed artefact is at 2049.273 - 1974.273 = 75 s.
    assert min(r["start"] for r in instructed) == pytest.approx(76.)


def test_sample_entrypoint_writes_audit_without_fabricating_results(tmp_path):
    from process import process
    process(Path(__file__).parents[1] / "sample", tmp_path)
    audit = json.loads((tmp_path / "data_audit.json").read_text())
    assert audit["status"] == "not_ready"
    assert not (tmp_path / "metrics.csv").exists()
