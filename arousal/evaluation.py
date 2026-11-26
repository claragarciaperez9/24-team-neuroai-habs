"""Participant AND video holdouts; trial metrics and the requested four figures."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from .alarm import AlarmConfig, alarm_metrics


def crossed_splits(table, folds=3, seed=7):
    """Cross two independent GroupKFold assignments.

    Test = held-out people × held-out videos. Train excludes BOTH held-out axes.
    Mixed rectangles are deliberately discarded, not quietly used for training.
    Crossing all K² combinations gives every trial exactly one test assignment.
    """
    assignments = []
    for column in ("participant", "video"):
        assignment = np.full(len(table), -1)
        for fold, (_, test) in enumerate(GroupKFold(folds, shuffle=True, random_state=seed).split(
                table, groups=table[column])):
            assignment[test] = fold
        assignments.append(assignment)
    people, videos = assignments
    for p, v in product(range(folds), repeat=2):
        train = np.flatnonzero((people != p) & (videos != v))
        test = np.flatnonzero((people == p) & (videos == v))
        if len(train) and len(test):
            yield f"p{p}-v{v}", train, test


def train_validation_split(table, seed=7):
    folds = min(3, table.participant.nunique(), table.video.nunique())
    if folds < 2:
        raise ValueError("Need more participants/videos for disjoint validation")
    for _, train, validation in crossed_splits(table, folds, seed):
        if set(table.iloc[train].target) == {0, 1, 2} and set(table.iloc[validation].target) == {0, 1, 2}:
            return train, validation
    raise ValueError("No participant/video-disjoint train/validation split containing all classes")


def assert_disjoint(train, validation, test):
    for a, b in ((train, validation), (train, test), (validation, test)):
        for column in ("participant", "video", "id"):
            if set(a[column]) & set(b[column]):
                raise AssertionError(f"Split leaks {column}")


def ece(y, probabilities, bins=10):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    total = 0.
    for i in range(bins):
        selected = (confidence >= i / bins) & ((confidence < (i+1) / bins) if i < bins-1 else (confidence <= 1))
        if selected.any():
            total += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(total)


def trial_metrics(rows):
    """Average reliable window probabilities per trial; expose abstention coverage."""
    count = rows.trial.nunique()
    available = rows[rows.reliable].groupby("trial").agg(
        target=("target", "first"), p_low=("p_low", "mean"),
        p_mid=("p_mid", "mean"), p_high=("p_high", "mean"))
    result = {"trials": count, "predicted_trials": len(available),
              "trial_coverage": len(available) / count if count else 0.,
              "window_coverage": float(rows.reliable.mean()) if len(rows) else 0.}
    if len(available):
        y = available.target.to_numpy()
        p = available[["p_low", "p_mid", "p_high"]].to_numpy()
        p = p / p.sum(axis=1, keepdims=True)  # Remove float32 softmax rounding drift.
        result.update(macro_f1=f1_score(y, p.argmax(1), labels=[0, 1, 2], average="macro", zero_division=0),
                      balanced_accuracy=balanced_accuracy_score(y, p.argmax(1)),
                      log_loss=log_loss(y, p, labels=[0, 1, 2]), ece=ece(y, p),
                      represented_classes=int(len(np.unique(y))))
    return result


def write_reports(predictions, out):
    """Reports include coverage so an aggressive gate cannot hide rejected trials."""
    from matplotlib import pyplot as plt
    metrics, times, alarms = [], [], []
    for (variant, condition, gating), rows in predictions.groupby(["variant", "condition", "gating"]):
        for sex in ("overall", "women", "men", "unknown"):
            subset = rows if sex == "overall" else rows[rows.sex == sex]
            if not len(subset):
                continue
            metrics.append({"variant": variant, "condition": condition, "gating": gating,
                            "sex": sex, **trial_metrics(subset)})
        if condition == "clean" and gating:
            for time, subset in rows.groupby("time"):
                times.append({"variant": variant, "time": time, **trial_metrics(subset)})
        if variant == "global_all" and condition == "clean" and gating:
            for threshold in np.linspace(.2, 1., 41):
                alarms.append({"threshold": threshold,
                               **alarm_metrics(rows, AlarmConfig(threshold=float(threshold)))})
    metrics, times, alarms = pd.DataFrame(metrics), pd.DataFrame(times), pd.DataFrame(alarms)
    metrics.to_csv(out / "metrics.csv", index=False)
    times.to_csv(out / "performance_by_time.csv", index=False)
    alarms.to_csv(out / "alarm_tradeoff.csv", index=False)
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant, frame in times.groupby("variant"):
        ax.plot(frame.time, frame.macro_f1 if "macro_f1" in frame else np.full(len(frame), np.nan), label=variant)
    ax.axhline(1/3, color="grey", linestyle="--", label="1/3 reference")
    ax.set(xlabel="Seconds since video onset", ylabel="Trial macro-F1", ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(out / "01_f1_vs_time.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    frame = metrics[(metrics.condition == "clean") & metrics.gating &
                    metrics.sex.isin(["overall", "women", "men"])]
    frame = frame[frame.variant.isin(["global", "sex_specific"])]
    if len(frame) and "macro_f1" in frame:
        frame.pivot(index="sex", columns="variant", values="macro_f1").plot.bar(ax=ax)
    ax.set(ylabel="Trial macro-F1", xlabel="Recorded gender group", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(out / "02_global_vs_sex.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    if len(alarms):
        ax.plot(alarms.false_alarms_per_hour, alarms.sensitivity, marker=".")
    ax.set(xlabel="False alarm episodes / non-high hour", ylabel="High-trial sensitivity", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(out / "03_alarm_tradeoff.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    frame = metrics[(metrics.variant == "global") & (metrics.sex == "overall")]
    if len(frame) and "macro_f1" in frame:
        frame.pivot(index="condition", columns="gating", values="macro_f1").rename(
            columns={False: "gate off", True: "gate on"}).plot.bar(ax=ax)
    ax.set(ylabel="Trial macro-F1 (see coverage in metrics.csv)", xlabel="Raw signal condition", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(out / "04_fault_gating.png", dpi=160); plt.close(fig)
