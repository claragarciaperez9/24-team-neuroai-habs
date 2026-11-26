"""Causal EMA alarm; an unreliable window clears accumulated alarm evidence."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class AlarmConfig:
    threshold: float = 1.01  # Disabled until chosen on validation data.
    hysteresis: float = .1
    duration: float = 1.5
    tau: float = 1.


class Alarm:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.ema = 0.
        self.above = 0.
        self.active = False

    def update(self, probability, reliable=True, dt=.5):
        if not reliable:
            self.reset()
            return False, False
        c = self.config
        self.ema += (1 - math.exp(-dt / c.tau)) * (probability - self.ema)
        if self.active and self.ema < c.threshold - c.hysteresis:
            self.active = False
        if not self.active:
            self.above = self.above + dt if self.ema >= c.threshold else 0.
            if self.above >= c.duration:
                self.active = True
                self.above = 0.
                return True, True
        return self.active, False


def alarm_metrics(rows, config):
    """False alarm episodes / observed non-high trial hours, including abstentions.

    Sensitivity is the fraction of all high trials with at least one alarm.
    Windows are sampled every .5 s; duration uses that monitored exposure only.
    """
    false_alarms, negative_seconds, hits, positives = 0, 0., 0, 0
    for _, trial in rows.groupby("trial", sort=False):
        alarm, detected = Alarm(config), False
        trial = trial.sort_values("time")
        target = int(trial.target.iloc[0])
        for row in trial.itertuples():
            _, onset = alarm.update(float(row.p_high) if row.reliable else 0., bool(row.reliable))
            if target != 2:
                false_alarms += int(onset)
                negative_seconds += .5
            else:
                detected |= onset
        if target == 2:
            positives += 1
            hits += int(detected)
    return {"sensitivity": hits / positives if positives else None,
            "false_alarms_per_hour": false_alarms / (negative_seconds / 3600) if negative_seconds else None,
            "false_alarm_episodes": false_alarms, "negative_exposure_s": negative_seconds,
            "high_trials": positives}


def choose_alarm(validation, target_fah=1.):
    candidates = []
    for threshold in [*np.linspace(.2, 1., 41), 1.01]:
        config = AlarmConfig(threshold=float(threshold))
        metrics = alarm_metrics(validation, config)
        candidates.append((config, metrics))
    if not (validation.target != 2).any() or not (validation.target == 2).any():
        raise ValueError("Alarm validation requires high and non-high trials")
    valid = [(c, m) for c, m in candidates if m["false_alarms_per_hour"] <= target_fah]
    config, metrics = max(valid, key=lambda cm: (cm[1]["sensitivity"], -cm[1]["false_alarms_per_hour"], -cm[0].threshold))
    return config, metrics
