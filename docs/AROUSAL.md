# Causal arousal decoding

The implementation lives in `arousal/`. `process.py` is the HABS entry point.
It produces low/mid/high probabilities at 1.0, 1.5, ..., 10.0 seconds, with
class boundaries `<3.5`, `3.5–6.5`, `>6.5`. An unreliable output has null
probabilities and no alarm. This is a research pipeline; no patient performance
claim is made by the bundled sample or synthetic tests.

## Run

Use Python 3.11+. On macOS, LightGBM needs `brew install libomp`.
Requirements pin torch/torchaudio 2.10 on macOS: newer wheels can crash alongside
LightGBM because of conflicting OpenMP runtimes, as documented in
[PyTorch #191933](https://github.com/pytorch/pytorch/issues/191933).
The model loader initializes one CPU thread before loading the quality models;
tests include loading a saved bundle in a fresh process.
On Linux, use PyTorch/torchaudio appropriate for your CPU or CUDA installation;
the HABS worker supplies its own CUDA build.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m arousal audit --data sample --out out/audit

# DATASET must contain CURATED/, labels.csv (or labels_train.csv), markers.csv.
python -m arousal train --data /data --out out/arousal \
  --device cuda --folds 3 --epochs 20 --target-fah 1
```

The checked-in sample has no video BDF and no labels table. Audit reports
`not_ready`; `process(sample, out)` writes the audit and returns. Explicit
`train` fails with an explanation, rather than writing invented metrics.
No data is downloaded automatically. The existing exploration notebook is left
as it was.

## Data and time boundaries

`data.py` loads selected native-rate channels with pyEDFlib and validates the
matching record map. BDF record padding is excluded using `TotalSamples`.
EEG/ECG use microvolts; stored GSR voltages are converted to microsiemens with
the release's resistance calibration. ECG, EDA or respiration absent from the
map are represented as missing and masked. EEG order is fixed: the release has
64 electrodes; F9 and F10 are omitted to give the requested 62 inputs. Change
the documented `EEG_NAMES` tuple if you choose another montage, then retrain.

Trials join labels on participant, day, start marker and video basename, with
end marker checked against `markers.csv`. Repeated ambiguous joins fail.
All rows containing `TESTSUBJECT`, flagged rows, missing flags, and unavailable
or out-of-range arousal ratings are excluded. Metadata arousal annotations do
not replace the labels table. Onsets are always
`onset_s - metadata_time_origin_s` for the relevant segment.

Every trial starts at exactly -5 seconds. The first released trial can lack
that baseline because the export starts at video onset: that trial is skipped.
Only baseline `[-5,0)` and response `[0,t)` enter physiological features. No
previous rating screen or samples after the video offset are loaded. At t=1,
the EEG input includes baseline `[-1,0)` plus response `[0,1)`, giving (62,500).
Each trial retains its own ID and one target across its 19 overlapping windows.
Shorter videos produce only windows before their offset.

## Causal preprocessing

`signals.py` uses stateful `sosfilt` only, with anti-alias filtering before
integer decimation to 250 Hz. It explicitly rejects input rates that are not
integer multiples of 250 Hz. Filter state and decimation phase survive any
chunk size. Filters reset at the start of each Prepare screen, preventing
previous-trial information from entering through filter state.

EEG uses a 0.5–45 Hz band-pass and 50 Hz notch, then a common average over
good channels selected using only the patient's pre-video rest. Bad channels
remain zero. EEG is unavailable if fewer than half the channels are usable;
good physiology can still support predictions. Exponential moving mean and
variance are estimated on rest after filter startup, and **frozen during all
video trials**. This is the
strict rest-only interpretation of exponential moving standardisation; online
adaptation on the video would conflict with rest-only normalization.
Filter states initialize from the first available sample at -5 s, avoiding
a fictitious SCR from starting a low-pass filter at zero on a constant baseline.

ECG uses a 5–35 Hz filter and rectified local R-peak candidates confirmed one
sample later, with a 300 ms refractory interval and a rest-derived threshold.
RR intervals outside 0.3–2 s are rejected; RMSSD uses only adjacent valid
intervals. EDA uses a 3 Hz low-pass, a 0.05 Hz tonic low-pass, their phasic
difference, and positive SCR peaks over 0.05 uS. Respiration uses 0.08–0.7 Hz,
peak-derived rate and amplitude summaries. The 25 physiological features are
8 EDA, 9 ECG and 8 respiration values; short-window HR/RMSSD availability bits
avoid treating an unavailable estimate as an observed zero. EDA and respiration
amplitudes are first scaled by the patient's rest. Additional feature scales
are fitted only on the training fold, separately from patient signal
normalization.

Tests require bit-for-bit agreement of offline and arbitrary-chunk streaming
signal and feature output, and verify that changing future data cannot change
an earlier prediction window.

## Artefact gate

`quality.py` extracts raw 1 s quality features every 0.5 s, before arousal
inference and before filters can hide faults: variance, 48–52 Hz power ratio,
power above 30 Hz, correlation to the three nearest good electrodes in the
standard 10-05 montage, flatness, range, jumps, nonfinite fraction and clipping.

Markers 10006–10018 are checked against the actual lookup: repeated blinking,
horizontal/vertical eye movement, horizontal/vertical head movement, jaw
clenching, swallowing. Markers 10000/10002/10004 are quiet reference tasks.
Use their metadata onsets, not lookup row numbers. P001's first instructed
blink starts at 2049.273 - 1974.273 = 75 s in its artefact recording.

EEG targets combine instructed positives with explicit blink/muscle/flatline/
line-noise rules. This implementation uses **rule-based weak labels, not
ICLabel or expert ground truth**. Quiet instructions do not override a detected
fault. Rest-only bad channels are excluded consistently from quality features.
The checked-in P001 sample has high raw line-noise ratios and supplies no
rule-clean EEG reference. A fold without both classes fails clearly; acquire
adequate clean calibration data or review the rules before running it. No
automatic relabelling fabricates clean examples.

Five injected faults—flatline, saturation, noise burst, electrode pop and
dropout—augment weak-clean reference data for each modality. Four small
LightGBM classifiers use only training participants' calibration data. The
same gate is shared by global and sex-specific arousal models for a controlled
comparison. Quality probability threshold defaults to 0.5. Nonfinite and
flatline inputs are hard failures as well.

EEG is masked if its trailing 2 s contain a flagged quality check. A physiology
modality is masked if *any* baseline/response quality check is flagged, because
its features use that whole context. With fewer than two clean modalities,
return `unreliable`, clear alarm evidence, and skip model inference. Modality
masks also zero branch inputs/embeddings so bias terms cannot stand in for a
missing EEG branch. The learned gate's decisions can include false negatives;
coverage is reported alongside predictive accuracy.

Quality reports give binary detection precision/recall against clean references
for every fault/instructed type. These are weak/injected-label metrics, not
clinical validation. Stream probes inject 1 s faults into held-out rest and
report detection delay (0.5 s cadence), misses, and false alarm episodes per
hour. Instructed EEG events also have delay probes: events already flagged
before onset are marked explicitly and have no estimable new-event delay.
Window false-positive rates are separately named as windows/hour.

## Models and splits

`model.py` uses Braindecode EEGNet convolution features, a linear projection
to 16 dimensions, and a 25→32→16 physiological MLP. Their concatenation with
time/10 and four modality masks feeds a 37→32→3 MLP. Braindecode 1.3 does not
provide EEGNet's requested `return_features=True` parameter; `EEGEncoder`
exposes that interface by removing the final classification layer. See the
[upstream API](https://braindecode.org/1.2/generated/braindecode.models.EEGNet.html).
Training is end to end, with class weights counted per trial, equal total
weight per trial, and modality dropout. One positive temperature is fitted to
validation trial log-loss.

`evaluation.py` crosses participant GroupKFold assignments with independent
video GroupKFold assignments. Test is the held-out participant/video
intersection. Training excludes **both** held-out sets; mixed rectangles are
discarded. All K² outer combinations give every trial exactly one test result.
A second crossed split holds out validation participants and videos, with
explicit disjointness assertions. Grouping participant/video pairs alone
would not satisfy the requested new-person, new-video evaluation.

Validation chooses the prediction-time cutoff among {0,2,3,4,5} s, training
a candidate for each. Here `t_start` means **earliest window end/prediction
time**, not earliest EEG sample; t_start=0 keeps all 19 windows. Earlier outputs
of a later-start model are `unreliable`. Validation is reused for epoch/start
selection, temperature and alarm tuning; the untouched outer tests estimate
generalization. Use a larger separate calibration cohort if available.

Every fold fits A (global) and B (women/men separately). The source field is
`gender`, used as the available proxy for the requested sex comparison. Unknown
values remain in global results and are excluded from the matched A/B figure.
If a group's training or validation lacks a target class, its model is marked
unavailable. The `global` versus `sex_specific` table uses exactly the same
remaining test trials; `global_all` also reports every test participant.
`training_counts.csv` and `run.json` expose actual men's participant/trial
counts and proportion. The expected approximately 29% men is a dataset
description, not an assumed or fabricated measured result.

Metrics average reliable probabilities per trial before macro-F1, balanced
accuracy, log-loss and 10-bin ECE. Fully rejected trials stay in coverage
denominators. Accuracy is conditional on accepted trials. The plot's 1/3 line
is a three-class reference (uniform balanced-accuracy chance); macro-F1 chance
depends on class proportions. Performance versus time uses one observation per
trial/time. Signal faults are injected into raw held-out data at response
[3,4) s, balanced deterministically across fault/modality combinations; gate
on/off use identical corrupted signals and the same trained model. This is an
inference ablation, not separately optimized models for the two conditions.

## Alarm and reports

`alarm.py` updates causal EMA P(high), with a 1 s time constant, 0.1 hysteresis,
and a 1.5 s minimum duration. Validation chooses a threshold for the requested
false alarm target (default one episode/non-high hour). If no useful threshold
meets it, 1.01 disables the alarm. This validation target is not a test/live
guarantee. Alarms reset at each trial and on any unreliable window. Test
threshold sweeps are descriptive curves, never used to choose the deployed
threshold. Sensitivity includes all high trials, including abstained trials.
Exposure is monitored 0.5 s bins; quiet gaps/rating screens are not counted.

Outputs in the chosen `--out` directory:

- `data_audit.json`, `trials.csv`, `splits.json`, `training_counts.csv`, `run.json`.
- `models/p*-v*/{global,women,men}/`: model, quality classifiers, scaler,
  temperature, start cutoff, alarm configuration and validation details.
- `window_predictions.csv`: probabilities or nulls, status, masks, alarms,
  participant/video IDs, fold and latency.
- `metrics.csv`, `performance_by_time.csv`, `alarm_tradeoff.csv`,
  `selected_alarm_test_metrics.csv`.
- `quality_label_provenance.csv`, `quality_metrics.csv`,
  `quality_detection_delay.csv`.
- Four PNGs: F1 versus time; matched global versus sex-specific overall/per
  group; alarm trade-off; gate on/off with/without injected faults.
- `latency.json`: cached feature-to-alarm timing. Replay below measures the
  complete step, including I/O and preprocessing, against 500 ms.

The disk cache stores roughly 2.4 MB of EEG windows per 10 s trial per condition,
plus quality/physiology arrays. Full training assembles a fold's selected
windows in RAM; use a worker with sufficient RAM/disk for the full release.
Three folds mean nine outer tests, and up to 135 candidate networks. There is
no silent cap on participants or trials. Delete `out/.../cache/` after changes
to preprocessing/feature code; source file changes invalidate cache entries.
Only load model/cache files you generated: quality files use joblib/pickle.

## Live LSL replay

Choose a saved outer-fold model and a trial from its held-out participants and
videos. Supply that patient's **pre-video rest**, and its BDF-relative onset.
For arbitrary sensor streams, call `Predictor.start(fs, rest_state)` once at
-5 s then `Predictor.push(block)` with timestamp-ordered, contiguous samples
in the documented units/channel order. Start a new stream per trial.

```bash
python -m arousal replay --model out/arousal/models/p0-v0/global \
  --bdf /data/CURATED/Pxxx/Pxxx_day-1_task-emotion-video_multimodal.bdf \
  --rest /data/CURATED/Pxxx/Pxxx_day-1_task-resting-state_multimodal.bdf \
  --onset 123.456 --duration 10 --out out/replay --lsl
```

This publishes canonical 65-channel raw samples to `NeuroAI-Replay`, consumes
them through a real LSL inlet, and publishes JSON status/probabilities/alarms
to `NeuroAI-Arousal` at 2 Hz. It also prints and saves `replay.jsonl`. Omit
`--lsl` for direct replay; `--speed 10` speeds up transport tests. Report
`replay_latency.json` on the target deployment machine: it includes BDF I/O,
LSL when enabled, causal preprocessing, quality checks and inference. No
hardware-independent <500 ms claim is made. An LSL dropout makes the remaining
scheduled outputs unreliable and suppresses alarms. It never closes timestamp
gaps by concatenating later samples. Custom acquisition code must likewise
represent discontinuities as missing data or reset the trial.
