# BCI Mental-Command Game Controller — Project Knowledge

Context dump for picking up this project in a new session. Written 2026-09-07.

## Goal

Detect mental commands (neutral / left-hand motor imagery / right-hand motor
imagery) from an Emotiv EPOC X headset in real time, and emulate Xbox 360
button presses so a video game can be played hands-free. Two parts:
1. **Training** (`BCI_Controller_train.py`): build the best classifier from
   available EEG data.
2. **Deployment** (`BCI_Controller_live.py`): stream live EEG, classify a
   sliding window, tap virtual-controller buttons.

The end user (deployment target) is the subject named **"LG"** in the data
(the person running this project, i.e. "you" in the code's comments). Any
other subject's data (e.g. "KD") is auxiliary -- a same-hardware/protocol
data source to try to transfer from, not a second deployment target.

## Data: `EDF data/`

Each subject has a folder `EDF data/<Name> Recordings EDF+/` (also an `EDF`
variant without markers/annotations -- always use the `EDF+` one). Per
subject: 5 recordings of right-hand imagined movement (filename token `LA`
= "Light Attack", marker `Light_Attack_(R_Arm)`, marker_value=10) and 5 of
left-hand (`Estus`, "Estus flask", marker `Estus_(L_Arm)`, marker_value=20).
Each `.edf` has a same-stem `..._intervalMarker.csv`.

**Protocol**: experimenter gives a 3s spoken countdown, then presses the
marker key exactly when the subject starts imagining the movement (subject
does not physically move -- confirmed motor **imagery**, not execution, for
every subject). Recordings are ~12-13s long with the marker a few seconds
in, EXCEPT each subject's very first recording of each class, which is ~60s
and has extra marker rows before the cue:

```csv
latency,duration,type,marker_value,key,timestamp,marker_id
4.11,15.01,Eyes_Opened,1,-1,...,1
24.12,15.01,Eyes_Closed,3,-1,...,2
45.11,0.00,Light_Attack_(R_Arm),10,10,...,3
```

The actual movement cue is **always the last row** -- earlier rows (when
present) are baseline segments. `load_edf_data` extracts several
non-overlapping 2s neutral windows from `Eyes_Opened` segments (not
`Eyes_Closed` -- closed-eyes rest has a strong posterior alpha rhythm
unrepresentative of "neutral" during real eyes-open gameplay).

Two real subjects exist so far: `LG Recordings EDF+` (yours, the deployment
target -- `EDF_ROOT` in code) and `KD Recordings EDF+` (auxiliary, a
different person, recorded 2026-08-25).

## `BCI_Controller_train.py` -- training pipeline

**Config** (top of file): `SAMPLE_FREQ=128`, `N_CHANNELS=14`, `N_CLASSES=3`,
`WINDOW_SECONDS=2.0` -> `WINDOW_SAMPLES=256`, `EMOTIV_CHANNELS` (14 names,
fixed hardware order), `CLASS_TO_ID={"neutral":0,"left":1,"right":2}`,
`FILENAME_LABEL_TOKENS={"LA":"right","Estus":"left"}`,
`NEUTRAL_BUFFER_SECONDS=0.3` (gap kept before the marker for the standard
pre-marker neutral window).

**Data loaders**:
- `load_edf_data(root_dir, ..., return_stats=False)` -- one subject folder ->
  `(X, y, recording_ids)`, or with per-channel `(channel_mean, channel_std)`
  appended if `return_stats=True`. Standardizes per-subject internally.
- `load_edf_datasets(root_dirs)` -- pools multiple subjects (each
  standardized independently first), returns `(X, y, subject_ids,
  recording_ids)` with globally-unique recording ids.
- `discover_auxiliary_subject_roots(edf_data_root=EDF_DATA_ROOT,
  exclude_root=EDF_ROOT)` -- globs every `"*Recordings EDF+"` folder except
  your own. **Dropping a new subject's folder into `EDF data/` and rerunning
  the script picks them up automatically** -- no code changes needed.
- `load_bnci2014_001(subject_ids)` -- public BCI-IV-2a motor-imagery dataset
  via braindecode/MOABB, spatially interpolated from its 22-channel montage
  onto the Emotiv 14-channel layout via `raw.interpolate_to(...,
  method="spline")`. Several non-obvious MNE API quirks were worked through
  here (see "Bugs fixed" below) -- treat this function as a reference for
  those gotchas if similar interpolation work comes up again.

**Model**: `build_model(pretrained_path=None)` builds a braindecode `EEGNet`
(~1,715 params, architecture-only in braindecode -- no pretrained weights
ship with the library). `freeze_backbone(model, train_last_n_modules=1,
module_names=None)` freezes everything then unfreezes the last N children
(or explicit named ones, for architectures where registration order !=
execution order). `build_augmentations` gives 5 braindecode transforms
(FTSurrogate, SmoothTimeMask, ChannelsDropout, GaussianNoise, FrequencyShift).
`finetune(...)` trains with `AugmentedDataLoader` (expands batches
`(1+n_augmentation)x`, not just in-place perturbation). `cross_validate(X, y,
groups, build_fn, ...)` does leave-one-group-out CV -- pass `recording_ids`
for leave-one-recording-out (right call for a single subject) or
`subject_ids`/pooled `recording_ids` for cross-subject.

**`__main__`** loads your (LG) data, runs 4 comparison arms, then does a
final production fit + save. Current arms:
- (a) EEGNet random init, CV on you only -- the baseline to beat.
- (b) EEGNet pretrained on BNCI2014_001, few-shot CV'd on you.
- (c) EEGNet pretrained on **all auto-discovered auxiliary subjects
  pooled**, few-shot CV'd on you. Skipped gracefully if none exist.
- (d) EEGNet random init, pooled CV across you + all auxiliary subjects
  (tests a shared/general classifier, NOT what gets deployed).

Final fit: whichever `pretrained_path` is set (currently `None` = train from
scratch on just your data) gets fine-tuned on your full dataset and saved to
`user_adapted.pt`, with matching normalization stats to `user_adapted_norm.npz`
(`np.savez(mean=..., std=...)`, shape `(14,)` each -- **required** because a
single live 2s window can't be standardized against itself; the live script
reuses these exact training-time stats).

## Experiments run so far (results)

All leave-one-recording-out CV, chance = 0.333. **Caveat that matters a lot**:
this is only ~10-20 recordings with 2 held-out trials/fold, so per-fold
accuracy is discrete (0, 0.5, or 1.0) and very noisy -- treat differences
under ~0.15 as not meaningful.

| Experiment | Accuracy | Notes |
|---|---|---|
| EEGNet random init (LG only), no augmentation | 0.400 ± 0.300 | |
| EEGNet random init (LG only), augmented (n_aug=4, prob=0.5) | 0.450-0.500, std 0.15-0.35 across 3 separate runs | This is the deployed recipe |
| EEGNet random init (LG only), stronger augmentation (n_aug=8, prob=0.7) | 0.500 ± 0.316 | No better than default, more variance, more compute -- default augmentation settings are already near-optimal |
| EEGNet pretrained on BNCI2014_001 | 0.300-0.400, std 0.24-0.37 | Consistently doesn't help -- likely negative transfer (different hardware/montage; EPOC X has no motor-cortex electrodes, so the interpolation is extrapolating the most relevant region from indirect data) |
| EEGNet pretrained on KD (auxiliary, same hardware/protocol) | 0.450 ± 0.150 | Ties baseline mean but **much lower variance** (9/10 folds landed on exactly 0.5) -- cheaper (145s vs 747s for BNCI) and more consistent, though not yet a clear accuracy win. Most promising direction if more auxiliary subjects get recorded. |
| EEGNet random init, pooled LG+KD | 0.475 ± 0.339 | Inconclusive; also not the deployment target (not personalized) |
| **Labram (pretrained EEG transformer foundation model)** | Not testable | Every available checkpoint needs 8-15s of context (pretraining objective requires long windows) -- incompatible with the ~2s window this project needs for real-time latency. Full writeup: `labram_investigation.md`. Dropped. |

**Current deployed model** (`user_adapted.pt` / `user_adapted_norm.npz`):
EEGNet trained from scratch on LG's own data only, with augmentation
(n_aug=4, prob=0.5), matching arm (a) -- because none of the pretraining
arms convincingly beat it yet.

## Bugs found and fixed (useful if similar patterns recur)

- Marker CSV row selection: was taking `marker_rows[0]`, which broke for
  recordings with baseline rows prepended (first row would be
  `Eyes_Opened`, not the cue). Fixed to `marker_rows[-1]` -- the cue is
  always last.
- `raw.interpolate_to(...)` (MNE): returns a **new** instance (must
  reassign, doesn't mutate in place), **drops annotations** entirely
  (rebuild via fresh `mne.Annotations(..., orig_time=None)` after, since the
  interpolated raw has no `meas_date` and the original annotations'
  `orig_time` isn't re-settable), needs a `DigMontage` not an `Info`, and
  needs `method="spline"` explicit (default resolution is broken in this
  MNE version for EEG).
- `mne.events_from_annotations(event_id=...)` needs a dict mapping
  description -> **integer** code, not description -> our string class
  label -- filter annotations to wanted descriptions first, then use
  `event_id="auto"` and map the auto-assigned codes back to labels
  ourselves.
- `torch.hub.load_state_dict_from_url` caches by **filename**, not URL -- an
  early wrong-URL attempt (`/blob/main/` HTML page instead of
  `/resolve/main/`) poisoned the cache under the real checkpoint's filename;
  every retry with the correct URL still silently loaded the cached garbage
  until the `file_name` was changed.
- `braindecode.models.EEGNetv4` was renamed to `EEGNet` in braindecode 1.6.1.

## `BCI_Controller_live.py` -- real-time deployment

Streams raw EEG from the Emotiv headset via the existing Cortex API wrapper
(`python/cortex.py`, reused as-is, not duplicated), classifies a sliding
2s window with the trained model, taps Xbox 360 buttons via `vgamepad`.

**Design decisions** (from user clarification):
- Right hand -> right bumper. Left hand -> left face button (`X` on Xbox
  layout). Neutral -> nothing.
- **Tap-on-onset + cooldown**, not hold-while-detected: a button fires once
  when a command is confirmed (`CONSECUTIVE_TO_TRIGGER=2` agreeing windows,
  ~0.5s), then further taps of the *same* command are paced by
  `COOLDOWN_SECONDS=0.4` (not reset after each tap) so a sustained mental
  command produces steady consecutive attacks without spamming. Switching to
  a different class resets the onset-confirmation streak.
- Credentials via `EMOTIV_CLIENT_ID` / `EMOTIV_CLIENT_SECRET` environment
  variables (not hardcoded, unlike `python/live_advance.py` /
  `python/record.py`'s example scripts, which both hardcode the same
  client_id/secret pair in plaintext).
- `PROBABILITY_THRESHOLD=0.5` softmax confidence gate (chance=0.333).
- All four timing/threshold constants are explicitly tunable once real
  in-game feel can be tested.

**Architecture**: `MentalCommandClassifier` (lock-protected
`deque(maxlen=WINDOW_SAMPLES)` buffer, applies saved train-time
mean/std, runs EEGNet forward pass) + `ButtonController` (onset-streak +
cooldown debounce logic, drives a `vg.VX360Gamepad()`) +
`LiveMentalCommandController` (owns the `Cortex` session; subscribes
directly to the raw `eeg` stream -- **not** Emotiv's own mental-command/
profile system, unlike `live_advance.py`, since we run our own model
instead; resolves the 14 `EMOTIV_CHANNELS` positions within the stream's
actual column list by name via the `new_data_labels` event, since the `eeg`
stream also carries COUNTER/INTERPOLATED/RAW_CQ columns in unguaranteed
positions).

**Dependencies installed into `venv`** (none were present before this
project touched this file): `vgamepad`, `websocket-client`,
`python-dispatch`, `importlib-metadata`.

**Verification status**: model-loading/classifier path fully verified
end-to-end (loads `user_adapted.pt` with no missing/unexpected keys, runs a
forward pass). `ButtonController` debounce logic verified with a recording
stand-in object. **`vg.VX360Gamepad()` itself has NOT been verified** --
`ViGEmBus` (the Windows driver vgamepad's virtual controller needs) is not
currently installed/attached on this machine, so real button-press behavior
is untested. Install from https://github.com/ViGEm/ViGEmBus/releases, then
set the two env vars, connect the headset, run Cortex/EmotivApp, and run
`BCI_Controller_live.py` for the first real end-to-end test.

## Other files in the repo (context, not modified this session)

- `python/cortex.py` -- Emotiv Cortex API websocket wrapper (JSON-RPC over
  `wss://localhost:6868`), reused by the live script. Was not previously
  installed/runnable in this venv (missing deps, now installed).
- `python/live_advance.py`, `python/record.py`, `python/marker.py`,
  `python/query_records.py`, `python/sub_data.py` -- older example/utility
  scripts for driving Emotiv's *own* built-in mental-command detection
  and recording sessions; `live_advance.py` is where the `vgamepad` +
  virtual-DS4-controller pattern this project's live script builds on
  originally came from (per git history: "Add light attack functionality in
  DS3 on mental command 'left' using virtual DS4 controller").
- `BCI_controller_TL.py` -- an old/unused prototype (braindecode tutorial
  boilerplate: `ShallowFBCSPNet` on raw `BNCI2014_001`, no connection to the
  current EDF/EEGNet pipeline). Superseded by `BCI_Controller_train.py`;
  not touched or referenced by current work.
- `Emotif BCI data/BCI Decrypt.py`, `unused/facial_expression_train.py` --
  not explored this session, presumably obsolete/unrelated.
- `labram_investigation.md` -- half-page writeup of why the Labram
  foundation model was dropped (see table above).
- `run_comparison.log`, `augmentation_sweep.log`, `full_comparison_with_KD.log`
  -- raw output logs from the three big experiment runs referenced above.
- `bnci_pretrained_eegnet.pt`, `auxiliary_pretrained_eegnet.pt` -- saved
  intermediate pretrained backbones from arms (b)/(c) (not the deployed
  model; that's `user_adapted.pt`).

## Working-style notes for whoever picks this up

- The user wants **non-trivial bash/python commands explained before
  running them**, even though edits can be applied without asking. This
  applies specifically to shell/Python commands being executed, not to file
  edits.
- Prefers descriptive variable/function names throughout (renamed `SFREQ`->
  `SAMPLE_FREQ`, rejected ambiguous `p1`/`p2`/`p3` naming earlier on).
  Wants helpful, informative print statements at each pipeline step.
- Comment style: descriptive but concise -- explain the non-obvious *why*,
  not the obvious *what*.
- Judgment calls with real behavioral/deployment consequences (e.g. tap vs.
  hold button semantics, which data pooling strategy to test, credential
  handling) get asked about explicitly rather than assumed; the user tends
  to answer decisively and is fine with "both" when an experiment is cheap
  enough to just run twice.
- Given multiple plausible interpretations of an ambiguous request, a short
  `AskUserQuestion` with a recommended option has worked well.

## Open items / natural next steps

- Verify `BCI_Controller_live.py` end-to-end with a real headset once
  ViGEmBus is installed; tune `PROBABILITY_THRESHOLD` /
  `CONSECUTIVE_TO_TRIGGER` / `COOLDOWN_SECONDS` / `TAP_DURATION_SECONDS`
  based on actual in-game feel.
- If more auxiliary subjects get recorded (same protocol), rerun
  `BCI_Controller_train.py` as-is -- arms (c)/(d) will pick them up
  automatically. The auxiliary-pretraining direction (arm c) is the most
  promising untapped lever so far (lower variance, cheap), worth more data
  before judging it conclusively.
- No current plan to revisit Labram/foundation models unless a
  short-context (<=4s) checkpoint becomes available.
