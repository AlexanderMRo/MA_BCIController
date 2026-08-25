"""
Few-shot subject adaptation for Motor Imagery EEG (Emotiv EPOC X).
Pipeline: load subject data -> preprocess -> (optional) augment ->
          load pretrained model -> freeze backbone -> fine-tune head.
"""

import csv
from pathlib import Path

import mne
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from braindecode.models import EEGNet  # swap for Labram/BENDR wrapper if desired
from braindecode.augmentation import (
    AugmentedDataLoader,
    FTSurrogate,
    SmoothTimeMask,
    ChannelsDropout,
    GaussianNoise,
    FrequencyShift,
)

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
SAMPLE_FREQ = 128      # Emotiv EPOC X: 128 Hz (256 Hz internal, exported often at 128)
N_CHANNELS = 14        # EPOC X electrodes
N_CLASSES = 3          # neutral, left hand movement, right hand movement
WINDOW_SECONDS = 2.0   # trial length; short enough to fit a neutral + a movement
                       # window into each ~12s recording without overlap
WINDOW_SAMPLES = int(round(SAMPLE_FREQ * WINDOW_SECONDS))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# EPOC X channel order (fixed by hardware)
EMOTIV_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                   "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# ---------------------------------------------------------------------------
# Subject recordings (EDF data/<Subject> Recordings EDF+): per subject, 5
# trials of right-hand movement ("LA" = Light Attack, R Arm keystroke marker)
# and 5 trials of left-hand movement ("Estus" = Estus flask, L Arm keystroke
# marker), one marker per file. There is no dedicated "neutral" recording, so
# a neutral window is carved out of each file's own pre-marker baseline.
# ---------------------------------------------------------------------------
EDF_ROOT = Path("EDF data") / "LG Recordings EDF+"  # this subject's export folder
FILENAME_LABEL_TOKENS = {
    "LA": "right",     # "Light Attack (R Arm)" marker, marker_value=10
    "Estus": "left",   # "Estus (L Arm)" marker, marker_value=20
}
CLASS_TO_ID = {"neutral": 0, "left": 1, "right": 2}
NEUTRAL_BUFFER_SECONDS = 0.3  # gap kept before the marker so anticipatory
                               # motor activity doesn't leak into "neutral"

# ---------------------------------------------------------------------------
# 1. Load your Emotiv data
# ---------------------------------------------------------------------------
def load_emotiv_data(X_path, y_path):
    """
    Expects:
      X: (n_trials, n_channels, n_times)  float32, in Volts (or microV -> scale!)
      y: (n_trials,) int labels in {0, ..., N_CLASSES-1}

    EmotivPRO exports CSV/EDF. Convert to epoched arrays first
    (e.g. with MNE: raw = mne.io.read_raw_edf(...), then epoch on markers).
    """
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)

    # Emotiv exports microvolts; braindecode/most models expect ~microvolt scale.
    # Standardize per-channel (recommended for cross-subject transfer):
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-6)
    return X, y


# ---------------------------------------------------------------------------
# 1b. Load a subject's .edf recording folder directly (no manual pre-conversion)
# ---------------------------------------------------------------------------
def load_edf_data(root_dir=EDF_ROOT, window_seconds=WINDOW_SECONDS,
                   neutral_buffer_seconds=NEUTRAL_BUFFER_SECONDS, sample_freq=SAMPLE_FREQ):
    """
    Iterates every .edf recording for a subject in `root_dir` (see record.py /
    marker.py), each paired with a same-named `..._intervalMarker.csv` holding
    the single keystroke marker for that trial. Per recording this yields two
    epochs: a movement epoch starting at the marker, labeled from the filename
    via `FILENAME_LABEL_TOKENS` ("LA" -> right, "Estus" -> left), and a neutral
    epoch taken from the pre-marker baseline of that same recording.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times), float32
    y : ndarray, shape (n_trials,), int64 -- see CLASS_TO_ID
    recording_ids : ndarray, shape (n_trials,), int64
        Index into the sorted list of .edf files each trial came from. Both
        the movement and neutral epoch pulled from the same recording share
        the same id, so this can be used as the `groups` argument to
        `cross_validate` for leave-one-recording-out validation.
    """
    window_samples = int(round(window_seconds * sample_freq))
    edf_paths = sorted(Path(root_dir).glob("*.edf"))
    if not edf_paths:
        raise FileNotFoundError(f"No .edf files found under {root_dir}")
    print(f"[load_edf_data] found {len(edf_paths)} .edf recordings under {root_dir}")

    epoch_list, label_list, recording_id_list = [], [], []

    for recording_idx, edf_path in enumerate(edf_paths):
        filename_tokens = edf_path.stem.split("_")
        class_label = next(
            (lbl for token, lbl in FILENAME_LABEL_TOKENS.items() if token in filename_tokens),
            None,
        )
        if class_label is None:
            print(f"[load_edf_data] skipping {edf_path.name}: no recognized class token in filename.")
            continue

        marker_csv_path = edf_path.with_name(edf_path.stem + "_intervalMarker.csv")
        with open(marker_csv_path, newline="") as marker_csv_file:
            marker_rows = list(csv.DictReader(marker_csv_file))
        if not marker_rows:
            print(f"[load_edf_data] skipping {edf_path.name}: marker csv has no rows.")
            continue
        marker_onset_s = float(marker_rows[0]["latency"])

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick(EMOTIV_CHANNELS)  # keep EEG sensors only, in a fixed order
        if raw.info["sfreq"] != sample_freq:
            raw.resample(sample_freq)
        eeg_microvolts = raw.get_data(units="uV").astype(np.float32)  # (n_channels, n_times)
        print(f"[load_edf_data] {edf_path.name}: class={class_label}, "
              f"marker@{marker_onset_s:.2f}s, recording={eeg_microvolts.shape[1] / sample_freq:.1f}s")

        # Movement epoch: starts at the marker (movement trigger).
        movement_start = int(round(marker_onset_s * sample_freq))
        movement_end = movement_start + window_samples
        if movement_end > eeg_microvolts.shape[1]:
            print(f"[load_edf_data]   -> skipping movement epoch: recording too short after marker.")
        else:
            epoch_list.append(eeg_microvolts[:, movement_start:movement_end])
            label_list.append(CLASS_TO_ID[class_label])
            recording_id_list.append(recording_idx)
            print(f"[load_edf_data]   -> movement epoch [{movement_start}:{movement_end}] labeled '{class_label}'")

        # Neutral epoch: window ending `neutral_buffer_seconds` before the
        # marker, so anticipatory motor activity doesn't leak in.
        neutral_end = movement_start - int(round(neutral_buffer_seconds * sample_freq))
        neutral_start = neutral_end - window_samples
        if neutral_start < 0:
            print(f"[load_edf_data]   -> skipping neutral epoch: not enough pre-marker baseline.")
        else:
            epoch_list.append(eeg_microvolts[:, neutral_start:neutral_end])
            label_list.append(CLASS_TO_ID["neutral"])
            recording_id_list.append(recording_idx)
            print(f"[load_edf_data]   -> neutral epoch [{neutral_start}:{neutral_end}] labeled 'neutral'")

    X = np.stack(epoch_list, axis=0)
    y = np.array(label_list, dtype=np.int64)
    recording_ids = np.array(recording_id_list, dtype=np.int64)

    # Per-channel standardization, as in `load_emotiv_data`. Done per subject,
    # *before* any pooling across subjects in `load_edf_datasets`, so one
    # subject's baseline offset/scale doesn't bleed into another's.
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-6)
    class_counts = {cls: int((y == idx).sum()) for cls, idx in CLASS_TO_ID.items()}
    print(f"[load_edf_data] done: X={X.shape}, class counts={class_counts}")
    return X, y, recording_ids


# ---------------------------------------------------------------------------
# 1c. Pool multiple subjects' recording folders (for collective pretraining)
# ---------------------------------------------------------------------------
def load_edf_datasets(root_dirs, window_seconds=WINDOW_SECONDS,
                       neutral_buffer_seconds=NEUTRAL_BUFFER_SECONDS, sample_freq=SAMPLE_FREQ):
    """
    Loads and concatenates several subjects' recording folders via
    `load_edf_data`. Each subject is standardized independently before
    pooling (see `load_edf_data`), so this is *not* the same as pointing
    `load_edf_data` at a parent folder -- call this instead once you have
    more than one subject's data.

    Parameters
    ----------
    root_dirs : list[str | Path]
        One recordings folder per subject.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times), float32
    y : ndarray, shape (n_trials,), int64 -- see CLASS_TO_ID
    subject_ids : ndarray, shape (n_trials,), int64 -- index into `root_dirs`
    recording_ids : ndarray, shape (n_trials,), int64
        Globally unique across subjects (unlike the per-subject ids returned
        by `load_edf_data` alone). Use either array as `groups` in
        `cross_validate` for leave-one-recording-out or leave-one-subject-out.
    """
    X_parts, y_parts, subject_id_parts, recording_id_parts = [], [], [], []
    for subject_idx, root_dir in enumerate(root_dirs):
        print(f"[load_edf_datasets] subject {subject_idx}: {Path(root_dir).name}")
        X_subject, y_subject, recording_ids_subject = load_edf_data(
            root_dir=root_dir, window_seconds=window_seconds,
            neutral_buffer_seconds=neutral_buffer_seconds, sample_freq=sample_freq,
        )
        X_parts.append(X_subject)
        y_parts.append(y_subject)
        subject_id_parts.append(np.full(len(y_subject), subject_idx, dtype=np.int64))
        # Offset so recording ids stay unique once concatenated across subjects.
        recording_id_parts.append(subject_idx * 100_000 + recording_ids_subject)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    subject_ids = np.concatenate(subject_id_parts, axis=0)
    recording_ids = np.concatenate(recording_id_parts, axis=0)
    print(f"[load_edf_datasets] pooled {len(root_dirs)} subjects -> X={X.shape}")
    return X, y, subject_ids, recording_ids


# ---------------------------------------------------------------------------
# 1d. Load BNCI2014_001 (BCI Competition IV 2a), a public motor-imagery
# dataset, spatially interpolated onto the Emotiv 14-channel layout so it
# can be pooled/pretrained on alongside your own subject's data.
# ---------------------------------------------------------------------------
BNCI_LABEL_TOKENS = {"left_hand": "left", "right_hand": "right"}
# feet/tongue trials are dropped: no matching class in CLASS_TO_ID, and
# BNCI2014_001 has no rest/neutral trials comparable to yours.


def load_bnci2014_001(subject_ids, window_seconds=WINDOW_SECONDS, sample_freq=SAMPLE_FREQ):
    """
    Loads BNCI2014_001 motor-imagery trials for `subject_ids` (1-9). Each
    recording's native 22-channel montage (Fz, FC3, ..., POz) is spatially
    interpolated onto the Emotiv 14-channel layout via `raw.interpolate_to`
    (spherical-spline interpolation; both montages are standard 10-20
    positions, fully covered by MNE's "standard_1020" template), then
    resampled to `sample_freq`. Only left_hand/right_hand trials are kept,
    epoched as a `window_seconds` window from cue onset.

    Standardization happens per (real) subject -- combining all of that
    subject's sessions/runs -- before pooling, same principle as
    `load_edf_data` / `load_edf_datasets`.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times), float32
    y : ndarray, shape (n_trials,), int64 -- see CLASS_TO_ID
    subject_ids : ndarray, shape (n_trials,), int64 -- the BNCI2014_001 subject number
    """
    from braindecode.datasets import MOABBDataset

    window_samples = int(round(window_seconds * sample_freq))
    dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=list(subject_ids))

    # interpolate_to() needs a DigMontage (not an Info) naming the target
    # channel positions -- build one from the Emotiv layout's standard 10-20
    # coordinates.
    standard_positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    target_montage = mne.channels.make_dig_montage(
        ch_pos={name: standard_positions[name] for name in EMOTIV_CHANNELS},
        coord_frame="head",
    )

    per_subject_epochs, per_subject_labels = {}, {}
    for recording in dataset.datasets:
        subject_id = int(recording.description["subject"])
        raw = recording.raw.copy()
        raw.pick("eeg")  # drop EOG/STI channels
        raw.set_montage("standard_1020", match_case=False)
        # interpolate_to() returns a *new* instance rather than modifying
        # raw in place (unlike pick/set_montage/resample above) -- must
        # reassign, or the original (wrong-channel-count) raw silently
        # passes through untouched. It also drops annotations entirely on
        # the returned instance (rebuilds via a fresh RawArray internally),
        # so capture and reapply them -- timing is unaffected since
        # interpolation only changes channel content, not the time axis.
        annotations = raw.annotations
        raw = raw.interpolate_to(target_montage, method="spline")
        # The interpolated raw has no meas_date, making the original
        # annotations' absolute orig_time ambiguous to reattach -- rebuild
        # with orig_time=None so onsets are taken relative to the recording
        # start instead (valid here since interpolation doesn't shift the
        # time axis). orig_time isn't settable on an existing Annotations.
        raw.set_annotations(mne.Annotations(
            onset=annotations.onset, duration=annotations.duration,
            description=annotations.description, orig_time=None,
        ))
        if raw.info["sfreq"] != sample_freq:
            raw.resample(sample_freq)

        # Filter to only the annotations we want *before* parsing events --
        # events_from_annotations's dict-based event_id filtering chokes on
        # some non-class annotations present on certain subjects/recordings
        # (e.g. boundary markers), even though it's documented to just skip
        # unlisted descriptions.
        keep = np.isin(raw.annotations.description, list(BNCI_LABEL_TOKENS.keys()))
        raw.set_annotations(raw.annotations[keep])

        # event_id here must map description -> *integer* code (that's the
        # MNE contract) -- BNCI_LABEL_TOKENS maps description -> our class
        # label string instead, so let MNE auto-assign codes (safe now that
        # annotations are filtered to just left_hand/right_hand) and map
        # those codes to our labels ourselves below.
        events, found_event_id = mne.events_from_annotations(
            raw, event_id="auto", verbose=False
        )
        if len(events) == 0:
            continue
        epochs = mne.Epochs(
            raw, events, event_id=found_event_id,
            tmin=0.0, tmax=(window_samples - 1) / sample_freq,
            baseline=None, preload=True, verbose=False,
        )
        eeg_microvolts = epochs.get_data(units="uV").astype(np.float32)
        code_to_label = {code: BNCI_LABEL_TOKENS[name] for name, code in found_event_id.items()}
        labels = [CLASS_TO_ID[code_to_label[code]] for code in epochs.events[:, 2]]

        per_subject_epochs.setdefault(subject_id, []).append(eeg_microvolts)
        per_subject_labels.setdefault(subject_id, []).extend(labels)

    X_parts, y_parts, subject_id_parts = [], [], []
    for subject_id in sorted(per_subject_epochs):
        X_subject = np.concatenate(per_subject_epochs[subject_id], axis=0)
        y_subject = np.array(per_subject_labels[subject_id], dtype=np.int64)
        X_subject = (X_subject - X_subject.mean(axis=(0, 2), keepdims=True)) / \
            (X_subject.std(axis=(0, 2), keepdims=True) + 1e-6)

        X_parts.append(X_subject)
        y_parts.append(y_subject)
        subject_id_parts.append(np.full(len(y_subject), subject_id, dtype=np.int64))
        print(f"[load_bnci2014_001] subject {subject_id}: {len(y_subject)} trials")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    subject_ids_out = np.concatenate(subject_id_parts, axis=0)
    print(f"[load_bnci2014_001] done: X={X.shape}")
    return X, y, subject_ids_out


# ---------------------------------------------------------------------------
# 2. Build / load the model
# ---------------------------------------------------------------------------
def build_model(pretrained_path=None):
    print(f"[build_model] EEGNet: {N_CHANNELS} channels, {WINDOW_SAMPLES} timepoints, "
          f"{N_CLASSES} classes, device={DEVICE}")
    model = EEGNet(
        n_chans=N_CHANNELS,
        n_outputs=N_CLASSES,
        n_times=WINDOW_SAMPLES,
    )

    if pretrained_path is not None:
        # Load weights pretrained on your collective/multi-subject dataset.
        print(f"[build_model] loading pretrained weights from {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=DEVICE)
        # strict=False lets you skip mismatched classification-head shapes,
        # which is exactly what you want when re-using a backbone for a new task.
        missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
        print("[build_model] missing keys:", missing)
        print("[build_model] unexpected keys:", unexpected)
    else:
        print("[build_model] no pretrained_path given, starting from random init")

    return model.to(DEVICE)


def freeze_backbone(model, train_last_n_modules=1, module_names=None):
    """
    Freeze everything, then unfreeze the module(s) to fine-tune. With only a
    few samples/class, freezing the backbone is usually essential to avoid
    catastrophic overfitting.

    module_names : list[str], optional
        Unfreeze exactly these top-level named children (e.g. ["final_layer"])
        instead of using `train_last_n_modules`. Needed for architectures
        where registration order != execution order -- e.g. InterpolatedLaBraM
        registers its interpolation layer *last* even though it runs first,
        so "last N children by position" would unfreeze the wrong module.
    """
    for param in model.parameters():
        param.requires_grad = False

    if module_names is not None:
        print(f"[freeze_backbone] freezing all layers, then unfreezing: {module_names}")
        named = dict(model.named_children())
        trainable_modules = [named[name] for name in module_names]
    else:
        print(f"[freeze_backbone] freezing all layers, then unfreezing the last {train_last_n_modules} module(s)")
        trainable_modules = list(model.children())[-train_last_n_modules:]

    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True

    n_trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"[freeze_backbone] trainable params: {n_trainable_params}")
    return model


# ---------------------------------------------------------------------------
# 3. Optional augmentation (crucial for few-shot)
# ---------------------------------------------------------------------------
def build_augmentations(sample_freq, window_samples, prob=0.5):
    """
    Braindecode transforms operate on-the-fly in the DataLoader. Combined with
    `n_augmentation` in `finetune`, they don't just perturb existing trials in
    place -- they materialize extra (1 + n_augmentation)x copies per batch,
    which is what actually grows the usable training set for a few-shot subject.
    Tune `probability` and magnitudes on a validation subject, not the test user.
    """
    return [
        # Frequency-domain surrogate: preserves power spectrum, randomizes phase.
        FTSurrogate(probability=prob, phase_noise_magnitude=0.5),
        # Masks a random time segment -> robustness to transient artifacts.
        SmoothTimeMask(probability=prob, mask_len_samples=int(0.1 * window_samples)),
        # Drops random channels -> robustness (helps with Emotiv's noisy contacts).
        ChannelsDropout(probability=prob, p_drop=0.2),
        # Additive noise.
        GaussianNoise(probability=prob, std=0.1),
        # Small frequency shift.
        FrequencyShift(probability=prob, sfreq=sample_freq, max_delta_freq=1.0),
    ]


# ---------------------------------------------------------------------------
# 4. Fine-tuning loop
# ---------------------------------------------------------------------------
def finetune(model, X, y, use_augmentation=True, n_augmentation=4, augmentation_prob=0.5,
             epochs=100, batch_size=8, lr=1e-3, sample_freq=SAMPLE_FREQ):
    """
    n_augmentation : int, optional
        With few-shot data (4-5 trials/class), a handful of real trials is
        rarely enough to fill a batch with useful variety. Each batch keeps
        its clean originals and appends `n_augmentation` independently
        augmented copies, i.e. the effective training set grows by
        (1 + n_augmentation)x. Set to 0 to fall back to in-place, non-expanding
        augmentation (batch size unchanged).
    augmentation_prob : float, optional
        Per-transform probability passed to `build_augmentations`. Higher
        means each augmented copy is more heavily perturbed on average.
    sample_freq : float, optional
        Actual sample rate of `X` -- must match, not just default to the
        Emotiv pipeline's SAMPLE_FREQ, since `FrequencyShift` computes its
        shift in Hz. Pass the Labram-path rate (200) when fine-tuning that
        architecture's 200Hz-resampled data.
    """
    window_samples = X.shape[-1]
    X_tensor = torch.as_tensor(X)
    y_tensor = torch.as_tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)

    if use_augmentation:
        transforms = build_augmentations(sample_freq, window_samples, prob=augmentation_prob)
        loader = AugmentedDataLoader(
            dataset, transforms=transforms,
            batch_size=batch_size, shuffle=True,
            n_augmentation=n_augmentation,
        )
        print(f"[finetune] {len(dataset)} real trials, augmentation on "
              f"(n_augmentation={n_augmentation} -> {1 + n_augmentation}x per batch)")
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"[finetune] {len(dataset)} real trials, augmentation off")

    # Only optimize params we unfroze.
    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-2,
    )
    criterion = nn.CrossEntropyLoss()
    print(f"[finetune] training for {epochs} epochs, batch_size={batch_size}, lr={lr}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss {total_loss / len(loader):.4f}")

    return model


# ---------------------------------------------------------------------------
# 5. Evaluation / cross-validation
# ---------------------------------------------------------------------------
def evaluate(model, X, y):
    """Classification accuracy of `model` on (X, y)."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.as_tensor(X).to(DEVICE)
        y_tensor = torch.as_tensor(y).to(DEVICE)
        predictions = model(X_tensor).argmax(dim=1)
        accuracy = (predictions == y_tensor).float().mean().item()
    model.train()
    return accuracy


def cross_validate(X, y, groups, build_fn, use_augmentation=True, n_augmentation=4,
                    augmentation_prob=0.5, epochs=100, batch_size=8, lr=1e-3, sample_freq=SAMPLE_FREQ):
    """
    Leave-one-group-out cross-validation: every unique value in `groups` is
    held out exactly once (its trials used only for validation), and a fresh
    model is trained on the rest -- no weights carry over between folds. Pass
    `recording_ids` for leave-one-recording-out (the right call with a single
    subject's data), or `subject_ids` for leave-one-subject-out once you have
    multiple subjects.

    build_fn : callable
        Called with no arguments at the start of every fold; must return a
        freshly-initialized model (already pretrained/frozen as desired) ready
        for `finetune`. Keeps this CV loop generic across architectures --
        e.g. `lambda: build_model(pretrained_path=None)` for a from-scratch
        EEGNet, or `lambda: freeze_backbone(build_labram_model(), module_names=["final_layer"])`.

    With so few trials this is more informative than a single train/test
    split: it tells you whether the model beats chance (1/N_CLASSES) on
    unseen trials at all, rather than just watching training loss go down.

    Returns
    -------
    fold_accuracies : dict[group_value, float]
    """
    fold_accuracies = {}
    for held_out_group in sorted(set(groups.tolist())):
        held_out_mask = groups == held_out_group
        X_train, y_train = X[~held_out_mask], y[~held_out_mask]
        X_val, y_val = X[held_out_mask], y[held_out_mask]
        if len(y_val) == 0 or len(y_train) == 0:
            continue

        print(f"[cross_validate] fold group={held_out_group}: "
              f"train={len(y_train)} trials, val={len(y_val)} trials")
        model = build_fn()
        model = finetune(model, X_train, y_train, use_augmentation=use_augmentation,
                          n_augmentation=n_augmentation, augmentation_prob=augmentation_prob,
                          epochs=epochs, batch_size=batch_size, lr=lr, sample_freq=sample_freq)

        accuracy = evaluate(model, X_val, y_val)
        fold_accuracies[held_out_group] = accuracy
        print(f"[cross_validate] fold group={held_out_group}: accuracy={accuracy:.3f}")

    accuracies = np.array(list(fold_accuracies.values()))
    print(f"[cross_validate] {len(accuracies)} folds -> mean accuracy "
          f"{accuracies.mean():.3f} +/- {accuracies.std():.3f} (chance={1 / N_CLASSES:.3f})")
    return fold_accuracies


# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    # Few-shot support set for this subject: 5 right-hand + 5 left-hand
    # movement trials, plus one neutral epoch carved from each recording's
    # own pre-marker baseline (see load_edf_data / EDF_ROOT above).
    X_subject, y_subject, recording_ids = load_edf_data()
    class_counts = {cls: int((y_subject == idx).sum()) for cls, idx in CLASS_TO_ID.items()}
    print(f"Loaded {len(y_subject)} trials: {class_counts}")

    # -- (a) Baseline: random-init EEGNet --------------------------------
    # Leave-one-recording-out CV: an honest read on whether the model beats
    # chance on trials it never trained on, before trusting anything trained
    # on all the data at once.
    print("\n===== (a) EEGNet, random init =====")
    t0 = time.time()
    acc_baseline = cross_validate(
        X_subject, y_subject, groups=recording_ids,
        build_fn=lambda: build_model(pretrained_path=None),
        use_augmentation=True, n_augmentation=4, epochs=100, batch_size=8,
    )
    time_baseline = time.time() - t0

    # -- (b) EEGNet pretrained on BNCI2014_001, then frozen-backbone -----
    #    few-shot fine-tuned on this subject
    print("\n===== (b) EEGNet, pretrained on BNCI2014_001 =====")
    t0 = time.time()
    X_bnci, y_bnci, _ = load_bnci2014_001(subject_ids=range(1, 10))
    bnci_model = finetune(build_model(pretrained_path=None), X_bnci, y_bnci,
                           use_augmentation=True, n_augmentation=2, epochs=50, batch_size=32)
    torch.save(bnci_model.state_dict(), "bnci_pretrained_eegnet.pt")
    print(f"[main] BNCI-pretrained backbone saved to bnci_pretrained_eegnet.pt")

    acc_bnci = cross_validate(
        X_subject, y_subject, groups=recording_ids,
        build_fn=lambda: freeze_backbone(
            build_model(pretrained_path="bnci_pretrained_eegnet.pt"), train_last_n_modules=1
        ),
        use_augmentation=True, n_augmentation=4, epochs=100, batch_size=8,
    )
    time_bnci = time.time() - t0

    # -- Summary -----------------------------------------------------------
    for name, accs, elapsed in [
        ("EEGNet (random init)", acc_baseline, time_baseline),
        ("EEGNet (BNCI2014_001-pretrained)", acc_bnci, time_bnci),
    ]:
        values = np.array(list(accs.values()))
        print(f"{name:38s} acc={values.mean():.3f} +/- {values.std():.3f}  ({elapsed:.0f}s)")

    # Point this at a checkpoint pretrained on a collective/multi-subject
    # dataset once you have one; None trains the whole (unfrozen) model
    # from scratch on just this subject's few-shot data.
    pretrained_path = None
    model = build_model(pretrained_path=pretrained_path)
    if pretrained_path is not None:
        model = freeze_backbone(model, train_last_n_modules=1)

    model = finetune(model, X_subject, y_subject,
                     use_augmentation=True, n_augmentation=4,
                     epochs=100, batch_size=8)

    torch.save(model.state_dict(), "user_adapted.pt")