"""
Few-shot subject adaptation for Motor Imagery EEG (Emotiv EPOC X).
Pipeline: load subject data -> preprocess -> (optional) augment ->
          load pretrained model -> freeze backbone -> fine-tune head.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from braindecode.models import EEGNetv4  # swap for Labram/BENDR wrapper if desired
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
SFREQ = 128            # Emotiv EPOC X: 128 Hz (256 Hz internal, exported often at 128)
N_CHANNELS = 14        # EPOC X electrodes
N_CLASSES = 3          # 2 or 3 MI classes
WINDOW_SAMPLES = SFREQ * 4   # e.g. 4-second trials
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# EPOC X channel order (fixed by hardware)
EMOTIV_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                   "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

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
# 2. Build / load the model
# ---------------------------------------------------------------------------
def build_model(pretrained_path=None):
    model = EEGNetv4(
        n_chans=N_CHANNELS,
        n_outputs=N_CLASSES,
        n_times=WINDOW_SAMPLES,
    )

    if pretrained_path is not None:
        # Load weights pretrained on your collective/multi-subject dataset.
        state = torch.load(pretrained_path, map_location=DEVICE)
        # strict=False lets you skip mismatched classification-head shapes,
        # which is exactly what you want when re-using a backbone for a new task.
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    return model.to(DEVICE)


def freeze_backbone(model, train_last_n_modules=1):
    """
    Freeze everything, then unfreeze the final module(s) for few-shot fine-tuning.
    With only 4-5 samples/class, freezing the backbone is usually essential
    to avoid catastrophic overfitting.
    """
    for p in model.parameters():
        p.requires_grad = False

    trainable = list(model.children())[-train_last_n_modules:]
    for module in trainable:
        for p in module.parameters():
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_train}")
    return model


# ---------------------------------------------------------------------------
# 3. Optional augmentation (crucial for few-shot)
# ---------------------------------------------------------------------------
def build_augmentations(sfreq, prob=0.5):
    """
    Braindecode transforms operate on-the-fly in the DataLoader.
    Tune `probability` and magnitudes on a validation subject, not the test user.
    """
    return [
        # Frequency-domain surrogate: preserves power spectrum, randomizes phase.
        FTSurrogate(probability=prob, phase_noise_magnitude=0.5),
        # Masks a random time segment -> robustness to transient artifacts.
        SmoothTimeMask(probability=prob, mask_len_samples=int(0.1 * WINDOW_SAMPLES)),
        # Drops random channels -> robustness (helps with Emotiv's noisy contacts).
        ChannelsDropout(probability=prob, p_drop=0.2),
        # Additive noise.
        GaussianNoise(probability=prob, std=0.1),
        # Small frequency shift.
        FrequencyShift(probability=prob, sfreq=sfreq, max_delta_freq=1.0),
    ]


# ---------------------------------------------------------------------------
# 4. Fine-tuning loop
# ---------------------------------------------------------------------------
def finetune(model, X, y, use_augmentation=True, epochs=100, batch_size=8, lr=1e-3):
    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y)
    dataset = TensorDataset(X_t, y_t)

    if use_augmentation:
        transforms = build_augmentations(SFREQ)
        loader = AugmentedDataLoader(
            dataset, transforms=transforms,
            batch_size=batch_size, shuffle=True,
        )
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Only optimize params we unfroze.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-2,
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss {total / len(loader):.4f}")

    return model


# ---------------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Few-shot support set from the new user: ~4-5 trials per class.
    X_user, y_user = load_emotiv_data("user_X.npy", "user_y.npy")

    model = build_model(pretrained_path="collective_pretrained.pt")
    model = freeze_backbone(model, train_last_n_modules=1)

    model = finetune(model, X_user, y_user,
                     use_augmentation=True, epochs=100, batch_size=8)

    torch.save(model.state_dict(), "user_adapted.pt")