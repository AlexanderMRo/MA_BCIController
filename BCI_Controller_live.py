"""
Real-time mental-command -> Xbox 360 button emulation.

Pipeline: Cortex 'eeg' stream -> sliding 2s window -> trained EEGNet
          (user_adapted.pt) -> debounced button tap via vgamepad.

Right hand movement -> right bumper. Left hand movement -> left face button
(X). Neutral -> no input. See BCI_Controller_train.py for how the model and
normalization stats were produced.

Prerequisites:
  - Emotiv headset connected, EmotivApp/Cortex running.
  - user_adapted.pt and user_adapted_norm.npz present (run
    BCI_Controller_train.py first).
  - EMOTIV_CLIENT_ID / EMOTIV_CLIENT_SECRET set in the environment.
  - ViGEmBus driver installed (vgamepad's virtual-controller backend --
    https://github.com/ViGEm/ViGEmBus/releases; not a pip package).
"""

import os
import sys
import time
import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch
import vgamepad as vg

# python/cortex.py isn't a package -- add its folder to sys.path so we can
# import the existing Cortex API wrapper without duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "python"))
from cortex import Cortex

from BCI_Controller_train import build_model, EMOTIV_CHANNELS, CLASS_TO_ID, WINDOW_SAMPLES, DEVICE

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
MODEL_PATH = "user_adapted.pt"
NORM_STATS_PATH = "user_adapted_norm.npz"

# Cortex app credentials -- read from the environment rather than hardcoded,
# unlike python/live_advance.py / python/record.py's example scripts.
CLIENT_ID = os.environ.get("EMOTIV_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("EMOTIV_CLIENT_SECRET", "")

INFERENCE_INTERVAL_SECONDS = 0.25  # sliding-window step: 4 predictions/s,
                                    # trivial against EEGNet's ~1.35ms cost
PROBABILITY_THRESHOLD = 0.5        # softmax confidence gate (chance=0.333)
CONSECUTIVE_TO_TRIGGER = 2         # ~0.5s of agreement before the *first*
                                    # tap of a newly-detected command
COOLDOWN_SECONDS = 0.4             # min gap between repeat taps of the same
                                    # already-confirmed command -- allows
                                    # consecutive attacks without spamming
TAP_DURATION_SECONDS = 0.1         # how long the button stays down per tap
# All four are tunable once you can feel how this plays in-game -- this
# model's leave-one-recording-out accuracy is ~0.5 on a 3-class task (see
# run_comparison.log / augmentation_sweep.log), so raw per-window
# predictions are noisy; these knobs trade responsiveness for stability.

ID_TO_CLASS = {class_id: class_name for class_name, class_id in CLASS_TO_ID.items()}
CLASS_TO_BUTTON = {
    "right": vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
    "left": vg.XUSB_BUTTON.XUSB_GAMEPAD_X,  # left face button, Xbox layout
}


# ---------------------------------------------------------------------------
# 1. Sliding-window EEG buffer + model inference
# ---------------------------------------------------------------------------
class MentalCommandClassifier:
    """Buffers streamed EEG samples and classifies the most recent window."""

    def __init__(self, model_path=MODEL_PATH, norm_stats_path=NORM_STATS_PATH):
        self.model = build_model(pretrained_path=model_path)
        self.model.eval()

        norm_stats = np.load(norm_stats_path)
        # Same per-channel mean/std computed at training time (see
        # BCI_Controller_train.py's load_edf_data) -- a single live window
        # can't be standardized against itself, so it must reuse these.
        self.channel_mean = norm_stats["mean"].reshape(-1, 1).astype(np.float32)
        self.channel_std = norm_stats["std"].reshape(-1, 1).astype(np.float32)

        self.sample_buffer = deque(maxlen=WINDOW_SAMPLES)
        self.buffer_lock = threading.Lock()

    def push_sample(self, channel_values):
        """channel_values: list[float], one value per EMOTIV_CHANNELS entry."""
        with self.buffer_lock:
            self.sample_buffer.append(channel_values)

    def buffer_ready(self):
        with self.buffer_lock:
            return len(self.sample_buffer) == WINDOW_SAMPLES

    def predict(self):
        """Runs one forward pass on the current window. Returns (class_name, probability)."""
        with self.buffer_lock:
            window_samples = list(self.sample_buffer)

        window = np.array(window_samples, dtype=np.float32).T  # (n_channels, n_times)
        window = (window - self.channel_mean) / self.channel_std
        window_tensor = torch.as_tensor(window).unsqueeze(0).to(DEVICE)  # (1, n_channels, n_times)

        with torch.no_grad():
            logits = self.model(window_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)
            predicted_id = int(probabilities.argmax())

        return ID_TO_CLASS[predicted_id], float(probabilities[predicted_id])


# ---------------------------------------------------------------------------
# 2. Debounced button controller
# ---------------------------------------------------------------------------
class ButtonController:
    """
    Turns a stream of per-window predictions into tapped button presses.
    Requires CONSECUTIVE_TO_TRIGGER agreeing, qualifying predictions before
    a command's *first* tap (raw per-window predictions are noisy -- see
    PROBABILITY_THRESHOLD comment above), then paces any further taps of
    that same still-active command by COOLDOWN_SECONDS -- so a sustained
    mental command produces steady consecutive taps rather than either a
    single tap or a spammed button.
    """

    def __init__(self, gamepad):
        self.gamepad = gamepad
        self.streak_class = None
        self.streak_length = 0
        self.last_tap_time = {}  # class_name -> time.monotonic() of last tap

    def update(self, predicted_class, probability):
        qualifying_class = predicted_class if probability >= PROBABILITY_THRESHOLD else "neutral"

        if qualifying_class == self.streak_class:
            self.streak_length += 1
        else:
            self.streak_class = qualifying_class
            self.streak_length = 1

        if qualifying_class not in CLASS_TO_BUTTON or self.streak_length < CONSECUTIVE_TO_TRIGGER:
            return

        now = time.monotonic()
        elapsed_since_last_tap = now - self.last_tap_time.get(qualifying_class, float("-inf"))
        if elapsed_since_last_tap >= COOLDOWN_SECONDS:
            self._tap(qualifying_class)
            self.last_tap_time[qualifying_class] = now

    def _tap(self, class_name):
        button = CLASS_TO_BUTTON[class_name]
        self.gamepad.press_button(button=button)
        self.gamepad.update()
        time.sleep(TAP_DURATION_SECONDS)
        self.gamepad.release_button(button=button)
        self.gamepad.update()
        print(f"[ButtonController] tapped '{class_name}'")

    def release_all(self):
        self.gamepad.reset()
        self.gamepad.update()


# ---------------------------------------------------------------------------
# 3. Cortex session: subscribe to raw EEG, drive the classifier + controller
# ---------------------------------------------------------------------------
class LiveMentalCommandController:
    """Owns the Cortex session, the classifier, and the button controller."""

    def __init__(self, client_id, client_secret):
        self.cortex = Cortex(client_id, client_secret, debug_mode=False)
        self.cortex.bind(create_session_done=self.on_create_session_done)
        self.cortex.bind(new_data_labels=self.on_new_data_labels)
        self.cortex.bind(new_eeg_data=self.on_new_eeg_data)
        self.cortex.bind(inform_error=self.on_inform_error)

        self.classifier = MentalCommandClassifier()
        self.gamepad = vg.VX360Gamepad()
        self.button_controller = ButtonController(self.gamepad)
        self.eeg_channel_indices = None  # resolved once stream labels arrive

    def start(self):
        """Blocks until the Cortex websocket closes (see cortex.py's open())."""
        self.cortex.open()

    def on_create_session_done(self, *args, **kwargs):
        # Subscribing directly to raw EEG -- unlike live_advance.py, we don't
        # need Emotiv's own profile/mental-command setup since we run our own
        # trained model instead of Cortex's built-in classifier.
        print("[live] session ready, subscribing to raw EEG stream")
        self.cortex.sub_request(["eeg"])

    def on_new_data_labels(self, *args, **kwargs):
        labels = kwargs.get("data")
        if labels["streamName"] != "eeg":
            return
        # The 'eeg' stream also carries non-EEG columns (COUNTER,
        # INTERPOLATED, RAW_CQ, ...) whose position isn't guaranteed, so
        # resolve our 14 channels by name instead of assuming a fixed layout.
        stream_labels = labels["labels"]
        self.eeg_channel_indices = [stream_labels.index(channel) for channel in EMOTIV_CHANNELS]
        print(f"[live] resolved EEG channel positions: {self.eeg_channel_indices}")

        inference_thread = threading.Thread(target=self.run_inference_loop, daemon=True)
        inference_thread.start()

    def on_new_eeg_data(self, *args, **kwargs):
        if self.eeg_channel_indices is None:
            return  # stream labels haven't arrived yet
        raw_row = kwargs.get("data")["eeg"]
        channel_values = [raw_row[index] for index in self.eeg_channel_indices]
        self.classifier.push_sample(channel_values)

    def run_inference_loop(self):
        print("[live] inference loop started, waiting for the window to fill...")
        while True:
            time.sleep(INFERENCE_INTERVAL_SECONDS)
            if not self.classifier.buffer_ready():
                continue
            predicted_class, probability = self.classifier.predict()
            print(f"[live] predicted={predicted_class:8s} p={probability:.2f}")
            self.button_controller.update(predicted_class, probability)

    def on_inform_error(self, *args, **kwargs):
        print("[live] Cortex error:", kwargs.get("error_data"))


# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------
def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "Set the EMOTIV_CLIENT_ID and EMOTIV_CLIENT_SECRET environment variables "
            "before running (see https://emotiv.gitbook.io/cortex-api#create-a-cortex-app)."
        )

    controller = LiveMentalCommandController(CLIENT_ID, CLIENT_SECRET)
    try:
        controller.start()
    except KeyboardInterrupt:
        print("\n[live] stopping, releasing any held buttons")
        controller.button_controller.release_all()
        controller.cortex.close()


if __name__ == "__main__":
    main()
