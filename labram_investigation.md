# Why a pretrained Labram foundation model doesn't fit this task

## Goal

Compare three ways to build the 3-class (neutral/left/right) mental-command
classifier, to decide what to deploy for real-time button-press emulation:
(a) EEGNet trained from scratch, (b) EEGNet pretrained on the public
BNCI2014_001 motor-imagery dataset, and (c) `InterpolatedLaBraM` -- a
braindecode wrapper around LaBraM, a real self-supervised-pretrained EEG
transformer, with actual downloadable weights (unlike EEGNet, which
braindecode ships as architecture-only).

## What got built

`InterpolatedLaBraM` looked like a clean fit: it wraps `Labram` with a
`ChannelInterpolationLayer` that projects an arbitrary electrode layout (our
14-channel Emotiv EPOC X) onto whatever montage the backbone expects,
handling the channel mismatch automatically. Single-trial inference measured
at 47.6ms on CPU (vs. EEGNet's 1.35ms) -- 35x heavier, but still trivial
against a ~2-second decision window, so latency was never the blocker.

## Where it broke: every available checkpoint needs 8-15 seconds of context

Three checkpoints were found and checked (`torch.hub`/Hugging Face `siblings`
listing, config files, and -- critically -- the actual saved tensor shapes,
not just documented metadata):

| Source | Channels | Real window (from weight shapes) | Notes |
|---|---|---|---|
| `braindecode/Labram-Braindecode` | 64 (fixed at save time) | 8s (`temporal_embedding`: 9 slots x 200-sample patches) | Also channel-mismatched with `InterpolatedLaBraM`'s built-in 128-channel target -- `strict=False` doesn't rescue shape mismatches, only missing/extra keys |
| `braindecode/labram-pretrained` (the repo the library's own docstring points `.from_pretrained()` at) | 128 -- matches `InterpolatedLaBraM` exactly, includes all 14 Emotiv channel names | 15s (`n_times=3000` in `config.json`) | Channels fit perfectly; window doesn't fit inside our 12s recordings at all |
| `eeg-telecom-paris/labram-base-official` | 22 (BCI-IV-2a motor-cortex montage -- same one used for BNCI2014_001) | **15s in practice** (`time_embed` shape is `(1, 16, 200)` = 15 patches, despite `model_cfg.yaml` claiming `input_window_seconds: 4`) | The published config describes a downstream fine-tuning setup, not what the saved backbone weights were actually pretrained with |

One genuinely good finding along the way: channel count turned out not to be
the hard constraint. Braindecode's `Labram` uses a **fixed 128-position
embedding table** regardless of `n_chans` -- it looks up whichever channels
you actually have by scalp position, not by resizing a tensor. Every
checkpoint checked was channel-flexible in principle.

The temporal side is the real wall, and it isn't a matter of picking a better
checkpoint: every LaBraM-family checkpoint found needs 8-15 seconds of
context, because that's how these models are actually pretrained (long-range
masked-patch prediction is close to the definition of the self-supervised
objective this architecture family uses). Forcing a ~2-4s window would mean
slicing 11-13 of the pretrained temporal positions out of `temporal_embedding`
-- discarding most of what the pretraining actually learned, and turning
"does real pretraining help" into "does a mutilated fragment of pretraining
help," which isn't the comparison this was meant to answer.

## Why the window length can't just be relaxed

This project settled on a ~2-second decision window specifically because
end-to-end latency for real-time game control is dominated by how much EEG
must be observed before a decision can be made, not by model inference time
(both EEGNet and Labram are fast enough to be irrelevant here). An 8-15
second requirement is a non-starter for that goal regardless of accuracy.
It also isn't just a latency compromise: even the *movement* epoch barely
fits inside a 12s recording at 8s, and no configuration leaves room for a
matching *neutral* epoch from the same file's pre-marker baseline, which is
how every recording's neutral trial is currently constructed.

## Bug found and fixed along the way (unrelated to the above)

`torch.hub.load_state_dict_from_url` caches downloads by filename, not URL.
An early debugging attempt used the wrong URL form
(`/blob/main/...`, HuggingFace's HTML viewer page) and silently cached an
88KB HTML stub under the real checkpoint's filename. Every subsequent
download attempt using that filename silently reused the corrupted cache
entry instead of re-downloading, producing `UnpicklingError: invalid load
key, '<'.` regardless of whether the URL was later corrected. Fixed by using
a distinct cache filename; the real 22.4MB checkpoint then downloaded and
loaded correctly (before the deeper shape-mismatch issue above surfaced).

## Conclusion

Dropped. Proceeding with the two EEGNet arms (random-init baseline vs.
pretrained on BNCI2014_001), both of which fit the 2s real-time window
natively and are already working.
