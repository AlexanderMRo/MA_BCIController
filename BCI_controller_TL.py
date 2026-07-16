from braindecode.datasets import MOABBDataset

import numpy as np
import torch

from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
    create_windows_from_events,
)




###
# 1.
# import pre-existing dataset of library
subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
###


#####
# 2.
# preprocess data
low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Preprocessor("pick_types", eeg=True, meg=False, stim=False),  # Keep EEG sensors
    Preprocessor(
        lambda data, factor: np.multiply(data, factor),  # Convert from V to uV
        factor=1e6,
    ),
    Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    Preprocessor(
        exponential_moving_standardize,  # Exponential moving standardization
        factor_new=factor_new,
        init_block_size=init_block_size,
    ),
]

# Preprocess the data
preprocess(dataset, preprocessors, n_jobs=-1)
####

####

###
# 3. (Extraction of windows)
# Extraction of the trials (windows) from the time series is based on the events inside the dataset. One event is
# the demarcation of the stimulus or the beginning of the trial. In this example, we want to analyse 0.5 [s]
# long before the corresponding event and the duration of the event itself. Therefore, we set the
# trial_start_offset_seconds to -0.5 [s] and the trial_stop_offset_seconds to 0 [s].
#
# We extract from the dataset the sampling frequency, which is the same for all datasets in this case, and we tested it.

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info["sfreq"]
assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
# Calculate the window start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

# Create windows using braindecode function for this. It needs parameters to
# define how windows should be used.
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)
###


###
# 4.
# Split into Train + Test dataset
splitted = windows_dataset.split("session")
train_set = splitted["0train"]  # Session train
test_set = splitted["1test"]  # Session evaluation

####################################
###### TRAINING
#
# 1. create model (currently used: shallowfbcspnet, want to use: BENDR(newer+better generalisation)/EEGNET (foundation model+ a lot of docs to it))
cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 4
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_channels = windows_dataset[0][0].shape[0]
n_times = windows_dataset[0][0].shape[1]

model = ShallowFBCSPNet(
    n_chans=n_channels,
    n_outputs=n_classes,
    n_times=n_times,
    final_conv_length="auto",
)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    model.cuda()

# TODO might also tryto just use the pretrained models and see if see if we can finetune it