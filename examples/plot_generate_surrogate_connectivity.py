#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl

from jumeg.jumeg_surrogates import Surrogates

from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
import mne

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'

raw = mne.io.read_raw_fif(fname_raw)
events = mne.read_events(fname_event)
inverse_operator = read_inverse_operator(fname_inv)

# add a bad channel
raw.info['bads'] += ['MEG 2443']

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       exclude='bads')

# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13))

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal", return_generator=False)

# get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

# average the source estimates within each label using sign-flips to reduce
# signal cancellations, also here we can return a generator
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip',
                                         return_generator=False)

# compute surrogates on the first STC extracted for 68 labels
n_surr = 10
surr_ts = Surrogates(label_ts[0])
surr_label_ts = surr_ts.compute_surrogates(n_surr=n_surr, return_generator=True)

# visualize the surrogates
pl.plot(label_ts[0][0, :], 'b')
for lts in surr_label_ts:
    pl.plot(lts[0, :], 'r')
pl.title('Extracted label time courses - real vs surrogates')
pl.show()
