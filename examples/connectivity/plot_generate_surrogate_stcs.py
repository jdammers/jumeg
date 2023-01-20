#!/usr/bin/env python

'''
=============
Generate surrogate STCs
=============


Example to show surrogate generation on STCs using the jumeg Surrogates
module.
'''

import os.path as op
import numpy as np
import matplotlib.pyplot as pl

from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs
import mne

from jumeg.jumeg_surrogates import Surrogates, check_power_spectrum

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')

fname_inv = op.join(data_path, 'MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif')
fname_raw = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw.fif')
fname_event = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')

stcs_testing = True

raw = mne.io.read_raw_fif(fname_raw)
events = mne.read_events(fname_event)
inverse_operator = read_inverse_operator(fname_inv)

# add a bad channel
raw.info['bads'] += ['MEG 2443']

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                       exclude='bads')

# define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13))


snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal", return_generator=False)

# compute surrogates for one STC
surr_stcs = Surrogates(stcs[0])
mysurr = surr_stcs.compute_surrogates(n_surr=10, return_generator=False)

# check if surrogates are correctly computed
assert not np.array_equal(mysurr[0].data,
                          mysurr[1].data), 'Surrogates mostly equal'
assert not np.array_equal(stcs[0].data,
                          mysurr[2].data), 'Surrogates equal to original'

# visualize results by plotting 1 the average across voxels
# of original and surrogate STC
pl.plot(stcs[0].data.mean(axis=0))
for i in mysurr:
    pl.plot(i.data.mean(axis=0), color='r')
pl.title('Averaged real vs surrogate source time courses')
pl.show()
