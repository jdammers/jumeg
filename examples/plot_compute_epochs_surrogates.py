#!/usr/bin/env python

'''
Example to show surrogate generation on Epochs using the jumeg Surrogates
module.
'''

import numpy as np
import matplotlib.pyplot as pl
from mne.datasets import sample
import mne

from jumeg.jumeg_surrogates import Surrogates, check_power_spectrum

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'

fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

epochs_testing = False

raw = mne.io.read_raw_fif(fname_raw)
events = mne.read_events(fname_event)

# add a bad channel
raw.info['bads'] += ['MEG 2443']

# pick MEG channels
picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False,
                       exclude='bads')

# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12))

# initialize the Surrogates object
surr_epochs = Surrogates(epochs)

n_surr = 10  # number of surrogates
mode = 'randomize_phase'
mysurr = surr_epochs.compute_surrogates(n_surr=n_surr,
                                        mode=mode, return_generator=False)

# get one epochs for plotting
# for epochs, a generator is always returned
first_surr_epoch = mysurr.next()

# visualize surrogates
fig, (ax1, ax2) = pl.subplots(2, 1)
epochs.average().plot(axes=ax1, show=False, titles='Evoked')
first_surr_epoch.average().plot(axes=ax2, show=False, titles='Surrogate Evoked')
pl.show()
