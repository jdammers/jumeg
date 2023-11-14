#!/usr/bin/env python
'''
==========
Plot data distribution
==========

Example script to plot a given data distribution compared with a standard
Gaussian distribution.
'''

import os.path as op
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from jumeg.jumeg_plot import plot_histo_fit_gaussian

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')

fname_raw = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw.fif')
fname_event = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')

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

data = epochs.get_data()

fig = plot_histo_fit_gaussian(data, nbins=100, fnout=None, show=True)
