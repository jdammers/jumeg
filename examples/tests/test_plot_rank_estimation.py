#!/usr/bin/env python
'''
Example script to estimate the rank of the given data array.
'''

import os.path as op
import matplotlib.pyplot as plt
import mne
from mne.datasets import sample
from jumeg.jumeg_utils import rank_estimation

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')

raw_fname = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw.fif')
fname_event = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')

raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(fname_event)

# add a bad channel
raw.info['bads'] += ['MEG 2443']

# pick MEG channels
picks = mne.pick_types(raw.info, meg='mag', eeg=False, stim=False, eog=False,
                       exclude='bads')

data = raw.get_data()[picks, :]

rank_all, rank_median = rank_estimation(data)

print('Ranks in order: MIBS, BIC, GAP, AIC, MDL, pct95, pct99: ', rank_all)
print('The median of the data is %f' % rank_median)
