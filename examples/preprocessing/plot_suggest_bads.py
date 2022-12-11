#!/usr/bin/env python
'''
Example code to use the jumeg suggest bads functionality.
'''

import os.path as op
import mne
from mne.datasets import sample
from jumeg import suggest_bads

# provide the path of the filename:
data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
raw_fname = op.join(data_path, 'MEG/sample/sample_audvis_raw.fif')

raw = mne.io.Raw(raw_fname, preload=True)

mybads, raw = suggest_bads(raw, show_raw=False, summary_plot=False)
