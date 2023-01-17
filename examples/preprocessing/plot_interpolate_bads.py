"""
Use suggest_bads to automatically identify bad MEG channels
and use interpolate_bads based on the center of mass of the
sensors for bad channel correction.
"""

import os.path as op

import mne
from mne.datasets import sample
from jumeg import suggest_bads
from jumeg import interpolate_bads

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')

raw_fname = op.join(data_path, 'MEG/sample/sample_audvis_raw.fif')

raw = mne.io.Raw(raw_fname, preload=True)
mybads, raw = suggest_bads(raw, show_raw=False, summary_plot=False)

# origin = None causes the method to use the sensor center of mass as origin
interpolate_bads(raw, origin=None, reset_bads=True)
