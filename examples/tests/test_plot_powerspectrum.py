"""
Compute infomax ICA on raw data.
"""

import os.path as op
import mne
from mne.datasets import sample
from jumeg.jumeg_plot import plot_powerspectrum

data_path = sample.data_path()

raw_fname = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw.fif')

raw = mne.io.Raw(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg='mag', exclude='bads')
raw.crop(0, 60)  # use 60s of data

psd_fname = plot_powerspectrum(raw_fname, raw=raw, picks=None, dir_plots=None,
                               tmin=None, tmax=None, fmin=0.0, fmax=None, n_fft=4096,
                               average=True)

print(psd_fname)
