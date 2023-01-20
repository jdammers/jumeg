#!/usr/bin/env python
'''
====================
Plot noise reducer
====================

Script to show the application of noise reducer on jusample data.
'''

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from jumeg.jumeg_noise_reducer import noise_reducer

import mne

plt.ion()

# load the jumeg sample data (has to be BTI)
data_dir = os.environ['JUSAMPLE_MEG_PATH']
subject = '207184'
raw_fname = op.join(data_dir, 'recordings', subject,
                    'sample_207184_rest_EC-raw.fif')

raw = mne.io.Raw(raw_fname, preload=True)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, sharey=True)

picks = mne.pick_types(raw.info, meg='mag', exclude='bads')
raw.plot_psd(fmin=0., fmax=100., tmin=None, tmax=60.,
             n_fft=None, picks=picks, ax=ax1);
ax1.set_title('Original')

# notch filter
raw_notch = raw.copy().notch_filter(np.arange(50, 251, 50), picks=picks,
                                    filter_length='auto',
                                    notch_widths=None, n_jobs=4, method='fir',
                                    phase='zero-double',
                                    fir_window='hamming', fir_design='firwin')
raw_notch.plot_psd(fmin=0., fmax=100., tmin=None, tmax=60.,
                  n_fft=None, picks=picks, ax=ax2);
ax2.set_title('Notch filter 50Hz applied')

# powerline removal using noise_reducer
raw_nr_notch = noise_reducer(raw_fname, raw=raw.copy(), detrending=False,
                         reflp=None, refhp=None, refnotch=[50., 100., 150.],
                         return_raw=True, verbose=False)
raw_nr_notch.plot_psd(fmin=0., fmax=100., tmin=None, tmax=60.,
                  n_fft=None, picks=picks, ax=ax3);
ax3.set_title('Noise reducer notch filter 50Hz applied')


# remove high freq noise (>0.1Hz) from ref channels
raw_nr2 = noise_reducer(raw_fname, raw=raw_nr_notch, detrending=False,
                        reflp=None, refhp=0.1, refnotch=None,
                        return_raw=True, verbose=False)
raw_nr2.plot_psd(fmin=0., fmax=100., tmin=None, tmax=60.,
                 n_fft=None, picks=picks, ax=ax4);
ax4.set_title('Noise reducer high pass filtered 0.1Hz')

# remove low freq noise (<5Hz) from ref channels
raw_nr = noise_reducer(raw_fname, raw=raw_nr2, detrending=False,
                       reflp=5., refhp=None, refnotch=None,
                       return_raw=True, verbose=False)
raw_nr.plot_psd(fmin=0., fmax=100., tmin=None, tmax=60.,
                 n_fft=None, picks=picks, ax=ax5);
ax5.set_title('Noise reducer low pass filtered 5Hz')

plt.tight_layout()
plt.show()
