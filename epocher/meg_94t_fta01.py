#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import mne
from mne.time_frequency import tfr_morlet


pin='/localdata/frank/data/MEG94T/mne/205386/MEG94T/120906_1401/1'
fraw_LLon='205386_MEG94T_120906_1401_1_c,rfDC,fihp1n,ocarta,ctpsbr-LLon-epo.fif'
fraw_LRon='205386_MEG94T_120906_1401_1_c,rfDC,fihp1n,ocarta,ctpsbr-LRon-epo.fif'

fraw_LLon='205386_MEG94T_120906_1401_1_c,rfDC,fihp1n,ocarta,ctpsbr_co-LLon-epo.fif'
fraw_LRon='205386_MEG94T_120906_1401_1_c,rfDC,fihp1n,ocarta,ctpsbr_co-LRon-epo.fif'



epo=mne.read_epochs(pin+'/'+fraw_LLon)

freqs = np.arange(6, 200, 3)  # define frequencies of interest
n_cycles = freqs / 6.  # different number of cycle per frequency
power, itc = tfr_morlet(epo, freqs=freqs, n_cycles=n_cycles, use_fft=False,
                        return_itc=True, decim=3, n_jobs=4)



power.plot_topo(baseline=(-0.2, 0), mode='logratio', title='Average power')

#power.plot([189], baseline=(-0.2, 0), mode='logratio')
#


from mne.time_frequency import induced_power
power, phase_lock = induced_power(epo, Fs=Fs, frequencies=frequencies, n_cycles=2, n_jobs=1)


from mne.time_frequency import tfr_stockwell


power, itc = tfr_stockwell(epo, fmin=6 ,fmax=100,return_itc=True,decim=2,n_jobs=4)



from mne.time_frequency import induced_power
power, phase_lock = induced_power(epochs_data, Fs=Fs, frequencies=frequencies, n_cycles=2, n_jobs=1)

# x(t)=cos(2*pi*5*t)+cos(2*pi*10*t)+cos(2*pi*20*t)+cos(2*pi*50*t)