#!/usr/bin/env python

# Script reads meg and eeg as fif files and combines them into one raw fif file. 
# jumeg_import_eeg2meg
#
# Authors: praveen.sripad@rwth-aachen.de
# License: BSD 3 clause

import mne, sys
import matplotlib.pyplot as pl

def jumeg_resample(l_sfreq, h_sfreq, samp_length):
    ''' 
    Downsampling function to downsample signal of samp_length from higher sampling frequency to lower sampling frequency.

    Parameters 
    ----------
    l_sfreq: Lower sampling frequency.
    h_sfreq: Higher sampling frequency.
    samp_length: Signal length or number of time points of signal with lower sampling frequency.

    Returns
    -------
    resamp_list: List of time points of downsampled signal.
    '''
    eps_limit = round((0.90 / h_sfreq), 3)
    resamp_list = []
    j = 0
    for i in xrange(0, samp_length):
        l_tp = round((i / l_sfreq), 3)
        while (l_tp - round((j / h_sfreq), 3) > eps_limit):
            j += 1
        resamp_list.append(j)
        j += 1
    return resamp_list

raw_fname = sys.argv[1]
eeg_fname = sys.argv[2]

def combine_meeg(raw_fname, eeg_fname):
    
    raw = mne.fiff.Raw(raw_fname, preload=True)
    eeg = mne.fiff.Raw(eeg_fname, preload=True)
    
    # Filter both signals from 1-200 Hz
    flow, fhigh = 1.0, 200.0
    filter_type = 'butter'
    filter_order = 4
    njobs = 2
    picks_fil = mne.fiff.pick_types(raw.info, meg=True, eog=True, ecg=True, exclude='bads')
    raw.filter(flow, fhigh, picks=picks_fil, n_jobs=njobs, method='iir', \
               iir_params={'ftype': filter_type, 'order': filter_order})
    picks_fil = mne.fiff.pick_types(eeg.info, meg=False, eeg=True, exclude='bads')
    eeg.filter(flow, fhigh, picks=picks_fil, n_jobs=njobs, method='iir', \
               iir_params={'ftype': filter_type, 'order': 2})
    
    # Find sync pulse S128 in stim channel of EEG signal.
    start_idx_eeg = mne.find_events(eeg, stim_channel='STI 014', output='onset')[0, 0]
    
    # Find sync pulse S128 in stim channel of MEG signal.
    start_idx_raw = mne.find_events(raw, stim_channel='STI 014', output='onset')[0, 0]
    
    # Start times for both eeg and meg channels
    start_time_eeg = eeg.index_as_time(start_idx_eeg)
    start_time_raw = raw.index_as_time(start_idx_raw)
    
    # Stop times for both eeg and meg channels
    stop_time_eeg = eeg.index_as_time(eeg.last_samp)
    stop_time_raw = raw.index_as_time(raw.last_samp)
    
    # Choose channel with shortest duration (usually MEG)
    meg_duration = stop_time_eeg - start_time_eeg
    eeg_duration = stop_time_raw - start_time_raw
    diff_time = min(meg_duration, eeg_duration)
    
    # Reset both the channel times based on shortest duration
    end_time_eeg = diff_time + start_time_eeg
    end_time_raw = diff_time + start_time_raw
    
    # Calculate the index of the last time points
    stop_idx_eeg = eeg.time_as_index(round(end_time_eeg, 3))[0]
    stop_idx_raw = raw.time_as_index(round(end_time_raw, 3))[0]
    print start_idx_raw, stop_idx_raw, start_idx_eeg, stop_idx_eeg
    
    eeg_data, eeg_times = eeg[:, start_idx_eeg:stop_idx_eeg]
    _, raw_times = raw[:, start_idx_raw:stop_idx_raw]
    
    # Resample eeg signal (jumeg_resample uses standard function)
    resamp_list = jumeg_resample(raw.info['sfreq'], eeg.info['sfreq'], raw_times.shape[0])
    
    # Update eeg signal
    eeg._data, eeg._times = eeg_data[:, resamp_list], eeg_times[resamp_list]
    
    # Update meg signal
    #raw._data, raw._times = raw_data, raw_times
    raw._data, raw._times = raw[:, start_idx_raw:stop_idx_raw]
    
    # Identify raw channels for ECG, EOG and STI and replace it with relevant data.
    raw._data[raw.ch_names.index('ECG 001')] = eeg._data[0]
    raw._data[raw.ch_names.index('EOG 001')] = eeg._data[1]
    raw._data[raw.ch_names.index('EOG 002')] = eeg._data[2]
    raw._data[raw.ch_names.index('STI 014')] = eeg._data[3]
    
    # Write the combined FIF file to disk.
    raw.save(raw_fname.split('.')[0] + '_processed' + '.fif')

'''
with open('list.txt') as temp_file:
  flist = [line.rstrip('\n') for line in temp_file]

for i in flist:
    combine_meeg(i+'-meg.fif', i+'-eeg_raw.fif')
'''

combine_meeg(raw_fname, eeg_fname)
