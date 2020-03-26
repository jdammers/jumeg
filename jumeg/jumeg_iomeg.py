#!/usr/bin/env python
'''
Functions to perform IO operations.

Authors: Praveen Sripad  <praveen.sripad@rwth-aachen.de>
         Frank Boers     <f.boers@fz-juelich.de>

License: BSD 3 clause
'''


def wrapper_brain_vision2fiff(header_fname):
    '''
    Python wrapper for mne_brin_vision2fiff binary.
    Please make sure the MNE_BIN_PATH environment variable is set correctly.
    Parameters
    ----------
    header_fname: Header file name. The .eeg data file is automatically found.
    '''

    import os

    if not os.environ['MNE_BIN_PATH']:
        print("MNE_BIN_PATH not correctly set.")
        return

    if header_fname is "" or not header_fname.endswith('vhdr'):
        print("Usage: .py <header_file>")
        print("Please use the original binary to pass other arguments.")
        return
    else:
        print("The header file name provided is %s" % (header_fname))

    mne_brain_vision2fiff_path = os.environ['MNE_BIN_PATH'] + \
                                 '/mne_brain_vision2fiff'
    os.system(mne_brain_vision2fiff_path + ' --header ' +
              header_fname + ' --out ' + header_fname.split('.')[0] + '-eeg')
    # mne_brain_vision2fiff always adds _raw.fif extension
    # to its files, so it has to be renamed to -eeg.fif.
    os.system('mv %s %s' % (header_fname.split('.')[0] +
                            '-eeg_raw.fif', header_fname.split('.')[0] + '-eeg.fif'))
    print("Output in FIF format can be found at %s" \
          % (header_fname.split('.')[0] + '-eeg.fif'))


def jumeg_resample(l_sfreq, h_sfreq, samp_length,
                   events=None):
    '''
    Downsampling function to resample signal of samp_length from
    higher sampling frequency to lower sampling frequency.

    Parameters
    ----------
    l_sfreq: Lower sampling frequency.
    h_sfreq: Higher sampling frequency.
    samp_length: Signal length or number of time points of signal
                    with lower sampling frequency.
    events: If set resampling is chosen in a way that all
                    events are included

    Returns
    -------
    resamp_list: List of time points of downsampled signal.
    '''
    import numpy as np

    eps_limit = round((0.90 / h_sfreq), 3)
    resamp_list = []
    j = 0
    for i in range(0, samp_length):
        l_tp = round((i / l_sfreq), 3)
        while l_tp - round((j / h_sfreq), 3) > eps_limit:
            j += 1
        resamp_list.append(j)
        j += 1

    # if stim channel is set, make sure that all events are
    # included
    if np.any(events):
        idx_events = events[:, 0]

        for idx in idx_events:
            if idx not in resamp_list:
                resamp_list[np.argmin(np.abs(resamp_list - idx))] = idx

    return resamp_list


def combine_meeg(raw_fname, eeg_fname, flow=0.6, fhigh=200,
                 filter_order=2, njobs=-1):
    '''
    Functions combines meg data with eeg data. This is done by: -
        1. Adjust MEG and EEG data length.
        2. Resampling EEG data channels to match sampling
           frequency of MEG signals.
        3. Write EEG channels into MEG fif file and write to disk.

    Parameters
    ----------
    raw_fname: FIF file containing MEG data.
    eeg_fname: FIF file containing EEG data.
    flow, fhigh: Low and high frequency limits for filtering.
                 (default 0.6-200 Hz)
    filter_order: Order of the Butterworth filter used for filtering.
    njobs : Number of jobs.

    Warning: Please make sure that the filter settings provided
             are stable for both MEG and EEG data.
    Only channels ECG 001, EOG 001, EOG 002 and STI 014 are written.
    '''

    import numpy as np
    import mne
    from mne.utils import logger

    if not raw_fname.endswith('-meg.fif') and \
            not eeg_fname.endswith('-eeg.fif'):
        logger.warning('Files names are not standard. \
                        Please use standard file name extensions.')

    raw = mne.io.Raw(raw_fname, preload=True)
    eeg = mne.io.Raw(eeg_fname, preload=True)

    # Filter both signals
    filter_type = 'butter'
    logger.info('The MEG and EEG signals will be filtered from %s to %s' \
                % (flow, fhigh))
    picks_fil = mne.pick_types(raw.info, meg=True, eog=True, \
                               ecg=True, exclude='bads')
    raw.filter(flow, fhigh, picks=picks_fil, n_jobs=njobs, method='iir', \
               iir_params={'ftype': filter_type, 'order': filter_order})
    picks_fil = mne.pick_types(eeg.info, meg=False, eeg=True, exclude='bads')
    eeg.filter(flow, fhigh, picks=picks_fil, n_jobs=njobs, method='iir', \
               iir_params={'ftype': filter_type, 'order': filter_order})

    # Find sync pulse S128 in stim channel of EEG signal.
    start_idx_eeg = mne.find_events(eeg, stim_channel='STI 014', \
                                    output='onset')[0, 0]

    # Find sync pulse S128 in stim channel of MEG signal.
    start_idx_raw = mne.find_events(raw, stim_channel='STI 014', \
                                    output='onset')[0, 0]

    # Start times for both eeg and meg channels
    start_time_eeg = eeg.times[start_idx_eeg]
    start_time_raw = raw.times[start_idx_raw]

    # Stop times for both eeg and meg channels
    stop_time_eeg = eeg.times[eeg.last_samp]
    stop_time_raw = raw.times[raw.last_samp]

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

    events = mne.find_events(eeg, stim_channel='STI 014', output='onset',
                             consecutive=True)
    events = events[np.where(events[:, 0] < stop_idx_eeg)[0], :]
    events = events[np.where(events[:, 0] > start_idx_eeg)[0], :]
    events[:, 0] -= start_idx_eeg

    eeg_data, eeg_times = eeg[:, start_idx_eeg:stop_idx_eeg]
    _, raw_times = raw[:, start_idx_raw:stop_idx_raw]

    # Resample eeg signal
    resamp_list = jumeg_resample(raw.info['sfreq'], eeg.info['sfreq'], \
                                 raw_times.shape[0], events=events)

    # Update eeg signal
    eeg._data, eeg._times = eeg_data[:, resamp_list], eeg_times[resamp_list]

    # Update meg signal
    raw._data, raw._times = raw[:, start_idx_raw:stop_idx_raw]
    raw._first_samps[0] = 0
    raw._last_samps[0] = raw._data.shape[1] - 1

    # Identify raw channels for ECG, EOG and STI and replace it with relevant data.
    logger.info('Only ECG 001, EOG 001, EOG002 and STI 014 will be updated.')
    raw._data[raw.ch_names.index('ECG 001')] = eeg._data[0]
    raw._data[raw.ch_names.index('EOG 001')] = eeg._data[1]
    raw._data[raw.ch_names.index('EOG 002')] = eeg._data[2]
    raw._data[raw.ch_names.index('STI 014')] = eeg._data[3]

    # Write the combined FIF file to disk.
    raw.save(raw_fname.split('-')[0] + ',meeg-raw.fif', overwrite=True)
