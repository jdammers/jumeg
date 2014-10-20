'''
Utilities module
'''
import mne
import sys
import os
import numpy as np
import scipy as sci
from mne.utils import logger


def check_jumeg_standards(fnames):
    '''
    Checks for file name extension and provides information on type of file

    fnames: str or list
    '''

    if isinstance(fnames, list):
        fname_list = fnames
    else:
        if isinstance(fnames, str):
            fname_list = list([fnames])
        else:
            fname_list = list(fnames)

    print fname_list
    # loop across all filenames
    for fname in fname_list:
        print fname
        if fname == '' or not fname.endswith('.fif'):
            print 'Empty string or not a FIF format filename.'
        elif fname.endswith('-meg.fif') or fname.endswith('-eeg.fif'):
            print 'Raw FIF file with only MEG or only EEG data.'
        elif fname.split('-')[-1] == 'raw.fif':
            print 'Raw FIF file - Subject %s, Experiment %s, Data %s, Time %s, \
                   Trial number %s.' \
                  %(fname.split('_')[0], fname.split('_')[1], fname.split('_')[2], \
                    fname.split('_')[3], fname.split('_')[4])
            print 'Processing identifier in the file %s.' \
                  %(fname.strip('-raw.fif').split('_')[-1])
        elif fname.split('-')[-1] == 'ica.fif':
            print 'FIF file storing ICA session.'
        elif fname.split('-')[-1] == 'evoked.fif':
            print 'FIF file with averages.'
        elif fname.split('-')[-1] == 'epochs.fif':
            print 'FIF file with epochs.'
        elif fname.split('-')[-1] == 'empty.fif':
            print 'Empty room FIF file.'
        else:
            print 'No known file info available. Filename does not follow conventions.'
        
        print 'Please verify if the information is correct and make the appropriate changes!'
    return


def mark_bads_batch(subject_list, subjects_dir=None):
    '''
    Opens all raw files ending with -raw.fif in subjects 
    directory for marking bads.

    Parameters
    ----------
    subject_list: List of subjects. 
    subjects_dir: The subjects directory. If None, the default SUBJECTS_DIR 
                  from environment will be considered.

    Output
    ------
    The raw files with bads marked are saved with _bcc (for bad channels checked)
    added to the file name. 
    '''
    for subj in subject_list:
        print "For subject %s"%(subj)
        if not subjects_dir: SUBJECTS_DIR = os.environ['SUBJECTS_DIR']
        dirname = SUBJECTS_DIR+'/'+subj
        for raw_fname in os.listdir(dirname):
            if raw_fname.endswith('-raw.fif'):
                print "Raw calculations for file %s"%(dirname+'/'+raw_fname)
                raw = mne.io.Raw(dirname+'/'+raw_fname)
                raw.plot(block=True)
                print 'The bad channels marked are'+raw.info['bads']
                raw.save(raw.info['filename'].split('/')[-1].split('.')[0]+ \
                         '_bcc-raw.fif')
                return

def rescale_artifact_to_signal(signal, artifact):
    ''' 
    Rescales artifact (ECG/EOG) to signal for plotting purposes 
    For evoked data, pass signal.data.mean(axis=0) and
    artifact.data.mean(axis=0).
    '''
    b = (signal.max() - signal.min()) / (artifact.max() + artifact.min())
    a = signal.max()
    rescaled_artifact = artifact * b + a
    return rescaled_artifact / 1e15

def peak_counter(signal):
    ''' Simple peak counter using scipy argrelmax function. '''
    return sci.signal.argrelmax(signal)[0].shape

def update_description(raw, comment):
    ''' Updates the raw description with the comment provided. '''
    raw.info['description'] = str(raw.info['description']) + ' ; ' + comment

def chop_raw_data(raw, start_time=60.0, stop_time=360.0):
    ''' 
    This function extracts specified duration of raw data 
    and write it into a fif file.
    Five mins of data will be extracted by default.

    Parameters
    ----------

    raw: Raw object. 
    start_time: Time to extract data from in seconds. Default is 60.0 seconds. 
    stop_time: Time up to which data is to be extracted. Default is 360.0 seconds.

    '''
    # Check if data is longer than required chop duration.
    if (raw.n_times / (raw.info['sfreq'])) < (stop_time + 60.0):
        logger.info("The data is not long enough.")
        return
    # Obtain indexes for start and stop times.
    assert start_time < stop_time, "Start time is greater than stop time."
    start_idx = raw.time_as_index(start_time)
    stop_idx = raw.time_as_index(stop_time)
    data, times = raw[:, start_idx:stop_idx]
    raw._data,raw._times = data, times
    dur = int((stop_time - start_time) / 60)
    raw.save(raw.info['filename'].split('/')[-1].split('.')[0]+'_'+str(dur)+'m.fif')
    # For the moment, simply warn.
    logger.warning('The file name is not saved in standard form.')
    return

def make_surrogates_sklearn(epochs):
    ''' 
    Make surrogate epochs using sklearn. Destroy time-phase relationship for each trial.
    
    Parameters
    ----------
    Epochs Object.

    Output
    ------
    Surrogate Epochs object
    '''
    surrogate = epochs.copy()
    surr = surrogate.get_data()
    for trial in range(len(surrogate)):
        for channel in range(len(surrogate.ch_names)):
            from sklearn.utils import check_random_state
            rng = check_random_state(channel)
            order = np.argsort(rng.randn(len(surrogate.times)))
            surr[trial, channel, :] = surr[trial, channel, order]
    surrogate._data = surr
    ps1 = np.abs(np.fft.fft(surr))**2
    ps2 = np.abs(np.fft.fft(epochs.get_data()))**2
    assert ps1.all() == ps2.all(), 'The power content does not match. Error.'
    return surrogate
    
def make_surrogates_shuffling(epochs):
    ''' 
    Make surrogate epochs by simply shuffling. Destroy time-phase relationship for each trial.
    
    Parameters
    ----------
    Epochs Object.

    Output
    ------
    Surrogate Epochs object
    '''
    surrogate = epochs.copy()
    surr = surrogate.get_data()
    for trial in range(len(epochs)):
        for channel in range(len(epochs.ch_names)):
            np.random.shuffle(surr[trial, channel, :])
    surrogate._data = surr
    ps1 = np.abs(np.fft.fft(surr))**2
    ps2 = np.abs(np.fft.fft(epochs.get_data()))**2
    assert ps1.all() == ps2.all(), 'The power content does not match. Error.'
    return surrogate
