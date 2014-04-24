import mne, sys
import numpy as np

def mark_bads_batch(subject_list, subjects_dir=None)
    '''
    Opens all raw files ending with -raw.fif in subjects directory for marking bads.

    Parameters
    ----------
    subject_list: List of subjects. 
    subjects_dir: The subjects directory. If None, the default SUBJECTS_DIR from environment will be considered.

    Output
    ------
    The raw files with bads marked are saved with _bads_marked.fif suffix. 
    '''
    for subj in subjects:
        print "For subject %s"%(subj)
        if not subjects_dir: SUBJECTS_DIR = os.environ['SUBJECTS_DIR']
        dirname = SUBJECTS_DIR+'/'+subj
        for raw_fname in os.listdir(dirname):
            if raw_fname.endswith('-raw.fif')
                print "Raw calculations for file %s"%(dirname+'/'+raw_fname)
                raw = mne.fiff.Raw(dirname+'/'+raw_fname)
                raw.plot(block=True)
                print 'The bad channels marked are'+raw.info['bads']
                raw.save(raw.info['filename'].split('/')[-1].split('.')[0]+'_bads_marked.fif')
                return

def rescale_artifact_to_signal(signal, artifact):
    ''' Rescales artifact (ECG/EOG) to signal for plotting purposes '''
    b = (signal.max() - signal.min()) / (artifact.max() + artifact.min())
    a = signal.max()
    rescaled_artifact = artifact * b + a
    return rescaled_artifact / 1e15

def rescale_evoked_artifact(evoked_meg_signal, evoked_artifact):
    """ Function to rescale an artifact evoked object to MEG evoked for display purposes. Only for grad. """
    b = (evoked_meg_signal.data.mean(axis=0).max() - evoked_meg_signal.data.mean(axis=0).min()) / \
        (evoked_artifact.max() + evoked_artifact.min())
    a = evoked_meg_signal.data.mean(axis=0).max() * 1e15
    rescaled_artifact = evoked_artifact * b * 1e15 + a
    return rescaled_artifact

def peak_counter(signal):
    ''' Simple peak counter '''
    count = 0
    thresh = signal.mean() + (signal.max() - signal.mean()) / 2
    for i in signal:
        if i > thresh:
            count += 1
    return count

def update_description(raw, comment):
    ''' Updates the raw description with the comment provided. '''
    raw.info['description'] = raw.info['description'] + ' \n ' + comment

def extract_five_mins_raw_data(raw):
    ''' This function extracts five mins of raw data excluding the first minute. '''
    # Check if data is longer than 6 mins. 
    if (raw.n_times / (raw.info['sfreq'] * 60)) < 6.0:
        print "The data is not long enough. Please provide a data longer than 6 minutes."
        return
    # Obtain indexes for the 60th and 360th second.
    start_time = raw.time_as_index(60.0)
    stop_time = raw.time_as_index(360.0)
    data, times = raw[:, start_time:stop_time]
    raw._data,raw._times = data, times
    raw.save(raw.info['filename'].split('/')[-1].split('.')[0]+'_5m.fif')
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
