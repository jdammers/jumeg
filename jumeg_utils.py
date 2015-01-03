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


def get_sytem_type(info):
    """
    Function to get type of the system used to record
    the processed MEG data
    """
    from mne.io.constants import FIFF
    chs = info.get('chs')
    coil_types = set([ch['coil_type'] for ch in chs])
    channel_types = set([ch['kind'] for ch in chs])
    has_4D_mag = FIFF.FIFFV_COIL_MAGNES_MAG in coil_types
    ctf_other_types = (FIFF.FIFFV_COIL_CTF_REF_MAG,
                       FIFF.FIFFV_COIL_CTF_REF_GRAD,
                       FIFF.FIFFV_COIL_CTF_OFFDIAG_REF_GRAD)
    has_CTF_grad = (FIFF.FIFFV_COIL_CTF_GRAD in coil_types or
                    (FIFF.FIFFV_MEG_CH in channel_types and
                     any([k in ctf_other_types for k in coil_types])))
    if has_4D_mag:
        system_type = 'magnesWH3600'
    elif has_CTF_grad:
        system_type = 'CTF-275'
    else:
        # ToDo: Expand method to also cope with other systems!
        print "System type not known!"
        system_type = None

    return system_type


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

#######################################################
#                                                     
# make surrogates from Epochs
#                                                     
#######################################################
def make_surrogates_epochs(epochs, check_power=False):
    ''' 
    Make surrogate epochs using sklearn. Destroy time-phase relationship for each trial.
    
    Parameters
    ----------
    Epochs Object.

    Output
    ------
    Surrogate Epochs object
    '''
    from sklearn.utils import check_random_state

    surrogate = epochs.copy()
    surr = surrogate.get_data()
    for trial in range(len(surrogate)):
        for channel in range(len(surrogate.ch_names)):            
            rng = check_random_state(channel)
            order = np.argsort(rng.randn(len(surrogate.times)))
            surr[trial, channel, :] = surr[trial, channel, order]
    surrogate._data = surr
    if (check_power):
        ps1 = np.abs(np.fft.fft(surr))**2
        ps2 = np.abs(np.fft.fft(epochs.get_data()))**2
        assert ps1.all() == ps2.all(), 'The power content does not match. Error.'

    return surrogate
    
# def make_surrogates_epoch_numpy(epochs):
#     ''' 
#     Make surrogate epochs by simply shuffling. Destroy time-phase relationship for each trial.
    
#     Parameters
#     ----------
#     Epochs Object.

#     Output
#     ------
#     Surrogate Epochs object
#     '''
#     surrogate = epochs.copy()
#     surr = surrogate.get_data()
#     for trial in range(len(epochs)):
#         for channel in range(len(epochs.ch_names)):
#             np.random.shuffle(surr[trial, channel, :])
#     surrogate._data = surr
#     ps1 = np.abs(np.fft.fft(surr))**2
#     ps2 = np.abs(np.fft.fft(epochs.get_data()))**2
#     assert ps1.all() == ps2.all(), 'The power content does not match. Error.'
#     return surrogate


#######################################################
#                                                     
# make surrogates CTPS phase trials
#                                                     
#######################################################
def make_surrogates_ctps(phase_array, nrepeat=1000, mode='shuffle', n_jobs=4, 
                        verbose=None):
    ''' calculate surrogates from an array of (phase) trials 
        by means of shuffling the phase

    Parameters
    ----------
    phase_trial : 4d ndarray of dimension [nfreqs x ntrials x nchan x nsamples]

    Optional:
    nrepeat: 

    mode: 2 different modi are allowed.
        'mode=shuffle' whill randomly shuffle the phase values. This is the default
        'mode=shift' whill randomly shift the phase values
    n_jobs: number of cpu nodes to use
    verbose:  verbose level (does not work yet)
    fnout: 'fnout=filename' will save the surrogates to (numpy) file


    Returns
    -------
    pt : shuffled phase trials

    '''

    from joblib import Parallel, delayed
    from mne.parallel import parallel_func
    from mne.preprocessing.ctps_ import kuiper
    from jumeg_math import shuffle_data
    
    nfreq, ntrials, nsources, nsamples  = phase_array.shape
    ks = np.zeros((nfreq,nrepeat,nsources, nsamples))        # Kuiper's stat
    pk = np.zeros((nfreq,nrepeat,nsources, nsamples))        # significance value of ks


    # create surrogates:  parallised over nrepeats 
    parallel, my_kuiper, _ = parallel_func(kuiper, n_jobs, verbose=verbose)
    for ifreq in range(nfreq):
        for isource in range(nsources):
            #print ">>> working on frequency: ",bp[ifreq,:],"   source: ",isource+1  
            print ">>> working on frequency range: ",ifreq+1,"   source: ",isource+1  
            pt = phase_array[ifreq, :, isource, :]  # extract [ntrials, nsamp]   

            # surrogate phase trails:  shuffle phase values for all repititions
            pt_s = Parallel(n_jobs=n_jobs, verbose=0)(delayed(shuffle_data)
                (pt, mode='shuffle') for i in range(nrepeat)) 
           
            # calculate Kuiper's statistics for each phase array
            out = parallel(my_kuiper(i) for i in pt_s)
            
            # store stat and pk in different arrays
            out = np.array(out)
            #ks[ifreq,:,isource,:] = out[:,0,:]  # is actually not needed
            pk[ifreq,:,isource,:] = out[:,1,:]  # [nrepeat, pk_idx, nsamp]

    return ks, pk


#######################################################
#                                                     
# calc stats on CTPS surrogates
#                                                     
#######################################################
def get_stats_surrogates_ctps(pksarr, verbose=False):
    ''' calculates some stats on the CTPS pk values obtain from surrogate tests. 

    Parameters
    ----------
    pksarr : 4d ndarray of dimension [nfreq x nrepeat x nsources x nsamples]

    Optional:
    verbose:  print some information on stdout 


    Returns
    -------
    stats : stats info stored in a python dictionary 

    '''

    import os
    import numpy as np

    nfreq, nrepeat, nsources, nsamples = pksarr.shape
    pks = np.reshape(pksarr, (nfreq, nrepeat*nsources*nsamples))  # [nsource * nrepeat, nbp]

    # stats for each frequency band
    pks_max = pks.max(axis=1)
    pks_min = pks.min(axis=1)
    pks_mean = pks.mean(axis=1)
    pks_std = pks.std(axis=1)
   
    # global stats
    pks_max_global = pks.max()
    pks_min_global = pks.min()
    pks_mean_global = pks.mean()
    pks_std_global = pks.std()

    pks_pct99_global = np.percentile(pksarr,99)

    # collect info and store into dictionary
    stats = {
        'path':  os.getcwd(),
        'fname': 'CTPS surrogates',
        'nrepeat': nrepeat,
        'nfreq': nfreq,
        'nsources': nsources,
        'nsamples': nsamples,
        'pks_min': pks_min,
        'pks_max': pks_max,
        'pks_mean': pks_mean,
        'pks_std': pks_std,
        'pks_min_global': pks_min_global,
        'pks_max_global': pks_max_global,
        'pks_mean_global': pks_mean_global,
        'pks_std_global': pks_std_global,
        'pks_pct99_global': pks_pct99_global
        }


    # mean and std dev
    if (verbose):
        print '>>> Stats from CTPS surrogates <<<'
        for i in range(nfreq):
            #print ">>> filter raw data: %0.1f - %0.1f..." % (flow, fhigh)
            print 'freq: ',i+1,'max/mean/std: ', pks_max[i], pks_mean[i], pks_std[i]  
        print    
        print 'overall stats:'
        print 'max/mean/std: ', pks_global_max, pks_global_mean, pks_global_std          
        print '99th percentile: ', pks_global_pct99 

    return stats