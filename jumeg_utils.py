'''
Utilities module for jumeg
'''

# Authors: Jurgen Dammers (j.dammers@fz-juelich.de)
#          Praveen Sripad (pravsripad@gmail.com)
#
# License: BSD (3-clause)

import sys
import os
import numpy as np
import scipy as sci
import mne
from mne.utils import logger
import matplotlib.cm as cmx
import matplotlib.colors as colors

def get_files_from_list(fin):
    ''' Return string of file or files as iterables lists '''
    if isinstance(fin, list):
        fout = fin
    else:
        if isinstance(fin, str):
            fout = list([fin])
        else:
            fout = list(fin)
    return fout


def retcode_error(command, subj):
    print '%s did not run successfully for subject %s.' % (command, subj)
    print 'Please check the arguments, and rerun for subject.'


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
                  % (fname.split('_')[0], fname.split('_')[1], fname.split('_')[2],
                     fname.split('_')[3], fname.split('_')[4])
            print 'Processing identifier in the file %s.' \
                  % (fname.strip('-raw.fif').split('_')[-1])
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
    elekta_types = (FIFF.FIFFV_COIL_VV_MAG_T3,
                    FIFF.FIFFV_COIL_VV_PLANAR_T1)
    has_CTF_grad = (FIFF.FIFFV_COIL_CTF_GRAD in coil_types or
                    (FIFF.FIFFV_MEG_CH in channel_types and
                     any([k in ctf_other_types for k in coil_types])))
    has_Elekta_grad = (FIFF.FIFFV_COIL_VV_MAG_T3 in coil_types or
                      (FIFF.FIFFV_MEG_CH in channel_types and
                       any([k in elekta_types for k in coil_types])))
    if has_4D_mag:
        system_type = 'magnesWH3600'
    elif has_CTF_grad:
        system_type = 'CTF-275'
    elif has_Elekta_grad:
        system_type = 'ElektaNeuromagTriux'
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
        print "For subject %s" % (subj)
        if not subjects_dir: subjects_dir = os.environ['SUBJECTS_DIR']
        dirname = subjects_dir + '/' + subj
        sub_file_list = os.listdir(dirname)
        for raw_fname in sub_file_list:
            if raw_fname.endswith('_bcc-raw.fif'): continue
            if raw_fname.endswith('-raw.fif'):
                print "Raw calculations for file %s" % (dirname + '/' + raw_fname)
                raw = mne.io.Raw(dirname + '/' + raw_fname, preload=True)
                raw.plot(block=True)
                print 'The bad channels marked are %s ' % (raw.info['bads'])
                save_fname = dirname + '/' + raw.info['filename'].split('/')[-1].split('-raw.fif')[0] + '_bcc-raw.fif'
                raw.save(save_fname)
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


def chop_raw_data(raw, start_time=60.0, stop_time=360.0, save=True):
    '''
    This function extracts specified duration of raw data
    and writes it into a fif file.
    Five mins of data will be extracted by default.

    Parameters
    ----------

    raw: Raw object or raw file name as a string.
    start_time: Time to extract data from in seconds. Default is 60.0 seconds.
    stop_time: Time up to which data is to be extracted. Default is 360.0 seconds.
    save: bool, If True the raw file is written to disk.

    '''
    if isinstance(raw, str):
        print 'Raw file name provided, loading raw object...'
        raw = mne.io.Raw(raw, preload=True)
    # Check if data is longer than required chop duration.
    if (raw.n_times / (raw.info['sfreq'])) < (stop_time + start_time):
        logger.info("The data is not long enough for file %s.") % (raw.info['filename'])
        return
    # Obtain indexes for start and stop times.
    assert start_time < stop_time, "Start time is greater than stop time."
    start_idx = raw.time_as_index(start_time)
    stop_idx = raw.time_as_index(stop_time)
    data, times = raw[:, start_idx:stop_idx]
    raw._data,raw._times = data, times
    dur = int((stop_time - start_time) / 60)
    if save:
        #raw.save(raw.info['filename'].split('/')[-1].split('.')[0] + '_' + str(dur) + 'm-raw.fif')
        raw.save(raw.info['filename'].split('-raw.fif')[0] + ',' + str(dur) + 'm-raw.fif')
    raw.close()
    return


##################################################
#
# destroy phase/time info by shuffling on 2D arrays
#
##################################################
def shuffle_data(data_trials, seed=None):
    '''
    Shuffling the time points of any data array. The probabiity density
    of the data samples is preserved.
    WARNING: This function simply reorders the time points and does not
    perform shuffling of the phases.

    Parameters
    ----------
    data_trials : 2d ndarray of dimension [ntrials x nsamples]
                  In each trial samples are randomly shuffled
    Returns
    -------
    dt : shuffled (time points only) trials
    '''
    np.random.seed(seed=None)   # for parallel processing => re-initialized
    ntrials, nsamples = data_trials.shape

    # shuffle all time points
    dt = data_trials.flatten()
    np.random.shuffle(dt)
    dt = dt.reshape(ntrials, nsamples)

    return dt


##################################################
#
# destroy phase/time info by shifting on 2D arrays
#
##################################################
def shift_data(data_trials, min_shift=0, max_shift=None, seed=None):
    '''
    Shifting the time points of any data array. The probability density of the data
    samples are preserved.
    WARNING: This function simply shifts the time points and does not
    perform shuffling of the phases in the frequency domain.

    Parameters
    ----------
    data_trials : 2d ndarray of dimension [ntrials x nsamples]
                  In each trial samples are randomly shifted

    Returns
    -------
    dt : Time shifted trials.
    '''

    np.random.seed(seed=None)   # for parallel processing => re-initialized
    ntrials, nsamples = data_trials.shape

    # random phase shifts for each trial
    dt = np.zeros((ntrials, nsamples), dtype=data_trials.dtype)
    # Limit shifts to the number of samples.
    if max_shift is None:
        max_shift = nsamples
    # shift array contacts maximum and minimum number of shifts
    assert (min_shift < max_shift) & (min_shift >= 0), 'min_shift is not less than max_shift'
    shift = np.random.permutation(np.arange(min_shift, max_shift))

    for itrial in range(ntrials):
        # random shift is picked from the range of min max values
        dt[itrial, :] = np.roll(data_trials[itrial, :], np.random.choice(shift))

    return dt


#######################################################
#
# make surrogates from Epochs
#
#######################################################
def make_surrogates_epochs(epochs, check_pdf=False, random_state=None):
    '''
    Make surrogate epochs using sklearn. Destroy each trial by shuffling the time points only.
    The shuffling is performed in the time domain only. The probability density function is
    preserved.

    Parameters
    ----------
    epochs : Epochs Object.
    check_pdf : Condition to test for equal probability density. (bool)
    random_state : Seed for random generator.

    Output
    ------
    Surrogate Epochs object
    '''
    from sklearn.utils import check_random_state
    rng = check_random_state(random_state)

    surrogate = epochs.copy()
    surr = surrogate.get_data()
    for trial in range(len(surrogate)):
        for channel in range(len(surrogate.ch_names)):
            order = np.argsort(rng.randn(len(surrogate.times)))
            surr[trial, channel, :] = surr[trial, channel, order]
    surrogate._data = surr

    if check_pdf:
        hist, _ = np.histogram(data_trials.flatten())
        hist_dt = np.histogram(dt.flatten())
        assert np.array_equal(hist, hist_dt), 'The histogram values are unequal.'

    return surrogate


def make_fftsurr_epochs(epochs, check_power=False):
    '''
    Make surrogate epochs using sklearn. Destroy each trial by shuffling the phase information.
    The shuffling is performed in the frequency domain only using fftsurr function from mlab.

    Parameters
    ----------
    Epochs Object.

    Output
    ------
    Surrogate Epochs object
    '''
    from matplotlib.mlab import fftsurr

    surrogate = epochs.copy()
    surr = surrogate.get_data()
    for trial in range(len(surrogate)):
        for channel in range(len(surrogate.ch_names)):
            surr[trial, channel, :] = fftsurr(surr[trial, channel, :])
    surrogate._data = surr
    if check_power:
        from mne.time_frequency import compute_epochs_psd
        ps1, _ = compute_epochs_psd(epochs, epochs.picks)
        ps2, _ = compute_epochs_psd(surrogate, surrogate.picks)
        assert np.allclose(ps1, ps2), 'The power content does not match. Error.'

    return surrogate


def make_phase_shuffled_surrogates_epochs(epochs, check_power=False):
    '''
    Make surrogate epochs using sklearn. Destroy phase information in each trial by randomization.
    The phases values are randomized in teh frequency domain.

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
            surr[trial, channel, :] = randomize_phase(surr[trial, channel, :])
    surrogate._data = surr
    if check_power:
        from mne.time_frequency import compute_epochs_psd
        ps1, _ = compute_epochs_psd(epochs, epochs.picks)
        ps2, _ = compute_epochs_psd(surrogate, surrogate.picks)
        # np.array_equal does not pass the assertion, due to minor changes in power.
        assert np.allclose(ps1, ps2), 'The power content does not match. Error.'

    return surrogate



#######################################################
#                                                     #
#      to extract the indices of the R-peak from      #
#               ECG single channel data               #
#                                                     #
#######################################################
def get_peak_ecg(ecg, sfreq=1017.25, flow=10, fhigh=20,
                 pct_thresh=95.0, default_peak2peak_min=0.5,
                 event_id=999):

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    from mne.filter import band_pass_filter
    from jumeg.jumeg_math import calc_tkeo
    from scipy.signal import argrelextrema as extrema

    # -------------------------------------------
    # filter ECG to get rid of noise and drifts
    # -------------------------------------------
    fecg = band_pass_filter(ecg, sfreq, flow, fhigh,
                            n_jobs=1, method='fft')
    ecg_abs = np.abs(fecg)

    # -------------------------------------------
    # apply Teager Kaiser energie Operator (TKEO)
    # -------------------------------------------
    tk_ecg = calc_tkeo(fecg)

    # -------------------------------------------
    # find all peaks of abs(EOG)
    # since we don't know if the EOG lead has a
    # positive or negative R-peak
    # -------------------------------------------
    ixpeak = extrema(tk_ecg, np.greater, axis=0)


    # -------------------------------------------
    # threshold for |R-peak|
    # ------------------------------------------
    peak_thresh_min = np.percentile(tk_ecg, pct_thresh, axis=0)
    ix = np.where(tk_ecg[ixpeak] > peak_thresh_min)[0]
    npeak = len(ix)
    if (npeak > 1):
        ixpeak = ixpeak[0][ix]
    else:
        return -1


    # -------------------------------------------
    # threshold for max Amplitude of R-peak
    # fixed to: median + 3*stddev
    # -------------------------------------------
    mag = fecg[ixpeak]
    mag_mean = np.median(mag)
    if (mag_mean > 0):
        nstd = 3
    else:
        nstd = -3

    peak_thresh_max = mag_mean + nstd * np.std(mag)
    ix = np.where(ecg_abs[ixpeak] < np.abs(peak_thresh_max))[0]
    npeak = len(ix)

    if (npeak > 1):
        ixpeak = ixpeak[ix]
    else:
        return -1


    # -------------------------------------------
    # => test if the R-peak is positive or negative
    # => we assume the the R-peak is the largest peak !!
    #
    # ==> sometime we have outliers and we should check
    #     the number of npos and nneg peaks -> which is larger?  -> note done yet
    #     -> we assume at least 2 peaks -> maybe we should check the ratio
    # -------------------------------------------
    ixp = np.where(fecg[ixpeak] > 0)[0]
    npos = len(ixp)
    ixn = np.where(fecg[ixpeak] < 0)[0]
    nneg = len(ixp)

    if (npos == 0 and nneg == 0):
        import pdb
        pdb.set_trace()
    if (npos > 3):
        peakval_pos = np.abs(np.median(ecg[ixpeak[ixp]]))
    else:
        peakval_pos = 0

    if (nneg > 3): peakval_neg = np.abs(np.median(ecg[ixpeak[ixn]]))
    else:
        peakval_neg = 0

    if (peakval_pos > peakval_neg):
        ixpeak  = ixpeak[ixp]
        ecg_pos = ecg
    else:
        ixpeak  = ixpeak[ixn]
        ecg_pos = - ecg

    npeak = len(ixpeak)
    if (npeak < 1):
        return -1


    # -------------------------------------------
    # check if we have peaks too close together
    # -------------------------------------------
    peak_ecg = ixpeak/sfreq
    dur = (np.roll(peak_ecg, -1)-peak_ecg)
    ix  = np.where(dur > default_peak2peak_min)[0]
    npeak = len(ix)
    if (npeak < 1):
        return -1

    ixpeak = np.append(ixpeak[0], ixpeak[ix])
    peak_ecg = ixpeak/sfreq
    dur = (peak_ecg-np.roll(peak_ecg, 1))
    ix  = np.where(dur > default_peak2peak_min)[0]
    npeak = len(ix)
    if (npeak < 1):
        return -1

    ixpeak = np.unique(np.append(ixpeak, ixpeak[ix[npeak-1]]))
    npeak = len(ixpeak)

    # -------------------------------------------
    # search around each peak if we find
    # higher peaks in a range of 0.1 s
    # -------------------------------------------
    seg_length = np.ceil(0.1 * sfreq)
    for ipeak in range(0, npeak-1):
        idx = [int(np.max([ixpeak[ipeak] - seg_length, 0])),
               int(np.min([ixpeak[ipeak]+seg_length, len(ecg)]))]
        idx_want = np.argmax(ecg_pos[idx[0]:idx[1]])
        ixpeak[ipeak] = idx[0] + idx_want


    # -------------------------------------------
    # to be confirm with mne implementation
    # -------------------------------------------
    ecg_events = np.c_[ixpeak, np.zeros(npeak),
                       np.zeros(npeak)+event_id]

    return ecg_events.astype(int)



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
#     assert np.aray_equal(ps1, ps2), 'The power content does not match. Error.'
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

    Returns
    -------
    pt : shuffled phase trials

    '''

    from joblib import Parallel, delayed
    from mne.parallel import parallel_func
    from mne.preprocessing.ctps_ import kuiper

    nfreq, ntrials, nsources, nsamples = phase_array.shape
    pk = np.zeros((nfreq, nrepeat, nsources, nsamples), dtype='float32')

    # create surrogates:  parallised over nrepeats
    parallel, my_kuiper, _ = parallel_func(kuiper, n_jobs, verbose=verbose)
    for ifreq in range(nfreq):
        for isource in range(nsources):
            # print ">>> working on frequency: ",bp[ifreq,:],"   source: ",isource+1
            print ">>> working on frequency range: ",ifreq + 1,"   source: ",isource + 1
            pt = phase_array[ifreq, :, isource, :]  # extract [ntrials, nsamp]

            if(mode=='shuffle'):
                # shuffle phase values for all repetitions
                pt_s = Parallel(n_jobs=n_jobs, verbose=0)(delayed(shuffle_data)
                                (pt) for i in range(nrepeat))
            else:
                # shift all phase values for all repetitions
                pt_s = Parallel(n_jobs=n_jobs, verbose=0)(delayed(shift_data)
                                (pt) for i in range(nrepeat))

            # calculate Kuiper's statistics for each phase array
            out = parallel(my_kuiper(i) for i in pt_s)

            # store stat and pk in different arrays
            out = np.array(out, dtype='float32')
            # ks[ifreq,:,isource,:] = out[:,0,:]  # is actually not needed
            pk[ifreq, :, isource, :] = out[:, 1, :]  # [nrepeat, pk_idx, nsamp]

    return pk


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
    pks = np.reshape(pksarr, (nfreq, nrepeat * nsources * nsamples))  # [nsource * nrepeat, nbp]

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

    pks_pct99_global = np.percentile(pksarr, 99)
    pks_pct999_global = np.percentile(pksarr, 99.9)
    pks_pct9999_global = np.percentile(pksarr, 99.99)

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
            'pks_pct99_global': pks_pct99_global,
            'pks_pct999_global': pks_pct999_global,
            'pks_pct9999_global': pks_pct9999_global
            }

    # mean and std dev
    if (verbose):
        print '>>> Stats from CTPS surrogates <<<'
        for i in range(nfreq):
            #print ">>> filter raw data: %0.1f - %0.1f..." % (flow, fhigh)
            print 'freq: ',i + 1, 'max/mean/std: ', pks_max[i], pks_mean[i], pks_std[i]
        print
        print 'overall stats:'
        print 'max/mean/std: ', pks_global_max, pks_global_mean, pks_global_std
        print '99th percentile: ', pks_global_pct99
        print '99.90th percentile: ', pks_global_pct999
        print '99.99th percentile: ', pks_global_pct9999

    return stats


###########################################################
#
# These functions copied from NIPY (http://nipy.org/nitime)
#
###########################################################
def threshold_arr(cmat, threshold=0.0, threshold2=None):
    """Threshold values from the input array.

    Parameters
    ----------
    cmat : array

    threshold : float, optional.
      First threshold.

    threshold2 : float, optional.
      Second threshold.

    Returns
    -------
    indices, values: a tuple with ndim+1

    Examples
    --------
    >>> np.set_printoptions(precision=4)  # For doctesting
    >>> a = np.linspace(0,0.2,5)
    >>> a
    array([ 0.  ,  0.05,  0.1 ,  0.15,  0.2 ])
    >>> threshold_arr(a,0.1)
    (array([3, 4]), array([ 0.15,  0.2 ]))

    With two thresholds:
    >>> threshold_arr(a,0.1,0.2)
    (array([0, 1]), array([ 0.  ,  0.05]))
    """
    # Select thresholds
    if threshold2 is None:
        th_low = -np.inf
        th_hi = threshold
    else:
        th_low = threshold
        th_hi = threshold2

    # Mask out the values we are actually going to use
    idx = np.where((cmat < th_low) | (cmat > th_hi))
    vals = cmat[idx]

    return idx + (vals,)


def thresholded_arr(arr, threshold=0.0, threshold2=None, fill_val=np.nan):
    """Threshold values from the input matrix and return a new matrix.

    Parameters
    ----------
    arr : array

    threshold : float
      First threshold.

    threshold2 : float, optional.
      Second threshold.

    Returns
    -------
    An array shaped like the input, with the values outside the threshold
    replaced with fill_val.

    Examples
    --------
    """
    a2 = np.empty_like(arr)
    a2.fill(fill_val)
    mth = threshold_arr(arr, threshold, threshold2)
    idx, vals = mth[:-1], mth[-1]
    a2[idx] = vals

    return a2


def rescale_arr(arr, amin, amax):
    """Rescale an array to a new range.

    Return a new array whose range of values is (amin,amax).

    Parameters
    ----------
    arr : array-like

    amin : float
      new minimum value

    amax : float
      new maximum value

    Examples
    --------
    >>> a = np.arange(5)

    >>> rescale_arr(a,3,6)
    array([ 3.  ,  3.75,  4.5 ,  5.25,  6.  ])
    """

    # old bounds
    m = arr.min()
    M = arr.max()
    # scale/offset
    s = float(amax - amin) / (M - m)
    d = amin - s * m

    # Apply clip before returning to cut off possible overflows outside the
    # intended range due to roundoff error, so that we can absolutely guarantee
    # that on output, there are no values > amax or < amin.
    return np.clip(s * arr + d, amin, amax)


def mask_indices(n, mask_func, k=0):
    """Return the indices to access (n,n) arrays, given a masking function.

    Assume mask_func() is a function that, for a square array a of size (n,n)
    with a possible offset argument k, when called as mask_func(a,k) returns a
    new array with zeros in certain locations (functions like triu() or tril()
    do precisely this).  Then this function returns the indices where the
    non-zero values would be located.

    Parameters
    ----------
    n : int
      The returned indices will be valid to access arrays of shape (n,n).

    mask_func : callable
      A function whose api is similar to that of numpy.tri{u,l}.  That is,
      mask_func(x,k) returns a boolean array, shaped like x.  k is an optional
      argument to the function.

    k : scalar
      An optional argument which is passed through to mask_func().  Functions
      like tri{u,l} take a second argument that is interpreted as an offset.

    Returns
    -------
    indices : an n-tuple of index arrays.
      The indices corresponding to the locations where mask_func(ones((n,n)),k)
      is True.

    Examples
    --------
    These are the indices that would allow you to access the upper triangular
    part of any 3x3 array:
    >>> iu = mask_indices(3,np.triu)

    For example, if `a` is a 3x3 array:
    >>> a = np.arange(9).reshape(3,3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    Then:
    >>> a[iu]
    array([0, 1, 2, 4, 5, 8])

    An offset can be passed also to the masking function.  This gets us the
    indices starting on the first diagonal right of the main one:
    >>> iu1 = mask_indices(3,np.triu,1)

    with which we now extract only three elements:
    >>> a[iu1]
    array([1, 2, 5])
    """
    m = np.ones((n, n), int)
    a = mask_func(m, k)
    return np.where(a != 0)


def triu_indices(n, k=0):
    """Return the indices for the upper-triangle of an (n,n) array.

    Parameters
    ----------
    n : int
      Sets the size of the arrays for which the returned indices will be valid.

    k : int, optional
      Diagonal offset (see triu() for details).

    Examples
    --------
    Commpute two different sets of indices to access 4x4 arrays, one for the
    upper triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> iu1 = triu_indices(4)
    >>> iu2 = triu_indices(4,2)

    Here is how they can be used with a sample array:
    >>> a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    >>> a
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12],
           [13, 14, 15, 16]])

    Both for indexing:
    >>> a[iu1]
    array([ 1,  2,  3,  4,  6,  7,  8, 11, 12, 16])

    And for assigning values:
    >>> a[iu1] = -1
    >>> a
    array([[-1, -1, -1, -1],
           [ 5, -1, -1, -1],
           [ 9, 10, -1, -1],
           [13, 14, 15, -1]])

    These cover almost the whole array (two diagonals right of the main one):
    >>> a[iu2] = -10
    >>> a
    array([[ -1,  -1, -10, -10],
           [  5,  -1,  -1, -10],
           [  9,  10,  -1,  -1],
           [ 13,  14,  15,  -1]])

    See also
    --------
    - tril_indices : similar function, for lower-triangular.
    - mask_indices : generic function accepting an arbitrary mask function.
    """
    return mask_indices(n, np.triu, k)


# Function obtained from scot (https://github.com/scot-dev/scot)
# Used to randomize the phase values of a signal.
def randomize_phase(data, random_state=None):
    '''
    Phase randomization.

    This function randomizes the input array's spectral phase along the first dimension.

    Parameters
    ----------
    data : array_like
        Input array

    Returns
    -------
    out : ndarray
        Array of same shape as `data`.

    Notes
    -----
    The algorithm randomizes the phase component of the input's complex fourier transform.

    Examples
    --------
    .. plot::
        :include-source:

        from pylab import *
        from scot.datatools import randomize_phase
        np.random.seed(1234)
        s = np.sin(np.linspace(0,10*np.pi,1000)).T
        x = np.vstack([s, np.sign(s)]).T
        y = randomize_phase(x)
        subplot(2,1,1)
        title('Phase randomization of sine wave and rectangular function')
        plot(x), axis([0,1000,-3,3])
        subplot(2,1,2)
        plot(y), axis([0,1000,-3,3])
        plt.show()
    '''
    from sklearn.utils import check_random_state
    data = np.asarray(data)
    data_freq = np.fft.rfft(data, axis=0)
    rng = check_random_state(random_state)
    data_freq = np.abs(data_freq) * np.exp(1j * rng.random_sample(data_freq.shape) * 2 * np.pi)
    return np.fft.irfft(data_freq, data.shape[0], axis=0)


def create_dummy_raw(data, ch_types, sfreq, ch_names, save=False,
                     raw_fname='output.fif'):
    '''
    A function that can be used to quickly create a raw object with the
    data provided.

    Inspired from https://gist.github.com/dengemann/e9b45f2ff3e3380907d3

    Parameters
    ----------
    data: ndarray, shape (n_channels, n_times)
    ch_types: list eg. ['misc'], ['eeg'] or ['meg']
    sfreq: float
        Sampling frequency.
    ch_names: list
        List of channel names.
    save : bool
        If True, the raw object will be saved as a fif. file.
    raw_fname : str
        If save is True, the name of the saved fif file.

    Returns
    -------
    raw : Instance of mne.io.Raw

    Example
    -------

    rng = np.random.RandomState(42)
    data = rng.random_sample((248, 2000))
    sfreq = 1e3
    ch_types = ['misc'] * 248
    ch_names = ['MISC {:03d}'.format(i + 1) for i in range(len(ch_types))]
    raw = create_dummy_raw(data, ch_types, sfreq, ch_names)

    '''
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    if save:
        raw.save(raw_fname)
    return raw


def create_dummy_epochs(data, events, ch_types, sfreq, ch_names, save=False,
                        epochs_fname='output-epo.fif'):
    '''
    A function that can be used to quickly create an Epochs object with the
    data provided.

    Inspired from https://gist.github.com/dengemann/e9b45f2ff3e3380907d3

    Parameters
    ----------
    data: ndarray, shape (n_channels, n_times)
    events: ndarray (n_events, 3)
        As returned by mne.find_events
    ch_types: list eg. ['misc'], ['eeg'] or ['meg']
    sfreq: float
        Sampling frequency.
    ch_names: list
        List of channel names.
    save : bool
        If True, the epochs object will be saved as a fif. file.
    epochs_fname : str
        If save is True, the name of the saved fif file.

    Returns
    -------
    epochs : Instance of mne.Epochs

    Example
    -------

    rng = np.random.RandomState(42)
    data = rng.random_sample((248, 2000))
    sfreq = 1e3
    ch_types = ['misc'] * 248
    ch_names = ['MISC {:03d}'.format(i + 1) for i in range(len(ch_types))]
    # make event with - event id 42, 10 events of duration 100 s each, 0 stim signal
    events = np.array((np.arange(0, 1000, 100), np.zeros((10)), np.array([42] * 10))).T
    epochs = create_dummy_epochs(data, events, ch_types, sfreq, ch_names)

    '''
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    epochs = mne.EpochsArray(data, info, events)
    if save:
        epochs.save(epochs_fname)
    return epochs


def put_pngs_into_html(regexp, html_out='output.html'):
    '''Lists all files in directory that matches pattern regexp
       and puts it into an html file with filename included.

    regexp : str
        String of dir path like '/home/kalka/*.png'
    html_out : str
        Output file name
    '''
    import glob
    files =  glob.glob(regexp)
    html_string = ''
    for fname in files:
        my_string = '<body><p>%s</p></body>' % (fname) + '\n' + '<img src=%s>' % (fname) + '\n'
        html_string += my_string
    f = open(html_out, 'w')
    message = """<html>
          <head></head>
          %s
          </html>""" % (html_string)
    f.write(message)
    f.close()


def check_env_variables(env_variable=None, key=None):
    '''Check the most important environment variables as
       (keys) - SUBJECTS_DIR, MNE_ROOT and FREESURFER_HOME.

    e.g. subjects_dir = check_env_variable(subjects_dir, key='SUBJECTS_DIR')
    If subjects_dir provided exists, then it is prioritized over the env variable.
    If not, then the environment variable pertaining to the key is returned. If both
    do not exist, then exits with an error message.
    Also checks if the directory exists.
    '''

    if key is None or not isinstance(key, str):
        print ('Please provide the key. Currently '
              'SUBJECTS_DIR, MNE_ROOT and FREESURFER_HOME as strings are allowed.')
        sys.exit()

    # Check subjects_dir
    if env_variable:
        os.environ[key] = env_variable
    elif env_variable is None and key in os.environ:
        env_variable = os.environ[key]
    else:
        print 'Please set the %s' % (key)
        sys.exit()

    if not os.path.isdir(env_variable):
        print 'Path %s is not a valid directory. Please check.' % (env_variable)
        sys.exit()

    return env_variable


def convert_annot2labels(annot_fname, subject='fsaverage', subjects_dir=None,
                         freesurfer_home=None):
    '''
    Convert an annotation to labels for a single subject for both hemispheres.
    The labels are written to '$SUBJECTS_DIR/$SUBJECT/label'.

    Parameters
    ----------
    annot_fname: str
        The name of the annotation (or parcellation).
    subject: str
        Subject name. Default is the fresurfer fsaverage.
    subjects_dir: str
        The subjects directory, if not provided, then the
        environment value is used.
    freesurfer_home: str
        The freeesurfer home path, if not provided, the
        environment value is used.

    Reference
    ---------
    https://surfer.nmr.mgh.harvard.edu/fswiki/mri_annotation2label
    '''
    from subprocess import call
    subjects_dir = check_env_variables(subjects_dir, key='SUBJECTS_DIR')
    freesurfer_home = check_env_variables(freesurfer_home, key='FREESURFER_HOME')
    freesurfer_bin = os.path.join(freesurfer_home, 'bin', '')
    outdir = os.path.join(subjects_dir, subject, 'label')
    print 'Convert annotation %s to labels' % (annot_fname)
    for hemi in ['lh', 'rh']:
        retcode = call([freesurfer_bin + '/mri_annotation2label', '--subject', subject, '--hemi', hemi,
                        '--annotation', annot_fname, '--outdir', outdir])
        if retcode != 0:
            retcode_error('mri_annotation2label')
            continue


def convert_label2label(annot_fname, subjects_list, srcsubject='fsaverage',
                        subjects_dir=None, freesurfer_home=None):
    '''
    Python wrapper for Freesurfer mri_label2label function.
    Converts all labels in annot_fname from source subject to target subject
    given the subjects directory. Both hemispheres are considered.
    The registration method used it surface.

    Parameters
    ----------
    annot_fname: str
        The name of the annotation (or parcellation).
    subjects_list: list or str
        Subject names to which the labels have to be transformed to (the target subjects).
        Can be provided as a list or a string.
    srcsubject: str
        The name of the source subject to be used. The source subject should
        contain the labels in the correct folders already. Default - fsaverage.
    subjects_dir: str
        The subjects directory, if not provided, then the
        environment value is used.
    freesurfer_home: str
        The freeesurfer home path, if not provided, the
        environment value is used.

    Reference:
    https://surfer.nmr.mgh.harvard.edu/fswiki/mri_label2label
    '''
    if subjects_list:
        subjects_list = get_files_from_list(subjects_list)
    else:
        raise RuntimeError('No subjects are specified.')

    subjects_dir = check_env_variables(subjects_dir, key='SUBJECTS_DIR')
    freesurfer_home = check_env_variables(freesurfer_home, key='FREESURFER_HOME')
    freesurfer_bin = os.path.join(freesurfer_home, 'bin', '')

    # obtain the names of labels in parcellation
    from mne.label import read_labels_from_annot
    labels = read_labels_from_annot(srcsubject, parc=annot_fname)
    lnames = [l.name.rsplit('-')[0] if l.hemi is 'lh' else '' for l in labels]
    lnames = filter(None, lnames)  # remove empty strings

    # convert the labels from source subject to target subject
    from subprocess import call
    for subj in subjects_list:
        # the target subject is subj provided
        print 'Converting labels from %s to %s' % (srcsubject, subj)
        for label in lnames:
            for hemi in ['lh', 'rh']:
                srclabel = os.path.join(subjects_dir, srcsubject, 'label', hemi + '.' + label + '.label')
                trglabel = os.path.join(subjects_dir, subj, 'label', hemi + '.' + label + '.label')
                retcode = call([freesurfer_bin + 'mri_label2label', '--srclabel', srclabel, '--srcsubject', srcsubject,
                    '--trglabel', trglabel, '--trgsubject', subj, '--regmethod', 'surface', '--hemi', hemi])
                if retcode != 0:
                    retcode_error('mri_label2label')
                    continue

    print 'Labels for %d subjects have been transformed from source %s' %(len(subjects_list), srcsubject)


def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


def subtract_overlapping_vertices(label, labels):
    '''
    Check if label overlaps with others in labels
    and return a new label without the overlapping vertices.
    The output label contains the original label vertices minus
    vertices from all overlapping labels in the list.

    label : instance of mne.Label
    labels : list of labels
    '''
    for lab in labels:
        if (lab.hemi == label.hemi and
           np.intersect1d(lab.vertices, label.vertices).size > 0 and
           lab is not label):
            label = label - lab

    if label.vertices.size > 0:
        return label
    else:
        print 'Label has no vertices left '
        return None


def apply_percentile_threshold(in_data, percentile):
    ''' Return ndarray with all values below percentile set to 0. '''
    in_data[in_data <= np.percentile(in_data, percentile)] = 0.
    return in_data
