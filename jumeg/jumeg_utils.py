'''
Utilities module for jumeg
'''

# Authors: Jurgen Dammers (j.dammers@fz-juelich.de)
#          Praveen Sripad (pravsripad@gmail.com)
#          Eberhard Eich (e.eich@fz-juelich.de) ()
#
# License: BSD (3-clause)

import sys
import os
import os.path as op
import fnmatch

import numpy as np
import scipy as sci
from sklearn.utils import check_random_state
import fnmatch

import mne
from mne.utils import logger


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
    print('%s did not run successfully for subject %s.' % (command, subj))
    print('Please check the arguments, and rerun for subject.')


def get_jumeg_path():
    '''Return the path where jumeg is installed.'''
    return os.path.abspath(os.path.dirname(__file__))


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

    print(fname_list)
    # loop across all filenames
    for fname in fname_list:
        print(fname)
        if fname == '' or not fname.endswith('.fif'):
            print('Empty string or not a FIF format filename.')
        elif fname.endswith('-meg.fif') or fname.endswith('-eeg.fif'):
            print('Raw FIF file with only MEG or only EEG data.')
        elif fname.split('-')[-1] == 'raw.fif':
            print('Raw FIF file - Subject %s, Experiment %s, Data %s, Time %s, \
                   Trial number %s.' \
                  % (fname.split('_')[0], fname.split('_')[1], fname.split('_')[2],
                     fname.split('_')[3], fname.split('_')[4]))
            print('Processing identifier in the file %s.' \
                  % (fname.strip('-raw.fif').split('_')[-1]))
        elif fname.split('-')[-1] == 'ica.fif':
            print('FIF file storing ICA session.')
        elif fname.split('-')[-1] == 'evoked.fif':
            print('FIF file with averages.')
        elif fname.split('-')[-1] == 'epochs.fif':
            print('FIF file with epochs.')
        elif fname.split('-')[-1] == 'empty.fif':
            print('Empty room FIF file.')
        else:
            print('No known file info available. Filename does not follow conventions.')

        print('Please verify if the information is correct and make the appropriate changes!')
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
        print("System type not known!")
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
        print("For subject %s" % (subj))
        if not subjects_dir: subjects_dir = os.environ['SUBJECTS_DIR']
        dirname = subjects_dir + '/' + subj
        sub_file_list = os.listdir(dirname)
        for raw_fname in sub_file_list:
            if raw_fname.endswith('_bcc-raw.fif'): continue
            if raw_fname.endswith('-raw.fif'):
                print("Raw calculations for file %s" % (dirname + '/' + raw_fname))
                raw = mne.io.Raw(dirname + '/' + raw_fname, preload=True)
                raw.plot(block=True)
                print('The bad channels marked are %s ' % (raw.info['bads']))
                save_fname = dirname + '/' + raw.filenames[0].split('/')[-1].split('-raw.fif')[0] + '_bcc-raw.fif'
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


def check_read_raw(raw_name, preload=True):
    '''
    Checks if raw_name provided is a filename of raw object.
    If it is a raw object, simply return, else read and return raw object.

    raw_name: instance of mne.io.Raw | str
        Raw object or filename to be read.
    preload: bool
        All data loaded to memory. Defaults to True.
    '''
    if isinstance(raw_name, mne.io.Raw):
        return raw_name
    elif isinstance(raw_name, str):
        raw = mne.io.Raw(raw_name, preload=preload)
        return raw
    else:
        raise RuntimeError('%s type not mne.io.Raw or string.' % raw_name)


def peak_counter(signal):
    ''' Simple peak counter using scipy argrelmax function. '''
    return sci.signal.argrelmax(signal)[0].shape


def update_description(raw, comment):
    ''' Updates the raw description with the comment provided. '''
    raw.info['description'] = str(raw.info['description']) + ' ; ' + comment


def chop_raw_data(raw, start_time=60.0, stop_time=360.0, save=True, return_chop=False):
    '''
    This function extracts specified duration of raw data
    and writes it into a fif file.
    Five mins of data will be extracted by default.

    Parameters
    ----------

    raw: Raw object or raw file name as a string.
    start_time: Time to extract data from in seconds. Default is 60.0 seconds.
    stop_time: Time up to which data is to be extracted. Default is 360.0 seconds.
    save: bool, If True the raw file is written to disk. (default: True)
    return_chop: bool, Return the chopped raw object. (default: False)

    '''
    if isinstance(raw, str):
        print('Raw file name provided, loading raw object...')
        raw = mne.io.Raw(raw, preload=True)
    # Check if data is longer than required chop duration.
    if (raw.n_times / (raw.info['sfreq'])) < (stop_time + start_time):
        logger.info("The data is not long enough for file %s.") % (raw.filenames[0])
        return
    # Obtain indexes for start and stop times.
    assert start_time < stop_time, "Start time is greater than stop time."
    crop = raw.copy().crop(tmin=start_time, tmax=stop_time)
    dur = int((stop_time - start_time) / 60)
    if save:
        crop.save(crop.filenames[0].split('-raw.fif')[0] + ',' + str(dur) + 'm-raw.fif')
    raw.close()
    if return_chop:
         return crop
    else:
        crop.close()
        return


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
    from mne.filter import filter_data
    from jumeg.jumeg_math import calc_tkeo
    from scipy.signal import argrelextrema as extrema

    # -------------------------------------------
    # filter ECG to get rid of noise and drifts
    # -------------------------------------------
    fecg = filter_data(ecg, sfreq, flow, fhigh,
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
            print(">>> working on frequency range: ",ifreq + 1,"   source: ",isource + 1)
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
        print('>>> Stats from CTPS surrogates <<<')
        for i in range(nfreq):
            #print ">>> filter raw data: %0.1f - %0.1f..." % (flow, fhigh)
            print('freq: ',i + 1, 'max/mean/std: ', pks_max[i], pks_mean[i], pks_std[i])
        print()
        print('overall stats:')
        print('max/mean/std: ', pks_global_max, pks_global_mean, pks_global_std)
        print('99th percentile: ', pks_global_pct99)
        print('99.90th percentile: ', pks_global_pct999)
        print('99.99th percentile: ', pks_global_pct9999)

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


def crop_images(regexp, crop_dims=(150, 150, 1450, 700), extension='crop'):
    '''Lists all files in directory that matches pattern regexp
       and puts it into an html file with filename included.

    regexp : str
        String of dir path like '/home/kalka/*.png'
    crop_dims : box tuple
        Dimensions to crop image (using PIL)
        (left, upper, right, lower) pixel values
    extension : str
        Output file name will be appended with extension.
    '''
    import glob
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError('For this method to work the PIL library is'
                           ' required.')
    files = glob.glob(regexp)
    for fname in files:
        orig = Image.open(fname)
        out_fname = op.splitext(fname)[0] + ',' + extension +\
                    op.splitext(fname)[1]
        cropim = orig.crop((150, 150, 1450, 700))
        print('Saving cropped image at %s' % out_fname)
        cropim.save(out_fname, fname.split('.')[1])


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
        print('Please set the %s' % (key))
        sys.exit()

    if not os.path.isdir(env_variable):
        print('Path %s is not a valid directory. Please check.' % (env_variable))
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
    print('Convert annotation %s to labels' % (annot_fname))
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
    lnames = [_f for _f in lnames if _f]  # remove empty strings

    # convert the labels from source subject to target subject
    from subprocess import call
    for subj in subjects_list:
        # the target subject is subj provided
        print('Converting labels from %s to %s' % (srcsubject, subj))
        for label in lnames:
            for hemi in ['lh', 'rh']:
                srclabel = os.path.join(subjects_dir, srcsubject, 'label', hemi + '.' + label + '.label')
                trglabel = os.path.join(subjects_dir, subj, 'label', hemi + '.' + label + '.label')
                retcode = call([freesurfer_bin + 'mri_label2label', '--srclabel', srclabel, '--srcsubject', srcsubject,
                    '--trglabel', trglabel, '--trgsubject', subj, '--regmethod', 'surface', '--hemi', hemi])
                if retcode != 0:
                    retcode_error('mri_label2label')
                    continue

    print('Labels for %d subjects have been transformed from source %s' %(len(subjects_list), srcsubject))


def get_cmap(N, cmap='hot'):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color. Can be used to generate N unique colors from a colormap.

    Usage:
    my_colours = get_cmap(3)
    for i in range(3):
        # print the RGB value of each of the colours
        print my_colours(i)

    '''
    import matplotlib.cm as cmx
    import matplotlib.colors as colors

    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)

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
        print('Label has no vertices left ')
        return None


def apply_percentile_threshold(in_data, percentile):
    ''' Return ndarray with all values below percentile set to 0. '''
    in_data[in_data <= np.percentile(in_data, percentile)] = 0.
    return in_data



def channel_indices_from_list(fulllist, findlist, excllist=None):
    """Get indices of matching channel names from list

    Parameters
    ----------
    fulllist: list of channel names
    findlist: list of (regexp) names to find
              regexp are resolved using mne.pick_channels_regexp()
    excllist: list of channel names to exclude,
              e.g., raw.info.get('bads')
    Returns
    -------
    chnpick: array with indices
    """
    chnpick = []
    for ir in range(len(findlist)):
        if findlist[ir].translate(str.maketrans('', '')).isalnum():
            try:
                chnpicktmp = ([fulllist.index(findlist[ir])])
                chnpick = np.array(np.concatenate((chnpick, chnpicktmp), axis=0),
                                   dtype=int)
            except:
                print(">>>>> Channel '%s' not found." % findlist[ir])
        else:
            chnpicktmp = (mne.pick_channels_regexp(fulllist, findlist[ir]))
            if len(chnpicktmp) == 0:
                print(">>>>> '%s' does not match any channel name." % findlist[ir])
            else:
                chnpick = np.array(np.concatenate((chnpick, chnpicktmp), axis=0),
                                   dtype=int)
    if len(chnpick) > 1:
        # Remove duplicates
        chnpick = np.sort(np.array(list(set(np.sort(chnpick)))))

    if excllist is not None and len(excllist) > 0:
        exclinds = [fulllist.index(excllist[ie]) for ie in range(len(excllist))]
        chnpick = list(np.setdiff1d(chnpick, exclinds))
    return chnpick


def time_shuffle_slices(fname_raw, shufflechans=None, tmin=None, tmax=None):
    """Permute time slices for specified channels.

    Parameters
    ----------
    fname_raw : (list of) rawfile names
    shufflechans : list of string
              List of channels to shuffle.
              If empty use the meg, ref_meg, and eeg channels.
              shufflechans may contain regexp, which are resolved
              using mne.pick_channels_regexp().
              All other channels are copied.
    tmin : lower latency bound for shuffle region [start of trace]
    tmax : upper latency bound for shuffle region [ end  of trace]
           Slice shuffling can be restricted to one region in the file,
           the remaining parts will contain plain copies.

    Outputfile
    ----------
    <wawa>,tperm-raw.fif for input <wawa>-raw.fif

    Returns
    -------
    TBD

    Bugs
    ----
    - it's the user's responsibility to keep track of shuffled chans
    - needs to load the entire data set for operation

    TODO
    ----
    Return raw object and indices of time shuffled channels.
    """
    from math import floor, ceil
    from mne.io.pick import pick_types, channel_indices_by_type

    fnraw = get_files_from_list(fname_raw)

    # loop across all filenames
    for fname in fnraw:
        if not op.isfile(fname):
            print('Exiting. File not present ', fname)
            sys.exit()
        raw = mne.io.Raw(fname, preload=True)
        # time window selection
        # slices are shuffled in [tmin,tmax], but the entire data set gets copied.
        if tmin is None:
            tmin = 0.
        if tmax is None:
            tmax = (raw.last_samp - raw.first_samp) / raw.info['sfreq']
        itmin = int(floor(tmin * raw.info['sfreq']))
        itmax = int(ceil(tmax * raw.info['sfreq']))
        if itmax-itmin < 1:
            raise ValueError("Time-window for slice shuffling empty/too short")
        print(">>> Set time-range to [%7.3f, %7.3f]" % (tmin, tmax))

        if shufflechans is None or len(shufflechans) == 0:
            shflpick = mne.pick_types(raw.info, meg=True, ref_meg=True,
                                      eeg=True, eog=False, stim=False)
        else:
            shflpick = channel_indices_from_list(raw.info['ch_names'][:],
                                                 shufflechans)

        nshfl = len(shflpick)
        if nshfl == 0:
            raise ValueError("No channel selected for slice shuffling")

        totbytype = ''
        shflbytype = ''
        channel_indices_by_type = mne.io.pick.channel_indices_by_type(raw.info)
        for k in list(channel_indices_by_type.keys()):
            tot4key = len(channel_indices_by_type[k][:])
            if tot4key>0:
                totbytype = totbytype + "%s:" % k + \
                            "%c%dd " % ('%', int(ceil(np.log10(tot4key+1)))) % tot4key
                shflbytype = shflbytype + "%s:" % k + \
                    "%c%dd " % ('%', int(ceil(np.log10(tot4key+1)))) % \
                    len(np.intersect1d(shflpick, channel_indices_by_type[k][:]))
        print(">>> %3d channels in file:  %s" % (len(raw.info['chs']), totbytype))
        print(">>> %3d channels shuffled: %s" % (len(shflpick), shflbytype))

        print("Calc shuffle-array...")
        numslice = raw._data.shape[1]
        lselbuf = np.arange(numslice)
        lselbuf[itmin:itmax] = itmin + np.random.permutation(itmax-itmin)

        print("Shuffling slices for selected channels:")
        data, times = raw[:, 0:numslice]
        # work on entire data stream
        for isl in range(raw._data.shape[1]):
            slice = np.take(raw._data, [lselbuf[isl]], axis=1)
            data[shflpick, isl] = slice[shflpick].flatten()
        # copy data to raw._data
        for isl in range(raw._data.shape[1]):
            raw._data[:, isl] = data[:, isl]

        shflname = os.path.join(os.path.dirname(fname),
                                os.path.basename(fname).split('-')[0]) + ',tperm-raw.fif'
        print("Saving '%s'..." % shflname)
        raw.save(shflname, overwrite=True)
    return


def rescale_data(data, times, baseline, mode='mean', copy=True, verbose=None):
    """Rescale aka baseline correct data.

    Parameters
    ----------
    data : array
        It can be of any shape. The only constraint is that the last
        dimension should be time.
    times : 1D array
        Time instants is seconds.
    baseline : tuple or list of length 2, ndarray or None
        The time interval to apply rescaling / baseline correction.
        If None do not apply it. If baseline is ``(bmin, bmax)``
        the interval is between ``bmin`` (s) and ``bmax`` (s).
        If ``bmin is None`` the beginning of the data is used
        and if ``bmax is None`` then ``bmax`` is set to the end of the
        interval. If baseline is ``(None, None)`` the entire time
        interval is used.
        If baseline is an array, then the given array will
        be used for computing the baseline correction i.e. the mean will be
        computed from the array provided. The array has to be the same length
        as the time dimension of the data. (Use case: if different prestim baseline
        needs to be applied on evoked signals around the response)
        If baseline is None, no correction is applied.
    mode : None | 'ratio' | 'zscore' | 'mean' | 'percent' | 'logratio' | 'zlogratio' # noqa
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline)), mean
        simply subtracts the mean power, percent is the same as applying ratio
        then mean, logratio is the same as mean but then rendered in log-scale,
        zlogratio is the same as zscore but data is rendered in log-scale
        first.
        If None no baseline correction is applied.
    copy : bool
        Whether to return a new instance or modify in place.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    data_scaled: array
        Array of same shape as data after rescaling.

    Note
    ----
    Function taken from mne.baseline.rescale in mne-python.
    (https://github.com/mne-tools/mne-python)
    """
    data = data.copy() if copy else data

    from mne.baseline import _log_rescale
    _log_rescale(baseline, mode)

    if baseline is None:
        return data

    if isinstance(baseline, np.ndarray):
        if times.size == baseline.size:
            # use baseline array as data
            use_array = baseline
        else:
            raise ValueError('Size of times and baseline should be the same')
    else:
        bmin, bmax = baseline
        if bmin is None:
            imin = 0
        else:
            imin = np.where(times >= bmin)[0]
            if len(imin) == 0:
                raise ValueError('bmin is too large (%s), it exceeds the largest '
                                 'time value' % (bmin,))
            imin = int(imin[0])
        if bmax is None:
            imax = len(times)
        else:
            imax = np.where(times <= bmax)[0]
            if len(imax) == 0:
                raise ValueError('bmax is too small (%s), it is smaller than the '
                                 'smallest time value' % (bmax,))
            imax = int(imax[-1]) + 1
        if imin >= imax:
            raise ValueError('Bad rescaling slice (%s:%s) from time values %s, %s'
                             % (imin, imax, bmin, bmax))
        use_array = data[..., imin:imax]

    # avoid potential "empty slice" warning
    if data.shape[-1] > 0:
        mean = np.mean(use_array, axis=-1)[..., None]
    else:
        mean = 0  # otherwise we get an ugly nan
    if mode == 'mean':
        data -= mean
    if mode == 'logratio':
        data /= mean
        data = np.log10(data)  # a value of 1 means 10 times bigger
    if mode == 'ratio':
        data /= mean
    elif mode == 'zscore':
        std = np.std(use_array, axis=-1)[..., None]
        data -= mean
        data /= std
    elif mode == 'percent':
        data -= mean
        data /= mean
    elif mode == 'zlogratio':
        data /= mean
        data = np.log10(data)
        std = np.std(use_array, axis=-1)[..., None]
        data /= std

    return data


def rank_estimation(data):
    '''
    Function to estimate the rank of the data using different rank estimators.
    '''
    from jumeg.decompose.ica import whitening
    from jumeg.decompose.dimension_selection import mibs, gap, aic, mdl, bic

    nchan, ntsl = data.shape

    # perform PCA to get sorted eigenvalues
    data_w, pca = whitening(data.T)

    # apply different rank estimators
    # MIBS, BIC, GAP, AIC, MDL, pct95, pct99
    rank1 = mibs(pca.explained_variance_, ntsl)  # MIBS
    rank2 = bic(pca.explained_variance_, ntsl)   # BIC
    rank3 = gap(pca.explained_variance_)  # GAP
    rank4 = aic(pca.explained_variance_)  # AIC
    rank5 = mdl(pca.explained_variance_)  # MDL
    rank6 = np.where(pca.explained_variance_ratio_.cumsum() <= 0.95)[0].size
    rank7 = np.where(pca.explained_variance_ratio_.cumsum() <= 0.99)[0].size
    rank_all = np.array([rank1, rank2, rank3, rank4, rank5, rank6, rank7])
    return (rank_all, np.median(rank_all))

def clip_eog2(eog, clip_to_value):
    '''
    Function to clip the EOG channel to a certain clip_to_value.
    All peaks higher than given value are pruned.
    Note: this may be used when peak detection for artefact removal fails due to
    abnormally high peaks in the EOG channel.

    Can be applied to a raw file using the below code:

    # apply the above function to one channel (here 276) of the raw object
    raw.apply_function(clip_eog2, clip_to_value=clip_to_value, picks=[276],
                       dtype=None, n_jobs=2)

    # saw the raw file
    raw.save(raw.info['filename'].split('-raw.fif')[0] + ',eogclip-raw.fif',
             overwrite=False)
    '''
    if clip_to_value < 0:
        eog_clipped = np.clip(eog, clip_to_value, np.max(eog))
    elif clip_to_value > 0:
        eog_clipped = np.clip(eog, np.min(eog), clip_to_value)
    else:
        print('Zero clip_to_value is ambigious !! Please check again.')
    return eog_clipped


def loadingBar(count, total, task_part=None):
    """ Provides user with a loadingbar line. See following:
          041/400 [==                  ]                     Subtask 793

          count/total [==                  ]                     'task_part'

    Parameters
    ----------
    count : str, float or int
          Current task count. Easy to access throught 'enumerate()'
    total : str, float or int
          Maximal number of all tasks
    task_part : String | Optional
          If the task is divided in subtask and you want to keep track of
          your functions progress in detail pass your subtask in string format.

    Example
    -------
    array = np.linspace(1, 1000, 400)
    for p, i in enumerate(array):
      loadingBar(count=p, total=array.shape[0],
                 task_part='Subtask')
    Returns
    -------
    stdout : Rewriteable String Output
          Generates a String Output for every of the progress steps
    """
    if task_part is None:
        task_part = ''
    percent = float(count + 1) / float(total) * 100
    size = 2

    sys.stdout.write("\r    "
                     + str(int(count + 1)).rjust(3, '0')
                     + "/" + str(int(total)).rjust(3, '0')
                     + ' [' + '=' * int(percent / 10) * size
                     + ' ' * (10 - int(percent / 10)) * size
                     + ']  %30s' % (task_part))
    if count + 1 == total:
        finish = '[done]'
        sys.stdout.write("\r    "
                         + str(int(count + 1)).rjust(3, '0')
                         + "/" + str(int(total)).rjust(3, '0')
                         + ' [' + '=' * int(percent / 10) * size
                         + ' ' * (10 - int(percent / 10)) * size
                         + ']  %30s\n' % (finish))


    return


def find_files(rootdir='.', pattern='*', recursive=False):
    '''
    Search and get list of filenames matching pattern.
    '''

    files = []
    for root, dirnames, filenames in os.walk(rootdir):
        if not recursive:
            del dirnames[:]
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    files = sorted(files)

    return files


def find_directories(rootdir='.', pattern='*'):
    '''
    Search and get a list of directories matching pattern.
    '''

    path = rootdir
    if path[-1] != '/':
        path += '/'

    # search for directories in rootdir
    dirlist=[]
    for filename in os.listdir(rootdir):
        if os.path.isdir(path+filename) == True:
            dirlist.append(filename)
    dirlist = sorted(dirlist)

    # select those which match pattern
    dirlist = fnmatch.filter(dirlist, pattern)

    return dirlist
