import os.path as op
import numpy as np
from utils import set_directory

import mne
from jumeg.decompose.ica_replace_mean_std import ICA, read_ica, apply_ica_replace_mean_std
from jumeg.jumeg_preprocessing import get_ics_cardiac, get_ics_ocular
from jumeg.jumeg_plot import plot_performance_artifact_rejection  # , plot_artefact_overview


def determine_chop_times_every_x_s(total_time, chop_length=60.):
    """
    Chop every X s where X=interval. If the last chop would have a length
    under X it is combined with the penultimate chop.

    Parameters
    ----------
    total_time : float
        Total length of the recording.
    chop_length : float
        Length of a chop.

    Returns
    -------
    chop_times : list of float
        Time points for when to chop the raw file
    """

    chop_times = []
    chop = 0.

    while total_time >= chop + 2 * chop_length:
        chop += chop_length
        chop_times.append(chop)

    return chop_times


def get_tmin_tmax(ct_idx, chop_times, sfreq):
    """
    Get tmin and tmax for the chop interval based on the
    time points given by chop_times.

    Parameters:
    -----------
    ct_idx : int
        Index corresponding to chop_times.
    chop_times : list of float
        List with the time points of when to chop the data.
    sfreq : float
        Sampling frequency of the measurement data.

    Returns:
    --------
    tmin : int
        Starting time of the chop interval in s.
    tmax : int
        Ending time of the chop interval in s.
    """

    if ct_idx == 0:
        tmin = 0
        tmax = chop_times[ct_idx] - 1. / sfreq
        print(int(tmin), int(tmax))

    elif ct_idx == len(chop_times):
        tmin = chop_times[ct_idx - 1]
        tmax = None
        print(int(tmin), "None")

    else:
        tmin = chop_times[ct_idx - 1]
        tmax = chop_times[ct_idx] - 1. / sfreq
        print(int(tmin), int(tmax))

    return tmin, tmax


def apply_ica_and_plot_performance(raw, ica, name_ecg, name_eog, raw_fname, clean_fname, picks=None,
                                   reject=None, replace_pre_whitener=True, save=False):
    """
    Applies ICA to the raw object and plots the performance of rejecting ECG and EOG artifacts.

    Parameters
    ----------
    raw : mne.io.Raw()
        Raw object ICA is applied to
    ica : ICA object
        ICA object being applied d to the raw object
    name_ecg : str
        Name of the ECG channel in the raw data
    name_eog : str
        Name of the (vertical) EOG channel in the raw data
    raw_fname : str | None
        Path for saving the raw object
    clean_fname : str | None
        Path for saving the ICA cleaned raw object
    picks : array-like of int | None
        Channels to be included for the calculation of pca_mean_ and _pre_whitener.
        This selection SHOULD BE THE SAME AS the one used in ica.fit().
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude. This parameter SHOULD BE
        THE SAME AS the one used in ica.fit().
        Valid keys are 'grad', 'mag', 'eeg', 'seeg', 'ecog', 'eog', 'ecg',
        'hbo', 'hbr'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
                          )

        It only applies if `inst` is of type Raw.
    replace_pre_whitener : bool
        If True, pre_whitener is replaced when applying ICA to
        unfiltered data otherwise the original pre_whitener is used.
    save : bool
        Save the raw object and cleaned raw object

    Returns
    -------
    raw_clean : mne.io.Raw()
        Raw object after ICA cleaning
    """

    # apply_ica_replace_mean_std processes in place -> need copy to plot performance
    raw_copy = raw.copy()
    ica = ica.copy()

    raw_clean = apply_ica_replace_mean_std(raw, ica, picks=picks, reject=reject,
                                           exclude=ica.exclude, n_pca_components=None,
                                           replace_pre_whitener=replace_pre_whitener)
    if save:
        if raw_fname is not None:
            raw_copy.save(raw_fname, overwrite=True)
        raw_clean.save(clean_fname, overwrite=True)

    overview_fname = clean_fname.rsplit('-raw.fif')[0] + ',overview-plot'
    plot_performance_artifact_rejection(raw_copy, ica, overview_fname,
                                        meg_clean=raw_clean,
                                        show=False, verbose=False,
                                        name_ecg=name_ecg,
                                        name_eog=name_eog)
    print('Saved ', overview_fname)

    raw_copy.close()

    return raw_clean


def fit_ica(raw, picks, reject, ecg_ch, eog_hor, eog_ver,
            flow_ecg, fhigh_ecg, flow_eog, fhigh_eog, ecg_thresh,
            eog_thresh, random_state):
    """
    Fit an ICA object to the raw file. Identify cardiac and ocular components
    and mark them for removal.

    Parameters:
    -----------
    inst : instance of Raw, Epochs or Evoked
        Raw measurements to be decomposed.
    picks : array-like of int
        Channels to be included. This selection remains throughout the
        initialized ICA solution. If None only good data channels are used.
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad', 'mag', 'eeg', 'seeg', 'ecog', 'eog', 'ecg',
        'hbo', 'hbr'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
                          )

        It only applies if `inst` is of type Raw.
    ecg_ch : array-like | ch_name | None
        ECG channel to which the sources shall be compared. It has to be
        of the same shape as the sources. If some string is supplied, a
        routine will try to find a matching channel. If None, a score
        function expecting only one input-array argument must be used,
        for instance, scipy.stats.skew (default).
    eog_hor : array-like | ch_name | None
        Horizontal EOG channel to which the sources shall be compared.
        It has to be of the same shape as the sources. If some string
        is supplied, a routine will try to find a matching channel. If
        None, a score function expecting only one input-array argument
        must be used, for instance, scipy.stats.skew (default).
    eog_ver : array-like | ch_name | None
        Vertical EOG channel to which the sources shall be compared.
        It has to be of the same shape as the sources. If some string
        is supplied, a routine will try to find a matching channel. If
        None, a score function expecting only one input-array argument
        must be used, for instance, scipy.stats.skew (default).
    flow_ecg : float
        Low pass frequency for ECG component identification.
    fhigh_ecg : float
        High pass frequency for ECG component identification.
    flow_eog : float
        Low pass frequency for EOG component identification.
    fhigh_eog : float
        High pass frequency for EOG component identification.
    ecg_thresh : float
        Threshold for ECG component idenfication.
    eog_thresh : float
        Threshold for EOG component idenfication.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the FastICA estimation.
        As the estimation is non-deterministic it can be useful to
        fix the seed to have reproducible results. Defaults to None.

    Returns:
    --------
    ica : mne.preprocessing.ICA
        ICA object for raw file with ECG and EOG components marked for removal.

    """
    # increased iteration to make it converge
    # fix the number of components to 60 to avoid diff numbers across different chops of data
    # 'extended-infomax', 'fastica', 'picard'
    ica = ICA(method='fastica', n_components=60, random_state=random_state,
              max_pca_components=None, max_iter=5000, verbose=False)
    ica.fit(raw, picks=picks, decim=None, reject=reject, verbose=True)

    #######################################################################
    # identify bad components
    #######################################################################

    # get ECG and EOG related components using MNE
    print('Computing scores and identifying components..')
    ecg_scores = ica.score_sources(raw, target=ecg_ch, score_func='pearsonr',
                                   l_freq=flow_ecg, h_freq=fhigh_ecg, verbose=False)
    # horizontal channel
    eog1_scores = ica.score_sources(raw, target=eog_hor, score_func='pearsonr',
                                    l_freq=flow_eog, h_freq=fhigh_eog, verbose=False)
    # vertical channel
    eog2_scores = ica.score_sources(raw, target=eog_ver, score_func='pearsonr',
                                    l_freq=flow_eog, h_freq=fhigh_eog, verbose=False)

    # print the top ecg, eog correlation scores
    ecg_inds = np.where(np.abs(ecg_scores) > ecg_thresh)[0]
    eog1_inds = np.where(np.abs(eog1_scores) > eog_thresh)[0]
    eog2_inds = np.where(np.abs(eog2_scores) > eog_thresh)[0]

    highly_corr = list(set(np.concatenate((ecg_inds, eog1_inds, eog2_inds))))
    highly_corr.sort()

    # get ECG/EOG related components using JuMEG
    ic_ecg = get_ics_cardiac(raw, ica, flow=flow_ecg, fhigh=fhigh_ecg,
                             thresh=ecg_thresh, tmin=-0.5, tmax=0.5, name_ecg=ecg_ch,
                             use_CTPS=True)[0]
    ic_eog = get_ics_ocular(raw, ica, flow=flow_eog, fhigh=fhigh_eog,
                            thresh=eog_thresh, name_eog_hor=eog_hor, name_eog_ver=eog_ver,
                            score_func='pearsonr')
    ic_ecg = list(set(ic_ecg))
    ic_eog = list(set(ic_eog))
    ic_ecg.sort()
    ic_eog.sort()

    # if necessary include components identified by correlation as well
    bads_list = list(set(list(ic_ecg) + list(ic_eog) + highly_corr))
    bads_list.sort()
    ica.exclude = bads_list

    highly_corr_ecg = list(set(ecg_inds))
    highly_corr_eog1 = list(set(eog1_inds))
    highly_corr_eog2 = list(set(eog2_inds))

    highly_corr_ecg.sort()
    highly_corr_eog1.sort()
    highly_corr_eog2.sort()

    print('Highly correlated artifact components are:')
    print('    ECG:  ', highly_corr_ecg)
    print('    EOG 1:', highly_corr_eog1)
    print('    EOG 2:', highly_corr_eog2)
    print('')
    print('Identified ECG components are: ', ic_ecg)
    print('Identified EOG components are: ', ic_eog)
    print("Plot ica sources to remove jumpy component for channels 4, 6, 8, 22")

    return ica


def chop_and_apply_ica(raw_filt_fname, pre_proc_ext, chop_length=60.,
                       flow_ecg=8., fhigh_ecg=20., flow_eog=1., fhigh_eog=20.,
                       ecg_thresh=0.25, eog_thresh=0.25, random_state=42, ecg_ch='EEG 002',
                       eog_hor='EEG 001', eog_ver='EEG 003', exclude='bads',
                       unfiltered=True, save=False):
    """
    Read raw file, chop it after trials and apply ica on the chops.
    Save the ICA object, cleaned raw chops and plots overview of the artefact
    rejection.

    The function has a lot of parameters hard coded within, please look
    through carefully and modify as seems fit.

    Parameters
    ----------
    raw_filt_fname : str
        The filtered raw file to clean.
    pre_proc_ext : str
        Extension listing the pre-processing steps done.
    chop_length : float
        Length of the chops in seconds.
    flow : float
        Lower frequency for the filtering applied to the raw file prior to chopping.
    fhigh : float
        Higher frequency for the filtering applied to the raw file prior to chopping.
    flow_ecg : float
        Lower frequency for the scoring of ECG sources.
    fhigh_ecg : float
        Higher frequency for the scoring of ECG sources.
    flow_eog : float
        Lower frequency for the scoring of EOG sources.
    fhigh_eog : float
        Higher frequency for the scoring of EOG sources.
    random_state : int
        Seed for pseudo random number generator.
    ecg_ch : str
        Name of the ECG channel.
    eog_hor : str
        Name of the horizontal EOG channel.
    eog_ver : str
        Name of the vertical EOG channel.
    ecg_thresh : float
        Threshold for independent ecg components.
    eog_thresh : float
        Threshold for independent eog components.
    exclude : list of string | str
        List of channels to exclude. If 'bads' (default), exclude channels
        in info['bads'].
    unfiltered : bool
        Apply ICA cleaning to unfiltered data as well.
    save : bool
        Save raw chops after applying ICA.

    Returns
    -------
    raw_chop_clean_filtered_list : list of mne.io.Raw instances
        List of the filtered raw chops after applying ICA cleaning.
    raw_chop_clean_unfiltered_list : list of mne.io.Raw instances
        List of the unfiltered raw chops after applying ICA cleaning.
    """

    raw_chop_clean_filtered_list = []
    raw_chop_clean_unfiltered_list = []

    print('Running chop_and_apply_ica on ', raw_filt_fname)

    reject = {'mag': 5e-12}

    raw_filt = mne.io.Raw(raw_filt_fname, preload=True, verbose=True)

    if unfiltered:
        raw_unfilt_fname = raw_filt_fname.replace(',fibp', '')
        raw_unfilt = mne.io.Raw(raw_unfilt_fname, preload=True, verbose=True)

    picks = mne.pick_types(raw_filt.info, meg=True, exclude=exclude)

    # you might want to determine the chop time in a more sophisticated way
    # to avoid accidentally chopping in the middle of a trial
    chop_times = determine_chop_times_every_x_s(raw_filt.n_times / raw_filt.info["sfreq"],
                                                chop_length=chop_length)

    # chop the data and apply filtering
    # avoid double counting of data point at chop: tmax = chop_times[i] - 1./raw.info["sfreq"]

    for i in range(0, len(chop_times) + 1):

        # get chop interval
        tmin, tmax = get_tmin_tmax(ct_idx=i, chop_times=chop_times,
                                   sfreq=raw_filt.info["sfreq"])

        #######################################################################
        # building the file names here
        #######################################################################

        info_filt = "fibp"

        if tmax is not None:
            tmaxi = int(tmax)
        else:
            tmaxi = tmax

        dirname = op.join(op.dirname(raw_filt_fname), 'chops')
        set_directory(dirname)
        prefix_filt = raw_filt_fname.rsplit('/')[-1].rsplit('-raw.fif')[0]
        ica_fname = op.join(dirname, prefix_filt + ',{}-{}-ica.fif'.format(int(tmin), tmaxi))

        # make sure to copy because the original is lost
        raw_filt_chop = raw_filt.copy().crop(tmin=tmin, tmax=tmax)
        clean_filt_fname = op.join(dirname, prefix_filt + ',{},ar,{}-{}-raw.fif'.format(info_filt, int(tmin), tmaxi))
        raw_filt_chop_fname = op.join(dirname, prefix_filt + ',{},{}-{}-raw.fif'.format(info_filt, int(tmin), tmaxi))

        if unfiltered:
            prefix_unfilt = prefix_filt.replace(',fibp', '')
            raw_unfilt_chop = raw_unfilt.copy().crop(tmin=tmin, tmax=tmax)
            clean_unfilt_fname = op.join(dirname, prefix_unfilt + ',ar,{}-{}-raw.fif'.format(int(tmin), tmaxi))
            raw_unfilt_chop_fname = op.join(dirname, prefix_unfilt + ',{}-{}-raw.fif'.format(int(tmin), tmaxi))

        #######################################################################
        # run the ICA on the chops
        #######################################################################

        print('Starting ICA...')
        if op.isfile(ica_fname):

            ica = read_ica(ica_fname)

        else:

            ica = fit_ica(raw=raw_filt_chop, picks=picks, reject=reject,
                          ecg_ch=ecg_ch, eog_hor=eog_hor, eog_ver=eog_ver,
                          flow_ecg=flow_ecg, fhigh_ecg=fhigh_ecg,
                          flow_eog=flow_eog, fhigh_eog=fhigh_eog,
                          ecg_thresh=ecg_thresh, eog_thresh=eog_thresh,
                          random_state=random_state)

        # plot topo-plots first because sometimes components are hard to identify
        # ica.plot_components()
        # do the most important manual check
        ica.plot_sources(raw_filt_chop, block=True)

        # save ica object
        ica.save(ica_fname)

        print('ICA components excluded: ', ica.exclude)

        #######################################################################
        # apply the ICA to data and save the resulting files
        #######################################################################

        print('Running cleaning on filtered data...')
        clean_filt_chop = apply_ica_and_plot_performance(raw_filt_chop, ica, ecg_ch, eog_ver,
                                                         raw_filt_chop_fname, clean_fname=clean_filt_fname,
                                                         picks=picks, replace_pre_whitener=True,
                                                         reject=reject, save=save)

        raw_chop_clean_filtered_list.append(clean_filt_chop)

        if unfiltered:

            print('Running cleaning on unfiltered data...')
            clean_unfilt_chop = apply_ica_and_plot_performance(raw_unfilt_chop, ica, ecg_ch, eog_ver,
                                                               raw_unfilt_chop_fname, clean_fname=clean_unfilt_fname,
                                                               picks=picks, replace_pre_whitener=True,
                                                               reject=reject, save=save)

            raw_chop_clean_unfiltered_list.append(clean_unfilt_chop)

        # if tmax is None, last chop is reached
        if tmax is None:
            break

    return raw_chop_clean_filtered_list, raw_chop_clean_unfiltered_list