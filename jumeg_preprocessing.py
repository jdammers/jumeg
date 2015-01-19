""" Preprocessing functions """

import os
import numpy as np
import matplotlib.pyplot as pl
import mne
import ctps


#################################################################
#
# filename conventions 
#
# >>> I assume that this will be provided in a different way <<<
# >>> probably by Frank's new routines (?) <<<
#
#################################################################
ext_raw = '-raw.fif'
ext_ave = '-ave.fif'
ext_ica = '-ica.fif'
ext_clean = ',ar-raw.fif'
ext_icap = ',ica-performance'     # figure extension provided by the routine
ext_empty_raw = '-empty.fif'
ext_empty_cov = ',empty-cov.fif'
prefix_filt = ',fibp'             # for now bp only 
prefix_ctps = ',ctpsbr-'        # e.g.: "...,ica,ctps-trigger.npy"



#################################################################
#
# apply filter on (raw) data
#
#################################################################
def apply_filter(fname_raw, flow=1, fhigh=45, order=4, njobs=4):

    ''' Applies the MNE butterworth filter to a list of raw files. '''

    filter_type = 'butter'
    filt_method = 'fft'

    fnraw = get_files_from_list(fname_raw)

    # loop across all filenames
    for fname in fnraw:
        print ">>> filter raw data: %0.1f - %0.1f..." % (flow, fhigh)
        # load raw data
        raw = mne.io.Raw(fname, preload=True)
        # filter raw data
        raw.filter(flow, fhigh, n_jobs=njobs, method=filt_method)
        # raw.filter(l_freq=flow_raw, h_freq=fhigh_raw, n_jobs=njobs, method='iir',
        #     iir_params={'ftype': filter_type, 'order': order})
        print ">>>> writing filtered data to disk..."
        name_raw = fname[:fname.rfind('-')]  # fname.split('-')[0]
        fnfilt = name_raw + prefix_filt + "%d-%d" % (flow, fhigh)
        fnfilt = fnfilt + fname[fname.rfind('-'):]  # fname.split('-')[1]
        print 'saving: ' + fnfilt
        raw.save(fnfilt, overwrite=True)


#######################################################
#
#  apply average on list of files
#
#######################################################
def apply_average(filenames, name_stim='STI 014', event_id=None, postfix=None,
                  tmin=-0.2, tmax=0.4, baseline=(None, 0), proj=False,
                  save_plot=True, show_plot=False):

    ''' Performs averaging to a list of raw files. '''

    # Trigger or Response ?
    if name_stim == 'STI 014':      # trigger
        trig_name = 'trigger'
    else:
        if name_stim == 'STI 013':   # response
            trig_name = 'response'
        else:
            trig_name = 'trigger'

    fnlist = get_files_from_list(filenames)

    # loop across raw files
    fnavg = []    # collect output filenames
    for fname in fnlist:
        name = os.path.split(fname)[1]
        print '>>> average raw data'
        print name
        # load raw data
        raw = mne.io.Raw(fname, preload=True)
        picks = mne.pick_types(raw.info, meg=True, ref_meg=False,
                               exclude='bads')

        # stim events
        stim_events = mne.find_events(raw, stim_channel=name_stim,
                                      consecutive=True)
        nevents = len(stim_events)

        if nevents > 0:
            # for a specific event ID
            if event_id:
                ix = np.where(stim_events[:, 2] == event_id)[0]
                stim_events = stim_events[ix, :]
            else:
                event_id = stim_events[0, 2]

            epochs = mne.Epochs(raw, events=stim_events,
                                event_id=event_id, tmin=tmin, tmax=tmax,
                                picks=picks, preload=True, baseline=baseline,
                                proj=proj)
            avg = epochs.average()

            # save averaged data
            if (fname.rfind(ext_raw) > -1):
                nchar = 8
            else:
                nchar = 4
            if (postfix):
                fnout = fname[0:len(fname) - nchar] + postfix + '.fif'
            else:
                fnout = fname[0:len(fname) - nchar] + ',' + trig_name + ext_ave

            avg.save(fnout)
            print 'saved:' + fnout
            fnavg.append(fnout)

            if (save_plot):
                plot_average(fnavg, show_plot=show_plot)

        else:
            event_id = None
            print '>>> Warning: Event not found in file: ' + fname


#######################################################
#
#  plot average from a list of files
#
#######################################################
def plot_average(filenames, save_plot=True, show_plot=False, dpi=100):

    ''' Plot Signal average from a list of averaged files. '''

    fname = get_files_from_list(filenames)

    # plot averages
    pl.ioff()  # switch off (interactive) plot visualisation
    factor = 1e15
    for fnavg in fname:
        name = fnavg[0:len(fnavg) - 4]
        basename = os.path.splitext(os.path.basename(name))[0]
        print fnavg
        # mne.read_evokeds provides a list or a single evoked based on condition.
        # here we assume only one evoked is returned (requires further handling)
        avg = mne.read_evokeds(fnavg)[0]
        ymin, ymax = avg.data.min(), avg.data.max()
        ymin *= factor * 1.1
        ymax *= factor * 1.1
        fig = pl.figure(basename, figsize=(10, 8), dpi=100)
        pl.clf()
        pl.ylim([ymin, ymax])
        pl.xlim([avg.times.min(), avg.times.max()])
        pl.plot(avg.times, avg.data.T * factor, color='black')
        pl.title(basename)

        # save figure
        fnfig = os.path.splitext(fnavg)[0] + '.png'
        pl.savefig(fnfig, dpi=dpi)

    pl.ion()  # switch on (interactive) plot visualisation


#######################################################
#
#  apply ICA for artifact rejection
#
#######################################################
def apply_ica(fname_filtered, n_components=0.99, decim=None,
              reject={'mag': 5e-12}, ica_method='fastica'):

    ''' Applies ICA to a list of (filtered) raw files. '''

    from mne.preprocessing import ICA

    fnfilt = get_files_from_list(fname_filtered)

    # loop across all filenames
    for fname in fnfilt:
        name = os.path.split(fname)[1]
        print ">>>> perform ICA signal decomposition on :  " + name
        # load filtered data
        raw = mne.io.Raw(fname, preload=True)
        picks = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude='bads')
        # ICA decomposition
        ica = ICA(method=ica_method, n_components=n_components,
                  max_pca_components=None)

        ica.fit(raw, picks=picks, decim=decim, reject=reject)

        # save ICA object
        fnica_out = fname[:fname.rfind(ext_raw)] + ext_ica
        # fnica_out = fname[0:len(fname)-4]+'-ica.fif'
        ica.save(fnica_out)


#######################################################
#
#  apply ICA-cleaning for artifact rejection
#
#######################################################
def apply_ica_cleaning(fname_ica, n_pca_components=None,
                       name_ecg='ECG 001', flow_ecg=10, fhigh_ecg=20,
                       name_eog_hor='EOG 001', name_eog_ver='EOG 002',
                       flow_eog=1, fhigh_eog=10, threshold=0.3,
                       unfiltered=False, notch_filter=True, notch_freq=50,
                       notch_width=None):

    ''' Performs artifact rejection based on ICA to a list of (ICA) files. '''

    fnlist = get_files_from_list(fname_ica)

    # loop across all filenames
    for fnica in fnlist:
        name = os.path.split(fnica)[1]
        #basename = fnica[0:len(fnica)-4]
        basename = fnica[:fnica.rfind(ext_ica)]
        fnfilt = basename + ext_raw
        fnclean = basename + ext_clean
        fnica_ar = basename + ext_icap
        print ">>>> perform artifact rejection on :"
        print '   ' + name

        # load filtered data
        meg_raw = mne.io.Raw(fnfilt, preload=True)
        picks = mne.pick_types(meg_raw.info, meg=True, ref_meg=False, exclude='bads')
        # ICA decomposition
        ica = mne.preprocessing.read_ica(fnica)

        # get ECG and EOG related components
        ic_ecg = get_ics_cardiac(meg_raw, ica,
                                 flow=flow_ecg, fhigh=fhigh_ecg, thresh=threshold)
        ic_eog = get_ics_ocular(meg_raw, ica,
                                flow=flow_eog, fhigh=fhigh_eog, thresh=threshold)
        ica.exclude += list(ic_ecg) + list(ic_eog)
        # ica.plot_topomap(ic_artefacts)
        ica.save(fnica)  # save again to store excluded

        # clean and save MEG data
        if n_pca_components:
            npca = n_pca_components
        else:
            npca = picks.size

        # check if cleaning should be applied
        # to unfiltered data
        if unfiltered:
            # adjust filenames to unfiltered data
            basename = basename[:basename.rfind(',')]
            fnfilt = basename + ext_raw
            fnclean = basename + ext_clean
            fnica_ar = basename + ext_icap

            # load raw unfiltered data
            meg_raw = mne.io.Raw(fnfilt, preload=True)

            # apply notch filter
            if notch_filter:

                from jumeg.filter import jumeg_filter

                # generate and apply filter
                # check if array of frequencies is given
                if type(notch_freq) in (tuple, list):
                    notch = np.array(notch_freq)
                elif type(np.ndarray) == np.ndarray:
                    notch = notch_freq
                # or a single frequency
                else:
                    notch = np.array([])

                fi_mne_notch = jumeg_filter(filter_method="mne", filter_type='notch',
                                            remove_dcoffset=False,
                                            notch=notch, notch_width=notch_width)

                # if only a single frequency is given generate optimal
                # filter parameter to also remove the harmonics
                if not type(notch_freq) in (tuple, list, np.ndarray):
                    fi_mne_notch.calc_notches(notch_freq)

                fi_mne_notch.apply_filter(meg_raw._data, picks=picks)

        # apply cleaning
        meg_clean = ica.apply(meg_raw, exclude=ica.exclude,
                              n_pca_components=npca, copy=True)
        meg_clean.save(fnclean, overwrite=True)

        # plot ECG, EOG averages before and after ICA
        print ">>>> create performance image..."
        plot_performance_artifact_rejection(meg_raw, ica, fnica_ar,
                                            show=False, verbose=False)


#######################################################
#
#  determine occular related ICs
#
#######################################################
def get_ics_ocular(meg_raw, ica, flow=1, fhigh=10,
                   name_eog_hor='EOG 001', name_eog_ver='EOG 002',
                   score_func='pearsonr', thresh=0.3):
    '''
    Find Independent Components related to ocular artefacts
    '''

    # Note: when using the following:
    #   - the filter settings are different
    #   - here we cannot define the filter range

    # vertical EOG
    # idx_eog_ver = [meg_raw.ch_names.index(name_eog_ver)]
    # eog_scores = ica.score_sources(meg_raw, meg_raw[idx_eog_ver][0])
    # eogv_idx = np.where(np.abs(eog_scores) > thresh)[0]
    # ica.exclude += list(eogv_idx)
    # ica.plot_topomap(eog_idx)

    # horizontal EOG
    # idx_eog_hor = [meg_raw.ch_names.index(name_eog_hor)]
    # eog_scores = ica.score_sources(meg_raw, meg_raw[idx_eog_hor][0])
    # eogh_idx = np.where(np.abs(eog_scores) > thresh)[0]
    # ica.exclude += list(eogh_idx)
    # ica.plot_topomap(eog_idx)
    # print [eogv_idx, eogh_idx]

    # vertical EOG
    if name_eog_ver in meg_raw.ch_names:
        idx_eog_ver = [meg_raw.ch_names.index(name_eog_ver)]
        eog_ver_filtered = mne.filter.band_pass_filter(meg_raw[idx_eog_ver, :][0],
                                                       meg_raw.info['sfreq'],
                                                       Fp1=flow, Fp2=fhigh)
        eog_ver_scores = ica.score_sources(meg_raw, target=eog_ver_filtered,
                                           score_func=score_func)
        # plus 1 for any()
        ic_eog_ver = np.where(np.abs(eog_ver_scores) >= thresh)[0] + 1
        if not ic_eog_ver.any():
            ic_eog_ver = np.array([0])
    else:
        print ">>>> NOTE: No vertical EOG channel found!"
        ic_eog_ver = np.array([0])

    # horizontal EOG
    if name_eog_hor in meg_raw.ch_names:
        idx_eog_hor = [meg_raw.ch_names.index(name_eog_hor)]
        eog_hor_filtered = mne.filter.band_pass_filter(meg_raw[idx_eog_hor, :][0],
                                                       meg_raw.info['sfreq'],
                                                       Fp1=flow, Fp2=fhigh)
        eog_hor_scores = ica.score_sources(meg_raw, target=eog_hor_filtered,
                                           score_func=score_func)
        # plus 1 for any()
        ic_eog_hor = np.where(np.abs(eog_hor_scores) >= thresh)[0] + 1
        if not ic_eog_hor.any():
            ic_eog_hor = np.array([0])
    else:
        print ">>>> NOTE: No horizontal EOG channel found!"
        ic_eog_hor = np.array([0])

    # combine both
    idx_eog = []
    for i in range(ic_eog_ver.size):
        ix = ic_eog_ver[i] - 1
        if (ix >= 0):
            idx_eog.append(ix)
    for i in range(ic_eog_hor.size):
        ix = ic_eog_hor[i] - 1
        if (ix >= 0):
            idx_eog.append(ix)

    return idx_eog


#######################################################
#
#  determine cardiac related ICs
#
#######################################################
def get_ics_cardiac(meg_raw, ica, flow=10, fhigh=20, tmin=-0.3, tmax=0.3,
                    name_ecg='ECG 001', use_CTPS=True, proj=False,
                    score_func='pearsonr', thresh=0.3):
    '''
    Identify components with cardiac artefacts
    '''

    from mne.preprocessing import find_ecg_events
    event_id_ecg = 999

    if name_ecg in meg_raw.ch_names:
        # get and filter ICA signals
        ica_raw = ica.get_sources(meg_raw)
        ica_raw.filter(l_freq=flow, h_freq=fhigh, n_jobs=2, method='fft')
        # get R-peak indices in ECG signal
        idx_R_peak, _, _ = find_ecg_events(meg_raw, ch_name=name_ecg,
                                           event_id=event_id_ecg, l_freq=flow,
                                           h_freq=fhigh, verbose=False)

        # -----------------------------------
        # default method:  CTPS
        #           else:  correlation
        # -----------------------------------
        if use_CTPS:
            # create epochs
            picks = np.arange(ica.n_components_)
            ica_epochs = mne.Epochs(ica_raw, events=idx_R_peak,
                                    event_id=event_id_ecg, tmin=tmin,
                                    tmax=tmax, baseline=None,
                                    proj=False, picks=picks, verbose=False)
            # compute CTPS
            _, pk, _ = ctps.compute_ctps(ica_epochs.get_data())

            pk_max = np.max(pk, axis=1)
            idx_ecg = np.where(pk_max >= thresh)[0]
        else:
            # use correlation
            idx_ecg = [meg_raw.ch_names.index(name_ecg)]
            ecg_filtered = mne.filter.band_pass_filter(meg_raw[idx_ecg, :][0],
                                                       meg_raw.info['sfreq'],
                                                       Fp1=flow, Fp2=fhigh)
            ecg_scores = ica.score_sources(meg_raw, target=ecg_filtered,
                                           score_func=score_func)
            idx_ecg = np.where(np.abs(ecg_scores) >= thresh)[0]

    else:
        print ">>>> NOTE: No ECG channel found!"
        idx_ecg = np.array([0])

    return idx_ecg


#######################################################
#
#  calculate the performance of artifact rejection
#
#######################################################
def calc_performance(evoked_raw, evoked_clean):
    ''' Gives a measure of the performance of the artifact reduction.
        Percentage value returned as output.
    '''
    from jumeg import jumeg_math as jmath

    diff = evoked_raw.data - evoked_clean.data
    rms_diff = jmath.calc_rms(diff, average=1)
    rms_meg = jmath.calc_rms(evoked_raw.data, average=1)
    arp = (rms_diff / rms_meg) * 100.0
    return np.round(arp)


#######################################################
#
#  calculate the frequency-correlation value
#
#######################################################
def calc_frequency_correlation(evoked_raw, evoked_clean):

    """
    Function to estimate the frequency-correlation value
    as introduced by Krishnaveni et al. (2006),
    Journal of Neural Engineering.
    """

    # transform signal to frequency range
    fft_raw = np.fft.fft(evoked_raw.data)
    fft_cleaned = np.fft.fft(evoked_clean.data)

    # get numerator
    numerator = np.sum(np.abs(np.real(fft_raw) * np.real(fft_cleaned)) + \
                       np.abs(np.imag(fft_raw) * np.imag(fft_cleaned)))

    # get denominator
    denominator = np.sqrt(np.sum(np.abs(fft_raw) ** 2) * \
                          np.sum(np.abs(fft_cleaned) ** 2))

    return np.round(numerator/denominator * 100.)



#######################################################
#
#  make/save plots to show the performance
#            of the ICA artifact rejection
#
#######################################################
def plot_performance_artifact_rejection(meg_raw, ica, fnout_fig,
                                        meg_clean=None, show=False,
                                        proj=False, verbose=False):
    '''
    Creates a performance image of the data before
    and after the cleaning process.

    returns array of performance values:
        perf_art_rej[0] --> performance related to cardiac artifacts
        perf_art_rej[1] --> performance related to ocular artifacts
    '''

    from mne.preprocessing import find_ecg_events, find_eog_events
    from jumeg import jumeg_math as jmath

    name_ecg = 'ECG 001'
    name_eog_hor = 'EOG 001'
    name_eog_ver = 'EOG 002'
    event_id_ecg = 999
    event_id_eog = 998
    tmin_ecg = -0.4
    tmax_ecg = 0.4
    tmin_eog = -0.4
    tmax_eog = 0.4

    picks = mne.pick_types(meg_raw.info, meg=True, ref_meg=False,
                           exclude='bads')
    # as we defined x% of the explained variance as noise (e.g. 5%)
    # we will remove this noise from the data
    if meg_clean:
        meg_clean_given = True
    else:
        meg_clean_given = False
        meg_clean = ica.apply(meg_raw, exclude=ica.exclude,
                              n_pca_components=ica.n_components_,
                              copy=True)

    # plotting parameter
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # check if ECG and EOG was recorded in addition
    # to the MEG data
    ch_names = meg_raw.info['ch_names']

    # ECG
    if name_ecg in ch_names:
        nstart = 0
        nrange = 1
    else:
        nstart = 1
        nrange = 1

    # EOG
    if name_eog_ver in ch_names:
        nrange = 2

    perf_art_rej = np.zeros(2)

    yFigSize = 6 * nrange

    # ToDo:  How can we avoid popping up the window if show=False ?
    pl.ioff()
    pl.figure('performance image', figsize=(12, yFigSize))
    pl.clf()

    # ECG, EOG:  loop over all artifact events
    for i in range(nstart, nrange):
        # get event indices
        if i == 0:
            baseline = (None, None)
            event_id = event_id_ecg
            idx_event, _, _ = find_ecg_events(meg_raw, event_id, ch_name=name_ecg,
                                              verbose=verbose)
            idx_ref_chan = meg_raw.ch_names.index(name_ecg)
            tmin = tmin_ecg
            tmax = tmax_ecg
            pl1 = nrange * 100 + 21
            pl2 = nrange * 100 + 22
            text1 = "CA: original data"
            text2 = "CA: cleaned data"
        elif i == 1:
            baseline = (None, None)
            event_id = event_id_eog
            idx_event = find_eog_events(meg_raw, event_id, ch_name=name_eog_ver,
                                        verbose=verbose)
            idx_ref_chan = meg_raw.ch_names.index(name_eog_ver)
            tmin = tmin_eog
            tmax = tmax_eog
            pl1 = nrange * 100 + 21 + (nrange - nstart - 1) * 2
            pl2 = nrange * 100 + 22 + (nrange - nstart - 1) * 2
            text1 = "OA: original data"
            text2 = "OA: cleaned data"

        # average the signals
        raw_epochs = mne.Epochs(meg_raw, idx_event, event_id, tmin, tmax,
                                picks=picks, baseline=baseline, proj=proj, verbose=verbose)
        cleaned_epochs = mne.Epochs(meg_clean, idx_event, event_id, tmin, tmax,
                                    picks=picks, baseline=baseline, proj=proj, verbose=verbose)
        ref_epochs = mne.Epochs(meg_raw, idx_event, event_id, tmin, tmax, picks=[idx_ref_chan],
                                baseline=baseline, proj=proj, verbose=verbose)

        raw_epochs_avg = raw_epochs.average()
        cleaned_epochs_avg = cleaned_epochs.average()
        ref_epochs_avg = np.average(ref_epochs.get_data(), axis=0).flatten() * -1.0
        times = raw_epochs_avg.times * 1e3
        if np.max(raw_epochs_avg.data) < 1:
            factor = 1e15
        else:
            factor = 1
        ymin = np.min(raw_epochs_avg.data) * factor
        ymax = np.max(raw_epochs_avg.data) * factor

        # plotting data before cleaning
        pl.subplot(pl1)
        pl.plot(times, raw_epochs_avg.data.T * factor, 'k')
        pl.title(text1)
        # plotting reference signal
        pl.plot(times, jmath.rescale(ref_epochs_avg, ymin, ymax), 'r')
        pl.xlim(times[0], times[len(times) - 1])
        pl.ylim(1.1 * ymin, 1.1 * ymax)
        # print some info
        textstr1 = 'num_events=%d\nEpochs: tmin, tmax = %0.1f, %0.1f' \
                   % (len(idx_event), tmin, tmax)
        pl.text(times[10], 1.09 * ymax, textstr1, fontsize=10,
                verticalalignment='top', bbox=props)

        # plotting data after cleaning
        pl.subplot(pl2)
        pl.plot(times, cleaned_epochs_avg.data.T * factor, 'k')
        pl.title(text2)
        # plotting reference signal again
        pl.plot(times, jmath.rescale(ref_epochs_avg, ymin, ymax), 'r')
        pl.xlim(times[0], times[len(times) - 1])
        pl.ylim(1.1 * ymin, 1.1 * ymax)
        # print some info
        #ToDo: would be nice to add info about ica.excluded

        perf_art_rej[i] = calc_performance(raw_epochs_avg, cleaned_epochs_avg)
        if meg_clean_given:
            textstr1 = 'Performance: %d\nFrequency Correlation: %d'\
                       % (perf_art_rej[i],
                          calc_frequency_correlation(raw_epochs_avg, cleaned_epochs_avg))
        else:
            textstr1 = 'Performance: %d\nFrequency Correlation: %d\n# ICs: %d\nExplained Var.: %d'\
                       % (perf_art_rej[i],
                          calc_frequency_correlation(raw_epochs_avg, cleaned_epochs_avg),
                          ica.n_components_, ica.n_components*100)

        pl.text(times[10], 1.09 * ymax, textstr1, fontsize=10,
                verticalalignment='top', bbox=props)

    if show:
        pl.show()

    # save image
    pl.savefig(fnout_fig + '.png', format='png')
    pl.close('performance image')
    pl.ion()

    return perf_art_rej



#######################################################
#
#  apply CTPS (for brain responses)
#
#######################################################
def apply_ctps(fname_ica, freqs=[(1, 4), (4, 8), (8, 12), (12, 16), (16, 20)],
               tmin=-0.2, tmax=0.4, name_stim='STI 014', event_id=None,
               baseline=(None, 0), proj=False):

    ''' Applies CTPS to a list of ICA files. '''

    from jumeg.filter import jumeg_filter

    fiws = jumeg_filter(filter_method="bw")
    fiws.filter_type = 'bp'   # bp, lp, hp
    fiws.dcoffset = True
    fiws.filter_attenuation_factor = 1

    nfreq = len(freqs)
    print '>>> CTPS calculation on: ', freqs


    # Trigger or Response ?
    if name_stim == 'STI 014':      # trigger
        trig_name = 'trigger'
    else:
        if name_stim == 'STI 013':   # response
            trig_name = 'response'
        else:
            trig_name = 'auxillary'

    fnlist = get_files_from_list(fname_ica)

    # loop across all filenames
    for fnica in fnlist:
        name = os.path.split(fnica)[1]
        #fname = fnica[0:len(fnica)-4]
        basename = fnica[:fnica.rfind(ext_ica)]
        fnraw = basename + ext_raw
        #basename = os.path.splitext(os.path.basename(fnica))[0]
        # load cleaned data
        raw = mne.io.Raw(fnraw, preload=True)
        picks = mne.pick_types(raw.info, meg=True, ref_meg=False, exclude='bads')

        # read (second) ICA
        print ">>>> working on: " + basename
        ica = mne.preprocessing.read_ica(fnica)
        ica_picks = np.arange(ica.n_components_)
        ncomp = len(ica_picks)

        # stim events
        stim_events = mne.find_events(raw, stim_channel=name_stim, consecutive=True)
        nevents = len(stim_events)

        if (nevents > 0):
            # for a specific event ID
            if event_id:
                ix = np.where(stim_events[:, 2] == event_id)[0]
                stim_events = stim_events[ix, :]
            else:
                event_id = stim_events[0, 2]
            # create ctps dictionary
            dctps = {'fnica': fnica,
                     'basename': basename,
                     'stim_channel': name_stim,
                     'trig_name': trig_name,
                     'ncomp': ncomp,
                     'nevent': nevents,
                     'event_id': event_id,
                     'nfreq': nfreq,
                     'freqs': freqs,
                     }
            # loop across all filenames
            pkarr = []
            ptarr = []
            pkmax_arr = []
            for ifreq in range(nfreq):
                ica_raw = ica.get_sources(raw)
                flow, fhigh = freqs[ifreq][0], freqs[ifreq][1]
                bp = str(flow) + '_' + str(fhigh)
                # filter ICA data and create epochs
                #tw=0.1
                # ica_raw.filter(l_freq=flow, h_freq=fhigh, picks=ica_picks,
                #     method='fft',l_trans_bandwidth=tw, h_trans_bandwidth=tw)
                # ica_raw.filter(l_freq=flow, h_freq=fhigh, picks=ica_picks,
                #                                                 method='fft')

                # filter ws settings
                # later we will make this as a one line call
                data_length = raw._data[0, :].size
                fiws.sampling_frequency = raw.info['sfreq']
                fiws.fcut1 = flow
                fiws.fcut2 = fhigh
                #fiws.init_filter_kernel(data_length)
                #fiws.init_filter(data_length)
                for ichan in ica_picks:
                    fiws.apply_filter(ica_raw._data[ichan, :])

                ica_epochs = mne.Epochs(ica_raw, events=stim_events,
                                        event_id=event_id, tmin=tmin,
                                        tmax=tmax, verbose=False,
                                        picks=ica_picks, baseline=baseline,
                                        proj=proj)
                # compute CTPS
                _, pk, pt = ctps.compute_ctps(ica_epochs.get_data())
                pkmax = pk.max(1)
                times = ica_epochs.times * 1e3
                pkarr.append(pk)
                ptarr.append(pt)
                pkmax_arr.append(pkmax)
            pkarr = np.array(pkarr)
            ptarr = np.array(ptarr)
            pkmax_arr = np.array(pkmax_arr)
            dctps['pk'] = np.float32(pkarr)
            dctps['pt'] = np.float32(ptarr)
            dctps['pkmax'] = np.float32(pkmax_arr)
            dctps['nsamp'] = len(times)
            dctps['times'] = np.float32(times)
            dctps['tmin'] = np.float32(ica_epochs.tmin)
            dctps['tmax'] = np.float32(ica_epochs.tmax)
            fnctps = basename + prefix_ctps + trig_name
            np.save(fnctps, dctps)
            # Note; loading example: dctps = np.load(fnctps).items()
        else:
            event_id = None

#######################################################
#
#  Perform CTPS surrogates tests
#
#######################################################
def apply_ctps_surrogates(fname_ctps, fnout, nrepeat=1000,
    mode='shuffle', save=True, n_jobs=4):

    ''' 
    Perform CTPS surrogate tests to estimate the significance level 
    for CTPS anaysis (a proper pK value ist estimated).

    It is most likely that the statistical reliability of this test
    is best improved by increasing the number of repetitions, while the
    number of different experiments/subjects have minor effects only

    Parameters
    ----------
    fname_ctps:  CTPS filename (or list of filenames)

    fnout: Output (text) filename to store surrogate stats across all files

    Options:
    nrepeat: number of repetitions used to estimate the pk threshold
             default is 1000

    mode: 2 different modi are allowed.
        'mode=shuffle' whill randomly shuffle the phase values. This is the default
        'mode=shift' whill randomly shift the phase values

    Return
    ------
    info string array containing the statistical values about the surrogate analysis

    '''
    import os, time
    from jumeg_utils import make_surrogates_ctps, get_stats_surrogates_ctps

    fnlist = get_files_from_list(fname_ctps)

    # loop across all filenames
    ifile = 1
    sep = '=========================================================================='
    info = [sep,'#','# Statistical analysis on CTPS surrogates','#',sep]  
    for fnctps in fnlist:
        path = os.path.dirname(fnctps)
        basename = os.path.basename(fnctps)
        name = os.path.splitext(basename)[0]
        print '>>> calc. surrogates based on: ' + basename
        # load CTPS data
        dctps = np.load(fnctps).item()
        phase_trials = dctps['pt']  # [nfreq, ntrials, nsources, nsamples]
        # create surrogate tests
        t_start = time.time()
        pks = make_surrogates_ctps(phase_trials,nrepeat=nrepeat,
            mode=mode,verbose=None,n_jobs=n_jobs)

        # perform stats on surrogates
        stats = get_stats_surrogates_ctps(pks, verbose=False)
        info.append(sep)
        info.append(path)
        info.append(basename)
        info.append('nfreq: '+ str(stats['nfreq']))
        info.append('nrepeat: '+ str(stats['nrepeat']))
        info.append('nsamples: '+ str(stats['nsamples']))
        info.append('nsources: '+ str(stats['nsources']))
        info.append('permutation mode: '+ mode)
        # info for each freq. band of the current data set
        info.append('# stats for each frequency band:')
        line_f    = 'freqs (Hz):'
        line_max  = 'pk max:    '
        line_mean = 'pk mean:   '
        line_min  = 'pk min:    '
        for i in range(stats['nfreq']):
            flow, fhigh = dctps['freqs'][i]
            line_f    += str('%5d-%d ' % (flow,fhigh))
            line_max  += str('%8.3f' % stats['pks_max'][i])
            line_mean += str('%8.3f' % stats['pks_mean'][i])
            line_min  += str('%8.3f' % stats['pks_min'][i])
        info.append(line_f)
        info.append(line_min)
        info.append(line_mean)
        info.append(line_max)
        # across freq. bands
        pks_min = stats['pks_min_global']
        pks_mean = stats['pks_mean_global']
        pks_std = stats['pks_std_global']
        pks_pct = stats['pks_pct99_global']
        pks_max = stats['pks_max_global']
        info.append('# stats across all frequency bands:')
        info.append('pk min:  '+ str('%8.3f' % pks_min))
        info.append('pk mean: '+ str('%8.3f' % stats['pks_mean_global']))
        info.append('pk std:  '+ str('%8.3f' % stats['pks_std_global']))
        info.append('pk pct99:'+ str('%8.3f' % stats['pks_pct99_global']))
        info.append('pk max:  '+ str('%8.3f' % stats['pks_max_global']))


        # combine global stats values of different ctps files into one global analysis
        if (ifile > 1):
            pks_all = np.concatenate((pks_all, pks.flatten()))
        else:
            pks_all = pks.flatten()
        ifile += 1

    if (ifile > 1):        
        info.append(sep)
        info.append('#')
        info.append('# stats across all files:')
        info.append('#')
        info.append('pk min:  '+ str('%8.3f' % pks_all.min()))
        info.append('pk mean: '+ str('%8.3f' % pks_all.mean()))
        info.append('pk std:  '+ str('%8.3f' % pks_all.std()))
        info.append('pk pct99:'+ str('%8.3f' % np.percentile(pks_all,99)))
        info.append('pk max:  '+ str('%8.3f' % pks_all.max()))

    info.append('#')
    duration = (time.time() - t_start)/ 60.0   # in minutes
    info.append('duration [min]: %0.2f' %duration)
    info.append('#')
    info.append(sep)
    
    # save surrogate stats
    if (save):
        np.savetxt(fnout, info, fmt='%s')

    return info


#######################################################
#
#  Select ICs from CTPS anaysis (for brain responses)
#
#######################################################
def apply_ctps_select_ic(fname_ctps, threshold=0.1):

    ''' Select ICs based on CTPS analysis. '''

    fnlist = get_files_from_list(fname_ctps)

    # loop across all filenames
    pl.ioff()  # switch off (interactive) plot visualisation
    ifile = 0
    for fnctps in fnlist:
        name = os.path.splitext(fnctps)[0]
        basename = os.path.splitext(os.path.basename(fnctps))[0]
        print '>>> working on: ' + basename
        # load CTPS data
        dctps = np.load(fnctps).item()
        freqs = dctps['freqs']
        nfreq = len(freqs)
        ncomp = dctps['ncomp']
        trig_name = dctps['trig_name']
        times = dctps['times']
        ic_sel = []
        # loop acros all freq. bands
        fig = pl.figure(ifile + 1, figsize=(16, 9), dpi=100)
        pl.clf()
        fig.subplots_adjust(left=0.08, right=0.95, bottom=0.05,
                            top=0.93, wspace=0.2, hspace=0.2)
        fig.suptitle(basename, fontweight='bold')
        nrow = np.ceil(float(nfreq) / 2)
        for ifreq in range(nfreq):
            pk = dctps['pk'][ifreq]
            pt = dctps['pt'][ifreq]
            pkmax = pk.max(1)
            ixmax = np.where(pkmax == pkmax.max())[0]
            ix = (np.where(pkmax >= threshold))[0]
            if np.any(ix):
                if (ifreq > 0):
                    ic_sel = np.append(ic_sel, ix + 1)
                else:
                    ic_sel = ix + 1

            # do construct names for title, fnout_fig, fnout_ctps
            frange = ' @' + str(freqs[ifreq][0]) + '-' + str(freqs[ifreq][1])
            x = np.arange(ncomp) + 1
            # do make bar plots for ctps thresh level plots
            ax = fig.add_subplot(nrow, 2, ifreq + 1)
            pl.bar(x, pkmax, color='steelblue')
            pl.bar(x[ix], pkmax[ix], color='red')
            pl.title(trig_name + frange, fontsize='small')
            pl.xlim([1, ncomp+1])
            pl.ylim([0, 0.5])
            pl.text(2, 0.45, 'ICs: ' + str(ix + 1))
        ic_sel = np.unique(ic_sel)
        nic = np.size(ic_sel)
        info = 'ICs (all): ' + str(ic_sel).strip('[]')
        fig.text(0.02, 0.01, info, transform=ax.transAxes)

        # save CTPS components found
        fntxt = name + '-ic_selection.txt'
        ic_sel = np.reshape(ic_sel, [1, nic])
        np.savetxt(fntxt, ic_sel, fmt='%i', delimiter=', ')
        ifile += 1

        # save figure
        fnfig = name + '-ic_selection.png'
        pl.savefig(fnfig, dpi=100)
    pl.ion()  # switch on (interactive) plot visualisation


#######################################################
#
#  apply ICA recomposition to select brain responses
#
#######################################################
def apply_ica_select_brain_response(fname_clean_raw, n_pca_components=None,
                                    conditions=['trigger'], include=None):

    ''' Performs ICA recomposition with selected brain response components to a list of (ICA) files.
        fname_clean_raw: raw data after ECG and EOG rejection.
        n_pca_commonents: ICA's recomposition parameter.
        conditions: the event kind to recompose the raw data, it can be 'trigger', 
                    'response' or include both conditions.
    '''

    fnlist = get_files_from_list(fname_clean_raw)

    # loop across all filenames
    for fn_clean in fnlist:
        #basename = fn_ctps_ics.rsplit('ctps')[0].rstrip(',')
        basename = fn_clean.split(ext_raw)[0]
        fnfilt = fn_clean
        fnarica = basename + ext_ica

        # load filtered and artefact removed data
        meg_raw = mne.io.Raw(fnfilt, preload=True)
        picks = mne.pick_types(meg_raw.info, meg=True, exclude='bads')
        # ICA decomposition
        ica = mne.preprocessing.read_ica(fnarica)

        # loop across different event IDs
        ctps_ics = []
        descrip_id = ''
        for event in conditions:
            fn_ics_eve = basename + prefix_ctps + event + '-ic_selection.txt'
            ctps_ics_eve = np.loadtxt(fn_ics_eve, dtype=int, delimiter=',')
            ctps_ics += (list(ctps_ics_eve - 1))
            descrip_id += ',' + event
        #To keep the index unique
        ctps_ics = list(set(ctps_ics))
        fnclean_eve = fn_ics_eve.split(',ctpsbr')[0] +\
            '%s,ctpsbr-raw.fif' % descrip_id

        # clean and save MEG data
        if n_pca_components:
            npca = n_pca_components
        else:
            npca = picks.size

        meg_clean = ica.apply(meg_raw, include=ctps_ics, n_pca_components=npca,
                              copy=True)
        if not meg_clean.info['description']:
            meg_clean.info['description'] = ''
            meg_clean.info['description'] += 'Raw recomposed from ctps selected\
                                              ICA components for brain\
                                              responses only.'
        meg_clean.save(fnclean_eve, overwrite=True)
        plot_compare_brain_responses(fname_clean_raw, fnclean_eve)


#######################################################
#
#  Plot and compare recomposed brain response data only.
#
#######################################################
def plot_compare_brain_responses(fname_orig, fname_new, event_id=1,
                                 tmin=-0.2, tmax=0.5, stim_name=None,
                                 proj=False, show=False):

    '''
    Function showing performance of signal with brain responses from
    selected components only. Plots the evoked (avg) signal of original
    data and brain responses only data along with difference between them.

    fname_orig, fname_new: str
    stim_ch: str (default STI 014)
    show: bool (default False)
    '''

    pl.ioff()
    if show:
        pl.ion()
    
    # Get the stimulus channel for special event from the fname_new
    #make a judgment, whether this raw data include more than one kind of event.
    #if True, use the first event as the start point of the epoches. 
    #Adjust the size of the time window based on different connditions
    basename = fname_new.split('-raw.fif')[0]

    # if stim_name is given we assume that the input data are raw and
    # cleaned data ('cleaned' means data were cardiac and ocular artifacts
    # were rejected)
    if stim_name:
        fnout_fig = basename + '-' + stim_name + '.png'
    else:
        stim_name = fname_new.rsplit(',ctpsbr')[0].rsplit('ar,')[1]
        # Construct file names.
        fnout_fig = basename + '.png'


    if ',' in stim_name:
        stim_ch = 'STI 014'
    elif stim_name == 'trigger':
        stim_ch = 'STI 014'
    elif stim_name == 'response':
        stim_ch = 'STI 013'

    
    # Read raw, calculate events, epochs, and evoked.
    raw_orig = mne.io.Raw(fname_orig, preload=True)
    raw_br = mne.io.Raw(fname_new, preload=True)

    events = mne.find_events(raw_orig, stim_channel=stim_ch, consecutive=True)
    events = mne.find_events(raw_br, stim_channel=stim_ch, consecutive=True)

    picks_orig = mne.pick_types(raw_orig.info, meg=True, exclude='bads')
    picks_br = mne.pick_types(raw_br.info, meg=True, exclude='bads')

    epochs_orig = mne.Epochs(raw_orig, events, event_id, proj=proj,
                             tmin=tmin, tmax=tmax, picks=picks_orig, preload=True)
    epochs_br = mne.Epochs(raw_br, events, event_id, proj=proj,
                           tmin=tmin, tmax=tmax, picks=picks_br, preload=True)

    evoked_orig = epochs_orig.average()
    evoked_br = epochs_br.average()

    times = evoked_orig.times * 1e3
    if np.max(evoked_orig.data) < 1:
        factor = 1e15
    else:
        factor = 1
    ymin = np.min(evoked_orig.data) * factor
    ymax = np.max(evoked_orig.data) * factor

    # Make the comparison plot.
    pl.figure('Compare raw data', figsize=(14, 5))
    pl.subplot(1, 2, 1)
    pl.plot(times, evoked_orig.data.T * factor, 'k', linewidth=0.5)
    pl.plot(times, evoked_br.data.T*factor, 'r', linewidth=0.5)
    pl.title('Signal before (black) and after (red) cleaning')
    pl.xlim(times[0], times[len(times) - 1])
    pl.ylim(1.1 * ymin, 1.1 * ymax)

    # print out some information
    textstr1 = 'Performance: %d\nFrequency Correlation: %d'\
               % (calc_performance(evoked_orig, evoked_br),
                  calc_frequency_correlation(evoked_orig, evoked_br))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    pl.text(times[10], 1.09 * ymax, textstr1, fontsize=10,
            verticalalignment='top', bbox=props)


    pl.subplot(1, 2, 2)
    evoked_diff = evoked_orig - evoked_br
    pl.plot(times, evoked_diff.data.T * factor, 'k', linewidth=0.5)
    pl.title('Difference signal')
    pl.xlim(times[0], times[len(times) - 1])
    pl.ylim(1.1 * ymin, 1.1 * ymax)

    pl.savefig(fnout_fig, format='png')
    pl.close('Compare raw data')
    pl.ion()


#######################################################
#                                                     #
# interface for creating the noise-covariance matrix  #
#                                                     #
#######################################################
def apply_create_noise_covariance(fname_empty_room, require_filter=True,
                                  verbose=None):

    '''
    Creates the noise covariance matrix from an empty room file.

    Parameters
    ----------
    fname_empty_room : String containing the filename
        of the empty room file (must be a fif-file)
    require_filter: bool
        If true, the empy room file is filtered before calculating
        the covariance matrix. (Beware, filter settings are fixed.)
    verbose : bool, str, int, or None
        If not None, override default verbose level
        (see mne.verbose).
        default: verbose=None
    '''

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    from mne import compute_raw_data_covariance as cp_covariance
    from mne import write_cov, pick_types
    from mne.io import Raw

    fner = get_files_from_list(fname_empty_room)

    nfiles = len(fner)

    # loop across all filenames
    for ifile in range(nfiles):
        fn_in = fner[ifile]
        print ">>> create noise covariance using file: "
        path_in, name = os.path.split(fn_in)
        print name

        if require_filter:
            print "Filtering with preset settings..."
            # filter empty room raw data
            apply_filter(fn_in, flow=1, fhigh=45, order=4, njobs=4)
            # reconstruct empty room file name accordingly
            fn_in = fn_in.split('-')[0] + ',bp1-45-empty.fif'

        # file name for saving noise_cov
        fn_out = fn_in[:fn_in.rfind(ext_empty_raw)] + ext_empty_cov

        # read in data
        raw_empty = Raw(fn_in, verbose=verbose)

        # pick MEG channels only
        picks = pick_types(raw_empty.info, meg=True, ref_meg=False, eeg=False,
                           stim=False, eog=False, exclude='bads')

        # calculate noise-covariance matrix
        noise_cov_mat = cp_covariance(raw_empty, picks=picks, verbose=verbose)

        # write noise-covariance matrix to disk
        write_cov(fn_out, noise_cov_mat)


#######################################################
#                                                     #
# small utility function to handle file lists         #
#                                                     #
#######################################################
def get_files_from_list(fin):
    ''' Return files as iterables lists '''
    if isinstance(fin, list):
        fout = fin
    else:
        if isinstance(fin, str):
            fout = list([fin])
        else:
            fout = list(fin)
    return fout
