# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.group_ica.py -------------------------------------
----------------------------------------------------------------------
 author     : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 09.11.2016
 version    : 1.0

----------------------------------------------------------------------
 Simple script to perform group ICA in source space
----------------------------------------------------------------------
"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# define some global file ending pattern
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

img_src_group_ica = ",src_group_ICA"



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  perform FourierICA on group data in source space
#  Note: here the parameters are optimized for resting
#  state data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def group_fourierICA_src_space_resting_state(fname_raw,
                               ica_method='fourierica',              # parameter for ICA method
                               nrep=50,                              # parameter for ICASSO
                               src_loc_method='dSPM', snr=1.0,       # parameter for inverse estimation
                               inv_pattern='-src-meg-fspace-inv.fif',
                               stim_name=None, stim_id=None,         # parameter for epoch generation
                               corr_event_picking=None,
                               stim_delay=0.0,
                               tmin=0.0, tmax=1.0,
                               average=False,
                               flow=4., fhigh=34.,                   # parameter for Fourier transformation
                               remove_outliers=True,
                               hamming_data=True,
                               dim_reduction='MDL',
                               pca_dim=None, cost_function='g2',     # parameter for complex ICA estimation
                               lrate=0.2, complex_mixing=True,
                               conv_eps=1e-9, max_iter=5000,
                               envelopeICA=False,
                               interpolate_bads=True,
                               decim_epochs=None,
                               fnout=None,                           # parameter for saving the results
                               verbose=True):

    """
    Module to perform group FourierICA on resting-state data (if wished
    in combination with ICASSO --> if 'nrep'=1 only FourierICA is performed,
    if 'nrep'>1 FourierICA is performed in combination with ICASSO).

    For information about the parameters see
    jumeg.decompose.group_ica.group_fourierICA_src_space()
    """

    # call routine for group FourierICA
    group_fourierICA_src_space(fname_raw, ica_method=ica_method,
                               nrep=nrep, src_loc_method=src_loc_method,
                               snr=snr, inv_pattern=inv_pattern,
                               stim_name=stim_name, stim_id=stim_id,
                               corr_event_picking=corr_event_picking,
                               stim_delay=stim_delay, tmin=tmin, tmax=tmax,
                               average=average, flow=flow, fhigh=fhigh,
                               remove_outliers=remove_outliers,
                               hamming_data=hamming_data,
                               dim_reduction=dim_reduction, pca_dim=pca_dim,
                               cost_function=cost_function, lrate=lrate,
                               complex_mixing=complex_mixing, conv_eps=conv_eps,
                               max_iter=max_iter, envelopeICA=envelopeICA,
                               interpolate_bads=interpolate_bads,
                               decim_epochs=decim_epochs, fnout=fnout,
                               verbose=verbose)



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  perform FourierICA on group data in source space
#  Note: Here the parameters are optimized for evoked
#  data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def group_fourierICA_src_space(fname_raw,
                               ica_method='fourierica',              # parameter for ICA method
                               nrep=50,                              # parameter for ICASSO
                               src_loc_method='dSPM', snr=1.0,       # parameter for inverse estimation
                               inv_pattern='-src-meg-fspace-inv.fif',
                               stim_name='STI 014', stim_id=1,       # parameter for epoch generation
                               corr_event_picking=None,
                               stim_delay=0.0,
                               tmin=-0.2, tmax=0.8,
                               average=False,
                               flow=4., fhigh=34.,                   # parameter for Fourier transformation
                               remove_outliers=False,
                               hamming_data=False,
                               dim_reduction='MDL',
                               pca_dim=None, cost_function='g2',     # parameter for complex ICA estimation
                               lrate=0.2, complex_mixing=True,
                               conv_eps=1e-9, max_iter=5000,
                               envelopeICA=False,
                               interpolate_bads=True,
                               decim_epochs=None,
                               fnout=None,                           # parameter for saving the results
                               verbose=True):


    """
    Module to perform group FourierICA (if wished in combination with
    ICASSO --> if 'nrep'=1 only FourierICA is performed, if 'nrep'>1
    FourierICA is performed in combination with ICASSO).

        Parameters
        ----------
        fname_raw: list of strings
            filename(s) of the pre-processed raw file(s)
        ica_method: string
            which ICA method should be used for the group ICA?
            You can chose between 'extended-infomax', 'fastica',
            'fourierica' and 'infomax'
            default: ica_method='fourierica'
        nrep: integer
            number of repetitions ICA, i.e. ICASSO, should be performed
            default: nrep=50
        src_loc_method: string
            method used for source localization.
            default: src_loc_method='dSPM'
        snr: float
            signal-to-noise ratio for performing source
            localization --> for single epochs an snr of 1.0 is recommended,
            if keyword 'average' is set one should use snr=3.0
            default: snr=1.0
        inv_pattern: string
            String containing the ending pattern of the inverse
            solution. Note, here fspace is used if the inverse
            solution is estimated in the Fourier space for later
            applying Fourier (i.e., complex) ICA
            default: inv_pattern='-src-meg-fspace-inv.fif'
        stim_name: string
            name of the stimulus channel. Note, for
            applying FourierCIA data are chopped around stimulus
            onset. If not set data are chopped in overlapping
            windows
            default: stim_names='STI 014'
        stim_id: integer or list of integers
            list containing the event IDs
            default: stim_id=1
        corr_event_picking: string
            if set should contain the complete python path and
            name of the function used to identify only the correct events
            default: corr_event_picking=None
        stim_delay: float
            Stimulus delay in seconds
            default: stim_delay=0.0
        tmin: float
            time of interest prior to stimulus onset for epoch
            generation (in seconds)
            default: tmin=-0.2
        tmax: float
            time of interest after the stimulus onset for epoch
            generation (in seconds)
            default: tmax=0.8
        average: bool
            should data be averaged across subjects before
            FourierICA application? Note, averaged data require
            less memory!
            default: average=False
        flow: float
            lower frequency border for estimating the optimal
            de-mixing matrix using FourierICA
            default: flow=4.0
        fhigh: float
            upper frequency border for estimating the optimal
            de-mixing matrix using FourierICA
            default: fhigh=34.0
            Note: here default flow and fhigh are choosen to
            contain:
                - theta (4-7Hz)
                - low (7.5-9.5Hz) and high alpha (10-12Hz),
                - low (13-23Hz) and high beta (24-34Hz)
        remove_outliers: If set outliers are removed from the Fourier
            transformed data.
            Outliers are defined as windows with large log-average power (LAP)

                 LAP_{c,t}=log \sum_{f}{|X_{c,tf}|^2

            where c, t and f are channels, window time-onsets and frequencies,
            respectively. The threshold is defined as |mean(LAP)+3 std(LAP)|.
            This process can be bypassed or replaced by specifying a function
            handle as an optional parameter.
            remove_outliers=False
        hamming_data: boolean
            if set a hamming window is applied to each
            epoch prior to Fourier transformation
            default: hamming_data=False
        dim_reduction: string {'', 'AIC', 'BIC', 'GAP', 'MDL', 'MIBS', 'explVar'}
            Method for dimension selection. For further information about
            the methods please check the script 'dimension_selection.py'.
            default: dim_reduction='MDL'
        pca_dim: Integer
            The number of components used for PCA decomposition.
            default: pca_dim=None
        cost_function: string
            which cost-function should be used in the complex
            ICA algorithm
            'g1': g_1(y) = 1 / (2 * np.sqrt(lrate + y))
            'g2': g_2(y) = 1 / (lrate + y)
            'g3': g_3(y) = y
            default: cost_function='g2'
        lrate: float
            learning rate which should be used in the applied
            ICA algorithm
            default: lrate=0.3
        complex_mixing: bool
            if mixing matrix should be real or complex
            default: complex_mixing=True
        conv_eps: float
            iteration stop when weight changes are smaller
            then this number
            default: conv_eps = 1e-9
        max_iter: integer
            maximum number of iterations used in FourierICA
            default: max_iter=5000
        envelopeICA: if set ICA is estimated on the envelope
            of the Fourier transformed input data, i.e., the
            mixing model is |x|=As
            default: envelopeICA=False
        interpolate_bads: bool
            if set bad channels are interpolated (using the
            mne routine raw.interpolate_bads()).
            default: interpolate_bads=True
        decim_epochs: integer
            if set the number of epochs will be reduced (per
            subject) to that number for the estimation of the demixing matrix.
            Note: the epochs were chosen randomly from the complete set of
            epochs.
            default: decim_epochs=None
        fnout: string
            output filename of the result structure. If not set the filename
            is generated automatically.
            default: fnout=None
        verbose: bool, str, int, or None
            If not None, override default verbose level
            (see mne.verbose).
            default: verbose=True


        Return
        ------
        groupICA_obj: dictionary
            Group ICA information stored in a dictionary. The dictionary
            has following keys:
            'fn_list': List of filenames which where used to estimate the
                group ICA
            'W_orig': estimated de-mixing matrix
            'A_orig': estimated mixing matrix
            'quality': quality index of the clustering between
                components belonging to one cluster
                (between 0 and 1; 1 refers to small clusters,
                i.e., components in one cluster have a highly similar)
            'icasso_obj': ICASSO object. For further information
                please have a look into the ICASSO routine
            'fourier_ica_obj': FourierICA object. For further information
                please have a look into the FourierICA routine
        fnout: string
            filename where the 'groupICA_obj' is stored
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from jumeg.decompose.icasso import JuMEG_icasso
    from mne import set_log_level
    import numpy as np
    from os.path import dirname, join
    from pickle import dump

    # set log level to 'WARNING'
    set_log_level('WARNING')


    # ------------------------------------------
    # check input parameter
    # ------------------------------------------
    # filenames
    if isinstance(fname_raw, list):
        fn_list = fname_raw
    else:
        fn_list = [fname_raw]


    # -------------------------------------------
    # set some path parameter
    # -------------------------------------------
    fn_inv = []
    for fn_raw in fn_list:
        fn_inv.append(fn_raw[:fn_raw.rfind('-raw.fif')] + inv_pattern)


    # ------------------------------------------
    # apply FourierICA combined with ICASSO
    # ------------------------------------------
    icasso_obj = JuMEG_icasso(nrep=nrep, fn_inv=fn_inv,
                              src_loc_method=src_loc_method,
                              morph2fsaverage=True,
                              ica_method=ica_method,
                              cost_function=cost_function,
                              dim_reduction=dim_reduction,
                              decim_epochs=decim_epochs,
                              tICA=False, snr=snr, lrate=lrate)

    W_orig, A_orig, quality, fourier_ica_obj \
        = icasso_obj.fit(fn_list, average=average,
                         stim_name=stim_name,
                         event_id=stim_id, pca_dim=pca_dim,
                         stim_delay=stim_delay,
                         tmin_win=tmin, tmax_win=tmax,
                         flow=flow, fhigh=fhigh,
                         max_iter=max_iter, conv_eps=conv_eps,
                         complex_mixing=complex_mixing,
                         envelopeICA=envelopeICA,
                         hamming_data=hamming_data,
                         remove_outliers=remove_outliers,
                         cost_function=cost_function,
                         interpolate_bads=interpolate_bads,
                         corr_event_picking=corr_event_picking,
                         verbose=verbose)


    # ------------------------------------------
    # save results to disk
    # ------------------------------------------
    # generate dictionary to save results
    groupICA_obj = {'fn_list': fn_list,
                    'W_orig': W_orig,
                    'A_orig': A_orig,
                    'quality': quality,
                    'icasso_obj': icasso_obj,
                    'fourier_ica_obj': fourier_ica_obj}


    # check if the output filename is already set
    if not fnout:
        # generate filename for output structure
        if isinstance(stim_id, (list, tuple)):
            fn_base = "group_FourierICA_combined"
            for id in np.sort(stim_id)[::-1]:
                fn_base += "_%ddB" % id
        elif isinstance(stim_id, int):
            fn_base = "group_FourierICA_%ddB" % stim_id
        else:
            fn_base = "group_ICA_resting_state.obj"

        # write file to disk
        fnout = join(dirname(dirname(fname_raw[0])), fn_base + ".obj")

    with open(fnout, "wb") as filehandler:
        dump(groupICA_obj, filehandler)

    # return dictionary
    return groupICA_obj, fnout




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  get time courses of FourierICA components
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_group_fourierICA_time_courses(groupICA_obj, event_id=None,
                                      resp_id=None, stim_delay=0,
                                      unfiltered=True,
                                      corr_event_picking=None,
                                      baseline=(None, None)):


    """
    Module to get time courses from the FourierICA components.
    Note, to save memory time courses are not saved during group
    ICA estimation. However, to estimate the time courses will
    take a while.

        Parameters
        ----------
        groupICA_obj: either filename of the group ICA object or an
            already swiped groupICA object
        event_id: Id of the event of interest to be considered in
            the stimulus channel.
            default: event_id=None
        resp_id: Response IDs for correct event estimation. Note:
            Must be in the order corresponding to the 'event_id'
            default: resp_id=None
        stim_delay: stimulus delay in milliseconds
            default: stim_delay=0
        unfiltered: if set data are not filtered prior to time-course
            generation
            default: unfiltered=True
        corr_event_picking: if set should contain the complete python
            path and name of the function used to identify only the
            correct events
            default: corr_event_picking=None
        baseline: If set baseline correction is applied to epochs
            prior to Fourier transformation
            default: baseline=(None, None)


        Return
        ------
        temporal_envelope_all: list of arrays containing
            the temporal envelopes.
        src_loc:  array
            3D array containing the source localization
            data used for FourierICA estimation
            (nfreq x nepochs x nvoxel)
        vert: list
            list containing two arrays with the order
            of the vertices.
        sfreq: float
            sampling frequency of the data
    """


    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    import numpy as np
    from mne.externals.six import string_types
    from scipy import fftpack

    # ------------------------------------------
    # test if 'groupICA_obj' is a string or
    # already the swiped object
    # ------------------------------------------
    if isinstance(groupICA_obj, string_types):
        from pickle import load
        with open(groupICA_obj, "rb") as filehandler:
            groupICA_obj = load(filehandler)


    icasso_obj = groupICA_obj['icasso_obj']
    fourier_ica_obj = groupICA_obj['fourier_ica_obj']
    fn_list = groupICA_obj['fn_list']

    if not isinstance(fn_list, list):
        fn_list = [fn_list]

    nfn_list = len(fn_list)


    # ------------------------------------------
    # check if FourierICA or temporal ICA was
    # performed
    # ------------------------------------------
    if fourier_ica_obj:
        average_epochs = False
        hamming_data = fourier_ica_obj.hamming_data
        remove_outliers = fourier_ica_obj.remove_outliers
    else:
        average_epochs = True
        hamming_data = False
        remove_outliers = False



    # check if we have more than one stimulus ID
    if not event_id:
        event_id = icasso_obj.event_id

    if not isinstance(event_id, (list, tuple)):
        event_id = [event_id]

    nstim = len(event_id)
    temporal_envelope_all = np.empty((nstim, 0)).tolist()


    # ------------------------------------------
    # loop over all stimulus IDs to get time
    # courses
    # ------------------------------------------
    for istim in range(nstim):

        # get current stimulus and response ID
        stim_id = event_id[istim]
        id_response = resp_id[istim]


        # ------------------------------------------
        # loop over all files
        # ------------------------------------------
        for idx in range(nfn_list):

            # ------------------------------------------
            # transform data to source space
            # ------------------------------------------
            # get some parameter
            fn_raw = fn_list[idx]
            tmin, tmax = icasso_obj.tmin_win, icasso_obj.tmax_win
            win_length_sec = (tmax - tmin)
            flow, fhigh = icasso_obj.flow, icasso_obj.fhigh

            _, src_loc, vert, _, _, sfreq, _ = \
                icasso_obj.prepare_data_for_fit(fn_raw, stim_name=icasso_obj.stim_name,
                                                tmin_stim=tmin, tmax_stim=tmax,
                                                flow=flow, fhigh=fhigh,
                                                event_id=[stim_id],
                                                resp_id=[id_response],
                                                stim_delay=stim_delay,
                                                hamming_data=hamming_data,
                                                corr_event_picking=corr_event_picking,
                                                fn_inv=icasso_obj.fn_inv[idx],
                                                averaged_epochs=average_epochs,
                                                baseline=baseline,
                                                remove_outliers=remove_outliers,
                                                unfiltered=unfiltered)

            # normalize source data
            fftsize, nwindows, nvoxel = src_loc.shape
            nrows_Xmat_c = fftsize*nwindows
            Xmat_c = src_loc.reshape((nrows_Xmat_c, nvoxel), order='F')
            dmean = np.mean(Xmat_c, axis=0).reshape((1, nvoxel))
            dstd = np.std(Xmat_c, axis=0).reshape((1, nvoxel))



            # -------------------------------------------
            # get some parameter
            # -------------------------------------------
            ncomp, nvoxel = groupICA_obj['W_orig'].shape
            if fourier_ica_obj:
                win_ntsl = int(np.floor(sfreq * win_length_sec))
            else:
                win_ntsl = fftsize

            startfftind = int(np.floor(flow * win_length_sec))
            fft_act = np.zeros((ncomp, win_ntsl), dtype=np.complex)


            # -------------------------------------------
            # define result arrays
            # -------------------------------------------
            if idx == 0:
                # we have to double the number of components as we separate the
                # results for left and right hemisphere
                # act = []
                temporal_envelope = []

            # act_cur = np.zeros((ncomp, nwindows, fftsize), dtype=np.complex)
            temporal_envelope_cur = np.zeros((nwindows, ncomp, win_ntsl))
            times = np.arange(win_ntsl)/sfreq + tmin


            # -------------------------------------------
            # loop over all epochs
            # -------------------------------------------
            for iepoch in range(nwindows):

                # get independent components
                src_loc_zero_mean = (src_loc[:, iepoch, :] - np.dot(np.ones((fftsize, 1)), dmean)) / \
                                    np.dot(np.ones((fftsize, 1)), dstd)

                # activations in both hemispheres
                act = np.dot(groupICA_obj['W_orig'], src_loc_zero_mean.transpose())

                # generate temporal profiles:
                # apply inverse STFT to get temporal envelope
                if fourier_ica_obj:
                    fft_act[:, startfftind:(startfftind+fftsize)] = act
                    temporal_envelope_cur[iepoch, :, :] = fftpack.ifft(fft_act, n=win_ntsl, axis=1).real
                else:
                    temporal_envelope_cur[iepoch, :, :] = act.transpose([1, 0, 2])


            # store data in list
            temporal_envelope.append(temporal_envelope_cur)
            # act.append(act_cur)


        # concatenate result data
        temporal_envelope = np.asarray(np.concatenate(temporal_envelope))
        # act = np.concatenate(act, axis=1)


        # -------------------------------------------
        # collecting time courses of interest
        # -------------------------------------------
        temporal_envelope_all[istim].append(temporal_envelope.real)

    return temporal_envelope_all, src_loc, vert, sfreq





# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  plot FourierICA results
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_group_fourierICA(fn_groupICA_obj,
                          stim_id=1, stim_delay=0,
                          resp_id=None,
                          corr_event_picking=None,
                          global_scaling=True,
                          subjects_dir=None,
                          bar_plot=False):

    """
    Interface to plot the results from group FourierICA

        Parameters
        ----------
        fn_groupICA_obj: filename of the group ICA object
        stim_id: Id of the event of interest to be considered in
            the stimulus channel. Only of interest if 'stim_name'
            is set
            default: event_id=1
        stim_delay: stimulus delay in milliseconds
            default: stim_delay=0
        resp_id: Response IDs for correct event estimation. Note:
            Must be in the order corresponding to the 'event_id'
            default: resp_id=None
        corr_event_picking: string
            if set should contain the complete python path and
            name of the function used to identify only the correct events
            default: corr_event_picking=None
        subjects_dir: string
            If the subjects directory is not confirm with
            the system variable 'SUBJECTS_DIR' parameter should be set
            default: subjects_dir=None
        bar_plot: boolean
            If set the results of the time-frequency analysis
            are shown as bar plot. This option is recommended
            when FourierICA was applied to resting-state data
            default: bar_plot=False
    """


    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from jumeg.decompose.fourier_ica_plot import plot_results_src_space
    from mne import set_log_level
    from os.path import exists
    from pickle import dump, load

    # set log level to 'WARNING'
    set_log_level('WARNING')


    # ------------------------------------------
    # read in group FourierICA object
    # ------------------------------------------
    with open(fn_groupICA_obj, "rb") as filehandler:
        groupICA_obj = load(filehandler)

    icasso_obj = groupICA_obj['icasso_obj']
    win_length_sec = icasso_obj.tmax_win - icasso_obj.tmin_win
    temp_profile_names = ["Event-ID %i" % i for i in groupICA_obj['icasso_obj'].event_id]

    # ------------------------------------------
    # check if time-courses already exist
    # ------------------------------------------
    fn_temporal_envelope = fn_groupICA_obj[:-4] + '_temporal_envelope.obj'
    # generate time courses if they do not exist
    if not exists(fn_temporal_envelope):
        # import necessary modules
        from jumeg.decompose.group_ica import get_group_fourierICA_time_courses

        # generate time courses
        temporal_envelope, src_loc, vert, sfreq = \
            get_group_fourierICA_time_courses(groupICA_obj, event_id=stim_id,
                                              stim_delay=stim_delay, resp_id=resp_id,
                                              corr_event_picking=corr_event_picking,
                                              unfiltered=False, baseline=(None, 0))

        # save data
        temp_env_obj = {'temporal_envelope': temporal_envelope,
                        'src_loc': src_loc, 'vert': vert, 'sfreq': sfreq}
        with open(fn_temporal_envelope, "wb") as filehandler:
            dump(temp_env_obj, filehandler)

    # when data are stored read them in
    else:
        # read data in
        with open(fn_temporal_envelope, "rb") as filehandler:
            temp_env_obj = load(filehandler)

        # get data
        temporal_envelope = temp_env_obj['temporal_envelope']
        src_loc = temp_env_obj['src_loc']
        vert = temp_env_obj['vert']


    # ------------------------------------------
    # check if classification already exists
    # ------------------------------------------
    if 'classification' in groupICA_obj and\
            'mni_coords' in groupICA_obj and\
            'labels' in groupICA_obj:
        classification = groupICA_obj['classification']
        mni_coords = groupICA_obj['mni_coords']
        labels = groupICA_obj['labels']
    else:
        classification = {}
        mni_coords = []
        labels = None


    # ------------------------------------------
    # plot "group" results
    # ------------------------------------------
    fnout_src_fourier_ica = fn_groupICA_obj[:fn_groupICA_obj.rfind('.obj')] + \
                            img_src_group_ica

    mni_coords, classification, labels =\
        plot_results_src_space(groupICA_obj['fourier_ica_obj'],
                               groupICA_obj['W_orig'], groupICA_obj['A_orig'],
                               src_loc_data=src_loc, vertno=vert,
                               subjects_dir=subjects_dir,
                               tpre=icasso_obj.tmin_win,
                               win_length_sec=win_length_sec,
                               flow=icasso_obj.flow, fhigh=icasso_obj.fhigh,
                               fnout=fnout_src_fourier_ica,
                               tICA=icasso_obj.tICA,
                               global_scaling=global_scaling,
                               temporal_envelope=temporal_envelope,
                               temp_profile_names=temp_profile_names,
                               classification=classification,
                               mni_coords=mni_coords, labels=labels,
                               bar_plot=bar_plot)


    # ------------------------------------------
    # adjust groupICA_obj with the new
    # parameters if they didn't exist before
    # ------------------------------------------
    if 'classification' not in groupICA_obj and\
            'mni_coords' not in groupICA_obj and\
            'labels' not in groupICA_obj:
        groupICA_obj['classification'] = classification
        groupICA_obj['mni_coords'] = mni_coords
        groupICA_obj['labels'] = labels

        # write new groupICA_obj back to disk
        with open(fn_groupICA_obj, "wb") as filehandler:
            dump(groupICA_obj, filehandler)


