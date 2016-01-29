# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.group_ica.py -------------------------------------
----------------------------------------------------------------------
 author     : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 27.11.2015
 version    : 1.0

----------------------------------------------------------------------
 Simple script to perform group ICA in source space
----------------------------------------------------------------------
"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# define some global file ending pattern
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
inv_pattern = "-src-meg-inv.fif"
img_src_group_ica = ",src_group_ICA"



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  perform FourierICA on group data in source space
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def group_fourierICA_src_space(fname_raw, average=False, stim_name=None,
                               stim_id=1, stim_delay=None, pca_dim=None,
                               ica_method='fourierica',
                               src_loc_method="dSPM", snr=3., nrep=1,
                               tmin=0, tmax=1.0, flow=4., fhigh=34.,
                               hamming_data=True, remove_outliers=True,
                               complex_mixing=True, max_iter=2000,
                               conv_eps=1e-9, lrate=0.2, cost_function='g2',
                               envelopeICA=False, verbose=True):


    """
    Module to perform group FourierICA.

        Parameters
        ----------
        fname_raw: filename(s) of the pre-processed raw file(s)
        average: Should data be averaged across subjects before
            FourierICA application? Note, averaged data require
            less memory!
            default: average=False
        stim_name: name of the stimulus channel. Note, for
            applying FourierCIA data are chopped around stimulus
            onset. If not set data are chopped in overlapping
            windows
            default: stim_names=None
        stim_id: Id of the event of interest to be considered in
            the stimulus channel. Only of interest if 'stim_name'
            is set
            default: event_id=1
        stim_delay: stimulus delay in milliseconds
            default: stim_delay=0
        pca_dim: the number of PCA components used to apply FourierICA.
            If pca_dim > 1 this refers to the exact number of components.
            If between 0 and 1 pca_dim refers to the variance which
            should be explained by the chosen components
            If pca_dim == None the minimum description length (MDL)
            (Wax and Kailath, 1985) criterion is used to estimation
            the number of components
            default: pca_dim=None
        ica_method: which ICA method should be used for the group ICA?
            You can chose between 'extended-infomax', 'fastica',
            'fourierica' and 'infomax'
            default: ica_method='fourierica'
        src_loc_method: method used for source localization.
            default: src_loc_method='dSPM'
        snr: signal-to-noise ratio for performing source
            localization
            default: snr=3.0
        nrep: number of repetitions ICA, i.e. ICASSO, should be performed
            default: nrep=1
        tmin: time of interest prior to stimulus onset.
            Important for generating epochs to apply FourierICA
            default = 0.0
        tmax: time of interest after stimulus onset.
            Important for generating epochs to apply FourierICA
            default = 1.0
        flow: ower frequency border for estimating the optimal
            de-mixing matrix using FourierICA
            default: flow=4.0
        fhigh: upper frequency border for estimating the optimal
            de-mixing matrix using FourierICA
            default: fhigh=34.0
            Note: here default flow and fhigh are choosen to
            contain:
                - theta (4-7Hz)
                - low (7.5-9.5Hz) and high alpha (10-12Hz),
                - low (13-23Hz) and high beta (24-34Hz)
        hamming_data: if set a hamming window is applied to each
            epoch prior to Fourier transformation
            default: hamming_data=True
        remove_outliers: If set outliers are removed from the Fourier
            transformed data.
            Outliers are defined as windows with large log-average power (LAP)

                 LAP_{c,t}=log \sum_{f}{|X_{c,tf}|^2

            where c, t and f are channels, window time-onsets and frequencies,
            respectively. The threshold is defined as |mean(LAP)+3 std(LAP)|.
            This process can be bypassed or replaced by specifying a function
            handle as an optional parameter.
            remove_outliers=True
        complex_mixing:
        max_iter:  maximum number od iterations used in FourierICA
            default: max_iter=2000
        conv_eps: teration stops when weight changes are smaller
            then this number
            default: conv_eps = 1e-9
        lrate: initial learning rate
            default: lrate=0.2
        cost_function: which cost-function should be used in the
            complex ICA algorithm
                'g1': g_1(y) = 1 / (2 * np.sqrt(lrate + y))
                'g2': g_2(y) = 1 / (lrate + y)
                'g3': g_3(y) = y
            default: cost_function = 'g2'
        envelopeICA: if set ICA is estimated on the envelope
            of the Fourier transformed input data, i.e., the
            mixing model is |x|=As
            default: envelopeICA=False
        verbose: bool, str, int, or None
            If not None, override default verbose level
            (see mne.verbose).
            default: verbose=True
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from jumeg.decompose.icasso import JuMEG_icasso
    from mne import set_log_level
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
                              envelopeICA=envelopeICA,
                              ica_method=ica_method,
                              tICA=False,
                              snr=snr, lrate=lrate)

    W_orig, A_orig, quality, fourier_ica_obj \
        = icasso_obj.fit(fn_list, average=average,
                         stim_name=stim_name,
                         event_id=stim_id, pca_dim=pca_dim,
                         stim_delay=stim_delay,
                         tmin_win=tmin, tmax_win=tmax,
                         flow=flow, fhigh=fhigh,
                         max_iter=max_iter, conv_eps=conv_eps,
                         complex_mixing=complex_mixing,
                         hamming_data=hamming_data,
                         remove_outliers=remove_outliers,
                         cost_function=cost_function,
                         verbose=verbose)


    # ------------------------------------------
    # save results to disk
    # ------------------------------------------
    groupICA_obj = {'fn_list': fn_list,
                    'W_orig': W_orig,
                    'A_orig': A_orig,
                    'quality': quality,
                    'icasso_obj': icasso_obj,
                    'fourier_ica_obj': fourier_ica_obj}


    if stim_id:
        fn_base = "group_ICA_%02ddB.obj" % (stim_id)
    else:
        fn_base = "group_ICA_resting_state.obj"

    fnout = join(dirname(dirname(fn_list[0])), fn_base)
    with open(fnout, "wb") as filehandler:
        dump(groupICA_obj, filehandler)

    # return dictionary
    return groupICA_obj, fnout




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  get time courses of FourierICA components
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_group_fourierICA_time_courses(groupICA_obj, event_id=None,
                                      stim_delay=0):


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
        stim_delay: stimulus delay in milliseconds
            default: stim_delay=0
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

        stim_id = event_id[istim]


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

            _, src_loc, vert, _, _, sfreq, _ = \
                icasso_obj.prepare_data_for_fit(fn_raw, stim_name=icasso_obj.stim_name,
                                                tmin_stim=tmin, tmax_stim=tmax,
                                                flow=icasso_obj.flow,
                                                fhigh=icasso_obj.fhigh,
                                                event_id=stim_id,
                                                stim_delay=stim_delay,
                                                hamming_data=hamming_data,
                                                fn_inv=icasso_obj.fn_inv[idx],
                                                averaged_epochs=average_epochs,
                                                remove_outliers=remove_outliers)

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

            startfftind = int(np.floor(icasso_obj.flow * win_length_sec))
            fft_act = np.zeros((ncomp, win_ntsl), dtype=np.complex)


            # -------------------------------------------
            # define result arrays
            # -------------------------------------------
            if idx == 0:
                # we have to double the number of components as we separate the
                # results for left and right hemisphere
                act = np.zeros((ncomp, nfn_list*nwindows, fftsize), dtype=np.complex)
                temporal_envelope = np.zeros((nfn_list*nwindows, ncomp, win_ntsl))

            idx_start = idx * nwindows


            # -------------------------------------------
            # loop over all epochs
            # -------------------------------------------
            for iepoch in range(nwindows):

                # get independent components
                src_loc_zero_mean = (src_loc[:, iepoch, :] - np.dot(np.ones((fftsize, 1)), dmean)) / \
                                    np.dot(np.ones((fftsize, 1)), dstd)

                # activations in both hemispheres
                act[:, idx_start+iepoch, :] = np.dot(groupICA_obj['W_orig'], src_loc_zero_mean.transpose())

                # generate temporal profiles:
                # apply inverse STFT to get temporal envelope
                if fourier_ica_obj:
                    fft_act[:, startfftind:(startfftind+fftsize)] = act[:, iepoch, :]
                    temporal_envelope[idx_start+iepoch, :, :] = fftpack.ifft(fft_act, n=win_ntsl, axis=1).real
                else:
                    temporal_envelope[idx_start+iepoch, :, :] = act.transpose([1, 0, 2])


        # -------------------------------------------
        # collecting time courses of interest
        # -------------------------------------------
        temporal_envelope_all[istim].append(temporal_envelope.real)



    return temporal_envelope_all, src_loc, vert, sfreq





# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  plot FourierICA results
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_group_fourierICA(fn_groupICA_obj, stim_name=None,
                          stim_id=1, stim_delay=0,
                          subjects_dir=None):

    """
    Interface to plot the results from group FourierICA

        Parameters
        ----------
        fn_groupICA_obj: filename of the group ICA object
        stim_name: name of the stimulus channel. Note, for
            applying FourierCIA data are chopped around stimulus
            onset. If not set data are chopped in overlapping
            windows
            default: stim_names=None
        stim_id: Id of the event of interest to be considered in
            the stimulus channel. Only of interest if 'stim_name'
            is set
            default: event_id=1
        stim_delay: stimulus delay in milliseconds
            default: stim_delay=0
        subjects_dir: If the subjects directory is not confirm with
            the system variable 'SUBJECTS_DIR' parameter should be set
            default: subjects_dir=None
    """


    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from jumeg.decompose.fourier_ica_plot import plot_results_src_space
    from mne import set_log_level
    from pickle import load

    # set log level to 'WARNING'
    set_log_level('WARNING')


    # ------------------------------------------
    # read in group FourierICA object
    # ------------------------------------------
    with open(fn_groupICA_obj, "rb") as filehandler:
        groupICA_obj = load(filehandler)

    icasso_obj = groupICA_obj['icasso_obj']
    win_length_sec = icasso_obj.tmax_win - icasso_obj.tmin_win


    # ------------------------------------------
    # get temporal envelope
    # ------------------------------------------
    temporal_envelope, src_loc, vert, sfreq = \
        get_group_fourierICA_time_courses(groupICA_obj)


    # ------------------------------------------
    # plot "group" results
    # ------------------------------------------
    if groupICA_obj.has_key('classification'):
        classification = groupICA_obj['classification']
    else:
        classification = []


    fnout_src_fourier_ica = fn_groupICA_obj[:fn_groupICA_obj.rfind('.obj')] + \
                            img_src_group_ica

    plot_results_src_space(groupICA_obj['fourier_ica_obj'],
                           groupICA_obj['W_orig'], groupICA_obj['A_orig'],
                           src_loc, vert, subjects_dir=subjects_dir,
                           tpre=icasso_obj.tmin_win,
                           win_length_sec=win_length_sec, sfreq=sfreq,
                           flow=icasso_obj.flow, fhigh=icasso_obj.fhigh,
                           fnout=fnout_src_fourier_ica,
                           tICA=icasso_obj.tICA,
                           morph2fsaverage=icasso_obj.morph2fsaverage,
                           temporal_envelope=temporal_envelope,
                           classification=classification,
                           show=False)




