# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.fourier_ica --------------------------------------
----------------------------------------------------------------------
 autor      : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 13.10.2016
 version    : 1.1

----------------------------------------------------------------------
 This simple implementation of FourierICA is part of the supplementary
 material of the publication:
----------------------------------------------------------------------

 A. Hyvaerinen, P. Ramkumar, L. Pakkonen, and R. Hari, 'Independent
 component analysis of short-time Fourier transforms for spontaneous
 EEG/MEG analysis', NeuroImage, 49(1):257-271, 2010.

 Should you use this code, we kindly request you to cite the
 aforementioned publication.

 <http://www.cs.helsinki.fi/group/neuroinf/code/fourierica/
fourierica.m DOWNLOAD FourierICA from here>

----------------------------------------------------------------------
 Overview
----------------------------------------------------------------------

 FourierICA is an unsupervised learning method suitable for the
 analysis of rhythmic activity in EEG/MEG recordings. The method
 performs independent component analysis (ICA) on short-time
 Fourier transforms of the data. As a result, more "interesting"
 sources with (amplitude modulated) oscillatory behaviour are
 uncovered and appropriately ranked. The method is also capable to
 reveal spatially-distributed sources appearing with different phases
 in different EEG/MEG channels by means of a complex-valued mixing
 matrix. For more details, please read the publication

 A. Hyvaerinen, P. Ramkumar, L. Pakkonen, and R. Hari, 'Independent
 component analysis of short-time Fourier transforms for spontaneous
 EEG/MEG analysis', NeuroImage, 49(1):257-271, 2010.

 or go to the webpage <http://www.cs.helsinki.fi/u/ahyvarin/>

----------------------------------------------------------------------
 How to use fourier_ica
----------------------------------------------------------------------

from jumeg.decompose import fourier_ica

ocarta_obj = fourier_ica.JuMEG_fourier_ica()
fourier_ica_obj.fit(meg_data)

--> for further comments we refer directly to the functions or to
    fourier_ica_test.py

----------------------------------------------------------------------
"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#              import necessary modules
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import scipy as sc


# #######################################################
# #                                                     #
# #              some general functions                 #
# #                                                     #
# #######################################################
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# estimate inverse kernel
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def calc_inv_kernel(fn_inv, method="dSPM", nave=1, snr=1.,
                    pick_ori="normal", verbose=None):

    """
    Interface for preparing the kernel of the inverse
    estimation.

        Parameters
        ----------
        fn_inv : String containing the filename
            of the inverse operator (must be a fif-file)
        method : string
            which source localization method should be used?
            MNE-based: "MNE" | "dSPM" | "sLORETA"
            default: method="dSPM"
        nave : number of averages used to regularize the solution
            default: nave=1
        snr : signal-to-noise ratio for source localization
            default: snr = 1.
        pick_ori : orientation of the sources for source
            localization
            default: pick_ori = "normal"
        verbose : bool, str, int, or None
            If not None, override default verbose level
            (see mne.verbose).
            default: verbose=None

        Returns
        -------
        kernel: kernel for source localization
        noise_norm: noise normalization matrix for source
            localization. Important when 'dSPM' or 'sLORETA'
            is used for source localization
        vertno: array of vertex assignment for source
            localization. For further information see mne
            python manual
    """

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    import mne.minimum_norm as min_norm
    from mne.minimum_norm.inverse import _assemble_kernel
    import numpy as np

    # -------------------------------------------
    # estimate inverse kernel
    # -------------------------------------------
    # load inverse solution
    import mne.minimum_norm as min_norm
    inv_operator = min_norm.read_inverse_operator(fn_inv, verbose=verbose)


    # set up the inverse according to the parameters
    lambda2      = 1. / snr ** 2.   # the regularization parameter.
    inv_operator = min_norm.prepare_inverse_operator(inv_operator, nave, lambda2, method)

    # estimate inverse kernel and noise normalization coefficient
    kernel, noise_norm, vertno = _assemble_kernel(inv_operator, None, method, pick_ori)

    if method == "MNE":
        noise_norm = np.ones((kernel.shape[0]/3))
        noise_norm = noise_norm[:, np.newaxis]


    # -------------------------------------------
    # return results
    # -------------------------------------------
    return kernel, noise_norm, vertno



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# estimate source localization for STFT transformed data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def stft_source_localization(data, fn_inv, method="dSPM",
                             morph2fsaverage=False,
                             nave=1, snr=1.,
                             pick_ori="normal", verbose=True):

    """
    Apply inverse operator to data. In general, the
    L2-norm inverse solution is computed.

        Parameters
        ----------
        data: array of MEG data
            (only the data, no MNE-data object!)
        fn_inv: String containing the filename
            of the inverse operator (must be a fif-file)
        method: string
            which source localization method should be used?
            MNE-based: "MNE" | "dSPM" | "sLORETA"
            default: method="dSPM"
        morph2fsaverage: If source localized data should be
            morphed to the 'fsaverage' brain (important when
            group analysis is applied)
            default: morph2fsaverage=False
        nave: number of averages used to regularize the solution
            default: nave=1
        snr: signal-to-noise ratio for source localization
            default: snr = 1.
        pick_ori: orientation of the sources for source
            localization
            default: pick_ori = "normal"
        verbose: bool, str, int, or None
            If not None, override default verbose level
            (see mne.verbose).
            default: verbose=None


        Returns
        -------
        src_loc_data: source localized data (only the
            array containing the data in the dimension
            #vertices x #timeslices)
        vertno: array of vertex assignment for source
            localization. For further information see mne
            python manual
    """

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    import numpy as np
    import types

    # check if data should be morphed
    if morph2fsaverage:
        from mne import compute_morph_matrix, grade_to_vertices, morph_data_precomputed
        from mne.source_estimate import SourceEstimate
        from os.path import basename, dirname


    # -------------------------------------------
    # estimate inverse kernel
    # -------------------------------------------
    kernel, noise_norm, vertno = calc_inv_kernel(fn_inv, method=method,
                                                 nave=nave, snr=snr,
                                                 pick_ori=pick_ori)

    # -------------------------------------------
    # get some information from the
    # input data
    # -------------------------------------------
    nfreq, nepochs, nchan = data.shape
    nvoxel = noise_norm.shape[0]

    if isinstance(data[0, 0, 0], complex):
        src_loc_data = np.zeros((nfreq, nepochs, nvoxel), dtype=np.complex)
    else:
        src_loc_data = np.zeros((nfreq, nepochs, nvoxel))


    # -------------------------------------------
    # read in morphing matrix
    # -------------------------------------------
    if morph2fsaverage:
        subject_id = basename(fn_inv)[:6]
        subjects_dir = dirname(dirname(fn_inv))
        vertices_to = grade_to_vertices('fsaverage', grade=4,
                                        subjects_dir=subjects_dir)

        morph_mat = compute_morph_matrix(subject_id, 'fsaverage',
                                         vertno, vertices_to,
                                         subjects_dir=subjects_dir)
        nvoxel_morph = 2 * len(vertices_to[0])

        if isinstance(data[0, 0, 0], complex):
            morphed_src_loc_data = np.zeros((nfreq, nepochs, nvoxel_morph), dtype=np.complex)
        else:
            morphed_src_loc_data = np.zeros((nfreq, nepochs, nvoxel_morph), dtype=np.complex)


    # -------------------------------------------
    # apply inverse operator for each time slice
    # -------------------------------------------
    for iepoch in range(nepochs):

        if verbose:
            from sys import stdout
            info = "\r" if iepoch > 0 else ""
            info += "... --> Epoch %d of %d done" % (iepoch+1, nepochs)
            stdout.write(info)
            stdout.flush()

        for ifreq in range(0, nfreq):
            # multiply measured data with inverse kernel
            loc_tmp = np.dot(kernel, data[ifreq, iepoch, :])

            if pick_ori != "normal":

                # estimate L2-norm and apply noise normalization
                src_loc_data[ifreq, iepoch, :] = loc_tmp[0::3].real ** 2 + 1j * loc_tmp[0::3].imag ** 2
                src_loc_data[ifreq, iepoch, :] += loc_tmp[1::3].real ** 2 + 1j * loc_tmp[1::3].imag ** 2
                src_loc_data[ifreq, iepoch, :] += loc_tmp[2::3].real ** 2 + 1j * loc_tmp[2::3].imag ** 2
                src_loc_data[ifreq, iepoch, :] = (np.sqrt(src_loc_data[ifreq, iepoch, :].real) +
                                                  1j * np.sqrt(src_loc_data[ifreq, iepoch, :].imag))
            else:
                src_loc_data[ifreq, iepoch, :] = loc_tmp

            src_loc_data[ifreq, iepoch, :] *= noise_norm[:, 0]


        if morph2fsaverage:
            SrcEst = SourceEstimate(src_loc_data[:, iepoch, :].T,
                                    vertno, 0, 1, verbose=verbose)
            SrcEst_morphed = morph_data_precomputed(subject_id, 'fsaverage',
                                                    SrcEst, vertices_to, morph_mat)

            morphed_src_loc_data[:, iepoch, :] = SrcEst_morphed.data.T

    if verbose:
         print("")

    if morph2fsaverage:
        src_loc_data = morphed_src_loc_data
        vertno = vertices_to

    return src_loc_data, vertno



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# routine to apply FourierICA combined with ICASSO to
# a data set (on sensor-level)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def apply_ICASSO_fourierICA(fn_raw, fn_inv=None, src_loc_method='dSPM',
                            snr=1.0, morph2fsaverage=True,
                            nrep=50, stim_name='STI 014', event_id=1,
                            tmin=-0.2, tmax=0.8, flow=4.0, fhigh=34.0,
                            pca_dim=None, decim_epochs=False,
                            corr_event_picking=None,
                            max_iter=10000, conv_eps=1e-10,
                            lrate=None, complex_mixing=True, hamming_data=False,
                            envelopeICA=False, remove_outliers=False,
                            cost_function=None, verbose=True,
                            plot_dir=None, fn_plot=None, plot_res=True):

    '''
    Interface for applying fourierICA combined with ICASSO.

        Parameters
        ----------
        fn_raw: file name or list of filenames for data where
            FourierICA and ICASSO should be applied to
        fn_inv: file name of inverse operator. If given
            FourierICA is applied on data transformed to
            source space
        src_loc_method: method used for source localization.
            Only of interest if 'fn_inv' is set
            default: src_loc_method='dSPM'
        snr: signal-to-noise ratio for performing source
            localization
            default: snr=1.0
        morph2fsaverage: should data be morphed to the
            'fsaverage' brain?
            default: morph2fsaverage=True
        nrep: number of repetitions for ICASSO estimation
            default: nrep=50
        stim_name:  name of the stimulus channel. Note, for
            applying FourierCIA data are chopped around stimulus
            onset. If not set data are chopped in overlapping
            windows
            default: stim_names=None
        event_id: ID of the event of interest to be considered in
            the stimulus channel. Only of interest if 'stim_name'
            is set
            default: event_id=1
        corr_event_picking: if set should contain the complete python
            path and name of the function used to identify only the
            correct events
        tmin: time of interest prior to stimulus onset for epoch
            generation (in seconds)
            default: tmin=-0.2
        tmax: time of interest after the stimulus onset for epoch
            generation (in seconds)
            default: tmin=0.8
        flow: lower frequency border for estimating the optimal
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
        pca_dim: The number of PCA components used to apply FourierICA.
            If pca_dim > 1 this refers to the exact number of components.
            If between 0 and 1 pca_dim refers to the variance which
            should be explained by the chosen components
        decim_epochs: integer. If set the number of epochs used
            to estimate the optimal demixing matrix is decimated
            to the given number.
            default: decim_epochs=False
        max_iter: maximum number od iterations used in FourierICA
            default: max_iter=10000
        conv_eps: iteration stops when weight changes are smaller
            then this number
            default: conv_eps=1e-10,
        lrate: float containg the learning rate which should be
            used in the applied ICA algorithm
            default: lrate=None
        complex_mixing: if mixing matrix should be real or complex
            default: complex_mixing=True
        hamming_data: if set a hamming window is applied to each
            epoch prior to Fourier transformation
            default: hamming_data=False
        envelopeICA: if set ICA is estimated on the envelope
            of the Fourier transformed input data, i.e., the
            mixing model is |x|=As
            default: envelopeICA=False
        remove_outliers: If set outliers are removed from the Fourier
            transformed data.
            Outliers are defined as windows with large log-average power (LAP)

                 LAP_{c,t}=log \sum_{f}{|X_{c,tf}|^2

            where c, t and f are channels, window time-onsets and frequencies,
            respectively. The threshold is defined as |mean(LAP)+3 std(LAP)|.
            This process can be bypassed or replaced by specifying a function
            handle as an optional parameter.
            default: remove_outliers=False
        cost_function: which cost-function should be used in the complex
            ICA algorithm
            'g1': g_1(y) = 1 / (2 * np.sqrt(lrate + y))
            'g2': g_2(y) = 1 / (lrate + y)
            'g3': g_3(y) = y
            default: cost_function=None
        verbose: bool, str, int, or None
            If not None, override default verbose level
            (see mne.verbose).
            default: verbose=True
        plot_dir: directory for saving the result plots.
            default: plot_dir=None
        fn_plot: filename for saving the plot
            default: None
        plot_res: If True resulting FourierICA components are
            plotted.
            default: plot_res=True

        Returns
        -------
        W_orig: estimated optimal de-mixing matrix
        A_orig: estimated mixing matrix
        quality: quality index of the clustering between
            components belonging to one cluster
            (between 0 and 1; 1 refers to small clusters,
            i.e., components in one cluster a highly similar)
        fourier_ica_obj: FourierICA object. For further information
            please have a look into the FourierICA routine
    '''

    # ------------------------------------------
    # import FourierICA module
    # ------------------------------------------
    from jumeg.decompose.icasso import JuMEG_icasso
    from mne import set_log_level


    # set log level to 'WARNING'
    set_log_level('WARNING')


    # ------------------------------------------
    # apply FourierICA combined with ICASSO
    # ------------------------------------------
    icasso_obj = JuMEG_icasso(fn_inv=fn_inv, nrep=nrep, envelopeICA=envelopeICA, lrate=lrate,
                              src_loc_method=src_loc_method, snr=snr, morph2fsaverage=morph2fsaverage)
    W_orig, A_orig, quality, fourier_ica_obj, _, _ = icasso_obj.fit(fn_raw,
                                                                    stim_name=stim_name, event_id=event_id,
                                                                    tmin_stim=tmin, tmax_stim=tmax,
                                                                    flow=flow, fhigh=fhigh,
                                                                    pca_dim=pca_dim,
                                                                    decim_epochs=decim_epochs,
                                                                    corr_event_picking=corr_event_picking,
                                                                    max_iter=max_iter, conv_eps=conv_eps,
                                                                    complex_mixing=complex_mixing,
                                                                    hamming_data=hamming_data,
                                                                    remove_outliers=remove_outliers,
                                                                    cost_function=cost_function,
                                                                    verbose=verbose)

    # ------------------------------------------
    # plot results
    # ------------------------------------------
    if plot_res:
        # ------------------------------------------
        # import FourierICA module
        # ------------------------------------------
        from .fourier_ica_plot import plot_results
        from mne import pick_types
        from mne.io import Raw
        from os import makedirs
        from os.path import basename, dirname, exists, join

        if plot_dir == None:
            # check if directory for result plots exist
            fn_dir = dirname(fn_raw)
            plot_dir = join(fn_dir, 'plots')
            if not exists(plot_dir):
                makedirs(plot_dir)

        # prepare data for plotting
        meg_raw = Raw(fn_raw, preload=True)
        meg_channels = pick_types(meg_raw.info, meg=True, eeg=False,
                                  eog=False, stim=False, exclude='bads')
        meg_data = meg_raw._data[meg_channels, :]

        # plot data
        if fn_plot:
            fn_fourier_ica = join(plot_dir, basename(fn_plot))
        else:
            fn_fourier_ica = join(plot_dir, basename(fn_raw[:fn_raw.rfind('-raw.fif')] + ',fourierICA'))

        pk_values = plot_results(fourier_ica_obj, meg_data, W_orig, A_orig, meg_raw.info,
                             meg_channels, cluster_quality=quality, fnout=fn_fourier_ica,
                             show=False)


    return W_orig, A_orig, quality, fourier_ica_obj



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  window data and apply short-time Fourier transform
# (STFT)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def apply_stft(origdata, events=[], tpre=0.0, sfreq=1017.25,
               flow=4.0, fhigh=34.0, win_length_sec=1.0,
               overlap_fac=0.5, hamming_data=False,
               remove_outliers=True, already_epoched=False,
               baseline=(None, None), decim_epochs=False,
               unfiltered=False, verbose=True):

    """
    The code determines the correct window size and
    frequency band according to the parameters specified
    by the user and then computes the short-time Fourier
    transform (STFT). Filtering is implemented by an ideal
    band-pass filter (gain=1) between the frequencies
    specified by the user.

        Parameters
        ----------
        origdata: 2D-array containing the data for
            fourier transformation (should have the form
            #channels x #time_slices)
        events: events used for generating the epochs. Only of
            interest for evoked data.
            default: events=[]
        tpre: time of interest prior to stimulus onset.
            default: tpre=0.0
        sfreq: sampling frequency of the input data
            default: sfreq=1017.25
        flow: lower frequency border for Fourier transformation
            default: flow=4.0
        fhigh: upper frequency border for Fourier transformation
            default: fhigh=34.0
        win_length_sec: length of the window around the stimulus
            onset.
            default: win_length_sec=1.0
        overlap_fac: How much to windows should overlap in percentage,
            i.e. 0.5 means 50 % overlap.
            Note: is ignored if events is set
            default: overlap_fac=0.5
        hamming_data: if set a hamming window is applied to each
            epoch prior to Fourier transformation
            default: hamming_data=False
        remove_outliers: If set outliers are removed from the Fourier
            transformed data.
            Outliers are defined as windows with large log-average
            power (LAP)

                 LAP_{c,t}=log \sum_{f}{|X_{c,tf}|^2

            where c, t and f are channels, window time-onsets and
            frequencies, respectively. The threshold is defined as
            |mean(LAP)+3 std(LAP)|. This process can be bypassed or
            replaced by specifying a function handle as an optional
            parameter.
            default: remove_outliers=True
        already_epoched: If set the already epoched data are directly
            used. The data should than have the form
            (#wtime_slices x #epochs x #nchan)
            default: already_epoched=False
        baseline: If set baseline correction is applied to epochs
            prior to Fourier transformation
            default: baseline=(None, None)
        decim_epochs: if set the number of epochs will be reduced
            (per subject) to that number for the estimation of the
            demixing matrix.
            Note: the epochs were chosen randomly from the complete
            set of epochs.
            default: decim_epochs=False
        unfiltered: bool
            If true data are not filtered to a certain frequency range
            default: unfiltered=False
        verbose: bool, str, int, or None
            If not None, override default verbose level
            (see mne.verbose).
            default: verbose=True

        Returns
        -------
        X: Array containing the Fourier transformed data
        events: events used for generating the epochs
    """

    # import FourierICA module
    from scipy import fftpack

    if verbose:
        print("... Generate sampling windows and STFT")

    # determine number of channels and time points in original data
    nchan = origdata.shape[0]
    ntsl = origdata.shape[1]

    # compute number of time points in one window based on other
    # parameter
    win_size = np.floor(win_length_sec*sfreq)
    win_inter = np.ceil(win_size * overlap_fac)

    if already_epoched:
        win_size, nwindows, nchan = origdata.shape

    elif len(events):
        events = events[events > np.abs(tpre * sfreq)]
        events = events[events < (ntsl-win_size-np.abs(tpre * sfreq))]
        if len(events):
            nwindows = len(events)
        else:
            print("Events not in the index range of the data!")

    else:
        nwindows = int(np.floor((ntsl-win_size)/win_inter+1))

    # compute frequency indices (for the STFT)
    startfftind = int(np.floor(flow*win_length_sec))
    if startfftind < 0:
        print("Minimal frequency must be positive!")
        import pdb
        pdb.set_trace()

    endfftind = int(np.floor(fhigh*win_length_sec+1))
    nyquistf = np.floor(win_size/2.0)
    if endfftind > nyquistf:
        print("Maximal frequency must be less than the Nyquist frequency!")
        import pdb
        pdb.set_trace()

    # check if data should be filtered
    if unfiltered:
        startfftind = 0
        endfftind = nyquistf - 1

    fftsize = int(endfftind-startfftind)

    # check if decimation should be applied
    win_sel = []
    if decim_epochs:

        # check if the chosen number is smaller than
        # the number of detected epochs
        if decim_epochs < nwindows:

            idx_sel = np.sort(np.random.choice(nwindows, decim_epochs,
                                               replace=False))
            nwindows = decim_epochs

            # check if events are given
            if len(events):
                events = events[idx_sel]

            # or if windows are generated without epochs
            else:
                win_sel = idx_sel * win_inter


    # initialization of tensor X, which is the main data matrix input
    # to the code that follows
    X = np.zeros((fftsize, nwindows, nchan), dtype=np.complex)

    # define window initial limits
    if len(events):
        idx_start = events[0] + np.floor(tpre * sfreq)
        window = [idx_start, idx_start+win_size]
    elif len(win_sel):
        window = [win_sel[0], win_sel[0]+win_size]
    else:
        window = [0, win_size]

    # construct hamming window if necessary
    if hamming_data:
        hamming_window = np.hamming(win_size)

    # short-time fourier transform (window sampling + fft)
    for iwin in range(0, nwindows):
        # extract data window
        if already_epoched:
            data_win = origdata[:, iwin, :].transpose()
        else:
            data_win = origdata[:, int(window[0]):int(window[1])]

        if baseline != (None, None):

            idx_base1 = int((baseline[0] - tpre) * sfreq) if baseline[0] != None else 0
            idx_base2 = int((baseline[1] - tpre) * sfreq) if baseline[1] != None else -1
            data_win = data_win - np.mean(data_win[:, idx_base1:idx_base2],
                                          axis=-1)[:, np.newaxis]

        # multiply by a hamming window if necessary
        if hamming_data:
            data_win = np.dot(data_win, np.diag(hamming_window))

        data_win_ft = fftpack.fft(data_win.transpose(), axis=0)
        X[:, iwin, :] = data_win_ft[startfftind:endfftind, :]

        # new data window interval
        if len(events) and (iwin+1 < nwindows):
            idx_start = events[iwin+1] + np.floor(tpre * sfreq)
            window = [idx_start, idx_start+win_size]
        elif len(win_sel) and (iwin+1 < nwindows):
            window = [win_sel[iwin+1], win_sel[iwin+1]+win_size]
        else:
            window += win_inter

    # REMOVE OUTLIERS
    # Outliers are defined as windows with large log-average power (LAP)
    #
    # LAP_{c,t}=log \sum_{f}{|X_{c,tf}|^2
    #
    # where c, t and f are channels, window time-onsets and frequencies,
    # respectively. The threshold is defined as |mean(LAP)+3 std(LAP)|.
    # This process can be bypassed or replaced by specifying a function
    # handle as an optional parameter.
    if remove_outliers:
        log_norms = np.log(np.sum(np.abs(X * X.conj()), axis=0))
        outlier_thres = np.mean(log_norms) + 3 * np.std(log_norms)
        outlier_indices = np.where(log_norms > outlier_thres)

        X[:, outlier_indices[0], outlier_indices[1]] = 0

        # print out some information if required
        if verbose:
            print("... Outliers removal: ")
            print(">>> removed %u windows" % len(outlier_indices[0]))

    return X, events



########################################################
#                                                      #
#               JuMEG_fourier_ica class                #
#                                                      #
########################################################
class JuMEG_fourier_ica(object):

    def __init__(self, events=[], tpre=0.0, overlap_fac=2.0, win_length_sec=1.0,
                 sfreq=1017.25, flow=4.0, fhigh=34.0, hamming_data=True,
                 remove_outliers=False, complex_mixing=True,
                 pca_dim=0.95, zero_tolerance=1e-7, ica_dim=200, max_iter=10000,
                 lrate=1.0, conv_eps=1e-16, cost_function='g2', envelopeICA=False,
                 decim_epochs=False):

        """
        Generate a fourier_ica object.

            Parameters
            ----------
            events: indices of events of interest. If set epochs are
                generated around these events. Otherwise windows are
                generated 'sliding window' like for the complete data
                set
                default: events=[]
            tpre: time of interest prior to stimulus onset.
                Important for generating epochs to apply FourierICA
                default=0.0
            win_length_sec: length of the epoch window in seconds
                default: win_length_sec=1.0
            overlap_fac: factor how much the windows overlap. Note, if
                keyword 'events' is set 'overlap_fac' is ignored
                default overlap_fac=2.0
            sfreq: sampling frequency of the input data
                default: sfreq=1017.25
            flow: lower frequency border for estimating the optimal
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
                remove_outliers=False
            complex_mixing: if mixing matrix should be real or complex
                default: complex_mixing=True
            pca_dim: the number of PCA components used to apply FourierICA.
                If pca_dim > 1 this refers to the exact number of components.
                If between 0 and 1 pca_dim refers to the variance which
                should be explained by the chosen components
                default: pca_dim=0.95
            zero_tolerance: threshold for eigenvalues to be considered (when
                applying PCA). All eigenvalues smaller than this threshold are
                discarded
                default: zero_tolerance=1e-7
            ica_dim: the number of ICA components used to apply FourierICA. If
                pca_dim is lower the number is reduced to that number.
                default: ica_dim = 200
            max_iter: maximum number od iterations used in FourierICA
                default: max_iter=10000
            lrate: initial learning rate
                default: lrate=1.0
            conv_eps: iteration stops when weight changes are smaller
                then this number
                default: conv_eps = 1e-16
            cost_function: which cost-function should be used in the complex
                ICA algorithm
                'g1': g_1(y) = 1 / (2 * np.sqrt(lrate + y))
                'g2': g_2(y) = 1 / (lrate + y)
                'g3': g_3(y) = y
            envelopeICA: if set ICA is estimated on the envelope
                of the Fourier transformed input data, i.e., the
                mixing model is |x|=As
                default: envelopeICA=False
            decim_epochs: if set the number of epochs will be reduced (per
                subject) to that number for the estimation of the demixing matrix.
                Note: the epochs were chosen randomly from the complete set of
                epochs.

            Returns
            -------
            object: FourierICA object

        """

        self._events = events
        self._tpre = tpre
        self._sfreq = sfreq
        self._flow = flow
        self._fhigh = fhigh
        self._win_length_sec = win_length_sec

        # Note: If events is set, overlap_fac is forced to be 1
        if len(self._events):
            self._overlap_fac = 1.0
        else:
            self._overlap_fac = overlap_fac

        self._hamming_data = hamming_data
        self._remove_outliers = remove_outliers
        self._complex_mixing = complex_mixing
        self._pca_dim = pca_dim
        self._zero_tolerance = zero_tolerance
        self._ica_dim = ica_dim
        self._max_iter = max_iter
        self._lrate = lrate
        self._conv_eps = conv_eps
        self.cost_function = cost_function
        self.envelopeICA = envelopeICA
        self.decim_epochs = decim_epochs
        self.dmean = []
        self.dstd = []


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get events
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_events(self, events):
        self._events = events
        # Note: If events is set, overlap_fac is forced to be 1
        if len(self._events):
            self._overlap_fac = 1.0

    def _get_events(self):
        return self._events

    events = property(_get_events, _set_events)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get time for start of the windows around events
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_tpre(self, tpre):
        self._events = tpre

    def _get_tpre(self):
        return self._tpre

    tpre = property(_get_tpre, _set_tpre)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get sampling frequency
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_sfreq(self, sfreq):
        self._sfreq = sfreq

    def _get_sfreq(self):
        return self._sfreq

    sfreq = property(_get_sfreq, _set_sfreq)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get start of frequency band of interest
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_flow(self, flow):
        self._flow = flow

    def _get_flow(self):
        return self._flow

    flow = property(_get_flow, _set_flow)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get end of frequency band of interest
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_fhigh(self, fhigh):
        self._fhigh = fhigh

    def _get_fhigh(self):
        return self._fhigh

    fhigh = property(_get_fhigh, _set_fhigh)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get window length in seconds (for FFT)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_win_length_sec(self, win_length_sec):
        self._win_length_sec = win_length_sec

    def _get_win_length_sec(self):
        return self._win_length_sec

    win_length_sec = property(_get_win_length_sec, _set_win_length_sec)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get window overlap factor (for FFT)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_overlap_fac(self, overlap_fac):
        # Note: If events is set, overlap_fac is forced to be 1
        if len(self.events):
            self._overlap_fac = 1.0
        else:
            self._overlap_fac = overlap_fac

    def _get_overlap_fac(self):
        return self._overlap_fac

    overlap_fac = property(_get_overlap_fac, _set_overlap_fac)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get hamming data, i.e. should a hamming window be
    # applied?
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_hamming_data(self, hamming_data):
        self._hamming_data = hamming_data

    def _get_hamming_data(self):
        return self._hamming_data

    hamming_data = property(_get_hamming_data, _set_hamming_data)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get remove outliers (from FFT windows)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_remove_outliers(self, remove_outliers):
        self._remove_outliers = remove_outliers

    def _get_remove_outliers(self):
        return self._remove_outliers

    remove_outliers = property(_get_remove_outliers, _set_remove_outliers)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get complex_mixing (for PCA decomposition)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_complex_mixing(self, complex_mixing):
        self._complex_mixing = complex_mixing

    def _get_complex_mixing(self):
        return self._complex_mixing

    complex_mixing = property(_get_complex_mixing, _set_complex_mixing)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get PCA dimension
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_pca_dim(self, pca_dim):
        self._pca_dim = pca_dim

    def _get_pca_dim(self):
        return self._pca_dim

    pca_dim = property(_get_pca_dim, _set_pca_dim)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get Zero tolerance
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_zero_tolerance(self, zero_tolerance):
        self._zero_tolerance = zero_tolerance

    def _get_zero_tolerance(self):
        return self._zero_tolerance

    zero_tolerance = property(_get_zero_tolerance, _set_zero_tolerance)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get ICA dimension
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_ica_dim(self, ica_dim):
        self._ica_dim = ica_dim

    def _get_ica_dim(self):
        return self._ica_dim

    ica_dim = property(_get_ica_dim, _set_ica_dim)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get maximum number of iterations (for ICA)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_max_iter(self, max_iter):
        self._max_iter = np.max((40 * self.pca_dim, max_iter))

    def _get_max_iter(self):
        return self._max_iter

    max_iter = property(_get_max_iter, _set_max_iter)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get maximum learning rate for ICA
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_lrate(self, lrate):
        self._lrate = lrate

    def _get_lrate(self):
        return self._lrate

    lrate = property(_get_lrate, _set_lrate)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get set threshold for stopping criterion
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_conv_eps(self, conv_eps):
        self._conv_eps = conv_eps

    def _get_conv_eps(self):
        return self._conv_eps

    conv_eps = property(_get_conv_eps, _set_conv_eps)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # get temporal envelope of independent components
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_temporal_envelope(self, origdata, W_orig, average=True):

        """
        Returns the temporal envelope of the independent
        components after FourierICA decomposition. Note, the
        'fit()' function must be applied before this routine
        can be used (to get W_orig).

            Parameters
            ----------
            origdata: array of data to be decomposed [nchan, ntsl].
            W_orig: estimated de-mixing matrix
            average: if set the temporal envelopes are averaged
                over all epochs
                default: average=True

            Returns
            -------
            temporal_envelope: temporal envelop of the independent
                components
        """

        # chop data into epochs and apply short-time Fourier transform (STFT)
        X, _ = apply_stft(origdata, events=self.events, tpre=self.tpre, sfreq=self.sfreq,
                          flow=self.flow, fhigh=self.fhigh, win_length_sec=self.win_length_sec,
                          overlap_fac=self.overlap_fac, hamming_data=self.hamming_data,
                          remove_outliers=False, verbose=False)

        # get some size information from data
        fftsize, nwindows, nchan = X.shape
        ntsl = int(np.floor(self.win_length_sec*self.sfreq))
        ncomp = W_orig.shape[0]
        temporal_envelope = np.zeros((nwindows, ncomp, ntsl))
        startfftind = int(np.floor(self.flow*self.win_length_sec))
        endfftind = int(startfftind+fftsize)
        fft_act = np.zeros((ncomp, ntsl), dtype=np.complex)
        act = np.zeros((ncomp, nwindows, fftsize), dtype=np.complex)

        # loop over all windows
        for iwin in range(0, nwindows):
            # transform data into FourierICA-space
            X_norm = (X[:, iwin, :] - np.dot(np.ones((fftsize, 1)), self.dmean)) / \
                          np.dot(np.ones((fftsize, 1)), self.dstd)
            act[:, iwin, :] = np.dot(W_orig, X_norm.transpose())
            # act = np.dot(W_orig, X[:, iwin, :].transpose())
            # apply inverse STFT to get temporal envelope
            fft_act[:, startfftind:endfftind] = act[:, iwin, :]
            temporal_envelope[iwin, :, :] = sc.fftpack.ifft(fft_act, n=ntsl, axis=1).real

        # average data if required
        if average:
            temporal_envelope = np.mean(temporal_envelope, axis=0)

        return temporal_envelope


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # get signal of each IC back-transformed to MEG space
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_reconstructed_signal(self, origdata, W_orig, A_orig,
                                 average=True):

        """
        Returns the reconstructed MEG signal obtained from each
        component separately. Note, the 'fit()'function must be
        applied before this routine can be used (to get W_orig).

            Parameters
            ----------
            origdata: array of data to be decomposed [nchan, ntsl].
            W_orig: estimated de-mixing matrix
            A_orig: estimated mixing matrix
            average: if set results are averaged over all epochs
                default: average=True

            Returns
            -------
            rec_signal_avg: reconstructed signal obtained from each
                component separately
            orig_avg: reconstructed signal obtained when using all
                components. Note, this might be different from the
                input 'origdata' as filtering is applied when performing
                Fourier transformation to convert data to FFT space
        """

        # chop data into epochs and apply short-time Fourier transform (STFT)
        X, _ = apply_stft(origdata, events=self.events, tpre=self.tpre, sfreq=self.sfreq,
                          flow=self.flow, fhigh=self.fhigh, win_length_sec=self.win_length_sec,
                          overlap_fac=self.overlap_fac, hamming_data=self.hamming_data,
                          remove_outliers=False, verbose=False)

        # get some size information from data
        fftsize, nwindows, nchan = X.shape
        ntsl = int(np.floor(self.win_length_sec*self.sfreq))
        ncomp = W_orig.shape[0]
        rec_signal = np.zeros((nwindows, ncomp, nchan, ntsl))
        orig_signal = np.zeros((nwindows, nchan, ntsl))


        startfftind = int(np.floor(self.flow*self.win_length_sec))
        endfftind = int(startfftind+fftsize)
        fft_data = np.zeros((nchan, ntsl), dtype=np.complex)

        # loop over all windows
        for iwin in range(0, nwindows):
            # transform data into FourierICA-space
            X_norm = (X[:, iwin, :] - np.dot(np.ones((fftsize, 1)), self.dmean)) / \
                     np.dot(np.ones((fftsize, 1)), self.dstd)
            act = np.dot(W_orig, X_norm.transpose())
            fft_data[:, startfftind:endfftind] = X[:, iwin, :].transpose()
            orig_signal[iwin, :, :] = sc.fftpack.ifft(fft_data, axis=1).real

            # back-transform signal to MEG-space for each IC separately
            for icomp in range(0, ncomp):
                A_cur = np.zeros((nchan, ncomp), dtype=np.complex)
                A_cur[:, icomp] = A_orig[:, icomp]
                fft_data[:, startfftind:endfftind] = np.dot(A_cur, act) * np.dot(np.ones((fftsize, 1)), self.dstd).transpose() + \
                                                     np.dot(np.ones((fftsize, 1)), self.dmean).transpose()
                rec_signal[iwin, icomp, :, :] = sc.fftpack.ifft(fft_data, axis=1).real

        # average signal over windows
        if average:
            rec_signal = np.mean(rec_signal, axis=0)
            orig_signal = np.mean(orig_signal, axis=0)

        return rec_signal, orig_signal


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # get Fourier amplitudes
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_fourier_ampl(self, origdata, W_orig):

        """
        Returns the Fourier amplitude of the input data after
        decomposition. Note, the 'fit()' function must be
        applied before this routine can be used (to get W_orig).

            Parameters
            ----------
            origdata: array of data to be decomposed [nchan, ntsl].
            W_orig: estimated de-mixing matrix

            Returns
            -------
            fourier_ampl: Fourier amplitude of the input data
        """

        # chop data into epochs and apply short-time Fourier transform (STFT)
        X, _ = apply_stft(origdata, events=self.events, tpre=self.tpre, sfreq=self.sfreq,
                          flow=self.flow, fhigh=self.fhigh, win_length_sec=self.win_length_sec,
                          overlap_fac=self.overlap_fac, hamming_data=self.hamming_data,
                          remove_outliers=False, verbose=False)

        # get some size information from data
        fftsize, nwindows, nchan = X.shape
        ncomp = W_orig.shape[0]
        act = np.zeros((ncomp, nwindows, fftsize))

        # loop over all windows
        for iwin in range(0, nwindows):
            X_norm = (X[:, iwin, :] - np.dot(np.ones((fftsize, 1)), self.dmean)) / \
                     np.dot(np.ones((fftsize, 1)), self.dstd)
            act[:, iwin, :] = np.abs(np.dot(W_orig, X_norm.transpose()))
            # act[:, iwin, :] = np.abs(np.dot(W_orig, X[:, iwin, :].transpose()))

        # average signal over windows and normalize to arbitrary units
        fourier_ampl = np.mean(act, axis=1)
        fourier_ampl /= np.max(fourier_ampl)

        return fourier_ampl



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # perform fourier based ICA signal decomposition
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fit(self, origdata, events=[], tpre=None, sfreq=None,
            flow=None, fhigh=None, win_length_sec=None,
            overlap_fac=None, hamming_data=None,
            remove_outliers=None,
            complex_mixing=None, pca_dim=None,
            zero_tolerance=None, ica_dim=None,
            max_iter=None, lrate=None, conv_eps=None,
            data_already_stft=False,
            data_already_normalized=False,
            whiten_mat=[], dewhiten_mat=[],
            pca_only=False, decim_epochs=False,
            verbose=False):

        """
        Apply FourierICA to given data set.

            Parameters
            ----------
            origdata: array of data to be decomposed [nchan, ntsl].
            data_already_stft: if set data are not transposed to
                Fourier space again (important when ICASSO is applied)
                default: data_already_stft=False
            data_already_normalized: if set data are not normalized
                prior to ICA estimation (important when ICASSO is applied)
                default: data_already_normalized=False
            whiten_mat: if set this matrix is used for whitening,
                i.e. PCA is not estimated again. This parameter is
                important when ICASSO is used.
                default: whiten_mat=[]
            dewhiten_mat: if set this matrix is used for de-whitening,
                i.e. PCA is not estimated again. This parameter is
                important when ICASSO is used.
                default: dewhiten_mat=[]
            pca_only: if set only PCA is estimated and not ICA
                default: pca_only=False
            verbose: bool, str, int, or None
                If not None, override default verbose level
                (see mne.verbose).
                default: verbose=True

            All other parameters are explained in the
            initialization of the FourierICA object.

            Returns
            -------
            W_orig: estimated de-mixing matrix
            A_orig: estimated mixing matrix
            S_FT: decomposed signal (Fourier coefficients)
            Smean: mean of the input data (in Fourier space)
            objective: values of the objectiv function used to
                sort the input data
            whiten_mat: whitening matrix
            dewhiten_mat: dewhitening matrix
        """

        # import necessary modules
        from jumeg.decompose.complex_ica import complex_ica


        # check input parameter
        if events:
            self.events = events
        if tpre:
            self.tpre = tpre
        if sfreq:
            self.sfreq = sfreq
        if flow:
            self.flow = flow
        if fhigh:
            self.fhigh = fhigh
        if win_length_sec:
            self.win_length_sec = win_length_sec
        if overlap_fac:
            self.overlap_fac = overlap_fac
        if hamming_data:
            self.hamming_data = hamming_data
        if remove_outliers:
            self.remove_outliers = remove_outliers
        if complex_mixing:
            self.complex_mixing = complex_mixing
        if pca_dim:
            self.pca_dim = pca_dim
        if zero_tolerance:
            self.zero_tolerance = zero_tolerance
        if ica_dim:
            self.ica_dim = ica_dim
        if max_iter:
            self.max_iter = max_iter
        if lrate:
            self.lrate = lrate
        if conv_eps:
            self._conv_eps = conv_eps
        if decim_epochs:
            self.decim_epochs = decim_epochs


        str_hamming_window = "True" if self.hamming_data else "False"
        str_complex_mixing = "True" if self.complex_mixing else "False"


        # print out some information
        if verbose:
            print(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<")
            print(">>>            Launching Fourier ICA           <<<")
            print(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<")
            print(">>>")
            print(">>> Sampling frequency set to: %d" % self.sfreq)
            print(">>> Start of frequency band set to: %d" % self.flow)
            print(">>> End of frequency band set to: %d" % self.fhigh)
            print(">>> Using hamming window: %s" % str_hamming_window)
            print(">>> Assume complex mixing: %s" % str_complex_mixing)
            print(">>>")
            print(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<")



        # ----------------------------------------------------------------
        # window data and apply short-time Fourier transform (STFT)
        # ----------------------------------------------------------------
        if data_already_stft:
            X = origdata.copy()
        else:
            X, events = apply_stft(origdata, events=self.events, tpre=self.tpre,
                                   sfreq=self.sfreq, flow=self.flow, fhigh=self.fhigh,
                                   win_length_sec=self.win_length_sec,
                                   overlap_fac=self.overlap_fac,
                                   hamming_data=self.hamming_data,
                                   remove_outliers=self.remove_outliers,
                                   decim_epochs=self.decim_epochs,
                                   verbose=verbose)
            self.events = events

        if not data_already_normalized:
            # concatenate STFT for consecutive windows in each channel
            fftsize, nwindows, nchan = X.shape
            nrows_Xmat_c = fftsize*nwindows
            Xmat_c = X.reshape((nrows_Xmat_c, nchan), order='F')
        else:
            Xmat_c = X

        del X

        # ----------------------------------------------------------------
        # apply ICA
        # ----------------------------------------------------------------
        # complex ICA
        W_orig, A_orig, S, Smean, Sstddev, objective, whiten_mat, dewhiten_mat = \
            complex_ica(Xmat_c, complex_mixing=self.complex_mixing,
                        pca_dim=self.pca_dim, ica_dim=self.ica_dim,
                        zero_tolerance=self.zero_tolerance,
                        conv_eps=self.conv_eps, max_iter=self.max_iter,
                        lrate=self.lrate, whiten_mat=whiten_mat,
                        dewhiten_mat=dewhiten_mat,
                        cost_function=self.cost_function,
                        envelopeICA=self.envelopeICA,
                        already_normalized=data_already_normalized,
                        pca_only=pca_only, verbose=verbose,
                        overwrite=True)


        # Independent components. If real mixing matrix, these are STFT's
        # of the sources
        self.ica_dim = W_orig.shape[0]
        self.dmean = Smean
        self.dstd = Sstddev

        if not data_already_normalized:
            S_FT = np.transpose(S.reshape((self.ica_dim, fftsize, nwindows)), (0, 2, 1))
        else:
            S_FT = S


        return W_orig, A_orig, S_FT, Smean, Sstddev, objective, whiten_mat, dewhiten_mat



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # save FourierICA object
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save(self, fnOut):

        """
        Method to save a FourierICA object to disk

            Parameters
            ----------
            fnOut: filename where the FourierICA object
                should be saved
        """

        # import necessary modules
        import pickle

        filehandler = open(fnOut, "wb")
        pickle.dump(self, filehandler)
        filehandler.close()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# read FourierICA object
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def read(fnIn):

    """
    Method to read a FourierICA object from disk

        Parameters
        ----------
        fnIn: filename of the FourierICA object
    """

    # import necessary modules
    import pickle

    filehandler = open(fnIn, "rb")
    fourier_ica_obj = pickle.load(filehandler)
    filehandler.close()

    return fourier_ica_obj


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   to simplify the call of the JuMEG_fourier_ica() help
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
fourier_ica = JuMEG_fourier_ica()