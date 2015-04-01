# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.fourier_ica --------------------------------------
----------------------------------------------------------------------
 autor      : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 31.03.2015
 version    : 1.0

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
#  window data and apply short-time Fourier transform
# (STFT)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def apply_stft(origdata, events=[], tpre=0.0, sfreq=1017.25,
               flow=1.0, fhigh=45.0, win_length_sec=1.0,
               overlap_fac=2.0, hamming_data=False,
               remove_outliers=True, fcnoutliers=[],
               verbose=True):

    """
    The code determines the correct window size and
    frequency band according to the parameters specified
    by the user and then computes the short-time Fourier
    transform (STFT). Filtering is implemented by an ideal
    band-pass filter (gain=1) between the frequencies
    specified by the user.

    --> overlap_fac ignored if events is set
    """

    from scipy import fftpack

    if verbose:
        print "... Sampling windows and STFT"

    # determine number of channels and time points in original data
    nchan = origdata.shape[0]
    ntsl = origdata.shape[1]

    # compute number of time points in one window based on other
    # parameter
    win_size = np.floor(win_length_sec*sfreq)
    win_inter = np.ceil(win_size/overlap_fac)

    if len(events):
        events = events[events > np.abs(tpre * sfreq)]
        events = events[events < (ntsl-win_size-np.abs(tpre * sfreq))]
        if len(events):
            nwindows = len(events)
        else:
            print "Events not in the index range of the data!"
    else:
        nwindows = int(np.floor((ntsl-win_size)/win_inter+1))


    # compute frequency indices (for the STFT)
    startfftind = int(np.floor(flow*win_length_sec))
    if startfftind < 0:
        print "Minimal frequency must be positive!"
        import pdb
        pdb.set_trace()

    endfftind = int(np.floor(fhigh*win_length_sec+1))
    nyquistf = np.floor(win_size/2.0)
    if endfftind > nyquistf:
        print "Maximal frequency must be less than the Nyquist frequency!"
        import pdb
        pdb.set_trace()

    fftsize = int(endfftind-startfftind)


    # initialization of tensor X, which is the main data matrix input
    # to the code that follows
    X = np.zeros((fftsize, nwindows, nchan), dtype=np.complex)

    # define window initial limits
    if len(events):
        idx_start = events[0] + np.floor(tpre * sfreq)
        window = [idx_start, idx_start+win_size]
    else:
        window = [0, win_size]

    # construct hamming window if necessary
    if hamming_data:
        hamming_window = np.hamming(win_size)


    # short-time fourier transform (window sampling + fft)
    for iwin in range(0, nwindows):
        # extract data window
        data_win = origdata[:, int(window[0]):int(window[1])]

        # multiply by a hamming window if necessary
        if hamming_data:
            data_win = np.dot(data_win, np.diag(hamming_window))

        data_win_ft = fftpack.fft(data_win.transpose(), axis=0)
        X[:, iwin, :] = data_win_ft[startfftind:endfftind, :]

        # new data window interval
        if len(events) and (iwin+1 < nwindows):
            idx_start = events[iwin+1] + np.floor(tpre * sfreq)
            window = [idx_start, idx_start+win_size]
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
        if fcnoutliers:
            X = fcnoutliers[X]
        else:
            log_norms = np.log(np.sum(np.abs(X * X.conj()), axis=0))
            outlier_thres = np.mean(log_norms) + 3 * np.std(log_norms)
            outlier_indices = np.where(log_norms > outlier_thres)

            X[:, outlier_indices[0], outlier_indices[1]] = 0

            # print out some information if required
            if verbose:
                print "... Outliers removal: "
                print ">>> removed %u windows" % len(outlier_indices[0])

    return X, events



########################################################
#                                                      #
#               JuMEG_fourier_ica class                #
#                                                      #
########################################################
class JuMEG_fourier_ica(object):

    def __init__(self, events=[], tpre=0.0, overlap_fac=2.0, win_length_sec=1.0,
                 sfreq=1017.25, flow=4.0, fhigh=34.0, hamming_data=True,
                 remove_outliers=False, fcnoutliers=[], complex_mixing=True,
                 pca_dim=0.95, zero_tolerance=1e-7, ica_dim=200, max_iter=10000,
                 lrate=1.0, conv_eps=1e-16):
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
            max_iter: maximum number od iterations used in FourierICA
                default: max_iter=10000
            lrate: initial learning rate
                default: lrate=1.0
            conv_eps: iteration stops when weight changes are smaller
                then this number
                default: conv_eps = 1e-16

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
        self._fcnoutliers = fcnoutliers
        self._complex_mixing =complex_mixing
        self._pca_dim = pca_dim
        self._zero_tolerance = zero_tolerance
        self._ica_dim = ica_dim
        self._max_iter = max_iter
        self._lrate = lrate
        self._conv_eps = conv_eps



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
    # set/get fcnoutliers
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_fcnoutliers(self, fcnoutliers):
        self._fcnoutliers = fcnoutliers

    def _get_fcnoutliers(self):
        return self._fcnoutliers

    fcnoutliers = property(_get_fcnoutliers, _set_fcnoutliers)


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
            pk_max: pk-values of the independent component
        """

        # import necessary modules
        from mne.preprocessing import ctps_ as ctps

        # chop data into epochs and apply short-time Fourier transform (STFT)
        X, _ = apply_stft(origdata, events=self.events, tpre=self.tpre, sfreq=self.sfreq,
                          flow=self.flow, fhigh=self.fhigh, win_length_sec=self.win_length_sec,
                          overlap_fac=self.overlap_fac, hamming_data=self.hamming_data,
                          remove_outliers=False, fcnoutliers=self.fcnoutliers,
                          verbose=False)

        # get some size information from data
        fftsize, nwindows, nchan = X.shape
        ntsl = int(np.floor(self.win_length_sec*self.sfreq))
        ncomp = W_orig.shape[0]
        temporal_envelope = np.zeros((nwindows, ncomp, ntsl))
        startfftind = int(np.floor(self.flow*self.win_length_sec))
        endfftind = int(startfftind+fftsize)
        fft_act = np.zeros((ncomp, ntsl), dtype=np.complex)

        # loop over all windows
        for iwin in range(0, nwindows):
            # transform data into FourierICA-space
            act = np.dot(W_orig, X[:, iwin, :].transpose())
            # apply inverse STFT to get temporal envelope
            fft_act[:, startfftind:endfftind] = act
            temporal_envelope[iwin, :, :] = sc.fftpack.ifft(fft_act, n=ntsl, axis=1).real

        # estimate pk-values
        _, pk, _ = ctps.ctps(temporal_envelope)
        pk_max = np.max(pk, axis=1)

        # average data if required
        if average:
            temporal_envelope = np.mean(temporal_envelope, axis=0)

        return temporal_envelope, pk_max


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
                          remove_outliers=False, fcnoutliers=self.fcnoutliers,
                          verbose=False)

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
            act = np.dot(W_orig, X[:, iwin, :].transpose())
            fft_data[:, startfftind:endfftind] = X[:, iwin, :].transpose()
            orig_signal[iwin, :, :] = sc.fftpack.ifft(fft_data, axis=1).real

            # back-transform signal to MEG-space for each IC separately
            for icomp in range(0, ncomp):
                A_cur = np.zeros((nchan, ncomp), dtype=np.complex)
                A_cur[:, icomp] = A_orig[:, icomp]
                fft_data[:, startfftind:endfftind] = np.dot(A_cur, act)
                rec_signal[iwin, icomp, :, :] = sc.fftpack.ifft(fft_data, axis=1).real

        # average signal over windows
        if average:
            rec_signal_avg = np.mean(rec_signal, axis=0)
            orig_avg = np.mean(orig_signal, axis=0)

        return rec_signal_avg, orig_avg


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
                          remove_outliers=False, fcnoutliers=self.fcnoutliers,
                          verbose=False)

        # get some size information from data
        fftsize, nwindows, nchan = X.shape
        ncomp = W_orig.shape[0]
        act = np.zeros((ncomp, nwindows, fftsize))

        # loop over all windows
        for iwin in range(0, nwindows):
            act[:, iwin, :] = np.abs(np.dot(W_orig, X[:, iwin, :].transpose()))

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
            remove_outliers=None, fcnoutliers=None,
            complex_mixing=None, pca_dim=None,
            zero_tolerance=None, ica_dim=None,
            max_iter=None, lrate=None, conv_eps=None,
            verbose=False):

        """
        Apply FourierICA to given data set.

            Parameters
            ----------
            origdata: array of data to be decomposed [nchan, ntsl].
            verbose: bool, str, int, or None
                If not None, override default verbose level
                (see mne.verbose).
                default: verbose=True

            For all other parameters are explained in the
            initialization of the FourierICA object

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
        if fcnoutliers:
            self.fcnoutliers = fcnoutliers
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



        str_hamming_window = "True" if self.hamming_data else "False"
        str_complex_mixing = "True" if self.complex_mixing else "False"


        # print out some information
        if verbose:
            print ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<"
            print ">>>            Launching Fourier ICA           <<<"
            print ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<"
            print ">>>"
            print ">>> Sampling frequency set to: %d" % self.sfreq
            print ">>> Start of frequency band set to: %d" % self.flow
            print ">>> End of frequency band set to: %d" % self.fhigh
            print ">>> Using hamming window: %s" % str_hamming_window
            print ">>> Assume complex mixing: %s" % str_complex_mixing
            print ">>>"
            print ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<"



        # ----------------------------------------------------------------
        # window data and apply short-time Fourier transform (STFT)
        # ----------------------------------------------------------------
        X, events = apply_stft(origdata, events=self.events, tpre=self.tpre,
                               sfreq=self.sfreq, flow=self.flow, fhigh=self.fhigh,
                               win_length_sec=self.win_length_sec,
                               overlap_fac=self.overlap_fac,
                               hamming_data=self.hamming_data,
                               remove_outliers=self.remove_outliers,
                               fcnoutliers=self.fcnoutliers, verbose=verbose)
        self.events = events
        fftsize, nwindows, nchan = X.shape


        # concatenate STFT for consecutive windows in each channel
        nrows_Xmat_c = fftsize*nwindows
        Xmat_c = X.reshape((nrows_Xmat_c, nchan), order='F')

        # ----------------------------------------------------------------
        # apply complex ICA
        # ----------------------------------------------------------------
        W_orig, A_orig, S, Smean, objective, whiten_mat, dewhiten_mat = complex_ica(Xmat_c, complex_mixing=self.complex_mixing,
                                                                                    pca_dim=self.pca_dim, ica_dim=self.ica_dim,
                                                                                    zero_tolerance=self.zero_tolerance,
                                                                                    conv_eps=self.conv_eps, max_iter=self.max_iter,
                                                                                    lrate=self.lrate, verbose=verbose)


        # Independent components. If real mixing matrix, these are STFT's
        # of the sources
        self.ica_dim = W_orig.shape[0]
        S_FT = np.transpose(S.reshape((self.ica_dim, fftsize, nwindows)), (0, 2, 1))


        return W_orig, A_orig, S_FT, Smean, objective, whiten_mat, dewhiten_mat



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   to simplify the call of the JuMEG_fourier_ica() help
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
fourier_ica = JuMEG_fourier_ica()