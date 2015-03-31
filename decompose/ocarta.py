# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.ocarta -------------------------------------------
----------------------------------------------------------------------
 author     : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 12.12.2014
 version    : 1.1 (NOTE: Current version is only able to handle data
                         recorded with the magnesWH3600 system)

----------------------------------------------------------------------
 Based on following publications:
----------------------------------------------------------------------

L. Breuer, J. Dammers, T.P.L. Roberts, and N.J. Shah, 'Ocular and
Cardiac Artifact Rejection for Real-Time Analysis in MEG',
Journal of Neuroscience Methods, Jun. 2014
(doi:10.1016/j.jneumeth.2014.06.016)

L. Breuer, J. Dammers, T.P.L. Roberts, and N.J. Shah, 'A Constrained
ICA Approach for Real-Time Cardiac Artifact Rejection in
Magnetoencephalography', IEEE Transactions on Biomedical Engineering,
Feb. 2014 (doi:10.1109/TBME.2013.2280143).

----------------------------------------------------------------------
 How to use the OCARTA?
----------------------------------------------------------------------

from jumeg.decompose import ocarta

ocarta_obj = ocarta.JuMEG_ocarta()
ocarta_obj.fit(fn_raw)

--> for further comments we refer directly to the functions

----------------------------------------------------------------------
"""


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#              import necessary modules
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
import random
import numpy as np
try:
    from sklearn.utils.extmath import fast_dot
except ImportError:
    fast_dot = np.dot



#######################################################
#                                                     #
#              some general functions                 #
#                                                     #
#######################################################
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  function to fit the sigmoidal function to the cdf of
#                         a signal
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _fit_sigmoidal_to_cdf(ref_signal):
    """
    Fits the sigmoidal function to the cumulative density
    function (cdf) of the input data by estimating the
    parameter a0 and a0 according to:

                      1.0
             ----------------------
             1.0 + a0 * exp(-a1 * x)
    """

    # import necessary modules
    from scipy.optimize import curve_fit
    from jumeg import jumeg_math as pre_math


    # rescale signal to the range [0, 1]
    ref_signal = pre_math.rescale(ref_signal, 0, 1)

    # estimate cdf
    num_bins = np.sqrt(ref_signal.shape[0])
    x = np.linspace(0, 1, num_bins)
    counts, _ = np.histogram(ref_signal, bins=num_bins, normed=True)
    cdf = np.cumsum(counts)
    # normalize cdf
    cdf /= cdf[cdf.shape[0]-1]

    # fit sigmoidal function to normalized cdf
    opt_para, _ = curve_fit(pre_math.sigm_func, x, cdf)

    # return optimal cost_function parameter
    return opt_para



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#    function to generate epochs around a given event
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def epochs(data, idx_event, sfreq, tpre, tpost):

    # get indices of the time window around the event
    idx_pre_event  = int(tpre * sfreq)
    idx_post_event = int(tpost * sfreq)

    # define some parameter
    nsamp       = idx_post_event - idx_pre_event + 1
    if len(data.shape) == 2:
        nchan, ntsl = data.shape
    else:
        nchan       = 1
        ntsl        = len(data)
        data        = data.reshape(nchan, ntsl)


    # check if time windows are in the data range
    if hasattr(idx_event, "__len__"):
        idx_event  = idx_event[((idx_event+idx_pre_event) > 0) & ((idx_event+idx_post_event) < ntsl)]
        nevents    = idx_event.shape[0]
        bool_array = True
    else:
        nevents    = 1
        bool_array = False

    if nevents == 0:
        return -1

    # create array for new epochs
    epoch_data = np.zeros((nevents, nchan, nsamp), dtype=np.float64)

    if bool_array is False:
        epoch_data[0, :, :] = data[:, int(idx_event+idx_pre_event):int(idx_event+idx_post_event+1)]
    else:
        for i in range(nevents):
            epoch_data[i, :, :] = data[:, int(idx_event[i]+idx_pre_event):int(idx_event[i]+idx_post_event+1)]

    # return epoch data
    return epoch_data



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         OCARTA constrained ICA implementation
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ocarta_constrained_ICA(data, initial_weights=None, lrate=None, block=None, wchange=1e-16,
                           annealdeg=60., annealstep=0.9, maxsteps=200, ca_idx=None,
                           ca_cost_func=[1., 1.], oa_idx=None, oa_cost_func=[1., 1.],
                           sphering=None, oa_template=[], fixed_random_state=None):
    """
    Run the OCARTA constrained ICA decomposition on raw data

        Parameters
        ----------
        data : data array [nchan, ntsl] for decomposition
        initial_weights : initialize weights matrix
            default: None --> identity matrix is used
        lrate : initial learning rate (for most applications 1e-3 is
            a  good start)
            --> smaller learining rates will slowering the convergence
            it merely indicates the relative size of the change in weights
            default:  lrate = 0.010d/alog(nchan^2.0)
        block : his block size used to randomly extract (in time) a chop
            of data
            default:  block = floor(sqrt(ntsl/3d))
        wchange : iteration stops when weight changes is smaller then this
            number
            default: wchange = 1e-16
        annealdeg : if angle delta is larger then annealdeg (in degree) the
            learning rate will be reduce
            default:  annealdeg = 60
        annealstep : the learning rate will be reduced by this factor:
            lrate  *= annealstep
            default:  annealstep = 0.9
        maxsteps : maximum number of iterations to be done
            default:  maxsteps = 200
        ca_idx: array
            indices of the columns of the weight matrix where 'ca_cost_func'
            should be used as cost-function
        ca_cost_func : array with 2 elements a0 and a1
            cost-function for cardiac activity:
            c(x) = 1.0 / (1.0 + a0 * exp(a1 * x))
            Note: Is only used if keyword 'ca_idx' is set
            default: [1., 1.] --> sigmoidal function is used
        oa_idx: array
            indices of the columns of the weight matrix where 'oa_cost_func'
            should be used as cost-function
        oa_cost_func : array with 2 elements a0 and a1
            cost-function for ocular activity:
            c(x) = 1.0 / (1.0 + a0 * exp(a1 * x))
            Note: Is only used if keyword 'oa_idx' is set
            default: [1., 1.] --> sigmoidal function is used
        sphering : sphering matrix used to whiten the data.
        oa_template : spatial template of ocular activity. If set one column
            of the demixing matrix is updated according to the template.
            default: oa_template=None

        Returns
        -------
        weights : un-mixing matrix
        activations : underlying sources
    """

    # import necessary modules
    from scipy.linalg import pinv
    from scipy.stats.stats import pearsonr
    from jumeg import jumeg_math as pre_math
    from math import copysign as sgn
    import math


    # define some default parameter
    default_max_weight   = 1e8
    default_restart_fac  = 0.9
    default_blowup       = 1e4
    default_blowup_fac   = 0.5
    default_nsmall_angle = 20
    degconst             = 180.0 / np.pi


    # check data shape
    ntsl, npc = data.shape

    # normalize data
    # --> to prevent an overflow in exp() estimation
    norm_factor = np.max(abs(data))
    data       /= norm_factor

    if (npc < 2) or (ntsl < npc):
        raise ValueError('Data size too small!')
    npc_square = npc ** 2


    # check input parameter
    # heuristic default - may need adjustment for
    # large or tiny data sets
    if lrate == None:
        lrate = 0.01/math.log(npc ** 2.0)

    if block == None:
        block = int(math.floor(math.sqrt(ntsl/3.0)))


    # collect parameter
    nblock = ntsl / block
    lastt  = (nblock - 1) * block + 1


    # initialize training
    if initial_weights == None:
        # initialize weights as identity matrix
        weights = np.identity(npc, dtype=np.float64)
    else:
        weights = initial_weights

    BI                = block * np.identity(npc, dtype=np.float64)
    bias              = np.zeros((npc, 1), dtype=np.float64)
    onesrow           = np.ones((1, block), dtype=np.float64)
    startweights      = weights.copy()
    oldweights        = startweights.copy()
    istep             = 0
    count_small_angle = 0
    wts_blowup        = False


    # ..................................
    # trainings loop
    # ..................................
    while istep < maxsteps:

        # ..................................
        # shuffel data at each step
        # ..................................
        if fixed_random_state:
            random.seed(istep)    # --> permutation is fixed but differs at each step
        else:
            random.seed(None)

        permute = range(ntsl)
        random.shuffle(permute)


        # ..................................
        # ICA training block
        # loop across block samples
        # ..................................
        for t in xrange(0, lastt, block):
            u_ = fast_dot(data[permute[t:t + block], :], weights) + fast_dot(bias, onesrow).T

            # ..................................
            # logistic ICA weights updates
            # ..................................
            y = pre_math.sigm_func(u_)
            if ca_idx is not None:
                y[:, ca_idx] = pre_math.sigm_func(u_[:, ca_idx], ca_cost_func[0], ca_cost_func[1])
            if oa_idx is not None:
                y[:, oa_idx] = pre_math.sigm_func(u_[:, oa_idx], oa_cost_func[0], oa_cost_func[1])

            weights += lrate * fast_dot(weights, BI + fast_dot(u_.T, (1.0 - 2.0 * y)))
            bias    += (lrate * np.sum((1.0 - 2.0 * y), axis=0, dtype=np.float64)).reshape(npc, 1)

            # check change limit
            max_weight_val = np.max(np.abs(weights))
            if max_weight_val > default_max_weight:
                wts_blowup = True

            if wts_blowup:
                break

        # ..................................
        # update weights for ocular activity
        # ..................................
        if len(oa_template) and (istep+1 > (0.5 * maxsteps)):

            # ..................................
            # generate spatial maps
            # ..................................
            spatial_maps       = fast_dot(sphering.T, pinv(weights.T)).T

            # ..................................
            # estimate correlation between
            # template and spatial maps
            # ..................................
            spatial_corr       = np.zeros(npc)
            for imap in range(npc):
                spatial_corr[imap] = pearsonr(spatial_maps[imap], oa_template)[0]

            # ..................................
            # update column of weights which
            # is most similar to ocular activity
            # ..................................
            imax               = np.argmax(np.abs(spatial_corr))
            c                  = np.abs(spatial_corr[imax])
            oa_min             = np.min(spatial_maps[imax])
            oa_max             = np.max(spatial_maps[imax])
            spatial_maps[imax] = c * spatial_maps[imax] + (1. - c) * \
                                 pre_math.rescale(sgn(1., spatial_corr[imax]) * oa_template, oa_min, oa_max)

            # ..................................
            # back-transform spatial maps
            # ..................................
            weights       = pinv(fast_dot(sphering, spatial_maps.T)).T


        # ..................................
        # here we continue after the for
        # loop over the ICA training blocks
        # if weights in bounds:
        # ..................................
        if not wts_blowup:
            oldwtchange = weights - oldweights
            istep      += 1
            angledelta  = 0.0
            delta       = oldwtchange.reshape(1, npc_square)
            change      = np.sum(delta * delta) #, dtype=np.float64)

            if istep > 1:
                angledelta = math.acos(np.sum(delta * olddelta)/math.sqrt(change * oldchange)) * degconst

            # ..................................
            # anneal learning rate
            # ..................................
            oldweights = weights.copy()

            if angledelta > annealdeg:
                lrate               *= annealstep    # anneal learning rate
                olddelta             = delta         # accumulate angledelta until annealdeg reached lrates
                oldchange            = change
                count_small_angle    = 0

            else:
                if istep == 1:                    # on first step only
                    olddelta  = delta             # initialize
                    oldchange = change

                count_small_angle += 1
                if (count_small_angle > default_nsmall_angle):
                    istep = maxsteps


            # ..................................
            # apply stopping rule
            # ..................................
                if (istep > 2) and (change < wchange):
                    istep  = maxsteps
                elif change > default_blowup:
                    lrate *= default_blowup_fac

        # ..................................
        # restart if weights blow up
        # (for lowering lrate)
        # ..................................
        else:
            istep        = 0                      # start again
            wts_blowup   = 0                      # re-initialize variables
            lrate       *= default_restart_fac    # with lower learning rate
            weights      = startweights.copy()
            oldweights   = startweights.copy()
            olddelta     = np.zeros((1, npc_square), dtype=np.float64)
            bias         = np.zeros((npc, 1), dtype=np.float64)



    # ..................................
    # prepare return values
    # ..................................
    data       *= norm_factor # reverse normalization (cf. line 226)
    weights     = weights.T   # to be conform with col/row convention outside this routine
    activations = fast_dot(weights, data.T)

    # return results
    return weights, activations



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#      function to identify ICs belonging to cardiac
#                        artifacts
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def identify_cardiac_activity(activations, idx_R_peak, sfreq=1017.25, ecg_flow=8,
                              ecg_fhigh=16, order=4, tpre=-0.3, tpost=0.7,
                              thresh_kui_ca=0.4):

    """
    Function to identify independent components (ICs) belonging
    to cardiac activity. The identification is based on cross-trial-
    phase-statistics (CTPS) as introduced be Dammers et al. (2008).

        Parameters
        ----------
        activations : data array [nchan, ntsl] of underlying sources
            (ICs) as achieved by ICA
        idx_R_peak : array containing the indices of the R-peaks
        sfreq : sampling frequency
            default: sfreq=1017.25
        order: filter order
            default: 4
        ecg_flow : float
            low cut-off frequency in Hz
        ecg_fhigh : float
            high cut-off frequency in Hz
        tpre : time before R-peak (to create Epochs) in seconds
            default: tpre=-0.3
        tpost : time after R-peak (to create Epochs) in seconds
            default: tpost=0.7
        thresh_kui_ca : float
            threshold for the normalized kuiper statistic to identify
            ICs belonging to cardiac activity. Must be in the range
            between 0. and 1.

        Returns
        -------
        idx_ca : array of indices of ICs belonging to cardiac
           activity
    """

    # import necessary modules
    from mne.preprocessing.ctps_ import ctps
    from jumeg.filter import jumeg_filter

    # first filter ICs to the main frequency
    # range of cardiac activity
    act_filtered = activations.copy()
    jfi_bw_bp    = jumeg_filter(filter_method='bw', filter_type='bp', fcut1=ecg_flow,
                                fcut2=ecg_fhigh, sampling_frequency=sfreq, order=order)
    jfi_bw_bp.apply_filter(act_filtered)

    # create epochs around the R-peak
    activations_epochs = epochs(act_filtered, idx_R_peak, sfreq, tpre, tpost)

    # estimate CTPS
    _, pk_dynamics, _ = ctps(activations_epochs, is_raw=True)
    del _
    pk_values         = np.max(pk_dynamics, axis=1)
    idx_ca            = np.where(pk_values >= thresh_kui_ca)[0]

    # check that at least one and at maximum
    # three ICs belong to CA
    if len(idx_ca) == 0:
        idx_ca = [np.argmax(pk_values)]
    elif len(idx_ca) > 3:
        idx_ca = np.argsort(pk_values)[-3:]

    # return indices
    return idx_ca



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#      function to identify ICs belonging to ocular
#                        artifacts
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def identify_ocular_activity(activations, eog_signals, spatial_maps,
                             oa_template, sfreq=1017.25, order=4,
                             eog_flow=1, eog_fhigh=10, thresh_corr_oa=0.8):

    """
    Function to identify independent components (ICs) belonging
    to ocular activity. The identification is based on correlation
    analysis between the ICs and the EOG signal

        Parameters
        ----------
        activations : data array [nchan, ntsl] of underlying sources
            (ICs) as achieved by ICA
        eog_signals : data vector containing EOG-signals
        spatial_maps : maps representing the spatial orientation
            of the ICs (when performing temporal ICA the spatial
            information is stored in the columns of the mixing-matrix)
        oa_template : spatial template of ocular activity
        sfreq : sampling frequency
            default: sfreq=1017.25
        order: filter order
            default: 4
        eog_flow : float
            low cut-off frequency in Hz
        eog_fhigh : float
            high cut-off frequency in Hz
        n_jobs : nt | str
            number of jobs to run in parallel. Can be 'cuda' if
            scikits.cuda is installed properly, CUDA is initialized,
            and method='fft'.
        thresh_corr_oa : float
            threshold for the correlation statistic to identify ICs
            belonging to cardiac activity. Should be in the range
            between 0. and 1.

        Returns
        -------
        idx_oa : array of indices of ICs belonging to ocular
           activity
    """

    # import necessary modules
    from jumeg.filter import jumeg_filter
    from scipy.stats.stats import pearsonr

    fi_bp_bw = jumeg_filter(filter_method='bw', filter_type='bp', fcut1=eog_flow,
                            fcut2=eog_fhigh, sampling_frequency=sfreq, order=order)

    # first filter ICs to the main frequency
    # range of ocular activity
    act_filtered = activations.copy()
    fi_bp_bw.apply_filter(act_filtered)
    eog_filtered = eog_signals.copy()
    fi_bp_bw.apply_filter(eog_filtered)

    # estimate person correlation
    nchan, _ = activations.shape
    temp_corr = np.zeros(nchan)
    spatial_corr = np.zeros(nchan)

    for i in range(nchan):
        temp_corr[i] = np.abs(pearsonr(act_filtered[i], eog_filtered)[0])
        spatial_corr[i] = np.abs(pearsonr(spatial_maps[i], oa_template)[0])

    # check where the correlation is above threshold
    idx_oa = np.arange(nchan)[(temp_corr+spatial_corr) > thresh_corr_oa]

    # check that at least one and at maximum
    # three ICs belong to OA
    if len(idx_oa) == 0:
        idx_oa = [np.argmax((temp_corr + spatial_corr))]
    elif len(idx_oa) > 3:
        idx_oa = np.argsort((temp_corr + spatial_corr))[-3:]

    # return results
    return idx_oa





########################################################
#                                                      #
#                  JuMEG_ocarta class                  #
#                                                      #
########################################################
class JuMEG_ocarta(object):

    def __init__(self, name_ecg='ECG 001', ecg_freq=[10, 20],
                 thresh_ecg=0.3, name_eog='EOG 002', eog_freq=[1, 10],
                 seg_length=30.0, shift_length=10.0,
                 percentile_eog=80, npc=None, explVar=0.95, lrate=None,
                 maxsteps=100, flow=1.0, fhigh=20.0):
        """
        Create ocarta object from raw data file.

            Optional parameters
            -------------------
            name_ecg : string
                Name of the ECG channel.
                default: name_ecg='ECG 001'
            ecg_freq:  two elementary int | float array
                [low, high] cut-off frequency in Hz for ECG signal to identify R-peaks
                default: ecg_freq=[10,20]
            name_eog : string
                Name of the EOG channel.
                default: name_eog='EOG 002'
            eog_freq : two elementary int | float array
                [low, high] cut-off frequency in Hz for EOG signal to identify eye-blinks
                default: eog_freq=[1,10]
            seg_length : int | float
                length of the data segments to be processed (in s).
                default: seg_length=30.0
            shift_length : int | float
                length of the shift from one to another data segment (in s).
                default: shift_length=10.0
            npc : int
                The number of PCA components used after ICA recomposition. The ensuing
                attribute allows to balance noise reduction against potential loss of
                features due to dimensionality reduction.
            explVar : float | None
                Must be between 0 and 1. If float, the number of components selected
                matches the number of components with a cumulative explained variance
                of 'explVar'
                default: explVar=0.95
            lrate : initial learning rate (for most applications 1e-3 is a good start)
                --> smaller learining rates will slowering the convergence it merely
                indicates the relative size of the change in weights
                default: lrate=None
            maxsteps: maximum number of iterations to be done
                default: maxsteps=50
            flow: if set data to estimate the optimal de-mixing matrix are filtered
                prior to estimation. Note, data cleaning is applied to unfiltered
                input data
            fhigh: if set data to estimate the optimal de-mixing matrix are filtered
                prior to estimation. Note, data cleaning is applied to unfiltered
                input data


            Returns
            -------
            ocarta_obj : instance of jumeg.decompose.ocarta.JuMEG_ocarta
        """

        self._block = 0
        self._ecg_freq = ecg_freq
        self._eog_freq = eog_freq
        self._eog_signals_tkeo = None
        self._explVar = explVar
        self._idx_eye_blink = None
        self._idx_R_peak = None
        self._lrate = lrate
        self._maxsteps = maxsteps
        self._name_ecg = name_ecg
        self._name_eog = name_eog
        self._npc = npc
        self._ntsl = 0
        self._opt_cost_func_cardiac = [1.0, 1.0]
        self._opt_cost_func_ocular = [1.0, 1.0]
        self._pca = None
        self._percentile_eog = percentile_eog
        self._picks = None
        self._seg_length = seg_length
        self._shift_length = shift_length
        self._system = None
        self._template_OA = None
        self._thresh_ca = thresh_ecg
        self._thresh_eog = 0.0
        self._performance_ca = 0.0
        self._performance_oa = 0.0
        self._freq_corr_ca = 0.0
        self._freq_corr_oa = 0.0
        self._flow = flow
        self._fhigh =fhigh



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get name of the ECG-channel
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_name_ecg(self, name_ecg):
        self._name_ecg = name_ecg

    def _get_name_ecg(self):
        return self._name_ecg

    name_ecg = property(_get_name_ecg, _set_name_ecg)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get optimal frequencies to identify heart beats
    # NOTE: Array with two elements expected
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_ecg_freq(self, ecg_freq):
        if len(ecg_freq) == 2:
            self._ecg_freq = ecg_freq
        else:
            print 'NOTE: Two elementary array expected!'

    def _get_ecg_freq(self):
        return self._ecg_freq

    ecg_freq = property(_get_ecg_freq, _set_ecg_freq)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get optimal threshold to identify cardiac activity
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_thresh_ecg(self, thresh_ecg):
        if abs(thresh_ecg) < 1.0:
            self._thresh_ca = abs(thresh_ecg)
        else:
            print 'NOTE: Threshold to identify cardiac activity must be between 0 and 1!'

    def _get_thresh_ecg(self):
        return self._thresh_ca

    thresh_ecg = property(_get_thresh_ecg, _set_thresh_ecg)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get indices of R-peaks
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_idx_R_peak(self, idx_R_peak):
        self._idx_R_peak = idx_R_peak

    def _get_idx_R_peak(self):
        return self._idx_R_peak

    idx_R_peak = property(_get_idx_R_peak, _set_idx_R_peak)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get name of the EOG-channel
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_name_eog(self, name_eog):
        self._name_eog = name_eog

    def _get_name_eog(self):
        return self._name_eog

    name_eog = property(_get_name_eog, _set_name_eog)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get optimal frequencies to identify eye blinks
    # NOTE: Array with two elements expected
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_eog_freq(self, eog_freq):
        if len(eog_freq) == 2:
            self._eog_freq = eog_freq
        else:
            print 'NOTE: Two elementary array expected!'

    def _get_eog_freq(self):
        return self._eog_freq

    eog_freq = property(_get_eog_freq, _set_eog_freq)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get indices of eye-blinks
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_idx_eye_blink(self, idx_eye_blink):
        self._idx_eye_blink = idx_eye_blink

    def _get_idx_eye_blink(self):
        return self._idx_eye_blink

    idx_eye_blink = property(_get_idx_eye_blink, _set_idx_eye_blink)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get optimale cost-function for cardiac activity
    # NOTE: Array with two elements expected
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_opt_cost_func_cardiac(self, cost_func):
        self._opt_cost_func_cardiac = cost_func

    def _get_opt_cost_func_cardiac(self):
        return self._opt_cost_func_cardiac

    opt_cost_func_cardiac = property(_get_opt_cost_func_cardiac, _set_opt_cost_func_cardiac)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get optimale cost-function for ocular activity
    # NOTE: Array with two elements expected
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_opt_cost_func_ocular(self, cost_func):
        self._opt_cost_func_ocular = cost_func

    def _get_opt_cost_func_ocular(self):
        return self._opt_cost_func_ocular

    opt_cost_func_ocular = property(_get_opt_cost_func_ocular, _set_opt_cost_func_ocular)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get length of the processed data segments (in s)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_seg_length(self, seg_length):
        self._seg_length = abs(seg_length)

    def _get_seg_length(self):
        return self._seg_length

    seg_length = property(_get_seg_length, _set_seg_length)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get length of the data shift between two data
    # segments (in s)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_shift_length(self, shift_length):
        self._seg_length = abs(shift_length)

    def _get_shift_length(self):
        return self._shift_length

    shift_length = property(_get_shift_length, _set_shift_length)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get explained variance for the number of components
    # used in the ICA
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_explVar(self, explVar):
        self._explVar = abs(explVar)

    def _get_explVar(self):
        return self._explVar

    explVar = property(_get_explVar, _set_explVar)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get the number of components used in the ICA
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_npc(self, npc):
        self._npc = abs(npc)

    def _get_npc(self):
        return self._npc

    npc = property(_get_npc, _set_npc)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get learning rate in the ICA implementation
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_lrate(self, lrate):
        self._lrate = abs(lrate)

    def _get_lrate(self):
        return self._lrate

    lrate = property(_get_lrate, _set_lrate)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get number of maximal steps performed in ICA
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_maxsteps(self, maxsteps):
        self._maxsteps = abs(maxsteps)

    def _get_maxsteps(self):
        return self._maxsteps

    maxsteps = property(_get_maxsteps, _set_maxsteps)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get performance value related to cardiac activity
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_perf_rej_ca(self, perf_val):
        self._performance_ca = abs(perf_val)

    def _get_perf_rej_ca(self):
        return self._performance_ca

    performance_ca = property(_get_perf_rej_ca, _set_perf_rej_ca)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get performance value related to ocular activity
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_perf_rej_oa(self, perf_val):
        self._performance_oa = abs(perf_val)

    def _get_perf_rej_oa(self):
        return self._performance_oa

    performance_oa = property(_get_perf_rej_oa, _set_perf_rej_oa)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get frequency correlation related to cardiac activity
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_freq_corr_ca(self, freq_corr):
        self._freq_corr_ca = abs(freq_corr)

    def _get_freq_corr_ca(self):
        return self._freq_corr_ca

    freq_corr_ca = property(_get_freq_corr_ca, _set_freq_corr_ca)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get frequency correlation related to ocular activity
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_freq_corr_oa(self, freq_corr):
        self._freq_corr_oa = abs(freq_corr)

    def _get_freq_corr_oa(self):
        return self._freq_corr_oa

    freq_corr_oa = property(_get_freq_corr_oa, _set_freq_corr_oa)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get low frequency range if data should be filtered
    # prior to the estimation of the demixing matrix
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_flow(self, flow):
        self._flow = abs(flow)

    def _get_flow(self):
        return self._flow

    flow = property(_get_flow, _set_flow)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get upper frequency range if data should be
    # filtered prior to the estimation of the demixing matrix
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_fhigh(self, fhigh):
        self._fhigh = abs(fhigh)

    def _get_fhigh(self):
        return self._fhigh

    fhigh = property(_get_fhigh, _set_fhigh)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #          spatial template of ocular activity
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _get_template_oa(self):
        """
        This function returns the optimal template for ocular
        activity dependent on the used MEG system.
        """
        if self._system == 'magnesWH3600':
            oa_template = np.zeros(248)
            idx = [41, 64, 65, 66, 91, 92, 93, 94, 95, 114, 115, 116, 123, 124, 125,
                   126, 127, 146, 147, 148, 152, 153, 154, 155, 173, 174, 175, 176,
                   177, 178, 192, 193, 194, 210, 211, 212, 226, 227, 228, 246, 247]
            oa_template[idx] = [ -0.21419708, -0.22414582, -0.23823837, -0.22548739,
                                 -0.20605918, -0.27002638, -0.28440455, -0.28815480,
                                 -0.24950478, 0.22117308,  0.29407277,  0.22017770,
                                 -0.27574748, -0.41399348, -0.38132934, -0.35345995,
                                 -0.26804101,  0.31008617, 0.41633716,  0.41061879,
                                 -0.63642773, -0.50244379, -0.39267986, -0.20910069,
                                 0.45186911,  0.65563883,  0.75937563, -0.73426719,
                                 -0.51053563, -0.40412956, 0.56405808,  0.76393096,
                                 1.26573280, 0.20691632, -0.52849269, -0.33448858,
                                 0.51931741,  0.86917479, -0.26111224, 0.25098986,
                                 0.44863074]

        elif self._system == 'CTF-275':
            oa_template = np.zeros(275)
            # ToDo: define optimal template for CTF system
        else:
            # ToDo: define optimal templates for Elekta system
            oa_template = None

        return oa_template



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # create a topoplot from the template of ocular activity
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def topoplot_oa(self, info, show=False, fn_img=None):
        """
        Creates a topoplot from the template of ocular
        activity.
        """

        # import necessary modules
        import matplotlib.pyplot as plt
        from mne.viz import plot_topomap
        from mne.channels.layout import _find_topomap_coords

        pos = _find_topomap_coords(info, self._picks)

        plt.ioff()
        fig = plt.figure('topoplot ocular activity', figsize=(12, 12))
        plot_topomap(self._template_OA[self._picks], pos, res=200, contours=0)
        plt.ion()

        # if desired show the image
        if show:
             fig.show()

        # save results
        if fn_img:
            fig.savefig(fn_img + '.png', format='png')



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  calculate optimal cost-function for cardiac activity
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def calc_opt_cost_func_cardiac(self, meg_raw):
        """
        Function to estimate the optimal parameter for a sigmoidal
        based cost-function for cardiac activity. The optimization
        is based on the ECG-signals which are recorded in synchrony
        with the MEG-signals.
        """

        # check if ECG channel exist in data
        if self.name_ecg in meg_raw.ch_names:

            # import necessary modules
            from mne.preprocessing import find_ecg_events
            from mne import Epochs, set_log_level

            # set logger level to WARNING
            set_log_level('WARNING')

            # define some parameter
            event_id_ecg = 999

            # first identify R-peaks in ECG signal
            idx_R_peak, _, _ = find_ecg_events(meg_raw, ch_name=self.name_ecg,
                                               event_id=event_id_ecg, l_freq=self.ecg_freq[0],
                                               h_freq=self.ecg_freq[1], verbose=None)

            self._set_idx_R_peak(idx_R_peak)

            # generate epochs around R-peaks and average signal
            picks = [meg_raw.info['ch_names'].index(self.name_ecg)]
            ecg_epochs = Epochs(meg_raw, events=idx_R_peak, event_id=event_id_ecg,
                                tmin=-0.3, tmax=0.3, baseline=None, picks=picks,
                                verbose=None, proj=False)
            ecg_signal = np.abs(ecg_epochs.get_data().flatten())

            # estimate optimal cost-function
            cost_func = _fit_sigmoidal_to_cdf(ecg_signal)

            self._set_opt_cost_func_cardiac(cost_func)

        # if no ECG channel is found use sigmoidal function as cost-function
        else:
            print ">>>> NOTE: No ECG channel found!"
            print ">>>>       Simoidal function used as cost-function for cardiac activity!"
            self._set_opt_cost_func_cardiac([1.0, 1.0])



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   calculate optimal cost-function for ocular activity
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def calc_opt_cost_func_ocular(self, meg_raw):
        """
        Function to estimate the optimal parameter for a sigmoidal
        based cost-function for ocular activity. The optimization
        is based on the EOG-signals which are recorded in synchrony
        with the MEG-signals.
        """

        # check if EOG channel exist in data
        if self.name_eog in meg_raw.ch_names:

            # import necessary modules
            from jumeg.jumeg_math import calc_tkeo
            from mne.preprocessing import find_eog_events
            from mne import Epochs, set_log_level
            from scipy.stats import scoreatpercentile as percentile

            # set logger level to WARNING
            set_log_level('WARNING')

            # define some parameter
            event_id_eog = 998

            # first identify R-peaks in ECG signal
            idx_eye_blink = find_eog_events(meg_raw, ch_name=self.name_eog,
                                            event_id=event_id_eog, l_freq=self.eog_freq[0],
                                            h_freq=self.eog_freq[1], verbose=None)
            self._set_idx_eye_blink(idx_eye_blink)

            # generate epochs around eye blinks and average signal
            picks = [meg_raw.info['ch_names'].index(self.name_eog)]
            eog_epochs = Epochs(meg_raw, events=idx_eye_blink, event_id=event_id_eog,
                                tmin=-0.3, tmax=0.3, baseline=None, picks=picks,
                                verbose=None, proj=False)
            eog_epochs.verbose = None
            eog_signal = np.abs(eog_epochs.get_data().flatten())

            # estimate optimal cost-function
            cost_func = _fit_sigmoidal_to_cdf(eog_signal)
            self._set_opt_cost_func_ocular(cost_func)

            # perform tkeo-transformation to EOG-signals
            self._eog_signals_tkeo = np.abs(calc_tkeo(meg_raw[picks][0]))
            # estimate threshold for ocular activity
            self._thresh_eog = percentile(self._eog_signals_tkeo, self._percentile_eog)


        # if no EOG channel is found use sigmoidal function as cost-function
        else:
            print ">>>> NOTE: No EOG channel found!"
            print ">>>>       Simoidal function used as cost-function for ocular activity!"
            self._set_opt_cost_func_ocular([1.0, 1.0])



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # interface to estimate the whitening matrix as well as
    # the current weight matrix W_(i) based on the previous
    # weight matrix W_(i-1).
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_weight_matrix(self, data, initial_weights=None,
                              ca_idx=None, oa_idx=None, annealstep=0.6):
        """
        Interface to estimate the whitening matrix as well as the current
        weight matrix W_(i) based on the previous weight matrix W_(i-1).
        """

        # import necessary modules
        from ica import whitening

        # estimate PCA structure
        if self._pca is None:
            pca_data, pca = whitening(data.T, npc=self.npc, explainedVar=self.explVar)
            self._pca = pca
            self.npc = len(pca_data[0])
        else:
            # perform centering and whitening
            dmean = data.mean(axis=-1)
            stddev = np.std(data, axis=-1)
            dnorm = (data - dmean[:, np.newaxis])/stddev[:, np.newaxis]

            # generate principal components
            if self.npc is None:
                if initial_weights is None:
                    self.npc = len(dnorm)
                else:
                    self.npc = initial_weights.shape[0]

            pca_data = fast_dot(dnorm.T, self._pca.components_[:self.npc].T)

            # update mean and standard-deviation in PCA object
            self._pca.mean_ = dmean
            self._pca.stddev_ = stddev


        # estimate weight matrix
        sphering = self._pca.components_[:self.npc].copy()

        weights, activations = ocarta_constrained_ICA(pca_data, initial_weights=initial_weights,
                                                      maxsteps=self.maxsteps, lrate=self.lrate, ca_idx=ca_idx,
                                                      ca_cost_func=self.opt_cost_func_cardiac, oa_idx=oa_idx,
                                                      oa_cost_func=self.opt_cost_func_ocular, sphering=sphering,
                                                      oa_template=self._template_OA[self._picks],
                                                      annealstep=annealstep)

        # return results
        return activations, weights



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # interface for updating cleaning information, i.e.
    # estimating the un-mixing matrix and identify ICs
    # related to cardiac and ocular artifacts
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _update_cleaning_information(self, meg_raw, idx_start, idx_end,
                                     initial_weights=None, ca_idx=None, oa_idx=None,
                                     annealstep=0.6):
        """
        Interface for updating cleaning information, i.e.
        estimating the un-mixing matrix and identifying
        independent components (ICs) related to ocular or
        cardiac artifacts.
        """

        # import necessary modules
        from scipy.linalg import pinv


        # (1) estimate optimal weight matrix
        act, weights = self._update_weight_matrix(meg_raw._data[self._picks, idx_start:idx_end],
                                                  initial_weights=initial_weights,
                                                  ca_idx=ca_idx, oa_idx=oa_idx,
                                                  annealstep=annealstep)


        # (2) identification of artifact ICs
        # ------------------------------------------------------
        # (a) for cardiac activity:
        # get indices of the ICs belonging to cardiac activity
        # --> using CTPS
        idx_R_peak = self._get_idx_R_peak().copy()[:, 0]
        idx_R_peak = idx_R_peak[idx_R_peak > idx_start]
        idx_R_peak = idx_R_peak[idx_R_peak < idx_end] - idx_start
        idx_ca = identify_cardiac_activity(act.copy(), idx_R_peak, thresh_kui_ca=self._get_thresh_ecg(),
                                           sfreq=meg_raw.info['sfreq'])

        # (b) for ocular activity
        # get indices of the ICs belonging to ocular activity
        # --> using correlation with EOG signals
        if self._get_name_eog() in meg_raw.ch_names:
            self._set_npc(weights.shape[0])
            spatial_maps = fast_dot(self._pca.components_[:self._get_npc()].T, pinv(weights)).T

            # make sure that ICs already identified as being related
            # to cardiac activity are not identified again
            idx_ok = np.arange(self._get_npc())
            idx_ok = np.setdiff1d(idx_ok, idx_ca)
            eog_signals = meg_raw._data[meg_raw.info['ch_names'].index(self._get_name_eog()), idx_start:idx_end]
            idx_oa = identify_ocular_activity(act[idx_ok], eog_signals, spatial_maps[idx_ok],
                                              self._template_OA[self._picks], sfreq=meg_raw.info['sfreq'])
            idx_oa = idx_ok[idx_oa]
        else:
            idx_oa = []

        # return results
        return weights, idx_ca, idx_oa


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   perform initial training to get starting values
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _initial_training(self, meg_raw, idx_start=None, idx_end=None):

        """
        Interface for estimating OCARTA on trainings data set
        in order to get optimal initial parameter for proper
        OCARTA estimation
        """

        # import necessary modules
        from jumeg import jumeg_math as pre_math
        from math import copysign as sgn
        from scipy.linalg import pinv
        from scipy.stats.stats import pearsonr


        # estimate optimal cost-functions for cardiac
        # and ocular activity
        self.calc_opt_cost_func_cardiac(meg_raw)
        self.calc_opt_cost_func_ocular(meg_raw)

        # get optimal spatial template for ocular activity
        self._template_OA = self._get_template_oa()

        # get indices of trainings data set
        # --> keep in mind that at least one eye-blink must occur
        if (idx_start == None) or (idx_end == None):
            if self.idx_eye_blink == None:
                idx_start = 0
            else:
                idx_start = self._get_idx_eye_blink()[0, 0] - (0.5 * self._block)

            if idx_start < 0:
                idx_start = 0

            idx_end = idx_start + self._block
            if idx_end > self._ntsl:
                idx_start = self._ntsl - self._block
                idx_end   = self._ntsl


        # perform ICA on trainings data set
        self._maxsteps *= 3
        weights, idx_ca, idx_oa = self._update_cleaning_information(meg_raw, idx_start, idx_end, annealstep=0.9)
        self._maxsteps /= 3

        # update template of ocular activity
        # (to have it individual for each subject)
        if len(idx_oa) > 0:
            oa_min = np.min(self._template_OA)
            oa_max = np.max(self._template_OA)
            oa_template = self._template_OA[self._picks].copy()  # np.zeros(len(self._template_OA[self._picks]))
            spatial_maps = fast_dot(self._pca.components_[:self.npc].T, pinv(weights)).T

            # loop over all components related to ocular activity
            for ioa in range(len(idx_oa)):
                orientation = sgn(1., pearsonr(spatial_maps[idx_oa[ioa]], self._template_OA[self._picks])[0])
                oa_template += pre_math.rescale(orientation * spatial_maps[idx_oa[ioa]], oa_min, oa_max)

            self._template_OA[self._picks] = oa_template

        # return results
        return weights, idx_ca, idx_oa


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   estimate performance values
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def performance(self, meg_raw, meg_clean):

        # import necessary modules
        from jumeg.jumeg_math import calc_performance, calc_frequency_correlation
        from mne import Epochs
        from mne.preprocessing import find_ecg_events, find_eog_events

        perf_ar = np.zeros(2)
        freq_corr_ar = np.zeros(2)

        # ECG, EOG:  loop over all artifact events
        for idx_ar in range(0, 2):

            # for cardiac artifacts
            if (idx_ar == 0) and self._get_name_ecg() in meg_raw.ch_names:
                event_id = 999
                idx_event, _, _ = find_ecg_events(meg_raw, event_id,
                                                  ch_name=self._get_name_ecg(),
                                                  verbose=False)
            # for ocular artifacts
            elif self._get_name_eog() in meg_raw.ch_names:
                event_id = 998
                idx_event = find_eog_events(meg_raw, event_id,
                                            ch_name=self._get_name_eog(),
                                            verbose=False)
            else:
                event_id = 0

            if event_id:
                # generate epochs
                raw_epochs = Epochs(meg_raw, idx_event, event_id, -0.4, 0.4,
                                    picks=self._picks, baseline=(None, None), proj=False,
                                    verbose=False)
                cleaned_epochs = Epochs(meg_clean, idx_event, event_id, -0.4, 0.4,
                                        picks=self._picks, baseline=(None, None), proj=False,
                                        verbose=False)

                raw_epochs_avg = raw_epochs.average()
                cleaned_epochs_avg = cleaned_epochs.average()

                # estimate performance and frequency correlation
                perf_ar[idx_ar] = calc_performance(raw_epochs_avg, cleaned_epochs_avg)
                freq_corr_ar[idx_ar] = calc_frequency_correlation(raw_epochs_avg, cleaned_epochs_avg)

        return perf_ar, freq_corr_ar


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   clean data using OCARTA
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fit(self, fn_raw, meg_raw=None, denoising=None,
            flow=None, fhigh=None, plot_template_OA=False, verbose=True,
            name_ecg=None, ecg_freq=None, thresh_ecg=None,
            name_eog=None, eog_freq=None, seg_length=None, shift_length=None,
            npc=None, explVar=None, lrate=None, maxsteps=None,
            fn_perf_img=None):


        """
        Function to fit OCARTA to input raw data file.

            Parameters
            ----------
            fn_raw : filename of the input data. Note, data should be
                filtered prior to ICA application.


            Optional parameters
            -------------------
            meg_raw : instance of mne.io.Raw. If set 'fn_raw' is ignored and
                the data stored in meg_raw are processed
                default: meg_raw=None
            denoising : If set data are denoised, i.e. when reconstructing the
                cleaned data set only the components explaining 'denoising'
                percentage of variance are taken. Must be between 0 and 1.
                default: denoising=None
            flow: if set data to estimate the optimal de-mixing matrix are filtered
                prior to estimation. Note, data cleaning is applied to unfiltered
                input data
                default: flow=1
            fhigh: if set data to estimate the optimal de-mixing matrix are filtered
                prior to estimation. Note, data cleaning is applied to unfiltered
                input data
                default: fhigh=20
            plot_template_OA: If set a topoplot of the template for ocular activity
                is generated
                default: plot_template_OA=False
            verbose : bool, str, int, or None
                If not None, override default verbose level
                (see mne.verbose).
                default: verbose=True

            for meaning of other optional parameter see JuMEG_ocarta.__init__, where
            the ocarta object is generated.


            Returns
            -------
            meg_clean : instance of mne.io.Raw. Cleaned version of the input data
            fn_out : filename of the cleaned data. It is constructed from the
                input filename by adding the extension ',ocarta-raw.fif'
        """

        # import necessary modules
        from jumeg.jumeg_plot import plot_performance_artifact_rejection as plt_perf
        from jumeg.jumeg_utils import get_sytem_type
        from mne import pick_types, set_log_level
        from mne.io import Raw
        from scipy.linalg import pinv


        # set log level to 'WARNING'
        set_log_level('WARNING')


        # read raw data in
        if meg_raw == None:
            meg_raw = Raw(fn_raw, preload=True, verbose=False)
            fn_out = fn_raw[:fn_raw.rfind('-raw.fif')] + ',ocarta-raw.fif'
        else:
            fn_raw = meg_raw.info['filename']
            fn_out = fn_raw[:fn_raw.rfind('-raw.fif')] + ',ocarta-raw.fif'


        # extract parameter from input data
        self._system = get_sytem_type(meg_raw.info)
        self._ntsl = int(meg_raw._data.shape[1])
        self._block = int(self._seg_length * meg_raw.info['sfreq'])
        self._picks = pick_types(meg_raw.info, meg=True, eeg=False,
                                 eog=False, stim=False, exclude='bads')


        # check input parameter
        if name_ecg:
            self.name_ecg = name_ecg
        if ecg_freq:
            self.ecg_freq = ecg_freq
        if thresh_ecg:
            self.thresh_ecg = thresh_ecg
        if name_eog:
            self.name_eog = name_eog
        if eog_freq:
            self.eog_freq = eog_freq
        if seg_length:
            self.seg_length = seg_length
        if shift_length:
            self.shift_length = shift_length
        if explVar:
            self.explVar = explVar
        if npc:
            self.npc = npc
        if lrate:
            self.lrate = lrate
        if maxsteps:
            self.maxsteps = maxsteps
        if flow:
            self.flow = flow
        if fhigh:
            self.fhigh = fhigh

        # make sure that everything is initialized well
        self._eog_signals_tkeo = None
        self._idx_eye_blink = None
        self._idx_R_peak = None
        self._pca = None
        self._template_OA = None
        self._thresh_eog = 0.0
        self._performance_ca = 0.0
        self._performance_oa = 0.0
        self._freq_corr_ca = 0.0
        self._freq_corr_oa = 0.0


        meg_clean = meg_raw.copy()
        meg_filt = meg_raw.copy()
        # check if data should be filtered prior to estimate
        # the optimal demixing parameter
        if self.flow or self.fhigh:

            # import filter module
            from jumeg.filter import jumeg_filter

            # define filter type
            if not self.flow:
                filter_type = 'lp'
                self.flow = self.fhigh
                filter_info = "          --> filter parameter    : filter type=low pass %d Hz" % self.flow
            elif not self.fhigh:
                filter_type = 'hp'
                filter_info = "          --> filter parameter    : filter type=high pass %d Hz" % self.flow
            else:
                filter_type = 'bp'
                filter_info = "          --> filter parameter    : filter type=band pass %d-%d Hz" % (self.flow, self.fhigh)

            fi_mne_notch = jumeg_filter(fcut1=self.flow, fcut2=self.fhigh,
                                        filter_type=filter_type,
                                        remove_dcoffset=False,
                                        sampling_frequency=meg_raw.info['sfreq'])
            fi_mne_notch.apply_filter(meg_filt._data, picks=self._picks)

        # perform initial training
        weights, idx_ca, idx_oa = self._initial_training(meg_filt)

        # get some parameter
        nchan = self._picks.shape[0]
        shift = int(self.shift_length * meg_filt.info['sfreq'])
        nsteps = np.floor((self._ntsl - self._block)/shift) + 1
        laststep = int(shift * nsteps)

        # print out some information
        if verbose:
            print ">>>> calculating OCARTA"
            print "       --> number of channels  : %d" % nchan
            print "       --> number of timeslices: %d" % self._ntsl
            print "       --> explained variance  : %g" % self.explVar
            print "       --> number of components: %d" % weights.shape[0]
            print "       --> block size (in s)   : %d" % self.seg_length
            print "       --> number of blocks    : %d" % nsteps
            print "       --> block shift (in s)  : %d" % self.shift_length
            print "       --> maxsteps training   : %d" % (3 * self.maxsteps)
            print "       --> maxsteps cleaning   : %d" % self.maxsteps
            print "       --> costfunction CA     : a0=%g, a1=%g" % (self.opt_cost_func_cardiac[0], self.opt_cost_func_cardiac[1])
            print "       --> costfunction OA     : a0=%g, a1=%g" % (self.opt_cost_func_ocular[0], self.opt_cost_func_ocular[1])
            print ""

            if self.flow or self.fhigh:
                print ">>>> NOTE: Optimal cleaning parameter are estimated from filtered data!"
                print "           However, cleaning is performed on unfiltered input data!"
                print filter_info
                print ""


        # check if denoising is desired
        sphering = self._pca.components_.copy()
        if denoising:
            full_var = np.sum(self._pca.explained_variance_)
            exp_var_ratio = self._pca.explained_variance_ / full_var
            npc_denoising = np.sum(exp_var_ratio.cumsum() <= denoising) + 1
            sphering[npc_denoising:, :] = 0.


        # now loop over all segments
        for istep, t in enumerate(range(0, laststep, shift)):

            # print out some information
            if verbose:
                print ">>>> Step %d of %d..." % (istep+1, nsteps)

            # --------------------------------------
            # Estimating un-mixing matrix and
            # identify ICs related to artifacts
            # --------------------------------------
            idx_end = t+self._block  # get index of last element
            if (idx_end+shift+1) > self._ntsl:
                idx_end = self._ntsl

            weights, idx_ca, idx_oa = self._update_cleaning_information(meg_filt, t, idx_end,
                                                                        initial_weights=weights.T,
                                                                        ca_idx=idx_ca, oa_idx=idx_oa)

            # get cleaning matrices
            iweights = pinv(weights)
            iweights[:, idx_ca] = 0.  # remove columns related to CA
            if len(idx_oa) > 0:
                iweights[:, idx_oa] = 0.  # remove columns related to OA

            # transform data to ICA space
            dnorm = (meg_raw._data[self._picks, t:idx_end] - self._pca.mean_[:, np.newaxis]) / self._pca.stddev_[:, np.newaxis]
            pc = fast_dot(dnorm.T, sphering.T)
            activations = fast_dot(weights, pc[:, :self.npc].T)  # transform to ICA-space

            # backtransform data
            pc[:, :self.npc] = fast_dot(iweights, activations).T                         # back-transform to PCA-space
            meg_clean._data[self._picks, t:idx_end] = fast_dot(pc, sphering).T * self._pca.stddev_[:, np.newaxis] + \
                                                      self._pca.mean_[:, np.newaxis]     # back-transform to sensor-space


        # write out some additional information
        if verbose:
            print ""
            print ">>>> cleaning done!"
            print ">>>> generate and save result files/images."



        # generate filenames for output files/images
        basename = fn_raw[:-8]
        if not fn_perf_img:
            fn_perf_img = basename + ',ocarta-performance'

        fn_topo = fn_perf_img[:fn_perf_img.rfind(',')] + ',ocarta_topoplot_oa'

        fn_out = basename + ',ocarta-raw.fif'


        # save cleaned data
        meg_clean.save(fn_out, overwrite=True, verbose=False)

        # generate topoplot image
        if plot_template_OA:
            self.topoplot_oa(meg_raw.info, fn_img=fn_topo)

        # generate performance image
        plt_perf(meg_raw, None, fn_perf_img, meg_clean=meg_clean)

        # estimate performance values/frequency correlation
        perf_ar, freq_corr_ar = self.performance(meg_raw, meg_clean)

        self.performance_ca = perf_ar[0]
        self.performance_oa = perf_ar[1]
        self.freq_corr_ca = freq_corr_ar[0]
        self.freq_corr_oa = freq_corr_ar[1]


        return meg_clean, fn_out




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   to simplify the call of the JuMEG_ocarta() help
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
ocarta = JuMEG_ocarta()

