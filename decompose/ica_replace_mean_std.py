import numpy as np

from mne.preprocessing import ICA as ICA_ORIG
from mne.preprocessing.ica import _check_start_stop, _check_for_unsupported_ica_channels
from mne.channels.channels import _contains_ch_type
from mne.io.base import BaseRaw
from mne.epochs import BaseEpochs
from mne.utils import _reject_data_segments, check_version
from mne.io.pick import pick_types, pick_info, _pick_data_channels, _DATA_CH_TYPES_SPLIT


def ica_update_mean_std(inst, ica, picks=None, start=None, stop=None, decim=None, reject=None,
                        flat=None, tstep=2.0, reject_by_annotation=True):
    """
    Returns the modified ica object after replacing pca_mean and _pre_whitener.

    This is necessary because in mne.preprocessing.ICA the standard deviation
    (_pre_whitener) of the MEG channels is calculated incorrectly. THE SAME PARAMETERS
    SHOULD BE USED AS THE ONES USED FOR ica.fit().

    Use case for unfiltered data:
    Sometimes it is best to fit ICA on filtered and cleaned data in order to improve
    the quality of decomposition. The fitted ICA object cannot be applied directly to
    the unfiltered data due to changes in the mean and standard deviation between the
    filtered and unfiltered data. This function can be used in this case. It computes
    the correct mean and std values from given unfiltered data and includes it in the
    ica object.

    Parameters
    ----------
    inst : instance of Raw or Epochs.
        The data to be processed. The instance is modified inplace.
    ica : mne.preprocessing.ica
        The ica object used for processing of the data.
    picks : array-like of int
        Channels to be included for the calculation of pca_mean_ and _pre_whitener.
        This selection SHOULD BE THE SAME AS the one used in ica.fit().
    start : int | float | None
        First sample to include. If float, data will be interpreted as
        time in seconds. If None, data will be used from the first sample.
    stop : int | float | None
        Last sample to not include. If float, data will be interpreted as
        time in seconds. If None, data will be used to the last sample.
    decim : int | None
        Increment for selecting each nth time slice. This parameter SHOULD BE THE
        SAME AS the one used in ica.fit(). If None, all samples within ``start`` and
        ``stop`` are used.
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
    flat : dict | None
        Rejection parameters based on flatness of signal.
        This parameter SHOULD BE THE SAME AS the one used in ica.fit().
        Valid keys are 'grad', 'mag', 'eeg', 'seeg', 'ecog', 'eog', 'ecg',
        'hbo', 'hbr'.
        Values are floats that set the minimum acceptable peak-to-peak
        amplitude. If flat is None then no rejection is done.
        It only applies if `inst` is of type Raw.
    tstep : float
        Length of data chunks for artifact rejection in seconds.
        This parameter SHOULD BE THE SAME AS the one used in ica.fit().
        It only applies if `inst` is of type Raw.
    reject_by_annotation : bool
        Whether to omit bad segments from the data before fitting.
        This parameter SHOULD BE THE SAME AS the one used in ica.fit().
        If True, annotated segments with a description that starts with 'bad' are
        omitted. Has no effect if ``inst`` is an Epochs or Evoked object.
        Defaults to True.

        .. versionadded:: mne v0.14.0

    Returns
    -------
    out : instance of mne.preprocessing.ICA
        The modified ica object with pca_mean_ and _pre_whitener replaced.
    """

    ica = ica.copy()

    pre_whitener = None

    if isinstance(inst, (BaseRaw, BaseEpochs)):
        _check_for_unsupported_ica_channels(picks, inst.info)
        if isinstance(inst, BaseRaw):
            pca_mean_, pre_whitener = get_pca_mean_and_pre_whitener_raw(inst, picks, start, stop, decim, reject, flat,
                                                                        tstep, pre_whitener, reject_by_annotation)
        elif isinstance(inst, BaseEpochs):
            pca_mean_, pre_whitener = get_pca_mean_and_pre_whitener_epochs(inst, picks, decim, pre_whitener)
    else:
        raise ValueError('Data input must be of Raw or Epochs type')

    ica.pca_mean_ = pca_mean_
    ica._pre_whitener = pre_whitener

    return ica


def apply_ica_replace_mean_std(inst, ica, picks=None, include=None, exclude=None, n_pca_components=None, start=None,
                               stop=None, decim=None, reject=None, flat=None, tstep=2.0, replace_pre_whitener=True,
                               reject_by_annotation=True):

    """
    Use instead of ica.apply().

    Returns the cleaned input data after calculating pca_mean_ and _pre_whitener based
    on the input data. This is necessary because in mne.preprocessing.ICA the standard
    deviation (_pre_whitener) of the MEG channels is calculated incorrectly.

    THE INPUT DATA AS WELL AS THE ICA OBJECT ARE MODIFIED IN PLACE. THE SAME PARAMETERS
    SHOULD BE USED AS THE ONES FOR ica.fit().

    Use case for unfiltered data:
    Sometimes it is best to fit ICA on filtered and cleaned data in order to improve
    the quality of decomposition. The fitted ICA object cannot be applied directly to
    the unfiltered data due to changes in the mean and standard deviation between the
    filtered and unfiltered data. This function can be used in this case. It computes
    the correct mean and std values from given unfiltered data and includes it in the
    ica object.

    Parameters
    ----------
    inst : instance of Raw or Epochs.
        The data to be processed. The instance is modified inplace.
    ica : mne.preprocessing.ica
        The ica object used for processing of the data.
    picks : array-like of int
        Channels to be included for the calculation of pca_mean_ and _pre_whitener.
        This selection SHOULD BE THE SAME AS the one used in ica.fit().
    include : array_like of int.
        The indices referring to columns in the ummixing matrix. The
        components to be kept.
    exclude : array_like of int.
        The indices referring to columns in the ummixing matrix. The
        components to be zeroed out.
    n_pca_components : int | float | None
        The number of PCA components to be kept, either absolute (int)
        or percentage of the explained variance (float). If None (default),
        all PCA components will be used.
    start : int | float | None
        First sample to include. If float, data will be interpreted as
        time in seconds. If None, data will be used from the first sample.
    stop : int | float | None
        Last sample to not include. If float, data will be interpreted as
        time in seconds. If None, data will be used to the last sample.
    decim : int | None
        Increment for selecting each nth time slice. This parameter SHOULD BE THE
        SAME AS the one used in ica.fit(). If None, all samples within ``start`` and
        ``stop`` are used.
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
    flat : dict | None
        Rejection parameters based on flatness of signal.
        This parameter SHOULD BE THE SAME AS the one used in ica.fit().
        Valid keys are 'grad', 'mag', 'eeg', 'seeg', 'ecog', 'eog', 'ecg',
        'hbo', 'hbr'.
        Values are floats that set the minimum acceptable peak-to-peak
        amplitude. If flat is None then no rejection is done.
        It only applies if `inst` is of type Raw.
    tstep : float
        Length of data chunks for artifact rejection in seconds.
        This parameter SHOULD BE THE SAME AS the one used in ica.fit().
        It only applies if `inst` is of type Raw.
    replace_pre_whitener : bool
        If True, pre_whitener is replaced otherwise the original pre_whitener is used.
    reject_by_annotation : bool
        Whether to omit bad segments from the data before fitting.
        This parameter SHOULD BE THE SAME AS the one used in ica.fit().
        If True, annotated segments with a description that starts with 'bad' are
        omitted. Has no effect if ``inst`` is an Epochs or Evoked object.
        Defaults to True.

        .. versionadded:: mne v0.14.0

    Returns
    -------
    out : instance of Raw or Epochs
        The cleaned data.
    """

    if replace_pre_whitener:
        pre_whitener = None
    else:
        pre_whitener = ica._pre_whitener

    if isinstance(inst, (BaseRaw, BaseEpochs)):
        _check_for_unsupported_ica_channels(picks, inst.info)
        if isinstance(inst, BaseRaw):
            pca_mean_, pre_whitener = get_pca_mean_and_pre_whitener_raw(inst, picks, start, stop, decim, reject, flat,
                                                                        tstep, pre_whitener, reject_by_annotation)
        elif isinstance(inst, BaseEpochs):
            pca_mean_, pre_whitener = get_pca_mean_and_pre_whitener_epochs(inst, picks, decim, pre_whitener)
    else:
        raise ValueError('Data input must be of Raw or Epochs type')

    ica.pca_mean_ = pca_mean_

    if replace_pre_whitener:
        ica._pre_whitener = pre_whitener

    return ica.apply(inst, include=include, exclude=exclude, n_pca_components=n_pca_components, start=start, stop=stop)


def get_pca_mean_and_pre_whitener_raw(raw, picks, start, stop, decim, reject, flat, tstep,
                                      pre_whitener, reject_by_annotation):
    """Aux method based on ica._fit_raw from mne v0.15"""

    if picks is None:  # just use good data channels
        picks = _pick_data_channels(raw.info, exclude='bads', with_ref_meg=False)

    info = pick_info(raw.info, picks)
    if info['comps']:
        info['comps'] = []

    start, stop = _check_start_stop(raw, start, stop)

    reject_by_annotation = 'omit' if reject_by_annotation else None
    # this will be a copy
    data = raw.get_data(picks, start, stop, reject_by_annotation)

    # this will be a view
    if decim is not None:
        data = data[:, ::decim]

    # this will make a copy
    if (reject is not None) or (flat is not None):
        data, drop_inds_ = _reject_data_segments(data, reject, flat, decim, info, tstep)
    # this may operate inplace or make a copy
    data, pre_whitener = pre_whiten(data, raw.info, picks, pre_whitener)

    pca_mean_ = np.mean(data, axis=1)

    return pca_mean_, pre_whitener


def get_pca_mean_and_pre_whitener_epochs(epochs, picks, decim, pre_whitener):

    """Aux method based on ica._fit_epochs from mne v0.15"""

    if picks is None:
        picks = _pick_data_channels(epochs.info, exclude='bads', with_ref_meg=False)

    # filter out all the channels the raw wouldn't have initialized
    info = pick_info(epochs.info, picks)
    if info['comps']:
        info['comps'] = []

    # this should be a copy (picks a list of int)
    data = epochs.get_data()[:, picks]
    # this will be a view
    if decim is not None:
        data = data[:, :, ::decim]

    # This will make at least one copy (one from hstack, maybe one more from _pre_whiten)
    data, pre_whitener = pre_whiten(np.hstack(data), epochs.info, picks, pre_whitener)

    pca_mean_ = np.mean(data, axis=1)

    return pca_mean_, pre_whitener


def pre_whiten(data, info, picks, pre_whitener=None):
    """Aux function based on ica._pre_whiten from mne v0.15

       pre_whitener[this_picks] = np.std(data[this_picks], axis=1)[:, None]
    """

    if pre_whitener is None:
        # use standardization as whitener
        # Scale (z-score) the data by channel type
        info = pick_info(info, picks)

        pre_whitener = np.empty([len(data), 1])
        for ch_type in _DATA_CH_TYPES_SPLIT + ['eog']:
            if _contains_ch_type(info, ch_type):
                if ch_type == 'seeg':
                    this_picks = pick_types(info, meg=False, seeg=True)
                elif ch_type == 'ecog':
                    this_picks = pick_types(info, meg=False, ecog=True)
                elif ch_type == 'eeg':
                    this_picks = pick_types(info, meg=False, eeg=True)
                elif ch_type in ('mag', 'grad'):
                    this_picks = pick_types(info, meg=ch_type)
                elif ch_type == 'eog':
                    this_picks = pick_types(info, meg=False, eog=True)
                elif ch_type in ('hbo', 'hbr'):
                    this_picks = pick_types(info, meg=False, fnirs=ch_type)
                else:
                    raise RuntimeError('Should not be reached.'
                                       'Unsupported channel {0}'
                                       .format(ch_type))
                pre_whitener[this_picks] = np.std(data[this_picks], axis=1)[:, None]

    data /= pre_whitener

    return data, pre_whitener


class ICA(ICA_ORIG):
    """
    This is based on mne.preprocessing.ICA with the only difference being
    that the standard deviation in _pre_whiten() is calculated on a per
    channel basis instead of per channel type.

    M/EEG signal decomposition using Independent Component Analysis (ICA).
    This object can be used to estimate ICA components and then
    remove some from Raw or Epochs for data exploration or artifact
    correction.
    Caveat! If supplying a noise covariance keep track of the projections
    available in the cov or in the raw object. For example, if you are
    interested in EOG or ECG artifacts, EOG and ECG projections should be
    temporally removed before fitting the ICA. You can say::

        >> projs, raw.info['projs'] = raw.info['projs'], []
        >> ica.fit(raw)
        >> raw.info['projs'] = projs

    .. note:: Methods implemented are FastICA (default), Infomax and
              Extended-Infomax. Infomax can be quite sensitive to differences
              in floating point arithmetic due to exponential non-linearity.
              Extended-Infomax seems to be more stable in this respect
              enhancing reproducibility and stability of results.

    .. warning:: ICA is sensitive to low-frequency drifts and therefore
                 requires the data to be high-pass filtered prior to fitting.
                 Typically, a cutoff frequency of 1 Hz is recommended. Note
                 that FIR filters prior to MNE 0.15 used the ``'firwin2'``
                 design method, which generally produces rather shallow filters
                 that might not work for ICA processing. Therefore, it is
                 recommended to use IIR filters for MNE up to 0.14. In MNE
                 0.15, FIR filters can be designed with the ``'firwin'``
                 method, which generally produces much steeper filters. This
                 method will be the default FIR design method in MNE 0.16. In
                 MNE 0.15, you need to explicitly set ``fir_design='firwin'``
                 to use this method. This is the recommended filter method for
                 ICA preprocessing.
    Parameters
    ----------
    n_components : int | float | None
        The number of components used for ICA decomposition. If int, it must be
        smaller then max_pca_components. If None, all PCA components will be
        used. If float between 0 and 1 components will be selected by the
        cumulative percentage of explained variance.
    max_pca_components : int | None
        The number of components used for PCA decomposition. If None, no
        dimension reduction will be applied and max_pca_components will equal
        the number of channels supplied on decomposing data. Defaults to None.
    n_pca_components : int | float
        The number of PCA components used after ICA recomposition. The ensuing
        attribute allows to balance noise reduction against potential loss of
        features due to dimensionality reduction. If greater than
        ``self.n_components_``, the next ``n_pca_components`` minus
        ``n_components_`` PCA components will be added before restoring the
        sensor space data. The attribute gets updated each time the according
        parameter for in .pick_sources_raw or .pick_sources_epochs is changed.
        If float, the number of components selected matches the number of
        components with a cumulative explained variance below
        `n_pca_components`.
    noise_cov : None | instance of mne.cov.Covariance
        Noise covariance used for whitening. If None, channels are just
        z-scored.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the FastICA estimation.
        As the estimation is non-deterministic it can be useful to
        fix the seed to have reproducible results. Defaults to None.
    method : {'fastica', 'infomax', 'extended-infomax'}
        The ICA method to use. Defaults to 'fastica'.
    fit_params : dict | None.
        Additional parameters passed to the ICA estimator chosen by `method`.
    max_iter : int, optional
        Maximum number of iterations during fit.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Attributes
    ----------
    current_fit : str
        Flag informing about which data type (raw or epochs) was used for
        the fit.
    ch_names : list-like
        Channel names resulting from initial picking.
        The number of components used for ICA decomposition.
    ``n_components_`` : int
        If fit, the actual number of components used for ICA decomposition.
    n_pca_components : int
        See above.
    max_pca_components : int
        The number of components used for PCA dimensionality reduction.
    verbose : bool, str, int, or None
        See above.
    ``pca_components_`` : ndarray
        If fit, the PCA components
    ``pca_mean_`` : ndarray
        If fit, the mean vector used to center the data before doing the PCA.
    ``pca_explained_variance_`` : ndarray
        If fit, the variance explained by each PCA component
    ``mixing_matrix_`` : ndarray
        If fit, the mixing matrix to restore observed data, else None.
    ``unmixing_matrix_`` : ndarray
        If fit, the matrix to unmix observed data, else None.
    exclude : list
        List of sources indices to exclude, i.e. artifact components identified
        throughout the ICA solution. Indices added to this list, will be
        dispatched to the .pick_sources methods. Source indices passed to
        the .pick_sources method via the 'exclude' argument are added to the
        .exclude attribute. When saving the ICA also the indices are restored.
        Hence, artifact components once identified don't have to be added
        again. To dump this 'artifact memory' say: ica.exclude = []
    info : None | instance of Info
        The measurement info copied from the object fitted.
    ``n_samples_`` : int
        the number of samples used on fit.
    ``labels_`` : dict
        A dictionary of independent component indices, grouped by types of
        independent components. This attribute is set by some of the artifact
        detection functions.
    """
    def __init__(self, n_components=None, max_pca_components=None,
                 n_pca_components=None, noise_cov=None, random_state=None,
                 method='fastica', fit_params=None, max_iter=200,
                 verbose=None):

        # check if version of mne is at most 0.17.0
        if not check_version('mne', '0.17.0'):
            print ""
            print ""
            print "jumeg.ica_replace_mean_std.ICA has only been tested with"
            print "mne-python up to version 0.17.0. Your Version of mne-python"
            print "is more recent."
            print "Please check if any arguments for initializing the ICA"
            print "object changed and implement these changes for the call"
            print "to the super class below. Furthermore, check if substan-"
            print "tial changes have been made to ICA._pre_whiten() and"
            print "implement these changes while making sure that the stan-"
            print "dard deviation is being calculated on a per-channel basis."
            print ""

            raise EnvironmentError("ICA has not been tested with your version of mne-python.")

        super(ICA, self).__init__(n_components=n_components, max_pca_components=max_pca_components,
                                  n_pca_components=n_pca_components, noise_cov=noise_cov,
                                  random_state=random_state, method=method, fit_params=fit_params,
                                  max_iter=max_iter, verbose=verbose)

    def _pre_whiten(self, data, info, picks):
        """Aux function."""
        has_pre_whitener = hasattr(self, 'pre_whitener_')
        if not has_pre_whitener and self.noise_cov is None:
            # use standardization as whitener
            # Scale (z-score) the data by channel
            info = pick_info(info, picks)
            pre_whitener = np.empty([len(data), 1])
            for ch_type in _DATA_CH_TYPES_SPLIT + ['eog']:
                if _contains_ch_type(info, ch_type):
                    if ch_type == 'seeg':
                        this_picks = pick_types(info, meg=False, seeg=True)
                    elif ch_type == 'ecog':
                        this_picks = pick_types(info, meg=False, ecog=True)
                    elif ch_type == 'eeg':
                        this_picks = pick_types(info, meg=False, eeg=True)
                    elif ch_type in ('mag', 'grad'):
                        this_picks = pick_types(info, meg=ch_type)
                    elif ch_type == 'eog':
                        this_picks = pick_types(info, meg=False, eog=True)
                    elif ch_type in ('hbo', 'hbr'):
                        this_picks = pick_types(info, meg=False, fnirs=ch_type)
                    else:
                        raise RuntimeError('Should not be reached.'
                                           'Unsupported channel {0}'
                                           .format(ch_type))

                    pre_whitener[this_picks] = np.std(data[this_picks], axis=1)[:, None]
            data /= pre_whitener
        elif not has_pre_whitener and self.noise_cov is not None:

            from mne.cov import compute_whitener

            pre_whitener, _ = compute_whitener(self.noise_cov, info, picks)
            assert data.shape[0] == pre_whitener.shape[1]
            data = np.dot(pre_whitener, data)
        elif has_pre_whitener and self.noise_cov is None:
            data /= self.pre_whitener_
            pre_whitener = self.pre_whitener_
        else:
            data = np.dot(self.pre_whitener_, data)
            pre_whitener = self.pre_whitener_

        return data, pre_whitener


def read_ica(fname, verbose=None):
    """
    Restore ICA solution from fif file.

    Parameters:
    ----------:
    fname : str
        Absolute path to fif file containing ICA matrices.
        The file name should end with -ica.fif or -ica.fif.gz.

    Returns:
    -------:
    ica : instance of ICA
        The ICA estimator.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    """

    from mne.utils import logger, check_fname
    from mne.io.open import fiff_open
    from mne.io.meas_info import read_meas_info
    from mne.io.tree import dir_tree_find
    from mne.io.constants import FIFF
    from mne.io.tag import read_tag
    from mne.preprocessing.ica import _deserialize
    from mne import Covariance
    from scipy import linalg
    from mne.fixes import _get_args

    check_fname(fname, 'ICA', ('-ica.fif', '-ica.fif.gz',
                               '_ica.fif', '_ica.fif.gz'))

    logger.info('Reading %s ...' % fname)
    fid, tree, _ = fiff_open(fname)

    try:
        # we used to store bads that weren't part of the info...
        info, meas = read_meas_info(fid, tree, clean_bads=True)
    except ValueError:
        logger.info('Could not find the measurement info. \n'
                    'Functionality requiring the info won\'t be'
                    ' available.')
        info = None

    ica_data = dir_tree_find(tree, FIFF.FIFFB_MNE_ICA)
    if len(ica_data) == 0:
        ica_data = dir_tree_find(tree, 123)  # Constant 123 Used before v 0.11
        if len(ica_data) == 0:
            fid.close()
            raise ValueError('Could not find ICA data')

    my_ica_data = ica_data[0]
    for d in my_ica_data['directory']:
        kind = d.kind
        pos = d.pos
        if kind == FIFF.FIFF_MNE_ICA_INTERFACE_PARAMS:
            tag = read_tag(fid, pos)
            ica_init = tag.data
        elif kind == FIFF.FIFF_MNE_ROW_NAMES:
            tag = read_tag(fid, pos)
            ch_names = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_WHITENER:
            tag = read_tag(fid, pos)
            pre_whitener = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_COMPONENTS:
            tag = read_tag(fid, pos)
            pca_components = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_EXPLAINED_VAR:
            tag = read_tag(fid, pos)
            pca_explained_variance = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_PCA_MEAN:
            tag = read_tag(fid, pos)
            pca_mean = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_MATRIX:
            tag = read_tag(fid, pos)
            unmixing_matrix = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_BADS:
            tag = read_tag(fid, pos)
            exclude = tag.data
        elif kind == FIFF.FIFF_MNE_ICA_MISC_PARAMS:
            tag = read_tag(fid, pos)
            ica_misc = tag.data

    fid.close()

    ica_init, ica_misc = [_deserialize(k) for k in (ica_init, ica_misc)]
    current_fit = ica_init.pop('current_fit')
    if ica_init['noise_cov'] == Covariance.__name__:
        logger.info('Reading whitener drawn from noise covariance ...')

    logger.info('Now restoring ICA solution ...')

    # make sure dtypes are np.float64 to satisfy fast_dot
    def f(x):
        return x.astype(np.float64)

    ica_init = dict((k, v) for k, v in ica_init.items()
                    if k in _get_args(ICA.__init__))
    ica = ICA(**ica_init)
    ica.current_fit = current_fit
    ica.ch_names = ch_names.split(':')
    ica.pre_whitener_ = f(pre_whitener)
    ica.pca_mean_ = f(pca_mean)
    ica.pca_components_ = f(pca_components)
    ica.n_components_ = unmixing_matrix.shape[0]
    ica._update_ica_names()
    ica.pca_explained_variance_ = f(pca_explained_variance)
    ica.unmixing_matrix_ = f(unmixing_matrix)
    ica.mixing_matrix_ = linalg.pinv(ica.unmixing_matrix_)
    ica.exclude = [] if exclude is None else list(exclude)
    ica.info = info
    if 'n_samples_' in ica_misc:
        ica.n_samples_ = ica_misc['n_samples_']
    if 'labels_' in ica_misc:
        labels_ = ica_misc['labels_']
        if labels_ is not None:
            ica.labels_ = labels_
    if 'method' in ica_misc:
        ica.method = ica_misc['method']

    logger.info('Ready.')

    return ica
