import operator
import numpy as np
from scipy import linalg

from mne.preprocessing import ICA as ICA_ORIG
from mne.preprocessing.ica import _deserialize, _band_pass_filter, _find_sources, _check_for_unsupported_ica_channels
from mne.channels.channels import _contains_ch_type
from mne.cov import compute_whitener
from mne.io.base import BaseRaw
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.utils import logger, check_fname, _reject_data_segments
from mne.io.pick import pick_types, pick_info, _pick_data_channels, _DATA_CH_TYPES_SPLIT
from mne.io.tag import read_tag
from mne.io.open import fiff_open
from mne.io.meas_info import read_meas_info
from mne.io.tree import dir_tree_find
from mne.io.constants import FIFF
from mne import Covariance
from mne.fixes import _get_args


def apply_ica_on_unfiltered(inst, ica, picks=None, include=None, exclude=None, n_pca_components=None, start=None,
                            stop=None, decim=None, reject=None, flat=None, tstep=2.0, replace_pre_whitener=True,
                            reject_by_annotation=True):

    """
    First, calculates pca_mean_ and _pre_whitener based on the unfiltered input
    data (inst) and modifies those in the ica object in place. The SAME PARAMETERS
    SHOULD BE USED AS the ones used for ica.fit() with filtered data.
    Then the altered ica object is applied to the unfiltered input data (inst).

    This method ensures that the correct mean and standard deviation are used for
    whitening the input data.

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
        The processed data.
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
    """ M/EEG signal decomposition using Independent Component Analysis (ICA).

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

        super(ICA, self).__init__(n_components=n_components, max_pca_components=max_pca_components,
                                  n_pca_components=n_pca_components, noise_cov=noise_cov,
                                  random_state=random_state, method=method, fit_params=fit_params,
                                  max_iter=max_iter, verbose=verbose)

        self.pca_mean_unfilt_ = None

    def _reset(self):
        """Aux method."""
        del self._pre_whitener
        del self._pre_whitener_unfilt
        del self.unmixing_matrix_
        del self.mixing_matrix_
        del self.n_components_
        del self.n_samples_
        del self.pca_components_
        del self.pca_explained_variance_
        del self.pca_mean_
        del self.pca_mean_unfilt_
        if hasattr(self, 'drop_inds_'):
            del self.drop_inds_

    def set_pca_mean_and_pre_whitener(self, inst, picks=None, start=None, stop=None, decim=None,
                                      reject=None, flat=None, tstep=2.0, reject_by_annotation=True,
                                      verbose=None):
        """
        Calculates pca_mean_unfilt_ based on the data. Use the same input as
        was used for ica.fit. Change inst only!!!
        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Raw measurements to be decomposed.
        picks : array-like of int
            Channels to be included. This selection remains throughout the
            initialized ICA solution. If None only good data channels are used.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
        decim : int | None
            Increment for selecting each nth time slice. If None, all samples
            within ``start`` and ``stop`` are used.
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
        flat : dict | None
            Rejection parameters based on flatness of signal.
            Valid keys are 'grad', 'mag', 'eeg', 'seeg', 'ecog', 'eog', 'ecg',
            'hbo', 'hbr'.
            Values are floats that set the minimum acceptable peak-to-peak
            amplitude. If flat is None then no rejection is done.
            It only applies if `inst` is of type Raw.
        tstep : float
            Length of data chunks for artifact rejection in seconds.
            It only applies if `inst` is of type Raw.
        reject_by_annotation : bool
            Whether to omit bad segments from the data before fitting. If True,
            annotated segments with a description that starts with 'bad' are
            omitted. Has no effect if ``inst`` is an Epochs or Evoked object.
            Defaults to True.

            .. versionadded:: 0.14.0

        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.
        Returns
        -------
        None
        """

        if isinstance(inst, (BaseRaw, BaseEpochs)):
            _check_for_unsupported_ica_channels(picks, inst.info)
            if isinstance(inst, BaseRaw):
                self._set_pca_mean_and_pre_whitener_raw(inst, picks, start, stop, decim, reject, flat,
                                                        tstep, reject_by_annotation, verbose)
            elif isinstance(inst, BaseEpochs):
                self._set_pca_mean_and_pre_whitener_epochs(inst, picks, decim, verbose)
        else:
            raise ValueError('Data input must be of Raw or Epochs type')

    def _set_pca_mean_and_pre_whitener_raw(self, raw, picks, start, stop, decim, reject, flat, tstep,
                          reject_by_annotation, verbose):
        """Aux method based on ica._fit_raw"""
        if self.current_fit != 'unfitted':
            self._reset()

        if picks is None:  # just use good data channels
            picks = _pick_data_channels(raw.info, exclude='bads',
                                        with_ref_meg=False)

        logger.info('Calculating pca_mean_unfilt_ using %i channels.' % len(picks))

        self.info = pick_info(raw.info, picks)
        if self.info['comps']:
            self.info['comps'] = []
        self.ch_names = self.info['ch_names']
        start, stop = _check_start_stop(raw, start, stop)

        reject_by_annotation = 'omit' if reject_by_annotation else None
        # this will be a copy
        data = raw.get_data(picks, start, stop, reject_by_annotation)

        # this will be a view
        if decim is not None:
            data = data[:, ::decim]

        # this will make a copy
        if (reject is not None) or (flat is not None):
            data, self.drop_inds_ = _reject_data_segments(data, reject, flat,
                                                          decim, self.info,
                                                          tstep)
        self.n_samples_ = data.shape[1]
        # this may operate inplace or make a copy
        data, self._pre_whitener = self._pre_whiten(data, raw.info, picks)

        self.pca_mean_ = np.mean(data, axis=1)

    def _set_pca_mean_and_pre_whitener_epochs(self, epochs, picks, decim, verbose):

        """Aux method based on _fit_epochs"""
        if self.current_fit != 'unfitted':
            self._reset()

        if picks is None:
            picks = _pick_data_channels(epochs.info, exclude='bads',
                                        with_ref_meg=False)

        logger.info('Calculating pca_mean_unfilt_ using %i channels.' % len(picks))

        # filter out all the channels the raw wouldn't have initialized
        self.info = pick_info(epochs.info, picks)
        if self.info['comps']:
            self.info['comps'] = []
        self.ch_names = self.info['ch_names']

        # this should be a copy (picks a list of int)
        data = epochs.get_data()[:, picks]
        # this will be a view
        if decim is not None:
            data = data[:, :, ::decim]

        self.n_samples_ = np.prod(data[:, 0, :].shape)

        # This will make at least one copy (one from hstack, maybe one more from _pre_whiten)
        data, self._pre_whitener = self._pre_whiten(np.hstack(data), epochs.info, picks)

        self.pca_mean_ = np.mean(data, axis=1)

    def _pre_whiten_unfilt(self, data, info, picks):
        """Aux function."""
        has_pre_whitener = hasattr(self, '_pre_whitener_unfilt')
        if not has_pre_whitener and self.noise_cov is None:
            # use standardization as whitener
            # Scale (z-score) the data by channel type
            info = pick_info(info, picks)
            pre_whitener_unfilt = np.empty([len(data), 1])
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
                    pre_whitener_unfilt[this_picks] = np.std(data[this_picks])
            data /= pre_whitener_unfilt
        elif not has_pre_whitener and self.noise_cov is not None:
            pre_whitener_unfilt, _ = compute_whitener(self.noise_cov, info, picks)
            assert data.shape[0] == pre_whitener_unfilt.shape[1]
            data = np.dot(pre_whitener_unfilt, data)
        elif has_pre_whitener and self.noise_cov is None:
            data /= self._pre_whitener_unfilt
            pre_whitener_unfilt = self._pre_whitener_unfilt
        else:
            data = np.dot(self._pre_whitener_unfilt, data)
            pre_whitener_unfilt = self._pre_whitener_unfilt

        return data, pre_whitener_unfilt

    def apply_on_unfilt(self, inst, include=None, exclude=None, n_pca_components=None, start=None, stop=None):
        """Remove selected components from the signal.

        Given the unmixing matrix, transform data,
        zero out components, and inverse transform the data.
        This procedure will reconstruct M/EEG signals from which
        the dynamics described by the excluded components is subtracted.
        The data is processed in place.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            The data to be processed. The instance is modified inplace.
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

        Returns
        -------
        out : instance of Raw, Epochs or Evoked
            The processed data.
        """

        if isinstance(inst, BaseRaw):
            out = self._apply_raw_unfilt(raw=inst, include=include,
                                         exclude=exclude,
                                         n_pca_components=n_pca_components,
                                         start=start, stop=stop)
        elif isinstance(inst, BaseEpochs):
            out = self._apply_epochs_unfilt(epochs=inst, include=include,
                                            exclude=exclude,
                                            n_pca_components=n_pca_components)
        elif isinstance(inst, Evoked):
            out = self._apply_evoked_unfilt(evoked=inst, include=include,
                                            exclude=exclude,
                                            n_pca_components=n_pca_components)
        else:
            raise ValueError('Data input must be of Raw, Epochs or Evoked '
                             'type')
        return out

    def _apply_raw_unfilt(self, raw, include, exclude, n_pca_components, start, stop):
        """Aux method."""
        if not raw.preload:
            raise ValueError('Raw data must be preloaded to apply ICA')

        if exclude is None:
            exclude = list(set(self.exclude))
        else:
            exclude = list(set(self.exclude + exclude))

        if n_pca_components is not None:
            self.n_pca_components = n_pca_components

        start, stop = _check_start_stop(raw, start, stop)

        picks = pick_types(raw.info, meg=False, include=self.ch_names,
                           exclude='bads', ref_meg=False)

        data = raw[picks, start:stop][0]
        data, self._pre_whitener_unfilt = self._pre_whiten_unfilt(data, raw.info, picks)

        data = self._pick_sources_unfilt(data, include, exclude)

        raw[picks, start:stop] = data
        return raw

    def _apply_epochs_unfilt(self, epochs, include, exclude, n_pca_components):
        """Aux method."""
        if not epochs.preload:
            raise ValueError('Epochs must be preloaded to apply ICA')

        picks = pick_types(epochs.info, meg=False, ref_meg=False,
                           include=self.ch_names,
                           exclude='bads')

        # special case where epochs come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Epochs don\'t match fitted data: %i channels '
                               'fitted but %i channels supplied. \nPlease '
                               'provide Epochs compatible with '
                               'ica.ch_names' % (len(self.ch_names),
                                                 len(picks)))

        if n_pca_components is not None:
            self.n_pca_components = n_pca_components

        data = np.hstack(epochs.get_data()[:, picks])
        data, self._pre_whitener_unfilt = self._pre_whiten_unfilt(data, epochs.info, picks)
        data = self._pick_sources_unfilt(data, include=include,
                                         exclude=exclude)

        # restore epochs, channels, tsl order
        epochs._data[:, picks] = np.array(np.split(data,
                                          len(epochs.events), 1))
        epochs.preload = True

        return epochs

    def _apply_evoked_unfilt(self, evoked, include, exclude, n_pca_components):
        """Aux method."""

        picks = pick_types(evoked.info, meg=False, ref_meg=False,
                           include=self.ch_names,
                           exclude='bads')

        # special case where evoked come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Evoked does not match fitted data: %i channels'
                               ' fitted but %i channels supplied. \nPlease '
                               'provide an Evoked object that\'s compatible '
                               'with ica.ch_names' % (len(self.ch_names),
                                                      len(picks)))

        if n_pca_components is not None:
            self.n_pca_components = n_pca_components

        data = evoked.data[picks]
        data, self._pre_whitener_unfilt = self._pre_whiten_unfilt(data, evoked.info, picks)
        data = self._pick_sources_unfilt(data, include=include,
                                         exclude=exclude)

        # restore evoked
        evoked.data[picks] = data

        return evoked

    def _pick_sources_unfilt(self, data, include, exclude):
        """Aux function."""
        if exclude is None:
            exclude = self.exclude
        else:
            exclude = list(set(self.exclude + list(exclude)))

        _n_pca_comp = self._check_n_pca_components(self.n_pca_components)

        if not(self.n_components_ <= _n_pca_comp <= self.max_pca_components):
            raise ValueError('n_pca_components must be >= '
                             'n_components and <= max_pca_components.')

        n_components = self.n_components_
        logger.info('Transforming to ICA space (%i components)' % n_components)

        # zero mean
        self.pca_mean_unfilt_ = np.mean(data, axis=1)
        data -= self.pca_mean_unfilt_[:, None]

        sel_keep = np.arange(n_components)
        if include not in (None, []):
            sel_keep = np.unique(include)
        elif exclude not in (None, []):
            sel_keep = np.setdiff1d(np.arange(n_components), exclude)

        logger.info('Zeroing out %i ICA components'
                    % (n_components - len(sel_keep)))

        unmixing = np.eye(_n_pca_comp)
        unmixing[:n_components, :n_components] = self.unmixing_matrix_
        unmixing = np.dot(unmixing, self.pca_components_[:_n_pca_comp])

        mixing = np.eye(_n_pca_comp)
        mixing[:n_components, :n_components] = self.mixing_matrix_
        mixing = np.dot(self.pca_components_[:_n_pca_comp].T, mixing)

        if _n_pca_comp > n_components:
            sel_keep = np.concatenate(
                (sel_keep, range(n_components, _n_pca_comp)))

        proj_mat = np.dot(mixing[:, sel_keep], unmixing[sel_keep, :])

        data = np.dot(proj_mat, data)

        # add mean back to data
        data += self.pca_mean_unfilt_[:, None]

        # restore scaling
        if self.noise_cov is None:  # revert standardization
            data *= self._pre_whitener_unfilt
        else:
            data = np.dot(linalg.pinv(self._pre_whitener_unfilt, cond=1e-14), data)

        return data

    def score_sources_unfilt(self, inst, target=None, score_func='pearsonr',
                             start=None, stop=None, l_freq=None, h_freq=None,
                             reject_by_annotation=True, verbose=None):
        """Assign score to components based on statistic or metric.

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            The object to reconstruct the sources from.
        target : array-like | ch_name | None
            Signal to which the sources shall be compared. It has to be of
            the same shape as the sources. If some string is supplied, a
            routine will try to find a matching channel. If None, a score
            function expecting only one input-array argument must be used,
            for instance, scipy.stats.skew (default).
        score_func : callable | str label
            Callable taking as arguments either two input arrays
            (e.g. Pearson correlation) or one input
            array (e. g. skewness) and returns a float. For convenience the
            most common score_funcs are available via string labels:
            Currently, all distance metrics from scipy.spatial and All
            functions from scipy.stats taking compatible input arguments are
            supported. These function have been modified to support iteration
            over the rows of a 2D array.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
        l_freq : float
            Low pass frequency.
        h_freq : float
            High pass frequency.
        reject_by_annotation : bool
            If True, data annotated as bad will be omitted. Defaults to True.

            .. versionadded:: 0.14.0

        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more). Defaults to self.verbose.

        Returns
        -------
        scores : ndarray
            scores for each source as returned from score_func
        """
        if isinstance(inst, BaseRaw):
            sources = self._transform_raw_unfilt(inst, start, stop,
                                                 reject_by_annotation)
        elif isinstance(inst, BaseEpochs):
            sources = self._transform_epochs_unfilt(inst, concatenate=True)
        elif isinstance(inst, Evoked):
            sources = self._transform_evoked_unfilt(inst)
        else:
            raise ValueError('Input must be of Raw, Epochs or Evoked type')

        if target is not None:  # we can have univariate metrics without target
            target = self._check_target(target, inst, start, stop,
                                        reject_by_annotation)

            if sources.shape[-1] != target.shape[-1]:
                raise ValueError('Sources and target do not have the same'
                                 'number of time slices.')
            # auto target selection
            if verbose is None:
                verbose = self.verbose
            if isinstance(inst, BaseRaw):
                sources, target = _band_pass_filter(self, sources, target,
                                                    l_freq, h_freq, verbose)

        scores = _find_sources(sources, target, score_func)

        return scores

    def _transform_unfilt(self, data):
        """Compute sources from data (operates inplace)."""
        if self.pca_mean_unfilt_ is not None:
            data -= self.pca_mean_unfilt_[:, None]
        # Apply first PCA
        pca_data = np.dot(self.pca_components_[:self.n_components_], data)
        # Apply unmixing to low dimension PCA
        sources = np.dot(self.unmixing_matrix_, pca_data)
        return sources

    def _transform_raw_unfilt(self, raw, start, stop, reject_by_annotation=False):
        """Transform raw data."""
        if not hasattr(self, 'mixing_matrix_'):
            raise RuntimeError('No fit available. Please fit ICA.')
        start, stop = _check_start_stop(raw, start, stop)

        picks = pick_types(raw.info, include=self.ch_names, exclude='bads',
                           meg=False, ref_meg=False)
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Raw doesn\'t match fitted data: %i channels '
                               'fitted but %i channels supplied. \nPlease '
                               'provide Raw compatible with '
                               'ica.ch_names' % (len(self.ch_names),
                                                 len(picks)))

        if reject_by_annotation:
            data = raw.get_data(picks, start, stop, 'omit')
        else:
            data = raw[picks, start:stop][0]

        data, self._pre_whitener_unfilt = self._pre_whiten_unfilt(data, raw.info, picks)
        self.pca_mean_unfilt_ = np.mean(data, axis=1)
        return self._transform_unfilt(data)

    def _transform_epochs_unfilt(self, epochs, concatenate):
        """Aux method."""
        if not hasattr(self, 'mixing_matrix_'):
            raise RuntimeError('No fit available. Please fit ICA')

        picks = pick_types(epochs.info, include=self.ch_names, exclude='bads',
                           meg=False, ref_meg=False)
        # special case where epochs come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Epochs don\'t match fitted data: %i channels '
                               'fitted but %i channels supplied. \nPlease '
                               'provide Epochs compatible with '
                               'ica.ch_names' % (len(self.ch_names),
                                                 len(picks)))

        data = np.hstack(epochs.get_data()[:, picks])
        data, self._pre_whitener_unfilt = self._pre_whiten_unfilt(data, epochs.info, picks)
        self.pca_mean_unfilt_ = np.mean(data, axis=1)
        sources = self._transform_unfilt(data)

        if not concatenate:
            # Put the data back in 3D
            sources = np.array(np.split(sources, len(epochs.events), 1))

        return sources

    def _transform_evoked_unfilt(self, evoked):
        """Aux method."""
        if not hasattr(self, 'mixing_matrix_'):
            raise RuntimeError('No fit available. Please first fit ICA')

        picks = pick_types(evoked.info, include=self.ch_names, exclude='bads',
                           meg=False, ref_meg=False)

        if len(picks) != len(self.ch_names):
            raise RuntimeError('Evoked doesn\'t match fitted data: %i channels'
                               ' fitted but %i channels supplied. \nPlease '
                               'provide Evoked compatible with '
                               'ica.ch_names' % (len(self.ch_names),
                                                 len(picks)))

        data, self._pre_whitener_unfilt = self._pre_whiten_unfilt(evoked.data[picks], evoked.info, picks)
        self.pca_mean_unfilt_ = np.mean(data, axis=1)
        sources = self._transform_unfilt(data)

        return sources


    def get_sources_unfilt(self, inst, add_channels=None, start=None, stop=None):
        """Estimate sources given the unmixing matrix.

        This method will return the sources in the container format passed.
        Typical usecases:

        1. pass Raw object to use `raw.plot` for ICA sources
        2. pass Epochs object to compute trial-based statistics in ICA space
        3. pass Evoked object to investigate time-locking in ICA space

        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Object to compute sources from and to represent sources in.
        add_channels : None | list of str
            Additional channels  to be added. Useful to e.g. compare sources
            with some reference. Defaults to None
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, the entire data will be used.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, the entire data will be used.

        Returns
        -------
        sources : instance of Raw, Epochs or Evoked
            The ICA sources time series.
        """
        if isinstance(inst, BaseRaw):
            sources = self._sources_as_raw_unfilt(inst, add_channels, start, stop)
        elif isinstance(inst, BaseEpochs):
            sources = self._sources_as_epochs_unfilt(inst, add_channels, False)
        elif isinstance(inst, Evoked):
            sources = self._sources_as_evoked_unfilt(inst, add_channels)
        else:
            raise ValueError('Data input must be of Raw, Epochs or Evoked '
                             'type')

        return sources

    def _sources_as_raw_unfilt(self, raw, add_channels, start, stop):
        """Aux method."""
        # merge copied instance and picked data with sources
        sources = self._transform_raw_unfilt(raw, start=start, stop=stop)
        if raw.preload:  # get data and temporarily delete
            data = raw._data
            del raw._data

        out = raw.copy()  # copy and reappend
        if raw.preload:
            raw._data = data

        # populate copied raw.
        start, stop = _check_start_stop(raw, start, stop)
        if add_channels is not None:
            raw_picked = raw.copy().pick_channels(add_channels)
            data_, times_ = raw_picked[:, start:stop]
            data_ = np.r_[sources, data_]
        else:
            data_ = sources
            _, times_ = raw[0, start:stop]
        out._data = data_
        out._times = times_
        out._filenames = [None]
        out.preload = True

        # update first and last samples
        out._first_samps = np.array([raw.first_samp +
                                     (start if start else 0)])
        out._last_samps = np.array([out.first_samp + stop
                                    if stop else raw.last_samp])

        out._projector = None
        self._export_info(out.info, raw, add_channels)
        out._update_times()

        return out

    def _sources_as_epochs_unfilt(self, epochs, add_channels, concatenate):
        """Aux method."""
        out = epochs.copy()
        sources = self._transform_epochs_unfilt(epochs, concatenate)
        if add_channels is not None:
            picks = [epochs.ch_names.index(k) for k in add_channels]
        else:
            picks = []
        out._data = np.concatenate([sources, epochs.get_data()[:, picks]],
                                   axis=1) if len(picks) > 0 else sources

        self._export_info(out.info, epochs, add_channels)
        out.preload = True
        out._raw = None
        out._projector = None

        return out

    def _sources_as_evoked_unfilt(self, evoked, add_channels):
        """Aux method."""
        if add_channels is not None:
            picks = [evoked.ch_names.index(k) for k in add_channels]
        else:
            picks = []

        sources = self._transform_evoked_unfilt(evoked)
        if len(picks) > 1:
            data = np.r_[sources, evoked.data[picks]]
        else:
            data = sources
        out = evoked.copy()
        out.data = data
        self._export_info(out.info, evoked, add_channels)

        return out

# ------------------------------------------------------------------


def _check_start_stop(raw, start, stop):
    """Aux function."""
    out = list()
    for st in (start, stop):
        if st is None:
            out.append(st)
        else:
            try:
                out.append(_ensure_int(st))
            except TypeError:  # not int-like
                out.append(raw.time_as_index(st)[0])
    return out


def _ensure_int(x, name='unknown', must_be='an int'):
    """Ensure a variable is an integer."""
    # This is preferred over numbers.Integral, see:
    # https://github.com/scipy/scipy/pull/7351#issuecomment-299713159
    try:
        x = int(operator.index(x))
    except TypeError:
        raise TypeError('%s must be %s, got %s' % (name, must_be, type(x)))
    return x


def read_ica(fname):
    """Restore ICA solution from fif file.

    Parameters
    ----------
    fname : str
        Absolute path to fif file containing ICA matrices.
        The file name should end with -ica.fif or -ica.fif.gz.

    Returns
    -------
    ica : instance of ICA
        The ICA estimator.
    """
    check_fname(fname, 'ICA', ('-ica.fif', '-ica.fif.gz'))

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
    ica._pre_whitener = f(pre_whitener)
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
