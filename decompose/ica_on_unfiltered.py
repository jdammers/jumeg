import numpy as np

from mne.preprocessing.ica import _check_start_stop, _check_for_unsupported_ica_channels
from mne.channels.channels import _contains_ch_type
from mne.io.base import BaseRaw
from mne.epochs import BaseEpochs
from mne.utils import _reject_data_segments
from mne.io.pick import pick_types, pick_info, _pick_data_channels, _DATA_CH_TYPES_SPLIT


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