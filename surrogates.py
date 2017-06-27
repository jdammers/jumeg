'''
Tools to generate surrogates.
'''

# Authors: Jurgen Dammers (j.dammers@fz-juelich.de)
#          Praveen Sripad (pravsripad@gmail.com)
#
# License: BSD (3-clause)

import numpy as np
import scipy as sci
from scipy.signal import welch
from mne.utils import logger
from sklearn.utils import check_random_state


def randomize_phase(data, axis=0):
    '''
    Phase randomization. This function randomizes the input array's spectral
    phase along the given dimension.

    Parameters
    ----------
    data: array_like
        Input array.
    axis: int
        Axis along which to operate.

    Returns
    -------
    out: ndarray
        Array of same shape as `data`.

    Notes
    -----
    The algorithm randomizes the phase component of the input's complex fourier transform.

    Function obtained from scot (https://github.com/scot-dev/scot)
    Used to randomize the phase values of a signal.
    '''
    data_freq = np.fft.rfft(np.asarray(data), axis=axis)
    rng = check_random_state(None)  # preferably always None
    data_freq = np.abs(data_freq) * np.exp(1j * rng.random_sample(data_freq.shape) * 2 * np.pi)
    surr = np.fft.irfft(data_freq, data.shape[axis], axis=axis)
    # check if psd of data and surr are the same TODO
    # assert np.array_equal(welch(data), welch(surr)), 'Surrogate computation failed.'
    return


def do_surrogates(inst):
    '''
    Perform surrogate computation based on the given mode on the data in the
    given instance.
    '''
    # if epochs
    if isinstance(inst, mne.Epochs):
        if not inst.preload:
            inst.load_data()
        surrogates = inst.copy()
        surr = np.zeros(inst.get_data.shape())
        for i, epoch in enumerate(surrogates):
            surr[i] = randomize_phase(epoch, axis=1)
        surrogates._data = surr
        assert not np.allclose(epochs._data, surrogates._data),\
            'Surrogates incorrectly computed !'
        return surr
