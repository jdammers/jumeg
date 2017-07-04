'''
Tools to generate surrogates.

# Heavily inspired from the pyunicorn package:
# URL: <http://www.pik-potsdam.de/members/donges/software>
# Reference: J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge,
# Q.-Y. Feng, L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan,
# H.A. Dijkstra, and J. Kurths, Unified functional network and
# nonlinear time series analysis for complex systems science:
# The pyunicorn package, Chaos 25, 113101 (2015), doi:10.1063/1.4934554,
# Preprint: arxiv.org:1507.01571 [physics.data-an].

'''

# Authors: Praveen Sripad (pravsripad@gmail.com)
# License: BSD (3-clause)

import numpy as np
from numpy import random
import scipy as sci
from scipy.signal import welch
from mne.utils import logger
from sklearn.utils import check_random_state

#     ps1 = np.abs(np.fft.fft(surr))**2
#     ps2 = np.abs(np.fft.fft(epochs.get_data()))**2
#     assert np.aray_equal(ps1, ps2), 'The power content does not match. Error.'


class Surrogates(object):
    '''
    The Surrogates class.
    '''
    def __init__(self, inst, picks=None):
        '''
        Initialize the Surrogates object.
        '''
        from mne.io.base import BaseRaw
        from mne.epochs import BaseEpochs
        from mne.source_estimate import SourceEstimate
        from mne.io.pick import _pick_data_channels

        # flags
        self._normalized = False
        self._fft_cached = False

        # cache
        self._original_data_fft = None

        if not isinstance(inst, (BaseEpochs, BaseRaw, SourceEstimate, np.ndarray)):
            raise ValueError('Must be an instance of ndarray, Epochs, Raw or'
                             'SourceEstimate. Got type {0}'.format(type(inst)))

        # make sure right picks are taken
        if picks is None and not isinstance(inst, (np.ndarray, SourceEstimate)):
            picks = _pick_data_channels(inst.info, with_ref_meg=False)
            print len(picks)

        # load the data if not loaded
        if isinstance(inst, (BaseEpochs, BaseRaw)):
            if not inst.preload:
                inst.load_data()

        if isinstance(inst, BaseRaw):
            self.original_data = inst.get_data(picks)
            # returns ndarray, shape (n_channels, n_times)
        elif isinstance(inst, BaseEpochs):
            self.original_data = inst.get_data()[:, picks, :]
            # returns array of shape (n_epochs, n_channels, n_times)
        elif isinstance(inst, SourceEstimate):  # SourceEstimate
            self.original_data = inst.data
            # array of shape (n_dipoles, n_times)
        else:  # must be ndarray
            self.original_data = inst

    @staticmethod
    def SimpleTestData():
        """
        Return Surrogates instance representing test a data set of 6 time
        series.

        :rtype: Surrogates instance
        :return: a Surrogates instance for testing purposes.
        """
        #  Create time series
        ts = np.zeros((6, 200))

        for i in xrange(6):
            ts[i, :] = np.sin(np.arange(200)*np.pi/15. + i*np.pi/2.) + \
                np.sin(np.arange(200) * np.pi / 30.)

        return Surrogates(inst=ts)

    @staticmethod
    def shuffle_time_points(data, axis=0, seed=None):
        '''
        Shuffling the time points of any data array. The probabiity density
        of the data samples is preserved.
        WARNING: This function simply reorders the time points and does not
        perform shuffling of the phases.

        Parameters
        ----------
        data : 2d ndarray of dimension [ntrials x nsamples]

        Returns
        -------
        shuffled_data : shuffled (time points only) trials
        '''
        np.random.seed(seed=None)   # for parallel processing => re-initialized
        shuffled_data = data.copy()

        # if shuffled_data.ndim == 1:
        #     shuffled_data = shuffled_data.reshape((1, -1))

        for i in xrange(shuffled_data.shape[0]):
            np.random.shuffle(shuffled_data[i, :])

        return shuffled_data

    @staticmethod
    def shift_data(data, min_shift=0, max_shift=None, seed=None):
        '''
        Shifting the time points of any data array.
        The probability density of the data samples are preserved.
        WARNING: This function simply shifts the time points and does not
        perform shuffling of the phases in the frequency domain.

        Parameters
        ----------
        data : 2d ndarray of dimension [ntrials x nsamples]
               In each trial samples are randomly shifted

        Returns
        -------
        shifted_data : Time shifted trials.
        '''

        np.random.seed(seed=None)  # for parallel processing => re-initialized
        shifted_data = data.copy()
        # limit shifts to the number of samples in last dimension
        if max_shift is None:
            max_shift = data.shape[-1]

        # shift array contacts maximum and minimum number of shifts
        assert (min_shift < max_shift) & (min_shift >= 0),\
            'min_shift is not less than max_shift'
        shift = np.random.permutation(np.arange(min_shift, max_shift))

        for itrial in xrange(shifted_data.shape[0]):
            # random shift is picked from the range of min max values
            shifted_data[itrial, :] = np.roll(shifted_data[itrial, :],
                                              np.random.choice(shift))

        return shifted_data

    @staticmethod
    def randomize_phase_scot(data, seed=None):
        '''
        Phase randomization. This function randomizes the input array's spectral
        phase along the given dimension.

        Parameters
        ----------
        data: array_like
            2D input array. The phase will be randomized for the last dimension.

        Returns
        -------
        phase_shuffled_data: ndarray
            Array of same shape as `data`.

        Notes
        -----
        The algorithm randomizes the phase component of the input's complex fourier transform.

        Function obtained from scot (https://github.com/scot-dev/scot)
        Used to randomize the phase values of a signal.
        '''
        data_freq = np.fft.rfft(np.asarray(data), axis=0)
        rng = check_random_state(None)  # preferably always None
        # include random phases between 0 and 2 * pi
        data_freq = (np.abs(data_freq) *
                     np.exp(1j * rng.random_sample(data_freq.shape) * 2 * np.pi))
        # compute the ifft and return the real part
        return np.real(np.fft.irfft(data_freq, data.shape[axis], axis=0))

    @staticmethod
    def randomize_phase(data):
        '''
        Surrogates by applying Fourier transform, randomizing the phases and
        computing the inverse fourier transform. The power spectrum of the data
        and the surrogates will be preserved.
        '''
        #  Calculate FFT of original_data time series
        #  The FFT of the original_data data has to be calculated only once,
        #  so it is stored in self._original_data_fft.
        if self._fft_cached:
            surrogates = self._original_data_fft
        else:
            surrogates = np.fft.rfft(original_data, axis=1)
            self._original_data_fft = surrogates
            self._fft_cached = True

        #  Get shapes
        (N, n_time) = original_data.shape
        len_phase = surrogates.shape[1]

        #  Generate random phases uniformly distributed in the
        #  interval [0, 2*Pi]
        phases = random.uniform(low=0, high=2 * np.pi, size=(N, len_phase))

        #  Add random phases uniformly distributed in the interval [0, 2*Pi]
        surrogates *= np.exp(1j * phases)

        #  Calculate IFFT and take the real part, the remaining imaginary part
        #  is due to numerical errors.
        return np.ascontiguousarray(np.real(np.fft.irfft(surrogates, n=n_time,
                                                         axis=1)))
