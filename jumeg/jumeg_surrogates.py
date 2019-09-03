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
from sklearn.utils import check_random_state

from mne.utils import logger
from mne.epochs import BaseEpochs
from mne.source_estimate import SourceEstimate


def check_power_spectrum(orig, surr):
    '''
    Check if power spectrum is conserved up to small numerical deviations.
    '''
    assert orig.shape == surr.shape, 'Shape mismatch.'
    # the first sample point and the Nyquist central point of the FFT always
    # differ between the real and the surrogate data
    check_length = int(orig.shape[-1] / 2 - 1)
    orig_ps = np.round(np.abs(np.fft.fft(orig, axis=1))[:, 1:check_length],
                       decimals=3)
    surr_ps = np.round(np.abs(np.fft.fft(surr, axis=1))[:, 1:check_length],
                       decimals=3)
    assert np.array_equal(orig_ps, surr_ps), 'Power spectrum not conserved.'
    print('Surrogates OK.')


class Surrogates(object):
    '''
    The Surrogates class.
    '''
    def __init__(self, inst, picks=None):
        '''
        Initialize the Surrogates object.

        #TODO Update documentation.
        '''
        from mne.io.pick import _pick_data_channels

        # flags
        self._normalized = False
        self._fft_cached = False

        # cache
        self._original_data_fft = None
        self.instance = None

        if not isinstance(inst, (BaseEpochs, SourceEstimate, np.ndarray)):
            raise ValueError('Must be an instance of ndarray, Epochs or'
                             'SourceEstimate. Got type {0}'.format(type(inst)))

        if isinstance(inst, BaseEpochs):
            # load the data if not loaded
            if not inst.preload:
                inst.load_data()
            # make sure right picks are taken
            if picks is None:
                picks = _pick_data_channels(inst.info, with_ref_meg=False)
            # returns array of shape (n_epochs, n_channels, n_times)
            self.original_data = inst.get_data()[:, picks, :]
            # cache the instance
            self.instance = inst.copy()

        elif isinstance(inst, SourceEstimate):  # SourceEstimate
            # array of shape (n_dipoles, n_times)
            self.original_data = inst.data
            # cache the instance
            self.instance = inst.copy()

        else:  # must be ndarray
            self.original_data = inst
            self.instance = inst.copy()

    def clear_cache(self):
        '''Clean up cache.'''
        try:
            del self._original_data_fft
        except AttributeError:
            pass

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

        for i in range(6):
            ts[i, :] = np.sin(np.arange(200)*np.pi/15. + i*np.pi/2.) + \
                np.sin(np.arange(200) * np.pi / 30.)

        return Surrogates(inst=ts)

    @staticmethod
    def shuffle_time_points(original_data, seed=None):
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
        rng = check_random_state(None)  # for parallel processing => re-initialized

        shuffled_data = original_data.copy()

        for i in range(shuffled_data.shape[0]):
            rng.shuffle(shuffled_data[i, :])

        return shuffled_data

    @staticmethod
    def shift_data(original_data, min_shift=0,
                   max_shift=None, seed=None):
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
        rng = check_random_state(None)  # for parallel processing => re-initialized

        shifted_data = original_data.copy()
        # limit shifts to the number of samples in last dimension
        if max_shift is None:
            max_shift = original_data.shape[-1]

        # shift array contacts maximum and minimum number of shifts
        assert (min_shift < max_shift) & (min_shift >= 0),\
            'min_shift is not less than max_shift'
        shift = rng.permutation(np.arange(min_shift, max_shift))

        for itrial in range(shifted_data.shape[0]):
            # random shift is picked from the range of min max values
            shifted_data[itrial, :] = np.roll(shifted_data[itrial, :],
                                              rng.choice(shift))

        return shifted_data

    @staticmethod
    def randomize_phase_scot(original_data, seed=None):
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
        data_freq = np.fft.rfft(np.asarray(original_data))
        rng = check_random_state(seed)  # preferably always None
        # include random phases between 0 and 2 * pi
        data_freq = (np.abs(data_freq) *
                     np.exp(1j * rng.random_sample(data_freq.shape) *
                     2 * np.pi))
        # compute the ifft and return the real part
        return np.real(np.fft.irfft(data_freq, original_data.shape[-1]))

    def randomize_phase(self, original_data, seed=None):
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
            # the last axis is to be used (axis=-1)
            surrogates = np.fft.rfft(original_data, axis=-1)
            self._original_data_fft = surrogates
            self._fft_cached = True

        rng = check_random_state(seed)
        #  get shapes
        n_times = original_data.shape[-1]  # last dimension is time

        #  Generate random phases uniformly distributed in the
        #  interval [0, 2*Pi]
        phases = rng.uniform(low=0, high=2*np.pi, size=(surrogates.shape))

        #  Add random phases uniformly distributed in the interval [0, 2*Pi]
        surrogates *= np.exp(1j * phases)

        #  Calculate IFFT and take the real part, the remaining imaginary part
        #  is due to numerical errors.
        return np.ascontiguousarray(np.real(np.fft.irfft(surrogates, n=n_times,
                                                         axis=-1)))

    def _generate_surrogates(self, mode='randomize_phase', n_surr=1,
                             seed=None, min_shift=0, max_shift=None):
        '''
        Private function to compute surrogates and return a generator.
        '''
        # do this n_surr times
        for i in range(n_surr):
            print('computing surrogate %d ' % i)

            if mode == 'shuffle':
                surrogate_data = self.shuffle_time_points(self.original_data,
                                                          seed=seed)
            elif mode == 'time_shift':
                surrogate_data = self.shift_data(self.original_data,
                                                 min_shift=min_shift,
                                                 max_shift=max_shift, seed=seed)
            elif mode == 'randomize_phase':
                surrogate_data = self.randomize_phase(self.original_data,
                                                      seed=seed)
            else:
                raise RuntimeError('Unknown mode provided, should be one of'
                                   'shuffle, time_shift or randomize_phase')

            assert self.original_data.shape == surrogate_data.shape,\
                ('Error: Shape mismatch after surrogate computation !')

            # now the surrogate is stored in surrogate_data
            if isinstance(self.instance, BaseEpochs):
                self.instance._data = surrogate_data
                yield self.instance
            elif isinstance(self.instance, SourceEstimate):
                # create a new instance with the new data
                new_instance = SourceEstimate(surrogate_data,
                                              self.instance.vertices,
                                              tmin=self.instance.tmin,
                                              tstep=self.instance.tstep,
                                              subject=self.instance.subject)
                yield new_instance
            elif isinstance(self.instance, np.ndarray):
                yield surrogate_data
            else:
                raise RuntimeError('Unknown instance.')

            del surrogate_data

    def compute_surrogates(self, mode='randomize_phase', n_surr=1,
                           seed=None, min_shift=0, max_shift=None,
                           return_generator=False):
        '''
        Compute the surrogates using given method and return the surrogate
        instance.

        n_surr: number of surrogates to compute.
        return_generator: If set, a generator will be returned instead of the data.
        mode: the type of surrogates to compute. One of shuffle, time_shift,
              or phase randomize
        min_shift and max_shift are required only when doing time_shift
        '''
        my_surrogate = self._generate_surrogates(mode=mode, n_surr=n_surr,
                                                 seed=seed, min_shift=min_shift,
                                                 max_shift=max_shift)

        if isinstance(self.instance, BaseEpochs):
            print(RuntimeWarning('WARNING: Currently surrogates on Epochs '
                                 'only returns a generator.'))
            return_generator = True

        if return_generator:  # simply return the generator
            return my_surrogate
        else:
            # make a list if generator is not required
            my_surrogate_list = list(my_surrogate)
            if isinstance(self.instance, np.ndarray):
                # if ndarray, return an array instead of a list
                my_surrogate_list = np.array(my_surrogate_list)
            return my_surrogate_list
