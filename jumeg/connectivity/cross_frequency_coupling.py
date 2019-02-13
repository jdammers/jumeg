#!/usr/bin/env python

# All the CFC relatd matlab code converted to python
# Translated from matlab code by Mark Kramer (math.bu.edu/people/mak/MA666/)

import numpy as np
from scipy.signal import hilbert
import mne
from mne.filter import filter_data


def filter_and_make_analytic_signal(data, sfreq, l_phase_freq, h_phase_freq,
                                    l_amp_freq, h_amp_freq, method='fft',
                                    n_jobs=1):
    """ Filter data to required range and compute analytic signal from it.

    Parameters
    ----------
    data : ndarray
        The signal to be analysed.
    l_phase_freq, h_phase_freq : float
        Low and high phase modulating frequencies.
    l_amp_freq, h_amp_freq : float
        Low and high amplitude modulated frequencies.
    method : 'fft' or 'iir'
        Filter method to be used. (mne.filter.filter_data)
    n_jobs : int
        Number of parallel jobs to run.

    Returns
    -------
    theta : ndarray
        Low frequency filtered signal (modulating)
    gamma : ndarray
        High frequency filtered signal (modulated)
    phase : ndarray
        Phase of low frequency signal above.
    amp : ndarray
        Amplitude envelope of the high freq. signal above.
    """
    # filter theta and gamma signals
    n_jobs = 4
    method = 'fft'
    l_phase_freq, h_phase_freq, l_amp_freq, h_amp_freq = 6, 10, 60, 150
    
    theta = filter_data(data, sfreq, l_phase_freq, h_phase_freq,
                        n_jobs=njobs, method=method)
    
    gamma = filter_data(data, sfreq, l_amp_freq, h_amp_freq,
                        n_jobs=njobs, method=method)

    # phase of the low freq modulating signal
    phase = np.angle(hilbert(theta))
    # amplitude envelope of the high freq modulated signal
    amp = np.abs(hilbert(gamma))

    return theta, gamma, phase, amp


def compute_and_plot_psd(data, sfreq, NFFT=512, show=True):
    """ Computes the power spectral density and produces a plot.

    Parameters
    ----------
    data : ndarray (n_times)
        The signal.
    sfreq : float
        Sampling frequency.
    NFFT : int (power of 2)
        Number of bins for each block of FFT.
    show : bool
        Display or hide plot. (Default is True)

    Returns
    -------
    power : ndarray
        The power spectral density of the signal.
    freqs : ndarray
        Frequencies.
    """
    import matplotlib.pyplot as pl
    if show is False:
        pl.ioff()
    power, freqs = pl.psd(data, Fs=sfreq, NFFT=NFFT)

    return power, freqs


def correlate_envelope_signal(signal, amp_envelope, n_surrogates=100,
                              random_state=None):
    """ Correlate the amplitude envelope with the signal.

    Parameters
    ----------
    signal : ndarrray (n_times)
        Low freq filtered signal.
    amp_envelope: ndarray (n_times)
        Amplitude envelope of high freq signal.
    n_surrogates : int
        Number of surrogates to be computed.
    random_state : int
        Seed value for random generator.

    Returns
    -------
    xcorr : ndarray (n_times)
        Cross correlation of the two signals.
    xcorr_surrogates : ndarray (n_surrogates)
        Cross correlation of the surrogate signals.
    max_surr : ndarray (n_surrogates)
        Maximum value of surrogate cross correlation.
    z_threshold : float
        Threshold value after z-scoring.
    """

    from sklearn.utils import check_random_state

    xcorr = np.correlate(signal, amp_envelope, 'full')
    xcorr_surrogates = np.zeros((n_surrogates, xcorr.size))
    max_surr = np.zeros((n_surrogates, 1))

    rng = check_random_state(random_state)  # initialize random generator
    for i in range(0, n_surrogates):
        order = np.argsort(rng.randn(len(amp_envelope)))
        xcorr_surrogates[i, :] = np.correlate(signal, amp_envelope[order], 'full')
        max_surr[i] = np.max(np.abs(xcorr_surrogates[i, :]))

    # compute some statistics
    #NOTE Needs to be rechecked. I want to check if the xcorr values
    # can come from the surrogates values (i.e. reject the null hypothesis
    # that xcorr computed is random.
    max_surr_mean = np.mean(max_surr)
    max_surr_std = np.std(max_surr)
    # compute zscores
    zscores = (xcorr - max_surr_mean) / max_surr_std
    from scipy import stats
    p_values = stats.norm.pdf(zscores)
    # perform fdr correction and compute threshold
    accept, _ = mne.stats.fdr_correction(p_values, alpha=0.001)
    z_threshold = np.abs(zscores[accept]).min()

    return xcorr, xcorr_surrogates, max_surr, z_threshold


def average_envelope_versus_phase(amplitude_envelope, phase):
    """ Computes the average amplitude envelope versus phase and plots it.
        (Buzsaki et al)

    Parameters
    ----------
    amplitude_envelope : ndarray
        Amplitude envelope signal.
    phase : ndarray
        Phase signal (in radians)

    """
    phase = phase * 360.0 / (2.0 * np.pi)  # use phase in degrees
    a_mean = np.zeros((36))      # divide phase into 36 bins
    a_std = np.zeros((36))       # each of width 10 degrees
    angle = np.zeros((36))       # label the phase with center phase

    for l, k in enumerate(range(-180, 180, 10)):
        indices = [i for (i, val) in enumerate(phase) if np.logical_and(val >= k, val < k + 10)]
        a_mean[l] = np.mean(amplitude_envelope[indices])
        a_std[l] = np.std(amplitude_envelope[indices])
        angle[l] = k + 5

    import matplotlib.pyplot as pl
    pl.figure('Average envelope versus phase')
    pl.plot(angle, a_mean, 'b')
    pl.fill_between(angle, a_mean + a_std, a_mean - a_std, color='gray')
    pl.xlabel('Phase (degrees)')
    pl.ylabel('Amplitude envelope')
    pl.xlim([-180, 180])
    pl.show()

    return


def event_related_average(gamma, data, win=100, show=True):
    """ Compute and plot event related averages triggered by peaks in high
        freq signal.

    Parameters
    ----------
    gamma : ndarray
        The high freq gamma signal.
    data : ndarray
        The original data signal.
    win : int
        Window length.
    show : bool
        Shows matplotlib figure. (Default is True)

    Returns
    -------
    average : ndarray
         Event related averages.

    References
    ----------
    [1] Bragin et al, J Neurosci, 1995.
    """
    # window length
    # indices and magnitudes of peaks
    locs, peaks = mne.preprocessing.peak_finder.peak_finder(gamma)
    locs = locs[np.logical_and(locs > win, locs < len(data) - win)]

    avg = np.zeros((len(locs), 2 * win))
    for i in range(0, len(locs)):
        avg[i, :] = data[locs[i] - win: locs[i] + win]

    average = np.mean(avg, axis=0)
    import matplotlib.pyplot as pl
    pl.plot(list(range(-win, win)), average)
    return average


# Compute the 1d modulation index
def modulation_index1d(amplitude_envelope, phase, sfreq, random_state=42):
    """ Computes the one dimensional modulation index.
        From Canolty et al, Science, 2006.

    Parameters
    ----------
    amplitude_envelope : ndarray
        Amplitude envelope signal
    phase: ndarray
        Phases. (phase values of the low freq signal)
    sfreq : float
        Sampling frequency.
    random_state : int or None
        Seed for random state genesrators.

    Returns
    -------
    mi : float
        Modulation index
    """
    n_samp = amplitude_envelope.size
    n_surr = 200  # Number of surrogates
    from mne.utils import check_random_state
    rng = check_random_state(random_state)
    shifts = np.ceil(n_samp * rng.rand(2 * n_surr))

    # reduce shifts to within one epoch of occurence
    # NOTE use sfreq for 1 second, n_samp / n_epochs for one epoch
    n_epochs = 10
    shifts = shifts[shifts > n_samp / n_epochs]
    shifts = shifts[shifts < n_samp - (n_samp / n_epochs)]
    shifts = shifts.astype(np.int)

    # construct the composite signal
    z = amplitude_envelope * np.exp(1j * phase)
    # mean of composite signal
    mean_raw = np.mean(z)

    surr_means = np.zeros((n_surr, 1))
    for i in range(0, n_surr):
        # shift the amplitude for each surrogate
        surr_amp = np.roll(amplitude_envelope, shifts[i])
        # compute the mean length
        surr_means[i] = np.abs(np.mean(surr_amp * np.exp(1j * phase)))

    # compare treu mean to surrogate
    surr_fits_mean = np.mean(surr_means, axis=0)
    surr_fits_std = np.std(surr_means, axis=0)

    # create a z score (m_norm_length)
    # which I hope is the modulation index?
    modulation_index = (np.abs(mean_raw) - surr_fits_mean) / surr_fits_std

    return modulation_index


def modulation_index2d(data, sfreq):
    """ Compute the two dimensional modulation index.

    Parameters
    ----------
    data : ndarray
        The signal data
    sfreq: float
        Sampling frequency

    Returns
    -------
    mod2d : ndarray
        2 dimensional modulation index
    """

    from mne.filter import filter_data
    from scipy.signal import hilbert

    flow = np.arange(2, 40, 1)
    flow_step = 1.0
    fhigh = np.arange(5, 205, 5)
    fhigh_step = 5.0

    mod2d = np.zeros((flow.size, fhigh.size))
    method = 'fft'
    n_jobs = 2

    for i in range(0, flow.size):
        theta = filter_data(data, sfreq, flow[i], flow[i] + flow_step,
                            method=method, n_jobs=n_jobs)
        theta = theta[sfreq: data.size - sfreq]
        phase = np.angle(hilbert(theta))

        for j in range(0, fhigh.size):
            gamma = filter_data(data, sfreq, fhigh[j], fhigh[j] + fhigh_step,
                                method=method, n_jobs=n_jobs)
            gamma = gamma[sfreq: data.size - sfreq]
            amp = np.abs(hilbert(gamma))

            # compute the modulation index
            m_norm_length = modulation_index1d(amp, phase, sfreq)
            mod2d[i, j] = m_norm_length

    return mod2d


def bicoherence(data, sfreq, fmax=50):
    """Compute the bicoherence (or bispectrum) of data.
       Bicoherence wiki entry - http://en.wikipedia.org/wiki/Bicoherence

    Parameters
    ----------
    data : ndarray (number of trials x times))
        Data array
    sfreq : float
        Sampling frequency
    fmax: float
        Maximum frequency of interest

    Returns
    -------
    bicoh :
        Bicoherence
    freqs : ndarray
        list of frequencies
    """
    data = data.reshape((1, data.size))
    n_samp = data.size  # number of samples
    n_trials = len(data)  # number of trials

    # frequency axis in Hz
    faxis = np.arange(0, n_samp/2, dtype=np.float32) / n_samp * sfreq
    faxis = faxis[faxis < fmax]

    b = np.zeros((faxis.size, faxis.size))

    numerator = np.zeros((faxis.size, faxis.size), dtype=np.complex64)
    power = np.zeros((n_samp))

    from scipy.signal import hann
    for k in range(0, n_trials):
        x = np.fft.fft(hann(n_samp) * data[k, :].T) #  take FFT of segment wit hanning
        num_temp = np.zeros((faxis.size, faxis.size), dtype=np.complex64)
        for i in range(0, faxis.size):
            for j in range(0, faxis.size):
                num_temp[i, j] = x[i] * x[j] * np.conj(x[i + j])

        numerator += num_temp / n_trials  # compute trial average of numerator
        power += np.abs(x * np.conj(x)) / n_trials  # compute FFT squared and avg over trials

    # Compute bicoherence
    for m in range(0, faxis.size):
        for n in range(0, faxis.size):
            b[m, n] = np.abs(numerator[m, n]) / np.sqrt(power[m] * power[n] * power[m + n])

    # return the normalized computed bicoherence and frequencies
    return b, faxis
