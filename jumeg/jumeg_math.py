import numpy as np


##################################################
#
# Function to rescale data
#
##################################################
def rescale(data_arr, minval, maxval):
    """ Function to rescale an array to the desired range. """
    min_data = -1.0 * np.min(data_arr)
    max_data = np.max(data_arr)
    if ((max_data+min_data) != 0):
        b = (maxval-minval)/(max_data+min_data)
        data_new = ((data_arr + min_data) * b) + minval
    # if data_arr is a constant function
    else:
        data_new = (max_data - min_data)/2.0
    return data_new


##################################################
#
# Function to calculate the RMS-value
#
##################################################
def calc_rms(data, average=None, rmsmean=None):
    ''' Calculate the rms value of the signal.
        Ported from Dr. J. Dammers IDL code.
    '''
    # check input
    sz      = np.shape(data)
    nchan   = np.size(sz)
    #  calc RMS
    rmsmean = 0
    if nchan == 1:
        ntsl = sz[0]
        rms  = np.sqrt(np.sum(data**2)/ntsl)
        rmsmean = rms
    elif nchan == 2:
        ntsl = sz[1]
        powe  = data**2
        rms  = np.sqrt(sum(powe, 2)/ntsl)
        rmsmean = np.mean(rms)
        if average:
            rms = np.sqrt(np.sum(np.sum(powe, 1)/nchan)/ntsl)
    else: return -1
    return rms


##################################################
#
# sigmoidal function
#
##################################################
def sigm_func(x, a0=1., a1=1.):
    """
    Sigmoidal function
    """
    return 1.0 / (1.0 + a0 * np.exp(-1.0 * a1 * x))


##################################################
#
# calculates the Taeger-Kaiser-Energy-Operator
#
##################################################
def calc_tkeo(signal):
    """
    Returns the Taeger-Kaiser-Energy-Operator:
       Y(n) = X^2(n) - X(n+1) * X(n-1)
    """
    # estimate tkeo
    s1       = signal ** 2.
    s2       = np.roll(signal, 1) * np.roll(signal, -1)
    tkeo     = s1 -s2

    # set first and last element to zero
    tkeo[0]  = 0.
    tkeo[-1] = 0.

    # return results
    return tkeo


#######################################################
#
#  calculate the performance of artifact rejection
#
#######################################################
def calc_performance(evoked_raw, evoked_clean):
    ''' Gives a measure of the performance of the artifact reduction.
        Percentage value returned as output.
    '''
    from jumeg import jumeg_math as jmath

    diff = evoked_raw.data - evoked_clean.data
    rms_diff = jmath.calc_rms(diff, average=1)
    rms_meg = jmath.calc_rms(evoked_raw.data, average=1)
    arp = (rms_diff / rms_meg) * 100.0
    return np.round(arp)


#######################################################
#
#  calculate the frequency-correlation value
#
#######################################################
def calc_frequency_correlation(evoked_raw, evoked_clean):

    """
    Function to estimate the frequency-correlation value
    as introduced by Krishnaveni et al. (2006),
    Journal of Neural Engineering.
    """

    # transform signal to frequency range
    fft_raw = np.fft.fft(evoked_raw.data)
    fft_cleaned = np.fft.fft(evoked_clean.data)

    # get numerator
    numerator = np.sum(np.abs(np.real(fft_raw) * np.real(fft_cleaned)) +
                       np.abs(np.imag(fft_raw) * np.imag(fft_cleaned)))

    # get denominator
    denominator = np.sqrt(np.sum(np.abs(fft_raw) ** 2) *
                          np.sum(np.abs(fft_cleaned) ** 2))

    return np.round(numerator / denominator * 100.)


##################################################
#
# Function calculate overlapping frequency windows
# fb 19.12.2014
#
##################################################
def calc_sliding_window(fmin, fmax, fstep):
    '''
    Calculate 50% overlapping frequency windows
    input:
    fmin    : start frequency
    fmax    : end frequency
    fstep   : window range f0<>f1

    return:  numpy array two dims with pairs of frequncy windows

    sliding window e.g. calculate frequency windos
    calc_sliding_window(4, 32, 8)
    out:
    array([[ 4, 12],
       [ 8, 16],
       [12, 20],
       [16, 24],
       [20, 28],
       [24, 32],
       [28, 36],
       [32, 40]])

    '''
    if fmin > 0:
       return np.array([ np.arange(fmin,fmax+1,fstep/2),np.arange(fmin + fstep,fmax+ fstep+fmin,fstep/2) ]).T
    else:
       fmin = 1
       return np.array([ np.arange(fmin,fmax+1,fstep/2),np.arange(fmin + fstep,fmax+ fstep+fmin,fstep/2) ]).T -1
