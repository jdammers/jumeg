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


##################################################
#
# destroy phase/time info on any data array
#
##################################################
def shuffle_data (data_trials, mode='shuffle'):
    ''' Shuffling the time information (phase) of any data array
    Parameters
    ----------
    data_trials : 2d ndarray of dimension [ntrials x nsamples]
    mode : 2 different modi are allowed.
    'mode=shuffle' whill randomly shuffle the phase values
    'mode=shift' whill randomly shift the phase values

    Returns
    -------
    s_trial : shuffled (phase) trials

    '''

    np.random.seed()     # for parallel processing it needs to be re-initialized
    ntrials, nsamples = data_trials.shape

    if (mode == 'shuffle'):                 
        # shuffle all phase entries
        dt = data_trials.flatten()
        np.random.shuffle(dt)       
        dt = dt.reshape(ntrials,nsamples)
    else:
        if (mode == 'shift'):
            # random phase shifts for each trial
            dt = np.zeros((ntrials, nsamples), dtype=data_trials.dtype)
            shift = np.random.permutation(np.arange(ntrials))
            for itrial in range(ntrials):
                dt[itrial,:] = np.roll(data_trials[itrial,:], shift[itrial]) 

    return dt

