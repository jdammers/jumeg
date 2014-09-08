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
    