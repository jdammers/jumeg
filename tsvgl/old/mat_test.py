
import numpy as np
import scipy.io as sio

ts=np.arange(815*7*119)*1.0
ts.dtype
ts.reshape(815,7,119)

ts_mat_dict = sio.loadmat('./test/ts.mat')
v_mat_dict  = sio.loadmat('./test/V.mat')

ts_mat = ts_mat_dict['data']
v_mat  = v_mat_dict['data']
