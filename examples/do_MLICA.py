"""
Compute ICA object based on filtered and downsampled data.
Identify ECG and EOG artifacts using MLICA and compare
results to correlation & ctps analysis.

Ahmad Hasasneh, Nikolas Kampel, Praveen Sripad, N. Jon Shah, and Jürgen Dammers
“Deep Learning Approach for Automatic Classification of Ocular and Cardiac
Artifacts in MEG Data,”
Journal of Engineering, vol. 2018, Article ID 1350692,10 pages, 2018.
https://doi.org/10.1155/2018/1350692
"""

import matplotlib.pylab as plt
plt.ion()
import numpy as np
import mne
from mne.preprocessing import ICA
from keras.models import load_model
from jumeg.jumeg_noise_reducer import noise_reducer
from jumeg.jumeg_preprocessing import get_ics_cardiac, get_ics_ocular

# config
MLICA_threshold = 0.8
n_components = 60
njobs = 4  # for downsampling
tmin = 0
tmax = tmin + 15000
flow_ecg, fhigh_ecg = 8, 20
flow_eog, fhigh_eog = 1, 20
ecg_thresh, eog_thresh = 0.3, 0.3
ecg_ch = 'ECG 001'
eog1_ch = 'EOG 001'
eog2_ch = 'EOG 002'
refnotch = [50., 100., 150., 200., 250., 300., 350., 400.]

# example filname
raw_fname = '/data/megraid21/sripad/cau_fif_data/jumeg_test_data/109925_CAU01A_100715_0842_2_c,rfDC-raw.fif'

# load the model for artifact rejection:
model_name = '/data/megraid26/meg_commons/MLICA/models_MLICA_paper_090418/x_validation_shuffle_v4/split_23/split_23-25-0.939.hdf5'
model = load_model(model_name)

# noise reducer
raw_nr = noise_reducer(raw_fname, reflp=5., return_raw=True)

raw_nr = noise_reducer(raw_fname, raw=raw_nr, refhp=0.1, noiseref=['RFG ...'], return_raw=True)

# 50HZ and 60HZ notch filter to remove noise
raw = noise_reducer(raw_fname, raw=raw_nr, refnotch=refnotch, return_raw=True)

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')

raw.filter(0., 45., picks=picks, filter_length='auto',
           l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           n_jobs=njobs, method='fir', phase='zero',
           fir_window='hamming')

# downsample the data to 250 Hz, necessary for the model
raw_ds = raw.copy().resample(250, npad='auto', window='boxcar', stim_picks=None, n_jobs=njobs, events=None)
raw_ds_chop = raw_ds.copy().crop(tmin=tmin*4./1000, tmax=tmax*4./1000)  # downsampled raw
raw_chop = raw.copy().crop(tmin=tmin*4./1000, tmax=tmax*4./1000)

ica = ICA(method='fastica', n_components=n_components, random_state=None,
          max_pca_components=None, max_iter=1000, verbose=None)

# do the ICA decomposition on downsampled raw
ica.fit(raw_ds_chop, picks=picks, reject={'mag': 5e-12},
        verbose=None)

sources = ica.get_sources(raw_ds_chop)._data

# extract temporal and spatial components
mm = np.float32(np.dot(ica.mixing_matrix_[:, :].T,
                       ica.pca_components_[:ica.n_components_]))

# use [:, :15000] to make sure it's 15000 data points
chop = sources[:, :15000]
chop_reshaped = np.reshape(chop, (len(chop), len(chop[0]), 1))

model_scores = model.predict([mm, chop_reshaped], verbose=1)

bads_MLICA = []

print model_scores

for idx in range(0, len(model_scores)):
    if model_scores[idx][0] > MLICA_threshold:
        bads_MLICA.append(idx)

ica.exclude = bads_MLICA

# visualisation
# ica.plot_sources(raw_ds_chop, block=True)

# compare MLICA to results from correlation and ctps analysis
ica.exclude = []

print 'Identifying components..'
# get ECG/EOG related components using JuMEG
ic_ecg = get_ics_cardiac(raw_chop, ica, flow=flow_ecg, fhigh=fhigh_ecg,
                         thresh=ecg_thresh, tmin=-0.5, tmax=0.5,
                         name_ecg=ecg_ch, use_CTPS=True)
ic_eog = get_ics_ocular(raw_chop, ica, flow=flow_eog, fhigh=fhigh_eog,
                        thresh=eog_thresh, name_eog_hor=eog1_ch,
                        name_eog_ver=eog2_ch, score_func='pearsonr')

bads_corr_ctps = list(ic_ecg) + list(ic_eog)
bads_corr_ctps = list(set(bads_corr_ctps))  # remove potential duplicates
bads_corr_ctps.sort()
ica.exclude = bads_corr_ctps  # to sort and remove repeats
# visualisation
# ica.plot_sources(raw_chop, block=True)

print 'Bad components from MLICA:', bads_MLICA
print 'Bad components from correlation & ctps:', ica.exclude