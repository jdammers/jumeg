"""
Compute ICA object based on filtered and downsampled data.
Identify ECG and EOG artifacts using the pre-trained DCNN model
and compare results using correlation & ctps analysis.

Apply ICA object to filtered and unfiltered data.

Ahmad Hasasneh, Nikolas Kampel, Praveen Sripad, N. Jon Shah, and Juergen Dammers
"Deep Learning Approach for Automatic Classification of Ocular and Cardiac
Artifacts in MEG Data"
Journal of Engineering, vol. 2018, Article ID 1350692,10 pages, 2018.
https://doi.org/10.1155/2018/1350692
"""

import os.path as op
import matplotlib.pylab as plt
plt.ion()
import numpy as np
import mne
from jumeg.decompose.ica_replace_mean_std import ICA, ica_update_mean_std
from keras.models import load_model
from jumeg.jumeg_noise_reducer import noise_reducer
from jumeg.jumeg_preprocessing import get_ics_cardiac, get_ics_ocular
from jumeg.jumeg_plot import plot_performance_artifact_rejection
from jumeg.jumeg_utils import get_jumeg_path

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# settings
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model_thresh = 0.8                        # >0.5 ..0.9
n_components = 40                         # 30 .. 60
njobs = 'cuda'
sfreq_new = 250                           # downsampling to 250 Hz
flow_raw, flow_high = 2, 45               # high pass filter prevents from false positives
flow_ecg, fhigh_ecg = 8, 20
flow_eog, fhigh_eog = 1, 20
ecg_thresh, eog_thresh = 0.3, 0.3
ecg_ch = 'ECG 001'
eog1_ch = 'EOG 001'
eog2_ch = 'EOG 002'
reject = {'mag': 5e-12}

# number time samples is fixed to 15000
nsamples_chop = 15000
ix_t1 = 0                              # time index: here we use the first chop
ix_t2 = ix_t1 + nsamples_chop

# ----------------------------------------------
# load DCNN model for artifact rejection
# the details of the model is provided in:
#       x_validation_shuffle_v4_split_23.txt
# model was trained on 4D data from Juelich
# ----------------------------------------------
model_path = op.join(get_jumeg_path(), 'data')
model_name = op.join(model_path, "dcnn_model.hdf5")
model = load_model(model_name)

# ----------------------------------------------
# read example data file
# ----------------------------------------------
path_data = '/data/megraid22/Common/DeepLearning/cau_data_validation/'
raw_fname = path_data + '109925_CAU01A_100715_0842_2_c,rfDC,t1,n_bcc,nr-raw.fif'
raw = mne.io.Raw(raw_fname, preload=True)
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')


# ----------------------------------------------
# filtering and down sampling
# ----------------------------------------------
# filter prior to ICA
raw_filtered = raw.copy().filter(flow_raw, flow_high, picks=picks, filter_length='auto',
                                 l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                                 n_jobs=njobs, method='fir', phase='zero',
                                 fir_window='hamming')
# downsample data
raw_ds = raw_filtered.copy().resample(sfreq_new, npad='auto', window='boxcar', stim_picks=None,
                                      n_jobs=njobs, events=None)

# ----------------------------------------------
# crop data to get first chop
# ----------------------------------------------
# get times to crop
dt = raw_ds.times[1]             # time period between two time samples
tmin = ix_t1 * dt
tmax = ix_t2 * dt - dt           #  subtract one sample
# crop all raw objects
raw_chop = raw.copy().crop(tmin=tmin, tmax=tmax)                     # raw
raw_filtered_chop = raw_filtered.copy().crop(tmin=tmin, tmax=tmax)   # raw filtered
raw_ds_chop = raw_ds.copy().crop(tmin=tmin, tmax=tmax)               # raw filtered downsampled
raw_filtered.close()

# ----------------------------------------------
# apply ICA
# ----------------------------------------------
ica = ICA(method='fastica', n_components=n_components, random_state=42,
          max_pca_components=None, max_iter=5000, verbose=None)
# do the ICA decomposition on downsampled raw
ica.fit(raw_ds_chop, picks=picks, reject=reject, verbose=None)
sources = ica.get_sources(raw_ds_chop)._data                     # get sources
sources = np.reshape(sources, (n_components,nsamples_chop, 1))   # reshape sources

# ----------------------------------------------
# model prediction
# identification of artifact components
# ----------------------------------------------
# compute base functions
mm = np.float32(np.dot(ica.mixing_matrix_[:, :ica.n_components_].T,
                       ica.pca_components_[:ica.n_components_, :ica.max_pca_components]))
# get model prediction
model_scores = model.predict([mm, sources], verbose=1)
# get ICs
bads_MLICA = list(np.where(model_scores[:,0] > model_thresh)[0])

# ----------------------------------------------
# order ICs for visualization
# ----------------------------------------------
var_order = sources.std(axis=1).flatten().argsort()[::-1]
good_ics = np.setdiff1d(var_order, bads_MLICA)
ic_order = list(np.concatenate([bads_MLICA, good_ics]))
# store components in ica object
ica.exclude = list(bads_MLICA)


# ----------------------------------------------
# compare MLICA results with correlation and ctps
# ----------------------------------------------
print('Identifying components..')
# get ECG/EOG related components using JuMEG
ic_ecg = get_ics_cardiac(raw_filtered_chop, ica, flow=flow_ecg, fhigh=fhigh_ecg,
                         thresh=ecg_thresh, tmin=-0.5, tmax=0.5,
                         name_ecg=ecg_ch, use_CTPS=True)[0]  # returns both ICs and scores (take only ICs)
ic_eog = get_ics_ocular(raw_filtered_chop, ica, flow=flow_eog, fhigh=fhigh_eog,
                        thresh=eog_thresh, name_eog_hor=eog1_ch,
                        name_eog_ver=eog2_ch, score_func='pearsonr')
bads_corr_ctps = list(ic_ecg) + list(ic_eog)
bads_corr_ctps = list(set(bads_corr_ctps))  # remove potential duplicates
bads_corr_ctps.sort()
print('Bad components from MLICA:', bads_MLICA)
print('Bad components from correlation & ctps:', bads_corr_ctps)


# ----------------------------------------------
# plot results
# ----------------------------------------------
# plot sources
fig = ica.plot_sources(raw_filtered_chop, picks=ic_order, title='MLICA', show=False)
#fig.savefig('MLICA_ica-sources.png')

# plot artifact rejection performance
fnout_fig = '109925_CAU01A_100715_0842_2_c,rfDC,0-45hz,ar-perf'
ica_filtered_chop = ica_update_mean_std(raw_filtered_chop, ica, picks=picks, reject=reject)
raw_filtered_chop_clean = ica_filtered_chop.apply(raw_filtered_chop, exclude=ica.exclude,
                                                  n_pca_components=None)
ica_unfiltered_chop = ica_update_mean_std(raw_chop, ica, picks=picks, reject=reject)
raw_unfiltered_chop_clean = ica_unfiltered_chop.apply(raw_chop, exclude=ica.exclude, n_pca_components=None)
plot_performance_artifact_rejection(raw.copy().crop(tmin=tmin, tmax=tmax), ica_unfiltered_chop, fnout_fig,
                                    meg_clean=raw_unfiltered_chop_clean,
                                    show=True, verbose=False,
                                    name_ecg=ecg_ch,
                                    name_eog=eog2_ch)
