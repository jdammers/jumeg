#!/usr/bin/env python3

'''
Perform Granger based causality analysis using Generalized Parital Directed
Coherence on example dataset.

Uses the data and example from mne-python combined with the Scot package
to perform the Granger Causality analysis.

Author: Praveen Sripad <pravsripad@gmail.com>
'''

import numpy as np
from scipy import stats

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from jumeg.jumeg_utils import get_jumeg_path
from jumeg.connectivity.causality import (compute_order, do_mvar_evaluation,
                                          prepare_causality_matrix)
from jumeg.connectivity import (plot_grouped_connectivity_circle,
                                plot_grouped_causality_circle)

import scot
import scot.connectivity_statistics as scs
from scot.connectivity import connectivity
import yaml

import time
t_start = time.time()

print(('Scot version -', scot.__version__))

yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

# Load data
inverse_operator = read_inverse_operator(fname_inv)
raw = mne.io.read_raw_fif(fname_raw)
events = mne.read_events(fname_event)

# Add a bad channel
raw.info['bads'] += ['MEG 2443']

# Pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))
if not epochs.preload:
    epochs.load_data()

# parameters, lots of parameters
snr = 1.0
lambda2 = 1.0 / snr ** 2
method = "MNE"  # use MNE method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal", return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('sample', parc='aparc',
                                    subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

# Average the source estimates within each label using sign-flips to reduce
# signal cancellations, also here we return a generator
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip',
                                         return_generator=False)
label_ts_ = np.array(label_ts)

bands = ['alpha']
freqs = [(8, 13)]
gcmethod = 'GPDC'
n_surr = 1  # number of surrogates
surr_thresh = 95  # percentile of surr threshold used
n_jobs = 1
nfft = 512

# normalize the representative ts
print('\nperform normalization using zscoring...')
label_ts = stats.zscore(label_ts_, axis=2)

morder = 15  # set fixed model order

# set this to find the optimal model order using the BIC criterion
# be advised, this takes a long time !!
# morder, bic = compute_order(label_ts, m_max=100)  # code provided by Qunxi
# print('the model order based on BIC is..', morder)

# evaluate the chosen model order
print(('\nShape of label_ts -', label_ts.shape))
# mvar needs (trials, channels, samples)
print(('\nRunning for model order - ', morder))

thr_cons, whit_min, whit_max = 0.8, 1., 3.
is_white, consistency, is_stable = do_mvar_evaluation(label_ts, morder,
                                                      whit_max, whit_min,
                                                      thr_cons)
print(('model_order, whiteness, consistency, stability: %d, %s, %f, %s\n'
      % (morder, str(is_white), consistency, str(is_stable))))

# compute the Granger Partial Directed Coherence values
print('computing GPDC connectivity...')

mvar = scot.var.VAR(morder)
# result : array, shape (`repeats`, n_channels, n_channels, nfft)
surr = scs.surrogate_connectivity(gcmethod, label_ts, mvar, nfft=nfft,
                                  n_jobs=n_jobs, repeats=n_surr)

mvar.fit(label_ts)
# mvar coefficients (n_channels, n_channels * model_order)
# mvar covariance matrix (n_channels, n_channels)
# result : array, shape (n_channels, n_channels, `nfft`)
cau = connectivity(gcmethod, mvar.coef, mvar.rescov, nfft=nfft)

# get the band averaged, thresholded connectivity matrix
caus, max_cons, max_surrs = prepare_causality_matrix(
    cau, surr, freqs, nfft=nfft,
    sfreq=epochs.info['sfreq'], surr_thresh=surr_thresh)

print(('Shape of causality matrix: ', caus.shape))

# read the label names used for plotting
# with open(labels_fname, 'r') as f:
#     label_names = pickle.load(f)

with open(labels_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

plot_grouped_causality_circle(caus[0], yaml_fname, label_names, n_lines=10,
                              labels_mode=None, replacer_dict=None,
                              out_fname='causality_sample.png',
                              colormap='Blues', colorbar=True,
                              arrowstyle='->,head_length=1,head_width=1',
                              figsize=(10, 6), show=False)

t_end = time.time()
total_time_taken = t_end - t_start
print(('Total time taken in minutes: %f' % (total_time_taken / 60.)))
