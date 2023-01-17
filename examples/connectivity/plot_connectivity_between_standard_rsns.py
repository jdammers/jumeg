'''
Modified MNE-Python example script to show connectivity between standard
resting state network labels obtained from [1].

[1] P. Garcés, M. C. Martín-Buro, and F. Maestú,
“Quantifying the Test-Retest Reliability of Magnetoencephalography
Resting-State Functional Connectivity,” Brain Connect., vol. 6, no. 6, pp.
448–460, 2016.

Author: Praveen sripad <pravsripad@gmail.com>
'''

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.viz import circular_layout

from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle

data_path = sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_inv = op.join(data_path, 'MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif')
fname_raw = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw.fif')
fname_event = op.join(data_path, 'MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')

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

# Compute inverse solution and for each epoch. By using "return_generator=True"
# stcs will be a generator object instead of a list.
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
method = "MNE"  # use dSPM method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
                            pick_ori="normal", return_generator=True)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot('sample', parc='standard_garces_2016',
                                    subjects_dir=subjects_dir)
labels = [lab for lab in labels if not lab.name.startswith('unknown')]
label_colors = [label.color for label in labels]

# Average the source estimates within each label using sign-flips to reduce
# signal cancellations, also here we return a generator
src = inverse_operator['src']
label_ts = mne.extract_label_time_course(stcs, labels, src, mode='mean_flip',
                                         return_generator=True)

fmin = 8.
fmax = 13.
sfreq = raw.info['sfreq']  # the sampling frequency
con_methods = ['coh', 'wpli']
con, freqs, times, n_epochs, n_tapers = spectral_connectivity_epochs(
    label_ts, method=con_methods, mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, mt_adaptive=True, n_jobs=4)

# con is a 3D array, get the connectivity for the first (and only) freq. band
# for each method
con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c[:, :, 0]

# Now, we visualize the connectivity using a circular graph layout
# First, we reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]

from jumeg import get_jumeg_path
yaml_fname = get_jumeg_path() + '/data/standard_garces_rsns_grouping.yaml'

import yaml
with open(yaml_fname, 'r') as f:
    xlabels = yaml.safe_load(f)

# the yaml file has been hand curated to follow the same order as label_names
# if not the node order has to be changed appropriately
node_order = list()
node_order.extend(label_names)

group_bound = [len(list(key.values())[0]) for key in xlabels]
group_bound = [0] + group_bound
group_boundaries = [sum(group_bound[:i+1]) for i in range(len(group_bound))]
group_boundaries.pop()

rsn_colors = ['m', 'b', 'y', 'c', 'r', 'g', 'w']

group_bound.pop(0)
label_colors = []
for ind, rep in enumerate(group_bound):
    label_colors += [rsn_colors[ind]] * rep
assert len(label_colors) == len(node_order), 'Number of colours do not match'

from mne.viz.circle import circular_layout
node_angles = circular_layout(label_names, label_names, start_pos=90,
                              group_boundaries=group_boundaries)

# Plot the graph using node colors from the FreeSurfer parcellation.
plot_connectivity_circle(con_res['wpli'], label_names, n_lines=300,
                         node_angles=node_angles, node_colors=label_colors,
                         title='Connectivity between standard RSNs')
# plt.savefig('circle.png', facecolor='black')
