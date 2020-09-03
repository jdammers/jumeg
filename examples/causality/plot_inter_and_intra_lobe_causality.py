"""
Group a causality matrix by lobes and plot the resulting
inter- and intra-lobe causality.

Author: Christian Kiefer <ch.kiefer@fz-juelich.de>
"""

import os
import os.path as op

import matplotlib.pyplot as plt
import mne
import numpy as np

from jumeg.connectivity.con_utils import group_con_matrix_by_lobe
from jumeg.connectivity.con_viz import plot_grouped_causality_circle
from jumeg.jumeg_utils import get_jumeg_path

###############################################################################
# Load the grouping files
###############################################################################

grouping_yaml_fname = op.join(get_jumeg_path(), 'data',
                              'desikan_aparc_cortex_based_grouping_ck.yaml')
lobe_grouping_yaml_fname = op.join(get_jumeg_path(), 'data',
                                   'lobes_grouping.yaml')

###############################################################################
# Load anatomical labels
###############################################################################

subjects_dir = os.environ['SUBJECTS_DIR']

full_labels = mne.read_labels_from_annot(subject='fsaverage', parc='aparc',
                                         hemi='both', subjects_dir=subjects_dir)

full_label_names = []
for full_label in full_labels:
    if full_label.name.startswith('unknown'):
        continue
    full_label_names.append(full_label.name)

###############################################################################
# create random causality matrix
###############################################################################

# create causality matrix
np.random.seed(42)
cau = np.random.uniform(-0.99, 0.01, (len(full_label_names), len(full_label_names)))

cau[cau < 0] = 0
cau = cau / 0.01  # values between 0 and 1

cau_grp, grp_label_names = group_con_matrix_by_lobe(con=cau, label_names=full_label_names,
                                                    grouping_yaml_fname=grouping_yaml_fname)

###############################################################################
# Compare original matrix with grouped matrix plot
###############################################################################

fig = plot_grouped_causality_circle(cau, grouping_yaml_fname, full_label_names,
                                    title='original causality matrix', n_lines=None,
                                    labels_mode=None, replacer_dict=None, out_fname=None,
                                    colormap='magma_r', colorbar=True, colorbar_pos=(-0.25, 0.05),
                                    arrowstyle='->,head_length=0.7,head_width=0.4',
                                    figsize=(9.1, 6), vmin=0., vmax=1.0, ignore_diagonal=True,
                                    show=True)

plt.close(fig)

fig = plot_grouped_causality_circle(cau_grp, lobe_grouping_yaml_fname, grp_label_names,
                                    title='test', n_lines=None, labels_mode=None,
                                    replacer_dict=None, out_fname=None, colormap='magma_r',
                                    colorbar=True, colorbar_pos=(-0.25, 0.05),
                                    arrowstyle='->,head_length=0.7,head_width=0.4',
                                    figsize=(9.1, 6), vmin=0., ignore_diagonal=False,
                                    show=True)

plt.close(fig)
