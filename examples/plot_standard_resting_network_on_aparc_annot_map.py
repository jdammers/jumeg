#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Plot the visualization of the standard resting state network on the
connectivity circle plot showing the Freesurfer aparc annotation map.

This provides a quick visualization of the primary connections of the standard
resting state network.

Uses the standard RSNs provided by [1]
[1] P. Garcés, M. C. Martín-Buro, and F. Maestú,
“Quantifying the Test-Retest Reliability of Magnetoencephalography
Resting-State Functional Connectivity,” Brain Connect., vol. 6, no. 6, pp. 448–460, 2016.
'''

import os.path as op
import numpy as np
import mne

from mne.datasets import sample
from jumeg.jumeg_utils import get_jumeg_path
from jumeg.connectivity import make_annot_from_csv
from jumeg.connectivity import plot_grouped_connectivity_circle

data_path = sample.data_path()
subject = 'sample'
subjects_dir = data_path + '/subjects'
parc_fname = 'standard_garces_2016'
csv_fname = op.join(get_jumeg_path(), 'data', 'standard_rsns.csv')

# set make_annot to True to save the annotation to disk
labels, coords, foci = make_annot_from_csv(subject, subjects_dir, csv_fname,
                                           parc_fname=parc_fname,
                                           make_annot=False,
                                           return_label_coords=True)

aparc = mne.read_labels_from_annot('sample', subjects_dir=subjects_dir)
aparc_names = [apa.name for apa in aparc]
lh_aparc = [mylab for mylab in aparc if mylab.hemi == 'lh']
rh_aparc = [mylab for mylab in aparc if mylab.hemi == 'rh']

# get the appropriate resting state labels
rst_aparc = []
for i, rst_label in enumerate(labels):
    myfoci = foci[i]  # get the vertex
    if rst_label.hemi == 'lh':  # vertex hemi is stored in the rst_label
        for mylab in lh_aparc:
            if myfoci in mylab.vertices:
                print 'Left: ', rst_label.name, myfoci, mylab
                rst_aparc.append(mylab)
    elif rst_label.hemi == 'rh':
        for mylab in rh_aparc:
            if myfoci in mylab.vertices:
                print 'Right: ', rst_label.name, myfoci, mylab
                rst_aparc.append(mylab)
    else:
        print 'ERROR: ', rst_label

# only 16 labels in aparc show up, there are no vertices in the left hemi for
# Frontoinsular_Median cingulate-lh
rst_indices = [aparc.index(rst) for rst in rst_aparc]

networks = {'Visual': ['lateraloccipital-lh', 'lateraloccipital-rh'],
            'Sensorimotor': ['supramarginal-lh', 'supramarginal-rh'],
            'Auditory': ['inferiortemporal-lh', 'middletemporal-rh'],
            'DMN': ['precuneus-lh', 'inferiorparietal-lh',
                    'inferiorparietal-rh', 'medialorbitofrontal-lh'],
            'Left_FP': ['inferiorparietal-lh', 'superiortemporal-lh'],
            'Right_FP': ['inferiorparietal-rh', 'superiortemporal-rh'],
            'Frontoinsular': ['inferiortemporal-lh', 'inferiortemporal-rh']}

# make a temporary connectivity matrix
n_nodes = 68
con = np.zeros((n_nodes, n_nodes))

rst_combindices = [[16, 31], [50, 14], [50, 15], [50, 28], [14, 15],
                   [14, 28], [15, 28], [22, 23], [14, 60], [62, 63],
                   [15, 61]]

# automatic assignment may also be done
# for comb in rst_combindices:
#     con[comb[0], comb[1]] = 1.

# assign different values to different networks manually
con[16, 31] = 0.3
con[50, 14] = con[50, 15] = con[50, 28] = con[14, 15] = con[14, 28] = con[15, 28] = 0.9  # DMN
con[22, 23] = 0.4
con[14, 60] = 0.5
con[62, 63] = 0.6
con[15, 61] = 0.8

con += con.T  # since we only add the combinations

# plot the connectivity circle showing standard RSNs
yaml_fname = get_jumeg_path() + '/examples/aparc_cortex_based_grouping.yaml'
plot_grouped_connectivity_circle(yaml_fname, con, aparc_names, n_lines=11,
                                 labels_mode=None, node_order_size=68,
                                 colormap='tab20',
                                 indices=None, out_fname='rsn_circle_plot.png')
