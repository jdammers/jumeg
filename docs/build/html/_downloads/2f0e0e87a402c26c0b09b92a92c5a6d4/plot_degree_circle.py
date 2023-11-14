#!/usr/bin/env python

'''
=============
Plot degree circle
=============

Plot degree values for a given set of nodes in a simple circle plot.
'''

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne_connectivity import degree
from jumeg import get_jumeg_path
from jumeg.connectivity import plot_degree_circle

import bct

orig_labels_fname = op.join(get_jumeg_path(), 'data/desikan_label_names.yaml')
yaml_fname = op.join(get_jumeg_path(), 'data/desikan_aparc_cortex_based_grouping.yaml')
con_fname = op.join(get_jumeg_path(), 'data/sample,aparc-con.npy')

con = np.load(con_fname)
con_ = con[0, :, :, 2] + con[0, :, :, 2].T

# compute the degree
degrees = degree(con_, threshold_prop=0.2)

fig, ax = plot_degree_circle(degrees, yaml_fname, orig_labels_fname)
