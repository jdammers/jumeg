#!/usr/bin/env python

'''
Plot degree values for a given set of nodes in a simple circle plot.
'''

import numpy as np
import matplotlib.pyplot as plt

import mne
from jumeg import get_jumeg_path
from jumeg.connectivity import plot_degree_circle

import bct

orig_labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
yaml_fname = get_jumeg_path() + '/data/desikan_aparc_cortex_based_grouping.yaml'
con_fname = get_jumeg_path() + '/data/sample,aparc-con.npy'

con = np.load(con_fname)
con_ = con[0, :, :, 2] + con[0, :, :, 2].T

# compute the degree
degrees = mne.connectivity.degree(con_, threshold_prop=0.2)

fig, ax = plot_degree_circle(degrees, yaml_fname, orig_labels_fname)
