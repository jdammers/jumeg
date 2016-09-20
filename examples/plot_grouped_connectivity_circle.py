#!/usr/bin/env python

'''
Read grouped aparc labels from yaml file.
Plot grouped connectivity circle with these grouped labels.
'''

import numpy as np
from jumeg import get_jumeg_path
from jumeg.connectivity import plot_grouped_connectivity_circle
import pickle

labels_fname = get_jumeg_path() + '/examples/label_names.list'
yaml_fname = get_jumeg_path() + '/examples/aparc_cortex_based_grouping.yaml'

with open('label_names.list', 'r') as f:
        label_names = pickle.load(f)

# make a random matrix with 68 nodes
# use simple seed for reproducibility
np.random.seed(42)
con = np.random.random((68, 68))
con[con < 0.5] = 0.

plot_grouped_connectivity_circle(yaml_fname, con, label_names, n_lines=10, colorbar=True)
