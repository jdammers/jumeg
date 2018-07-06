#!/usr/bin/env python2

import numpy as np

from jumeg import get_jumeg_path
from jumeg.connectivity import (plot_grouped_connectivity_circle,
                                plot_grouped_causality_circle)
import pickle

# load the yaml grouping of Freesurfer labels
yaml_fname = get_jumeg_path() + '/examples/example.yaml'
labels_fname = get_jumeg_path() + '/examples/fruits.list'

yaml_fname = get_jumeg_path() + '/examples/aparc_cortex_based_grouping.yaml'
labels_fname = get_jumeg_path() + '/examples/label_names.list'

with open(labels_fname, 'r') as f:
    label_names = pickle.load(f)

# make a random causality matrix
n_nodes = 68  # currently needs to be always this number
caus = np.random.random((n_nodes, n_nodes))
caus[np.diag_indices_from(caus)] = 0.
caus[caus < 0.7] = 0.

plot_grouped_causality_circle(caus, yaml_fname, label_names, n_lines=10,
                              labels_mode='cortex_only',
                              out_fname='causality_circle.png',
                              colormap='Blues', colorbar=True,
                              figsize=(10, 6), show=False)
