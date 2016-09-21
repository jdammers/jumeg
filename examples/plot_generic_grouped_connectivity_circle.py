#!/usr/bin/env python

import numpy as np
from jumeg.connectivity import plot_generic_grouped_circle, plot_fica_grouped_circle
from jumeg import get_jumeg_path

# load the yaml grouping of Freesurfer labels
yaml_fname = get_jumeg_path() + '/examples/rsn_aparc_cortex_grouping.yaml'

# make a random matrix with 68 nodes
# use simple seed for reproducibility
np.random.seed(42)
con = np.random.random((34, 34))
con[con < 0.5] = 0.

# load the label names in the original order
# this should be same order as the connectivity matrix
labels_fname = get_jumeg_path() + '/examples/fica_names.txt'
with open(labels_fname, 'r') as f:
    orig_labels = [line.rstrip('\n') for line in f]


# plot the connectivity circle grouped
plot_generic_grouped_circle(yaml_fname, con, orig_labels,
                            node_order_size=34,
                            out_fname='fica_circle.png',
                            show=False, n_lines=20,
                            title='Groups')

# plot the connectivity circle with outer group labels ring
plot_fica_grouped_circle(yaml_fname, con,
                         node_order_size=34,
                         out_fname='fica_circle_with_names.png',
                         show=False, n_lines=20,
                         title='Groups + Labels')
