#!/usr/bin/env python

"""
Example how to create a custom label groups and plot grouped connectivity
circle with these labels.

Author: Praveen Sripad <pravsripad@gmail.com>
        Christian Kiefer <ch.kiefer@fz-juelich.de>
"""

import matplotlib.pyplot as plt
from jumeg import get_jumeg_path
from jumeg.connectivity import (plot_grouped_connectivity_circle,
                                generate_random_connectivity_matrix)

import yaml

labels_fname = get_jumeg_path() + '/data/desikan_label_names.yaml'
replacer_dict_fname = get_jumeg_path() + '/data/replacer_dictionaries.yaml'

with open(labels_fname, 'r') as f:
    label_names = yaml.safe_load(f)['label_names']

with open(replacer_dict_fname, 'r') as f:
    replacer_dict = yaml.safe_load(f)['replacer_dict_aparc']

# make a random matrix with 68 nodes
# use simple seed for reproducibility
con = generate_random_connectivity_matrix(size=(68, 68), symmetric=True)

# make groups based on lobes
occipital = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']
parietal = ['superiorparietal', 'inferiorparietal', 'precuneus',
            'postcentral', 'supramarginal']
temporal = ['bankssts', 'temporalpole', 'superiortemporal', 'middletemporal',
            'transversetemporal', 'inferiortemporal', 'fusiform',
            'entorhinal', 'parahippocampal']
insula = ['insula']
cingulate = ['rostralanteriorcingulate', 'caudalanteriorcingulate',
             'posteriorcingulate', 'isthmuscingulate']
frontal = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal',
           'parsopercularis', 'parsorbitalis', 'parstriangularis',
           'lateralorbitofrontal', 'medialorbitofrontal', 'precentral',
           'paracentral', 'frontalpole']

# we need a list of dictionaries, one dict for each group to denote grouping
label_groups = [{'occipital': occipital}, {'parietal': parietal},
                {'temporal': temporal}, {'insula': insula},
                {'cingulate': cingulate},
                {'frontal': frontal}]

n_colors = len(label_groups)
cmap = plt.get_cmap('Pastel1')
cortex_colors = cmap.colors[:n_colors] + cmap.colors[:n_colors][::-1]

# plot simple connectivity circle with cortex based grouping and colors
plot_grouped_connectivity_circle(label_groups, con, label_names,
                                 labels_mode='replace',
                                 replacer_dict=replacer_dict,
                                 cortex_colors=cortex_colors, vmin=0., vmax=1.,
                                 out_fname='fig_grouped_con_circle_cortex.png',
                                 colorbar_pos=(0.1, 0.1), n_lines=50, colorbar=True,
                                 colormap='viridis')
