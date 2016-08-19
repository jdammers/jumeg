README.txt

This directory contains all the files related to cauality analysis project by Qunxi Dong.

The main analysis steps involved are:

1. Spatio temporal clustering on evoked data. 
	stat_cluster.py - functions
	spatial_clustering.py - main script
2. Identification of ROIs based on results of spatio temporal clustering.
	apply_merge.py - functions
	ROIs_definition.py - main script
3. Application of causality analysis.
	apply_causality_whole.py - functions
	causality_analysis_whole.py - main script
4. Plotting.
	plot_causal_spectral.py
	plot_labels.py

The scripts require setting the subjects directory from which the files
are automatically read and the results stored in directories relative
to the base directory.

External dependency:
ScoT - (https://github.com/scot-dev/scot)
mne-python - (https://github.com/mne-tools/mne-python)
