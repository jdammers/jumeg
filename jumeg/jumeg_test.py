#!/usr/bin/env python

import jumeg
import os.path

raw_fname = "109925_CAU01A_100715_0842_2_c,rfDC-raw.fif"
if not os.path.isfile(raw_fname):
    print("Please find the test file at the below location on the meg_store2 network drive - \
           cp /data/meg_store2/fif_data/jumeg_test_data/109925_CAU01A_100715_0842_2_c,rfDC-raw.fif .")

# Function to check and explain the file naming standards
#jumeg.jumeg_utils.check_jumeg_standards(raw_fname)

# Function to apply noise reducer
jumeg.jumeg_noise_reducer.noise_reducer(raw_fname, verbose=True)

# Filter functions
#jumeg.jumeg_preprocessing.apply_filter(raw_fname)

fclean = raw_fname[:raw_fname.rfind('-raw.fif')] + ',bp1-45Hz-raw.fif'

# Evoked functions
#jumeg.jumeg_preprocessing.apply_average(fclean)

# ICA functions
#jumeg.jumeg_preprocessing.apply_ica(fclean)
fica_name = fclean[:fclean.rfind('-raw.fif')] + '-ica.fif'

# Perform ECG/EOG rejection using ICA
#jumeg.jumeg_preprocessing.apply_ica_cleaning(fica_name)
#jumeg.jumeg_preprocessing.apply_ica_cleaning(fica_name, unfiltered=True)

# OCARTA cleaning
from jumeg.decompose import ocarta
ocarta_obj = ocarta.JuMEG_ocarta()
ocarta_obj.fit(fclean, unfiltered=False, verbose=True)

# CTPS functions

#jumeg.jumeg_preprocessing.apply_ctps(fica_name)
fctps_name = '109925_CAU01A_100715_0842_2_c,rfDC,bp1-45Hz,ctps-trigger.npy'
#jumeg.jumeg_preprocessing.apply_ctps_select_ic(fctps_name)

# Function recompose brain response components only
fname_ctps_ics = '109925_CAU01A_100715_0842_2_c,rfDC,bp1-45Hz,ctps-trigger-ic_selection.txt'
#jumeg.jumeg_preprocessing.apply_ica_select_brain_response(fname_ctps_ics)

# Function to process empty file
empty_fname = '109925_CAU01A_100715_0844_2_c,rfDC-empty.fif'
#jumeg.jumeg_preprocessing.apply_create_noise_covariance(empty_fname, verbose=True)
