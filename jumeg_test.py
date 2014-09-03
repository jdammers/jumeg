#!/usr/bin/env python

import jumeg
import os.path

raw_fname = "109925_CAU01A_100715_0842_2_c,rfDC-raw.fif"
if not os.path.isfile(raw_fname):
    print "Please find the test file at the below location on the meg_store2 network drive - \
           cp /data/meg_store2/fif_data/jumeg_test_data/109925_CAU01A_100715_0842_2_c,rfDC-raw.fif ."

jumeg.jumeg_utils.check_jumeg_standards(raw_fname)

# Filter Functions
#jumeg.jumeg_preprocessing.apply_filter(raw_fname)

fclean = raw_fname.split('-')[0] + ',bp1-45Hz-raw.fif'

# Evoked Functions
#jumeg.jumeg_preprocessing.apply_average(fclean)

# ICA Functions
#jumeg.jumeg_preprocessing.apply_ica(fclean)

fica_name = fclean.strip('-raw.fif') + '-ica.fif'
#jumeg.jumeg_utils.check_jumeg_standards(fica_name)
#jumeg.jumeg_preprocessing.apply_ica_cleaning(fica_name)

# CTPS Functions

#jumeg.jumeg_preprocessing.apply_ctps(fica_name)
fctps_name = '109925_CAU01A_100715_0842_2_c,rfDC,bp1-45Hz,ctps-trigger.npy'
#jumeg.jumeg_preprocessing.apply_ctps_select_ic(fctps_name)
