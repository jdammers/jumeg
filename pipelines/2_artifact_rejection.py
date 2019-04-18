import os
import os.path as op
import yaml

from chop_and_apply_ica import chop_and_apply_ica

with open('config_file.yaml', 'r') as f:
    config = yaml.load(f)

###############################################################################
# Get settings from config
###############################################################################

# directories
basedir = config['basedir']
recordings_dir = op.join(basedir, config['recordings_dir'])

# subject list
subjects = config['subjects']

ica_cfg = config['ica']
unfiltered = ica_cfg['unfiltered']

pre_proc_ext = config['pre_proc_ext']

for subj in subjects:
    print("Filtering for subject %s" % subj)

    dirname = op.join(recordings_dir, subj)
    sub_file_list = os.listdir(dirname)

    for raw_fname in sub_file_list:

        if raw_fname.endswith(pre_proc_ext + '-raw.fif'):

            raw_filt_fname = op.join(dirname, raw_fname)

            if raw_fname.endswith('-raw.fif'):
                clean_filt_fname = raw_filt_fname.rsplit('-raw.fif')[0] + ',ar-raw.fif'

            clean_unfilt_fname = clean_filt_fname.replace(',fibp', '')

            if not op.exists(clean_filt_fname) or (unfiltered and not op.exists(clean_unfilt_fname)):

                # creates list of small, cleaned chops
                clean_filt, clean_unfilt = chop_and_apply_ica(raw_filt_fname, ica_cfg)

                clean_filt.save(clean_filt_fname)

                if unfiltered:
                    clean_unfilt.save(clean_unfilt_fname)


