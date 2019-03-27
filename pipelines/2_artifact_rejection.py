import os
import os.path as op
import mne
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


ecg_ch = config['ica']['ecg_ch']
eog_hor_ch = config['ica']['eog_hor_ch']
eog_ver_ch = config['ica']['eog_ver_ch']

flow_ecg = config['ica']['flow_ecg']
fhigh_ecg = config['ica']['fhigh_ecg']
flow_eog = config['ica']['flow_eog']
fhigh_eog = config['ica']['fhigh_eog']

ecg_thresh = config['ica']['ecg_thresh']
eog_thresh = config['ica']['eog_thresh']
use_jumeg = config['ica']['use_jumeg']
random_state = config['ica']['random_state']
unfiltered = config['ica']['unfiltered']

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
                clean_filt_list, clean_unfilt_list = chop_and_apply_ica(raw_filt_fname, pre_proc_ext, chop_length=60,
                                                                        flow_ecg=flow_ecg, fhigh_ecg=fhigh_ecg,
                                                                        flow_eog=flow_eog, fhigh_eog=fhigh_eog,
                                                                        ecg_thresh=ecg_thresh, eog_thresh=eog_thresh,
                                                                        random_state=random_state, ecg_ch=ecg_ch,
                                                                        eog_hor=eog_hor_ch, eog_ver=eog_ver_ch,
                                                                        exclude='bads', use_jumeg=use_jumeg,
                                                                        unfiltered=unfiltered, save=True)

                clean_filt_concat = mne.concatenate_raws(clean_filt_list)
                clean_filt_concat.save(clean_filt_fname)

                if unfiltered:
                    clean_unfilt_concat = mne.concatenate_raws(clean_unfilt_list)
                    clean_unfilt_concat.save(clean_unfilt_fname)


