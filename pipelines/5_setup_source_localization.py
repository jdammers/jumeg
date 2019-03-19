#!/usr/bin/env python

"""
Python wrapper scripts to setup SUBJECTS_DIR for source localization.

1. Create / setup directories for source localization.
2. Construct brain surfaces from MR.
3. Setup the source space.
4. Perform coregistration
  With subject MRI
- https://de.slideshare.net/mne-python/mnepythyon-coregistration-28598463
  Without subject MRI
- https://de.slideshare.net/mne-python/mne-pythyon-coregistration

Setting up source space done, start forward and inverse model computation.
"""

import os
import os.path as op
import mne
from mne.commands import mne_make_scalp_surfaces
from utils import mksubjdirs

# using subprocess to run freesurfer commands (other option os.system)
# e.g. call(["ls", "-l"])
from subprocess import call

import yaml

with open('config_file.yaml', 'r') as f:
    config = yaml.load(f)

###############################################################################
# Get settings from config
###############################################################################

# directories
basedir = config['basedir']
subjects_dir = op.join(basedir, config['subjects_dir'])

# subject list
subjects = config['subjects']

###############################################################################
# Set environment variables
###############################################################################

os.environ['SUBJECTS_DIR'] = subjects_dir

mri_dir = '/mri/orig/001.mgz'
nii_fname = '_1x1x1mm.nii'

mne_bin_path = '/Users/kiefer/mne/MNE-2.7.3-3268-MacOSX-i386/bin/'  # Path to MNE-C binary files
freesurfer_home = '/Users/kiefer/mne/freesurfer/'
freesurfer_bin = freesurfer_home + 'bin/'

# determiny granularity of source space
spacing = 4  # e.g., 4 for ico4

###############################################################################
# loop across subject list for recon-all (time intensive: 11+ h per subject)
###############################################################################

# to convert from dicom to nii use mri_convert:
# mri_convert -it siemens_dicom -i path_to_dicom -ot nii -o recordings_dir/subjid/subjid_1x1x1mm_orig.nii.gz
for subj in subjects:

    print('Setting up freesurfer surfaces and source spaces for %s' % subj)

    # Makes subject directories and dir structures
    # alternative to using freesurfer's mksubjdirs, which is kind of buggy
    mksubjdirs(subjects_dir, subj)

    # Convert NIFTI files to mgz format that is read by freesurfer
    # do not do this step if file is already in 001.mgz format

    call([freesurfer_bin + 'mri_convert',
          op.join(subjects_dir, subj, subj + nii_fname),
          op.join(subjects_dir, subj + mri_dir)])

    # Reconstruct all surfaces and basically everything else.
    # Computationally intensive! (11+ h)
    # If talairach fails use: tkregister2 --mgz --s subj for manual adjustment
    # https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/Talairach_freeview

    call([freesurfer_bin + 'recon-all', '-autorecon-all', '-subjid', subj])

    # Set up the MRI data for forward model.
    call([mne_bin_path + 'mne_setup_mri', '--subject', subj, '--overwrite'])
    # $MNE_BIN_PATH/mne_setup_mri --subject $i --overwrite (if needed)

    # Setting up of Triangulation files
    # call([mne_bin_path + 'mne_watershed_bem', '--subject', subj])
    # $MNE_BIN_PATH/mne_watershed_bem --subject $i --overwrite (available in python, use the python version)


###############################################################################
# loop across subject list for source space and bem mesh construction
###############################################################################

for subj in subjects:

    # creates brain, inner_skull, outer_skull, and outer_skin and
    # saves them in the current folder
    # creates the files in bem/watershed/
    bem = mne.bem.make_watershed_bem(subject=subj, subjects_dir=subjects_dir, overwrite=True)

    # Set up the source space - creates source space description in fif format in /bem
    # setting it up as ico is useful to create labels in the future. - 06.10.14
    src = mne.setup_source_space(subj, spacing='ico%d' % spacing, surface='white',
                                 subjects_dir=subjects_dir, n_jobs=1)

    # save source spaces
    src_fname = op.join(subjects_dir, subj, 'bem', subj + '-ico-%d-src.fif' % spacing)
    src.save(src_fname, overwrite=True)

    # make lh.smseghead
    call([freesurfer_bin + 'mkheadsurf', '-subjid', subj])

    head = op.join(subjects_dir, subj, 'bem', subj + '-head.fif')
    mne_make_scalp_surfaces._run(subjects_dir=subjects_dir, subject=subj,
                                 force='--force', no_decimate=True, overwrite=True,
                                 verbose=True)

    # refer to MNE cookbook for more details
    head_medium = op.join(subjects_dir, subj, 'bem', subj + '-head-medium.fif')
    print ('linking %s as main head surface..' % head_medium)
    call(['ln', '-s', head_medium, head])

    # create the -bem.fif, .surf and -bem-sol.fif files
    success = call([mne_bin_path + 'mne_setup_forward_model', '--subject', subj,
                    '--ico', str(spacing), '--surf'])

    # if mne_setup_forward_model returns an error when checking the layers
    # adjust scalp surface etc. with one of the following commands (try different values):

    # call([mne_bin_path + 'mne_setup_forward_model', '--subject', subj,
    #    '--ico', '4', '--surf', '--outershift', '40', '--scalpshift', '40'])
    # call([mne_bin_path + 'mne_setup_forward_model', '--subject', subj,
    #    '--ico', '4', '--surf', '--outershift', '10', '--scalpshift', '10'])
    # call([mne_bin_path + 'mne_setup_forward_model', '--subject', subj, '--ico',
    #    '4', '--surf', '--innershift', '-10'])

    if not success == 0:
        raise RuntimeError("mne_setup_forward_model failed.")

# Align the coordinate frames using either mne coreg in terminal (recommended)
# or mne.gui.coregistration() in python - results in a subj-trans.fif file
