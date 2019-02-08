#!/usr/bin/env python

'''Python wrapper scripts to setup SUBJECTS_DIR for source localization,
   compute geometry surfaces and compute the forward operator.
'''

import os
import sys
from mne import setup_source_space
from .jumeg_utils import (get_files_from_list, check_env_variables,
                          retcode_error)

# using subprocess to run freesurfer commands (other option os.system)
# e.g. call(["ls", "-l"]) or call('ls -lrt', shell=True)
from subprocess import call


def make_subject_dirs(subjects_list, freesurfer_home=None):
    ''' Python wrapper for mksubjdirs.'''

    freesurfer_home = check_env_variables(freesurfer_home, key='FREESURFER_HOME')
    freesurfer_bin = os.path.join(freesurfer_home, 'bin', '')

    for subj in subjects_list:
        print('Making freeesurfer directories %s' % (subj))
        # Makes subject directories and dir structures
        retcode = call([freesurfer_bin + 'mksubjdirs', subj])
        if retcode != 0:
            retcode_error('mksubjdirs', subj)
            continue


def setup_mri_surfaces(subjects_list, subjects_dir=None, mne_root=None,
                       freesurfer_home=None, mri_convert=False, recon_all=False,
                       mri_extn='mri/orig/001.mgz', nii_extn='_1x1x1mm.nii'):
    '''Function to perform complete surface reconstruction and related operations.
       The following commands are executed:
       mksubjdirs, mri_convert, recon-all, mne_setup_mri, mne.setup_source_space,
       mne_watershed_bem, mne_make_scalp_surfaces.

       mri_convert is used to convert files from NIFTI (nii) to Freesurfer (mgz) format.
       It is best to run the conversion via command line beforehand.
    '''
    from mne.commands import mne_make_scalp_surfaces

    subjects_dir = check_env_variables(subjects_dir, key='SUBJECTS_DIR')
    freesurfer_home = check_env_variables(freesurfer_home, key='FREESURFER_HOME')
    mne_bin_path = os.path.join(mne_root, 'bin', '')
    freesurfer_bin = os.path.join(freesurfer_home, 'bin', '')
    print(subjects_dir, freesurfer_home, mne_bin_path, freesurfer_bin)

    # some commands requires it, setting it here for safety
    os.environ['DISPLAY'] = ':0'

    for subj in subjects_list:

        print('Setting up freesurfer surfaces and source spaces for %s' % (subj))

        if mri_convert:
            # Convert NIFTI files to mgz format that is read by freesurfer
            nii_file = os.path.join(subjects_dir, subj, subj + nii_extn)
            mri_file = os.path.join(subjects_dir, subj, mri_extn)
            print(nii_file, mri_file)
            retcode = call([freesurfer_bin + 'mri_convert', nii_file, mri_file])
            if retcode != 0:
                retcode_error('mri_convert', subj)
                continue

        if recon_all:
            # Reconstruct all surfaces and basically everything else.
            retcode = call([freesurfer_bin + 'recon-all', '-autorecon-all', '-subjid', subj])
            if retcode != 0:
                retcode_error('recon-all', subj)
                continue

        # Set up the MRI data for forward model.
        retcode = call([mne_bin_path + 'mne_setup_mri', '--subject', subj])
        if retcode != 0:
            retcode_error('mne_setup_mri', subj)
            continue

        # $MNE_BIN_PATH/mne_setup_mri --subject $i --overwrite (if needed)

        # Set up the source space - creates source space description in fif format in /bem
        # setting it up as ico is useful to create labels in the future. - 06.10.14
        try:
            setup_source_space(subj, fname=True, spacing='ico4',
                               surface='white', overwrite=False,
                               subjects_dir=subjects_dir, n_jobs=2)
        except:
            retcode_error('mne.setup_source_space', subj)
            continue

        # Setting up of Triangulation files
        retcode = call([mne_bin_path + 'mne_watershed_bem', '--overwrite', '--subject', subj])
        if retcode != 0:
            retcode_error('mne_watershed_bem', subj)
            continue

        # $MNE_BIN_PATH/mne_watershed_bem --subject $i --overwrite (soon available in python)

        # Setting up the surface files
        watershed_dir = os.path.join(subjects_dir, subj, 'bem/watershed', '')
        surf_dir = os.path.join(subjects_dir, subj, 'bem', '')
        if not os.path.isdir(watershed_dir) or not os.path.isdir(surf_dir):
            print('BEM directories /bem/watershed or /bem/ not found.')

        call(['ln', '-s', watershed_dir + subj + '_brain_surface', surf_dir + subj + '-brain.surf'])
        call(['ln', '-s', watershed_dir + subj + '_inner_skull_surface', surf_dir + subj + '-inner_skull.surf'])
        call(['ln', '-s', watershed_dir + subj + '_outer_skin_surface', surf_dir + subj + '-outer_skin.surf'])
        call(['ln', '-s', watershed_dir + subj + '_outer_skull_surface', surf_dir + subj + '-outer_skull.surf'])

        # making fine level surfaces
        retcode = call([freesurfer_bin + 'mkheadsurf', '-s', subj])
        if retcode != 0:
            retcode_error('mkheadsurf', subj)
            continue

        try:
            mne_make_scalp_surfaces._run(subjects_dir, subj, force=True,
                                         overwrite=True, no_decimate=False,
                                         verbose=True)
        except:
            retcode_error('mne_make_scalp_surfaces', subj)
            continue
        # refer to MNE cookbook for more details

        head = os.path.join(subjects_dir, subj, 'bem/', subj + '-head.fif')
        head_bkp = os.path.join(subjects_dir, subj, 'bem/', subj + '-head.fif_orig')
        if os.path.isfile(head):
            os.rename(head, head_bkp)
        head_medium = os.path.join(subjects_dir, subj, 'bem/', subj + '-head-medium.fif')
        print('linking %s as main head surface..' % (head_medium))
        call(['ln', '-s', head_medium, head])

        print('Surface reconstruction routines completed for subject %s' % (subj))


def setup_forward_model(subjects_list, subjects_dir=None, mne_root=None):
    '''Setup forward model.

    Setting up of the forward model - computes Boundary Element Model geometry files.
    Creates the -bem.fif, .surf and -bem-sol.fif files

    If skin and/or skull surfaces intersect causing errors, it is possible to try and
    shift the inner skull surface inwards or the our skin surface outwards.

    The corresponding command to use would be:
    mne_setup_forward_model --subject subj --ico 4 --surf --outershift 10 --scalpshift 10
    mne_setup_forward_model --subject subj --ico 4 --surf --innershift 10
    '''

    subjects_dir = check_env_variables(subjects_dir, key='SUBJECTS_DIR')
    mne_root = check_env_variables(mne_root, key='MNE_ROOT')
    mne_bin_path = os.path.join(mne_root, 'bin', '')

    for subj in subjects_list:

        print('Setting up BEM boundary model for %s' % (subj))

        # call([mne_bin_path + 'mne_setup_forward_model', '--subject', subj, '--ico', '4',
        #      '--surf', '--outershift', '10', '--scalpshift', '10'])
        retcode = call([mne_bin_path + 'mne_setup_forward_model', '--subject', subj, '--ico', '4', '--surf'])
        if retcode != 0:
            retcode_error('mne_setup_forward_model', subj)
            continue

    print ('Next step is to align the MRI and MEG coordinate frames using '
           'mne.coregistration.gui() resulting in a -trans.fif file.')


def compute_forward_solution(fname_raw, subjects_dir=None, spacing='ico4',
                             mindist=5, eeg=False, overwrite=False):
    '''Performs forward solution computation using mne_do_foward_solution
       (uses MNE-C binaries).

       Requires bem sol files, and raw meas file)

    Input
    -----
    fname_raw : str or list
    List of raw files for which to compute the forward operator.

    Returns
    -------
    None. Forward operator -fwd.fif will be saved.
    '''

    from mne import do_forward_solution

    fnames = get_files_from_list(fname_raw)
    subjects_dir = check_env_variables(subjects_dir, key='SUBJECTS_DIR')

    for fname in fnames:
        print('Computing fwd solution for %s' % (fname))

        basename = os.path.basename(fname).split('-raw.fif')[0]
        subject = basename.split('_')[0]
        meas_fname = os.path.basename(fname)
        fwd_fname = meas_fname.rsplit('-raw.fif')[0] + '-fwd.fif'
        src_fname = subject + '-ico-4-src.fif'
        bem_fname = subject + '-5120-5120-5120-bem-sol.fif'
        trans_fname = subject + '-trans.fif'

        fwd = do_forward_solution(subject, meas_fname, fname=fwd_fname, src=src_fname,
                                  spacing=spacing, mindist=mindist, bem=bem_fname, mri=trans_fname,
                                  eeg=eeg, overwrite=overwrite, subjects_dir=subjects_dir)

        # fwd['surf_ori'] = True
        # to read forward solutions
        # fwd = mne.read_forward_solution(fwd_fname)

        print('Forward operator saved in file %s' % (fwd_fname))
