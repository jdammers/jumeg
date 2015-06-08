#!/usr/bin/env python

'''Python wrapper scripts to setup SUBJECTS_DIR for source localization, compute geometry surfaces
   and compute the forward operator.
'''

import sys
import os
import mne

from mne.utils import get_subjects_dir
from jumeg.jumeg_utils import get_files_from_list

# using subprocess to run freesurfer commands (other option os.system)
# e.g. call(["ls", "-l"]) or call('ls -lrt', shell=True)
from subprocess import call


def retcode_error(command, subj):
    print '%s did not run successfully for subject %s.' % (command, subj)
    print 'Please check the arguments, and rerun for subject.'
 

def setup_mri_surfaces(subjects_list, subjects_dir=None, mne_root=None, freesurfer_home=None,
                       mri_convert=False, mri_file='/mri/orig/001.mgz',
                       nii_file='_1x1x1mm.nii'):
    '''Function to perform complete surface reconstruction and related operations.
       The following commands are executed: 
       mksubjdirs, mri_convert, recon-all, mne_setup_mri, mne.setup_source_space,
       mne_watershed_bem, mne_make_scalp_surfaces.

       mri_convert is used to convert files from NIFTI (nii) to Freesurfer (mgz) format.
       It is best to run the conversion via command line beforehand.
    '''

    if subjects_dir is None and 'SUBJECTS_DIR' in os.environ:
        subjects_dir = os.environ['SUBJECTS_DIR']
    else:
        print 'Please set SUBJECTS_DIR.'

    if mne_root is None and 'MNE_ROOT' in os.environ:
        mne_root = os.environ['MNE_ROOT']
    else:
        print 'Please set MNE_ROOT.'
    mne_bin_path = mne_root + '/bin/'
    
    if freesurfer_home is None and 'FREESURFER_HOME' in os.environ:
        freesurfer_home = os.environ['FREESURFER_HOME']
    else:
        print 'Please set FREESURFER_HOME.'
    freesurfer_bin = freesurfer_home + '/bin/'

    # some commands requires it, setting it here for safety
    os.environ['DISPLAY'] = ':0'

    for subj in subjects_list:
    
        print 'Setting up freesurfer surfaces and source spaces for %s' % (subj)
    
        # Makes subject directories and dir structures
        retcode = call([freesurfer_bin + 'mksubjdirs', '-p', subj])
        if retcode != 0:
            retcode_error('mksubjdirs', subj)
            continue
    
        if mri_convert:
            # Convert NIFTI files to mgz format that is read by freesurfer
            retcode = call([freesurfer_bin + 'mri_convert', subj + '/' + subj + nii_file, subj + mri_file])
            if retcode != 0:
                retcode_error('mri_convert', subj)
                continue
    
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

        #$MNE_BIN_PATH/mne_setup_mri --subject $i --overwrite (if needed)
    
        # Set up the source space - creates source space description in fif format in /bem
        # setting it up as ico is useful to create labels in the future. - 06.10.14
        try: 
            mne.setup_source_space(subj, fname=True, spacing='ico4',
                                   surface='white', overwrite=False,
                                   subjects_dir=subjects_dir, n_jobs=2)
        except:
            retcode_error('mne.setup_source_space', subj)
            continue
    
        # Setting up of Triangulation files
        retcode = call([mne_bin_path + 'mne_watershed_bem', '--subject', subj])
        if retcode != 0:
            retcode_error('mne_watershed_bem', subj)
            continue

        # $MNE_BIN_PATH/mne_watershed_bem --subject $i --overwrite (soon available in python)
    
        # Setting up the surface files
        watershed_dir = subjects_dir + subj + '/bem/watershed/' + subj
        surf_dir = subjects_dir + subj + '/bem/' + subj
        if not os.path.isdir(watershed_dir) or not os.path.isdir(surf_dir):
            print 'BEM directories /bem/watershed or /bem/ not found.'

        call(['ln', '-s', watershed_dir + '_brain_surface', surf_dir + '-brain.surf'])
        call(['ln', '-s', watershed_dir + '_inner_skull_surface', surf_dir + '-inner_skull.surf'])
        call(['ln', '-s', watershed_dir + '_outer_skin_surface',  surf_dir + '-outer_skin.surf'])
        call(['ln', '-s', watershed_dir + '_outer_skull_surface', surf_dir + '-outer_skull.surf'])
    
        from mne.commands import mne_make_scalp_surfaces
        try: 
            mne_make_scalp_surfaces._run(subjects_dir, subj, force=True, overwrite=True, verbose=True)
        except:
            retcode_error('mne_make_scalp_surfaces', subj)
            continue
        # refer to MNE cookbook for more details

        head = subjects_dir + subj + '/bem/' + subj + '-head.fif'
        head_medium = subjects_dir + subj + '/bem/' + subj + '-head-medium.fif'
        print 'linking %s as main head surface..' % (head_medium)
        call(['ln', '-s', head_medium, head])

        print 'Surface reconstruction routines completed for subject %s' % (subj)


def setup_forward_model(subjects, subjects_dir=None, mne_root=None):
    '''Setup forward model.

    Setting up of the forward model - computes Boundary Element Model geometry files.
    Creates the -bem.fif, .surf and -bem-sol.fif files

    If skin and/or skull surfaces intersect causing errors, it is possible to try and
    shift the inner skull surface inwards or the our skin surface outwards.

    The corresponding command to use would be: 
    mne_setup_forward_model --subject subj --ico 4 --surf --outershift 10 --scalpshift 10
    mne_setup_forward_model --subject subj --ico 4 --surf --innershift 10
    '''

    if subjects_dir is None and 'SUBJECTS_DIR' in os.environ:
        subjects_dir = os.environ['SUBJECTS_DIR']
    else:
        print 'Please set SUBJECTS_DIR.'

    if mne_root is None and 'MNE_ROOT' in os.environ:
        mne_root = os.environ['MNE_ROOT']
    else:
        print 'Please set MNE_ROOT.'
    mne_bin_path = mne_root + '/bin/'

    for subj in subjects_list:

        print 'Setting up BEM boundary model for %s' % (subj)

        #call([mne_bin_path + 'mne_setup_forward_model', '--subject', subj, '--ico', '4', '--surf', '--outershift', '10', '--scalpshift', '10'])
        retcode = call([mne_bin_path + 'mne_setup_forward_model', '--subject', subj, '--ico', '4', '--surf'])
        if retcode != 0:
            retcode_error('mne_setup_forward_model', subj)
            continue
    
    print 'Next step is to align the MRI and MEG coordinate frames using mne.coregistration.gui() resulting in a -trans.fif file.'


def compute_forward_solution(fname_raw, subjects_dir=None, spacing='ico4', mindist=5, eeg=False, overwrite=False):
    '''Performs forward solution computation using mne_do_foward_solution (uses MNE-C binaries)
       Requires bem sol files, and raw meas file)

    Input
    -----
    fname_raw : str or list
    List of raw files for which to compute the forward operator.

    Returns
    -------
    None. Forward operator -fwd.fif will be saved.
    '''
    fnames = get_files_from_list(fname_raw)

    if subjects_dir is None and 'SUBJECTS_DIR' in os.environ:
        subjects_dir = os.environ['SUBJECTS_DIR']
    else:
        print 'Please set SUBJECTS_DIR.'

    for fname in fnames:
        print 'Computing fwd solution for %s' % (fname)

        basename = os.path.basename(fname).split('-raw.fif')[0]
        subject = basename.split('_')[0]
        meas_fname = os.path.basename(fname)
        fwd_fname = meas_fname.rsplit('-raw.fif')[0] + '-fwd.fif'
        src_fname = subject + '-ico-4-src.fif'
        bem_fname = subject + '-5120-5120-5120-bem-sol.fif'
        trans_fname = subject + '-trans.fif'
            
        fwd = mne.do_forward_solution(subject, meas_fname, fname=fwd_fname, src=src_fname,
                                      spacing=spacing, mindist=mindist, bem=bem_fname, mri=trans_fname,
                                      eeg=eeg, overwrite=overwrite, subjects_dir=subjects_dir)

        #fwd['surf_ori'] = True
        # to read forward solutions
        # fwd = mne.read_forward_solution(fwd_fname)

        print 'Forward operator saved in file %s' % (fwd_fname)
