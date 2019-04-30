#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Authors: Daniel van de Velden (d.vandevelden@yahoo.de)
#
# License: BSD (3-clause)

from jumeg.jumeg_utils import loadingBar
import mne
import numpy as np
from scipy.optimize import leastsq
from sklearn.neighbors import BallTree
from mne.transforms import (rotation, rotation3d, scaling,
                            translation, apply_trans)
from mne.source_space import _get_lut, _get_lut_id, _get_mgz_header
import nibabel as nib
from scipy import linalg
from scipy.spatial.distance import cdist
from time import localtime, strftime
import time as time2
import os
import os.path as op
from scipy.interpolate import griddata
from mne.source_estimate import _write_stc
from nilearn import plotting
from nilearn.image import index_img
from nibabel.affines import apply_affine
import matplotlib.pyplot as plt
from mne.source_estimate import VolSourceEstimate
# from matplotlib import cm
from matplotlib.ticker import LinearLocator

from nilearn.plotting.img_plotting import _MNI152Template
from functools import reduce

import logging
logger = logging.getLogger("root")

MNI152TEMPLATE = _MNI152Template()

# =============================================================================
#
# =============================================================================
def convert_to_unicode(inlist):
    if type(inlist) != str:
        inlist = inlist.decode('utf-8')
        return inlist
    else:
        return inlist


def read_vert_labelwise(fname_src, subject, subjects_dir):
    """Read the labelwise vertice file and remove duplicates.
        
    Parameters
    ----------
    fname_src : string
        Path to a source space file.
    subject : str | None
        The subject name. It is necessary to set the
        subject parameter to avoid analysis errors.
    subjects_dir : string
        Path to SUBJECTS_DIR if it is not set in the environment.
    
    Returns
    -------
    label_dict : dict
        A dict containing all labels available for the subject's source space
        and their respective vertex indices
    
    """
    fname_labels = fname_src[:-4] + '_vertno_labelwise.npy'
    label_dict = np.load(fname_labels, encoding='latin1').item()
    subj_vert_src = mne.read_source_spaces(fname_src)
    label_dict = _remove_vert_duplicates(subject, subj_vert_src, label_dict,
                                         subjects_dir)
    del subj_vert_src

    return label_dict


def _point_cloud_error_balltree(subj_p, temp_tree):
    """Find the distance from each source point to its closest target point.
    Uses sklearn.neighbors.BallTree for greater efficiency"""
    dist, _ = temp_tree.query(subj_p)
    err = dist.ravel()
    return err


def _point_cloud_error(src_pts, tgt_pts):
    """Find the distance from each source point to its closest target point.
    Parameters."""
    Y = cdist(src_pts, tgt_pts, 'euclidean')
    dist = Y.min(axis=1)
    return dist


def _trans_from_est(params):
    """Convert transformation parameters into a transformation matrix."""
    i = 0
    trans = []
    x, y, z = params[:3]
    trans.insert(0, translation(x, y, z))
    i += 3
    x, y, z = params[i:i + 3]
    trans.append(rotation(x, y, z))
    i += 3
    x, y, z = params[i:i + 3]
    trans.append(scaling(x, y, z))
    i += 3
    trans = reduce(np.dot, trans)
    return trans


def _get_scaling_factors(s_pts, t_pts):

    """
    Calculate scaling factors to match the size of the subject
    brain and the template brain.

    Paramters:
    ----------
    s_pts : np.array
        Coordinates of the vertices in a given label from the
        subject source space.
    t_pts :  np.array
        Coordinates of the vertices in a given label from the
        template source space.

    Returns:
    --------

    """
    # Get the x-,y-,z- min and max Limits to create the span for each axis
    s_x, s_y, s_z = s_pts.T
    s_x_diff = np.max(s_x) - np.min(s_x)
    s_y_diff = np.max(s_y) - np.min(s_y)
    s_z_diff = np.max(s_z) - np.min(s_z)
    t_x, t_y, t_z = t_pts.T
    t_x_diff = np.max(t_x) - np.min(t_x)
    t_y_diff = np.max(t_y) - np.min(t_y)
    t_z_diff = np.max(t_z) - np.min(t_z)

    # Calculate a scaling factor for the subject to match template size
    # and avoid 'Nan' by zero division

    # instead of comparing float with zero, check absolute value up to a given precision
    precision = 1e-18
    if np.fabs(t_x_diff) < precision or np.fabs(s_x_diff) < precision:
        x_scale = 0.
    else:
        x_scale = t_x_diff / s_x_diff

    if np.fabs(t_y_diff) < precision or np.fabs(s_y_diff) < precision:
        y_scale = 0.
    else:
        y_scale = t_y_diff / s_y_diff

    if np.fabs(t_z_diff) < precision or np.fabs(s_z_diff) < precision:
        z_scale = 0.
    else:
        z_scale = t_z_diff / s_z_diff

    return x_scale, y_scale, z_scale


def _get_best_trans_matrix(init_trans, s_pts, t_pts, template_spacing, e_func):
    """
    Calculate the least squares error for different variations of
    the initial transformation and return the transformation with
    the minimum error

    Parameters:
    -----------
    init_trans : np.array of shape (4, 4)
        Numpy array containing the initial transformation matrix.
    s_pts : np.array
        Coordinates of the vertices in a given label from the
        subject source space.
    t_pts :  np.array
        Coordinates of the vertices in a given label from the
        template source space.
    template_spacing : float
        Grid spacing for the template source space.
    e_func : str
        Either 'balltree' or 'euclidean'.

    Returns:
    --------

    trans :

    err_stats : [dist_mean, dist_max, dist_var, dist_err]
    """

    if e_func == 'balltree':
        errfunc = _point_cloud_error_balltree
        temp_tree = BallTree(t_pts)
    elif e_func == 'euclidean':
        errfunc = _point_cloud_error
        temp_tree = None

    # Find calculate the least squares error for variation of the initial transformation
    poss_trans = find_optimum_transformations(init_trans, s_pts, t_pts, template_spacing,
                                              e_func, temp_tree, errfunc)
    dist_max_list = []
    dist_mean_list = []
    dist_var_list = []
    dist_err_list = []
    for tra in poss_trans:
        points_to_match = s_pts
        points_to_match = apply_trans(tra, points_to_match)

        if e_func == 'balltree':
            template_pts = temp_tree
        elif e_func == 'euclidean':
            template_pts = t_pts

        dist_mean_list.append(np.mean(errfunc(points_to_match[:, :3], template_pts)))
        dist_var_list.append(np.var(errfunc(points_to_match[:, :3], template_pts)))
        dist_max_list.append(np.max(errfunc(points_to_match[:, :3], template_pts)))
        dist_err_list.append(errfunc(points_to_match[:, :3], template_pts))

        del points_to_match

    dist_mean_arr = np.asarray(dist_mean_list)

    # Select the best fitting Transformation-Matrix
    idx1 = np.argmin(dist_mean_arr)

    # Collect all values belonging to the optimum solution
    trans = poss_trans[idx1]
    dist_max = dist_max_list[idx1]
    dist_mean = dist_mean_list[idx1]
    dist_var = dist_var_list[idx1]
    dist_err = dist_err_list[idx1]

    del poss_trans
    del dist_mean_arr
    del dist_mean_list
    del dist_max_list
    del dist_var_list
    del dist_err_list

    err_stats = [dist_mean, dist_max, dist_var, dist_err]

    return trans, err_stats


def auto_match_labels(fname_subj_src, label_dict_subject,
                      fname_temp_src, label_dict_template,
                      subjects_dir, volume_labels, template_spacing,
                      e_func, fname_save, save_trans=False):
    """
    Matches a subject's volume source space labelwise to another volume
    source space
    
    Parameters
    ----------
    fname_subj_src : string
        Filename of the first volume source space.
    label_dict_subject : dict
        Dictionary containing all labels and the numbers of the
        vertices belonging to these labels for the subject.
    fname_temp_src : string
        Filename of the second volume source space to match on.
    label_dict_template : dict
        Dictionary containing all labels and the numbers of the
        vertices belonging to these labels for the template.
    volume_labels : list of volume Labels
        List of the volume labels of interest
    subjects_dir : str
        Path to the subject directory.
    template_spacing : int | float
        The grid distances of the second volume source space in mm
    e_func : string | None
        Error function, either 'balltree' or 'euclidean'. If None, the
        default 'balltree' function is used.
    fname_save : str
        File name under which the transformation matrix is to be saved.
    save_trans : bool
        If it is True the transformation matrix for each label is saved
        as a dictionary. False is default
  
    Returns
    -------
    label_trans_dic : dict
        Dictionary of all the labels transformation matrizes
    label_trans_dic_err : dict
        Dictionary of all the labels transformation matrizes distance
        errors (mm)
    label_trans_dic_mean_dist : dict
        Dictionary of all the labels transformation matrizes mean
        distances (mm)
    label_trans_dic_max_dist : dict
        Dictionary of all the labels transformation matrizes max
        distance (mm)
    label_trans_dic_var_dist : dict
        Dictionary of all the labels transformation matrizes distance
        error variance (mm)
    """

    if e_func == 'balltree':
        err_function = 'BallTree Error Function'
    elif e_func == 'euclidean':
        err_function = 'Euclidean Error Function'
    else:
        print('No or invalid error function provided, using BallTree instead')
        err_function = 'BallTree Error Function'

    subj_src = mne.read_source_spaces(fname_subj_src)
    x, y, z = subj_src[0]['rr'].T
    # subj_p contains the coordinates of the vertices
    subj_p = np.c_[x, y, z]
    subject = subj_src[0]['subject_his_id']

    temp_src = mne.read_source_spaces(fname_temp_src)
    x1, y1, z1 = temp_src[0]['rr'].T
    # temp_p contains the coordinates of the vertices
    temp_p = np.c_[x1, y1, z1]
    template = temp_src[0]['subject_his_id']

    print("""\n#### Attempting to match %d volume source space labels from
    Subject: '%s' to Template: '%s' with
    %s...""" % (len(volume_labels), subject, template, err_function))

    # make sure to remove duplicate vertices before matching
    label_dict_subject = _remove_vert_duplicates(subject, subj_src, label_dict_subject,
                                                 subjects_dir)

    label_dict_template = _remove_vert_duplicates(template, temp_src, label_dict_template,
                                                  subjects_dir)

    vert_sum = 0
    vert_sum_temp = 0

    for label_i in volume_labels:
        vert_sum = vert_sum + label_dict_subject[label_i].shape[0]
        vert_sum_temp = vert_sum_temp + label_dict_template[label_i].shape[0]

        # check for overlapping labels
        for label_j in volume_labels:
            if label_i != label_j:
                h = np.intersect1d(label_dict_subject[label_i], label_dict_subject[label_j])
                if h.shape[0] > 0:
                    raise ValueError("Label %s contains %d vertices from label %s" % (label_i,
                                                                                      h.shape[0],
                                                                                      label_j))

    print('    # N subject vertices:', vert_sum)
    print('    # N template vertices:', vert_sum_temp)

    # Prepare empty containers to store the possible transformation results
    label_trans_dic = {}
    label_trans_dic_err = {}
    label_trans_dic_var_dist = {}
    label_trans_dic_mean_dist = {}
    label_trans_dic_max_dist = {}
    start_time = time2.time()
    del subj_src, temp_src

    for label_idx, label in enumerate(volume_labels):
        loadingBar(count=label_idx, total=len(volume_labels),
                   task_part='%s' % label)
        print('')

        # Select coords for label and check if they exceed the label size limit
        s_pts = subj_p[label_dict_subject[label]]
        t_pts = temp_p[label_dict_template[label]]

        # IIRC: the error function in find_optimum_transformations needs at least
        # 6 points. if all points are the same then this point is taken as
        # minimum -> for clarifications ask Daniel

        if s_pts.shape[0] == 0:
            raise ValueError("The label does not contain any vertices for the subject.")

        elif s_pts.shape[0] < 6:
            while s_pts.shape[0] < 6:
                s_pts = np.concatenate((s_pts, s_pts))

        if t_pts.shape[0] == 0:
            # Append the Dictionaries with the zeros since there is no label to
            # match the points
            trans = _trans_from_est(np.zeros([9, 1]))
            trans[0, 0], trans[1, 1], trans[2, 2] = 1., 1., 1.
            label_trans_dic.update({label: trans})
            label_trans_dic_mean_dist.update({label: np.min(0)})
            label_trans_dic_max_dist.update({label: np.min(0)})
            label_trans_dic_var_dist.update({label: np.min(0)})
            label_trans_dic_err.update({label: 0})
        else:

            # Calculate a scaling factor for the subject to match template size
            x_scale, y_scale, z_scale = _get_scaling_factors(s_pts, t_pts)

            # Find center of mass
            cm_s = np.mean(s_pts, axis=0)
            cm_t = np.mean(t_pts, axis=0)
            initial_transl = (cm_t - cm_s)

            # Create the the initial transformation matrix
            init_trans = np.zeros([4, 4])
            init_trans[:3, :3] = rotation3d(0., 0., 0.) * [x_scale, y_scale, z_scale]

            init_trans[0, 3] = initial_transl[0]
            init_trans[1, 3] = initial_transl[1]
            init_trans[2, 3] = initial_transl[2]
            init_trans[3, 3] = 1.

            # Calculate the least squares error for different variations of
            # the initial transformation and return the transformation with
            # the minimum error
            trans, err_stats = _get_best_trans_matrix(init_trans, s_pts, t_pts,
                                                      template_spacing, e_func)

            # TODO: test that the results are still the same
            [dist_mean, dist_max, dist_var, dist_err] = err_stats

            # Append the Dictionaries with the result and error values
            label_trans_dic.update({label: trans})
            label_trans_dic_mean_dist.update({label: dist_mean})
            label_trans_dic_max_dist.update({label: dist_max})
            label_trans_dic_var_dist.update({label: dist_var})
            label_trans_dic_err.update({label: dist_err})

    if save_trans:
        print('\n    Writing Transformation matrices to file..')
        fname_lw_trans = fname_save
        mat_mak_trans_dict = dict()
        mat_mak_trans_dict['ID'] = '%s -> %s' % (subject, template)
        mat_mak_trans_dict['Labeltransformation'] = label_trans_dic
        mat_mak_trans_dict['Transformation Error[mm]'] = label_trans_dic_err
        mat_mak_trans_dict['Mean Distance Error [mm]'] = label_trans_dic_mean_dist
        mat_mak_trans_dict['Max Distance Error [mm]'] = label_trans_dic_max_dist
        mat_mak_trans_dict['Distance Variance Error [mm]'] = label_trans_dic_var_dist
        mat_mak_trans_dict_arr = np.array([mat_mak_trans_dict])
        np.save(fname_lw_trans, mat_mak_trans_dict_arr)
        print('    [done] -> Calculation Time: %.2f minutes.' % (
            ((time2.time() - start_time) / 60)))

        return

    else:
        return (label_trans_dic, label_trans_dic_err, label_trans_dic_mean_dist,
                label_trans_dic_max_dist, label_trans_dic_var_dist)


def find_optimum_transformations(init_trans, s_pts, t_pts, template_spacing,
                                 e_func, temp_tree, errfunc):
    """
    Vary the initial transformation by a translation of up to three times the
    grid spacing and compute the transformation with the smallest least square
    error.

    Parameters:
    -----------
    init_trans : 4-D transformation matrix
        Initial guess of the transformation matrix from the subject brain to
        the template brain.
    s_pts :
        Vertex coordinates in the subject brain.
    t_pts :
        Vertex coordinates in the template brain.
    template_spacing : float
        Grid spacing of the vertices in the template brain.
    e_func : str
        Error function to use. Either 'balltree' or 'euclidian'.
    temp_tree :
        BallTree(t_pts) if e_func is 'balltree'.
    errfunc :
        The error function for the computation of the least squares error.

    Returns:
    --------
    poss_trans : list of 4-D transformation matrices
        List of one transformation matrix for each variation of the intial
        transformation with the smallest least squares error.

    """

    # template spacing in meters
    tsm = template_spacing / 1e3

    # Try different initial translations in space to avoid local minima
    # No label should require a translation by more than 3 times the grid spacing (tsm)
    auto_match_iters = np.array([[0., 0., 0.],
                                 [0., 0., tsm], [0., 0., tsm * 2], [0., 0., tsm * 3],
                                 [tsm, 0., 0.], [tsm * 2, 0., 0.], [tsm * 3, 0., 0.],
                                 [0., tsm, 0.], [0., tsm * 2, 0.], [0., tsm * 3, 0.],
                                 [0., 0., -tsm], [0., 0., -tsm * 2], [0., 0., -tsm * 3],
                                 [-tsm, 0., 0.], [-tsm * 2, 0., 0.], [-tsm * 3, 0., 0.],
                                 [0., -tsm, 0.], [0., -tsm * 2, 0.], [0., -tsm * 3, 0.]])

    # possible translation matrices
    poss_trans = []
    for p, ami in enumerate(auto_match_iters):

        # vary the initial translation value by adding ami
        tx, ty, tz = init_trans[0, 3] + ami[0], init_trans[1, 3] + ami[1], init_trans[2, 3] + ami[2]
        sx, sy, sz = init_trans[0, 0], init_trans[1, 1], init_trans[2, 2]
        rx, ry, rz = 0, 0, 0

        # starting point for finding the transformation matrix trans which
        # minimizes the error between np.dot(s_pts, trans) and t_pts
        x0 = (tx, ty, tz, rx, ry, rz)

        def error(x):
            tx, ty, tz, rx, ry, rz = x
            trans0 = np.zeros([4, 4])
            trans0[:3, :3] = rotation3d(rx, ry, rz) * [sx, sy, sz]
            trans0[0, 3] = tx
            trans0[1, 3] = ty
            trans0[2, 3] = tz
            # rotate and scale
            est = np.dot(s_pts, trans0[:3, :3].T)
            # translate
            est += trans0[:3, 3]
            if e_func == 'balltree':
                err = errfunc(est[:, :3], temp_tree)
            elif e_func == 'euclidean':
                err = errfunc(est[:, :3], t_pts)
            return err

        est, _, info, msg, _ = leastsq(error, x0, full_output=True)
        est = np.concatenate((est, (init_trans[0, 0],
                                    init_trans[1, 1],
                                    init_trans[2, 2])
                              ))
        trans = _trans_from_est(est)
        poss_trans.append(trans)

    return poss_trans


def _transform_src_lw(vsrc_subject_from, label_dict_subject_from,
                      volume_labels, subject_to,
                      subjects_dir, label_trans_dic=None):
    """Transformes given Labels of interest from one subjects' to another.
    
    Parameters
    ----------
    vsrc_subject_from : instance of SourceSpaces
        The source spaces that will be transformed.
    label_dict_subject_from : dict

    volume_labels : list
        List of the volume labels of interest
    subject_to : str | None
        The template subject.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    label_trans_dic : dict | None
        Dictionary containing transformation matrices for all labels (acquired
        by auto_match_labels function). If label_trans_dic is None the method
        will attempt to read the file from disc.
    
    Returns
    -------
    transformed_p : array
        Transformed points from subject volume source space to volume source
        space of the template subject.
    idx_vertices : array
        Array of idxs for all transformed vertices in the volume source space.
    """
    subj_vol = vsrc_subject_from
    subject = subj_vol[0]['subject_his_id']
    x, y, z = subj_vol[0]['rr'].T
    subj_p = np.c_[x, y, z]
    label_dict = label_dict_subject_from
    print("""\n#### Attempting to transform %s source space labelwise to 
    %s source space..""" % (subject, subject_to))

    if label_trans_dic is None:
        print('\n#### Attempting to read MatchMaking Transformations from file..')
        indiv_spacing = (np.abs(subj_vol[0]['rr'][0, 0]) -
                         np.abs(subj_vol[0]['rr'][1, 0])) * 1e3

        fname_lw_trans = op.join(subjects_dir, subject,
                                 '%s_%s_vol-%.2f_lw-trans.npy' % (subject, subject_to,
                                                                  indiv_spacing))

        try:
            mat_mak_trans_dict_arr = np.load(fname_lw_trans, encoding='latin1')

        except IOError:
            print('MatchMaking Transformations file NOT found:')
            print(fname_lw_trans, '\n')
            print('Please calculate the transformation matrix dictionary by using')
            print('the jumeg.jumeg_volmorpher.auto_match_labels function.')

            import sys
            sys.exit(-1)

        label_trans_ID = mat_mak_trans_dict_arr[0]['ID']
        print('    Reading MatchMaking file %s..' % label_trans_ID)
        label_trans_dic = mat_mak_trans_dict_arr[0]['Labeltransformation']
    else:
        label_trans_dic = label_trans_dic

    vert_sum = []
    for label_i in volume_labels:
        vert_sum.append(label_dict[label_i].shape[0])
        for label_j in volume_labels:
            if label_i != label_j:
                h = np.intersect1d(label_dict[label_i], label_dict[label_j])
                if h.shape[0] > 0:
                    print("In Label:", label_i, """ are vertices from
                           Label:""", label_j, "(", h.shape[0], ")")
                    break

    transformed_p = np.array([[0, 0, 0]])
    idx_vertices = []

    for idx, label in enumerate(volume_labels):
        loadingBar(idx, len(volume_labels), task_part=label)
        idx_vertices.append(label_dict[label])
        trans_p = subj_p[label_dict[label]]
        trans = label_trans_dic[label]
        # apply trans
        trans_p = apply_trans(trans, trans_p)
        del trans
        transformed_p = np.concatenate((transformed_p, trans_p))
        del trans_p
    transformed_p = transformed_p[1:]
    idx_vertices = np.concatenate(np.asarray(idx_vertices))
    print('    [done]')

    return transformed_p, idx_vertices


def set_unwanted_to_zero(vsrc, stc_data, volume_labels, label_dict):
    """

    Parameters:
    -----------
    vsrc : mne.VolSourceSpace
    stc_data : np.array
        data from source time courses.
    volume_labels : list of str
        List with volume labels of interest
    label_dict : dict
        Dictionary containing for each label the indices of the
        vertices which are part of the label.

    Returns:
    --------
    stc_data_mod : np.array()
        The modified stc_data array with data set to zero for
        vertices which are not part of the labels of interest.
    """

    # label of interest
    LOI_idx = list()

    for p, labels in enumerate(volume_labels):

        label_verts = label_dict[labels]

        for i in range(0, label_verts.shape[0]):

            LOI_idx.append(np.where(label_verts[i] == vsrc[0]['vertno']))

    LOI_idx = np.asarray(LOI_idx)

    stc_data_mod = np.zeros(stc_data.shape)
    stc_data_mod[LOI_idx, :] = stc_data[LOI_idx, :]

    return stc_data_mod


def volume_morph_stc(fname_stc_orig, subject_from, fname_vsrc_subject_from,
                     volume_labels, subject_to, fname_vsrc_subject_to,
                     cond, interpolation_method, normalize, subjects_dir,
                     unwanted_to_zero=True, label_trans_dic=None, run=None,
                     n_iter=None, fname_save_stc=None, save_stc=False, plot=False):
    """
    Perform volume morphing from one subject to a template.
    
    Parameters
    ----------
    fname_stc_orig : string
        Filepath of the original stc
    subject_from : string
        Name of the original subject as named in the SUBJECTS_DIR
    fname_vsrc_subject_from : str
        Filepath of the subjects volume source space
    volume_labels : list of volume Labels
        List of the volume labels of interest
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    fname_vsrc_subject_to : string
        Filepath of the template subjects volume source space
    interpolation_method : str
        Only 'linear' seeems to be working for 3D data. 'balltree' and
        'euclidean' only work for 2D?.
    cond : str (Not really needed)
        Experimental condition under which the data was recorded.
    normalize : bool
        If True, normalize activity patterns label by label before and after
        morphing.
    subjects_dir : str | None
        Path to SUBJECTS_DIR if it is not set in the environment.
    unwanted_to_zero : bool
       If True, set all non-Labels-of-interest in resulting stc to zero.
    label_trans_dic : dict | None
        Dictionary containing transformation matrices for all labels (acquired
        by auto_match_labels function). If label_trans_dic is None the method
        will attempt to read the file from disc.
    run : int | None
        Specifies the run if multiple measurements for the same condition
        were performed.
    n_iter : int | None
        If MFT was used for the inverse solution, n_iter is the
        number of iterations.
    fname_save_stc : str | None
        File name for the morphed volume stc file to be saved under.
        If fname_save_stc is None, use the standard file name convention.
    save_stc : bool
        True to save. False is default
    plot : bool
        Plot the morphed stc.
  
    Returns
    -------
    In Case of save_stc=True:
      stc_morphed : VolSourceEstimate
            Volume source estimate for the destination subject.
            
    In Case of save_stc=False:
      new_data : dict
          One or more new stc data array
    """

    print('####                  START                ####')
    print('####             Volume Morphing           ####')

    if cond is None:
        str_cond = ''
    else:
        str_cond = ' | Cond.: %s' % cond
    if run is None:
        str_run = ''
    else:
        str_run = ' | Run: %d' % run
    if n_iter is None:
        str_niter = ''
    else:
        str_niter = ' | Iter. :%d' % n_iter

    string = '    Subject: %s' % subject_from + str_run + str_cond + str_niter

    print(string)
    print('\n#### Reading essential data files..')
    # STC 
    stc_orig = mne.read_source_estimate(fname_stc_orig)
    stcdata = stc_orig.data
    nvert, ntimes = stc_orig.shape
    tmin, tstep = stc_orig.times[0], stc_orig.tstep

    # Source Spaces
    subj_vol = mne.read_source_spaces(fname_vsrc_subject_from)
    temp_vol = mne.read_source_spaces(fname_vsrc_subject_to)

    ###########################################################################
    # get dictionaries with labels and their respective vertices
    ###########################################################################

    fname_label_dict_subject_from = (fname_vsrc_subject_from[:-4] +
                                     '_vertno_labelwise.npy')
    label_dict_subject_from = np.load(fname_label_dict_subject_from,
                                      encoding='latin1').item()

    fname_label_dict_subject_to = (fname_vsrc_subject_to[:-4] +
                                   '_vertno_labelwise.npy')
    label_dict_subject_to = np.load(fname_label_dict_subject_to,
                                    encoding='latin1').item()

    # Check for vertex duplicates
    label_dict_subject_from = _remove_vert_duplicates(subject_from, subj_vol,
                                                      label_dict_subject_from,
                                                      subjects_dir)

    ###########################################################################
    # Labelwise transform the whole subject source space
    ###########################################################################

    transformed_p, idx_vertices = _transform_src_lw(subj_vol,
                                                    label_dict_subject_from,
                                                    volume_labels, subject_to,
                                                    subjects_dir,
                                                    label_trans_dic)
    xn, yn, zn = transformed_p.T

    stcdata_sel = []
    for p, i in enumerate(idx_vertices):
        stcdata_sel.append(np.where(idx_vertices[p] == subj_vol[0]['vertno']))
    stcdata_sel = np.asarray(stcdata_sel).flatten()
    stcdata_ch = stcdata[stcdata_sel]

    ###########################################################################
    # Interpolate the data
    ###########################################################################

    print('\n#### Attempting to interpolate STC Data for every time sample..')
    print('    Interpolation method: ', interpolation_method)

    st_time = time2.time()

    xt, yt, zt = temp_vol[0]['rr'][temp_vol[0]['inuse'].astype(bool)].T
    inter_data = np.zeros([xt.shape[0], ntimes])

    for i in range(0, ntimes):

        loadingBar(i, ntimes, task_part='Time slice: %i' % (i + 1))
        inter_data[:, i] = griddata((xn, yn, zn), stcdata_ch[:, i], (xt, yt, zt),
                                    method=interpolation_method, rescale=True)

    if interpolation_method == 'linear':
        inter_data = np.nan_to_num(inter_data)

    if unwanted_to_zero:
        print('#### Setting all unknown vertex values to zero..')

        # set all vertices that do not belong to a label of interest (given by
        # label_dict IIRC) to zero
        inter_data = set_unwanted_to_zero(temp_vol, inter_data, volume_labels, label_dict_subject_to)

        # do the same for the original data for normalization purposes (I think)
        data_utz = set_unwanted_to_zero(subj_vol, stc_orig.data, volume_labels, label_dict_subject_from)

        stc_orig.data = data_utz

    if normalize:

        print('\n#### Attempting to normalize the vol-morphed stc..')
        normalized_new_data = inter_data.copy()

        for p, labels in enumerate(volume_labels):

            lab_verts = label_dict_subject_from[labels]
            lab_verts_temp = label_dict_subject_to[labels]

            # get for the subject brain the indices of all vertices for the given label
            subj_vert_idx = []
            for i in range(0, lab_verts.shape[0]):
                subj_vert_idx.append(np.where(lab_verts[i] == subj_vol[0]['vertno']))
            subj_vert_idx = np.asarray(subj_vert_idx)

            # get for the template brain the indices of all vertices for the given label
            temp_vert_idx = []
            for i in range(0, lab_verts_temp.shape[0]):
                temp_vert_idx.append(np.where(lab_verts_temp[i] == temp_vol[0]['vertno']))
            temp_vert_idx = np.asarray(temp_vert_idx)

            # The original implementation by Daniel did not use the absolute
            # value for normalization. This is probably because he used MFT
            # for the inverse solution which only provides positive activity
            # values.

            # a = np.sum(stc_orig.data[subj_vert_idx], axis=0)
            # b = np.sum(inter_data[temp_vert_idx], axis=0)
            # norm_m_score = a / b

            # The LCMV beamformer can result in positive as well as negative
            # values which can cancel each other out, e.g., after morphing
            # there are more vertices in a "negative value area" than before
            # resulting in a smaller sum 'b' -> norm_m_score becomes large.

            afabs = np.sum(np.fabs(stc_orig.data[subj_vert_idx]), axis=0)
            bfabs = np.sum(np.fabs(inter_data[temp_vert_idx]), axis=0)
            norm_m_score = afabs / bfabs

            normalized_new_data[temp_vert_idx] *= norm_m_score

        new_data = normalized_new_data

    else:

        new_data = inter_data

    print('    [done] -> Calculation Time: %.2f minutes.' % (
            (time2.time() - st_time) / 60.
    ))

    if save_stc:
        print('\n#### Attempting to write interpolated STC Data to file..')

        if fname_save_stc is None:
            fname_stc_morphed = fname_stc_orig[:-7] + '_morphed_to_%s_%s-vl.stc'
            fname_stc_morphed = fname_stc_morphed % (subject_to, interpolation_method)

        else:
            fname_stc_morphed = fname_save_stc

        print('    Destination:', fname_stc_morphed)

        _write_stc(fname_stc_morphed, tmin=tmin, tstep=tstep,
                   vertices=temp_vol[0]['vertno'], data=new_data)

        stc_morphed = mne.read_source_estimate(fname_stc_morphed)

        if plot:
            _volumemorphing_plot_results(stc_orig, stc_morphed,
                                         subj_vol, label_dict_subject_from,
                                         temp_vol, label_dict_subject_to,
                                         volume_labels, subjects_dir=subjects_dir,
                                         cond=cond, run=run, n_iter=n_iter, save=True)

        print('####            Volume Morphing            ####')
        print('####                 DONE                  ####')

        return stc_morphed

    print('####  Volume morphed stc data NOT saved..  ####\n')
    print('####            Volume Morphing            ####')
    print('####                 DONE                  ####')

    return new_data


def _volumemorphing_plot_results(stc_orig, stc_morphed,
                                 volume_orig, label_dict_from,
                                 volume_temp, label_dict_to,
                                 volume_labels, subjects_dir,
                                 cond, run=None, n_iter=None,
                                 save=False):
    """
    Plot before and after morphing results.
    
    Parameters
    ----------
    stc_orig : VolSourceEstimate
        Volume source estimate for the original subject.
    stc_morphed : VolSourceEstimate
        Volume source estimate for the destination subject.
    volume_orig : instance of SourceSpaces
        The original source space that were morphed to the current
        subject.
    label_dict_from : list
        Equivalent label vertex dict to the original source space
    volume_temp : instance of SourceSpaces
        The template source space that is  morphed on.
    label_dict_to : list
        Equivalent label vertex dict to the template source space
    volume_labels : list of volume Labels
        List of the volume labels of interest
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    cond : str
        Evoked condition as a string to give the plot more intel.
    run : int | None
        Specifies the run if multiple measurements for the same condition
        were performed.
    n_iter : int | None
        If MFT was used for the inverse solution, n_iter is the
        number of iterations.

    Returns
    -------
    if save == True : None
        Automatically creates matplotlib.figure and writes it to disk.
    if save == False : returns matplotlib.figure
    
    """
    if run is None:
        run_title = ''
        run_fname = ''
    else:
        run_title = ' | Run: %d' % run
        run_fname = ',run%d' % run

    if n_iter is None:
        n_iter_title = ''
        n_iter_fname = ''
    else:
        n_iter_title = ' | Iter.: %d' % n_iter
        n_iter_fname = ',iter-%d' % n_iter

    subj_vol = volume_orig
    subject_from = volume_orig[0]['subject_his_id']
    temp_vol = volume_temp
    temp_spacing = (abs(temp_vol[0]['rr'][0, 0]
                        - temp_vol[0]['rr'][1, 0]) * 1000).round()
    subject_to = volume_temp[0]['subject_his_id']
    label_dict = label_dict_from
    label_dict_template = label_dict_to
    new_data = stc_morphed.data
    indiv_spacing = make_indiv_spacing(subject_from, subject_to,
                                       temp_spacing, subjects_dir)

    print('\n#### Attempting to save the volume morphing results ..')
    directory = op.join(subjects_dir , subject_from, 'plots', 'VolumeMorphing')
    if not op.exists(directory):
        os.makedirs(directory)

    # Create new figure and two subplots, sharing both axes
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True,
                                   num=999, figsize=(16, 9))
    fig.text(0.985, 0.75, 'Amplitude [T]', color='white', size='large',
             horizontalalignment='right', verticalalignment='center',
             rotation=-90, transform=ax1.transAxes)
    fig.text(0.985, 0.25, 'Amplitude [T]', color='white', size='large',
             horizontalalignment='right', verticalalignment='center',
             rotation=-90, transform=ax2.transAxes)

    suptitle = 'VolumeMorphing from %s to %s' % (subject_from, subject_to)
    suptitle = suptitle + ' | Cond.: %s' % cond
    suptitle = suptitle + run_title + n_iter_title

    plt.suptitle(suptitle, fontsize=16, color='white')
    fig.set_facecolor('black')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.04, top=0.94,
                        left=0.0, right=0.97)
    t = int(np.where(np.sum(stc_orig.data, axis=0)
                     == np.max(np.sum(stc_orig.data, axis=0)))[0])

    plot_vstc(vstc=stc_orig, vsrc=volume_orig, tstep=stc_orig.tstep,
              subjects_dir=subjects_dir, time_sample=t, coords=None,
              figure=999, axes=ax1, save=False)

    plot_vstc(vstc=stc_morphed, vsrc=volume_temp, tstep=stc_orig.tstep,
              subjects_dir=subjects_dir, time_sample=t, coords=None,
              figure=999, axes=ax2, save=False)

    if save:
        fname_save_fig = '%s_to_%s' + run_fname
        fname_save_fig = fname_save_fig + ',vol-%.2f,%s'
        fname_save_fig = fname_save_fig % (subject_from, subject_to, indiv_spacing, cond)
        fname_save_fig = fname_save_fig + n_iter_fname
        fname_save_fig = op.join(directory, fname_save_fig + ',volmorphing-result.png')

        plt.savefig(fname_save_fig, facecolor=fig.get_facecolor(),
                    format='png', edgecolor='none')
        plt.close()
    else:
        plt.show()

    print("""\n#### Attempting to compare subjects activity and interpolated
        activity in template for all labels..""")
    subj_lab_act = {}
    temp_lab_act = {}
    for label in volume_labels:
        lab_arr = label_dict[str(label)]
        lab_arr_temp = label_dict_template[str(label)]
        subj_vert_idx = np.array([], dtype=int)
        temp_vert_idx = np.array([], dtype=int)
        for i in range(0, lab_arr.shape[0]):
            subj_vert_idx = np.append(subj_vert_idx,
                                      np.where(lab_arr[i]
                                               == subj_vol[0]['vertno']))
        for i in range(0, lab_arr_temp.shape[0]):
            temp_vert_idx = np.append(temp_vert_idx,
                                      np.where(lab_arr_temp[i]
                                               == temp_vol[0]['vertno']))
        lab_act_sum = np.array([])
        lab_act_sum_temp = np.array([])
        for t in range(0, stc_orig.times.shape[0]):
            lab_act_sum = np.append(lab_act_sum,
                                    np.sum(stc_orig.data[subj_vert_idx, t]))
            lab_act_sum_temp = np.append(lab_act_sum_temp,
                                         np.sum(stc_morphed.data[temp_vert_idx, t]))
        subj_lab_act.update({label: lab_act_sum})
        temp_lab_act.update({label: lab_act_sum_temp})
    print('    [done]')

    # Stc per label  
    # fig, axs = plt.subplots(len(volume_labels) / 5, 5, figsize=(15, 15),
    #                         facecolor='w', edgecolor='k')
    fig, axs = plt.subplots(int(np.ceil(len(volume_labels) / 5.)), 5, figsize=(15, 15),
                            facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.6, wspace=.255,
                        bottom=0.089, top=.9,
                        left=0.03, right=0.985)
    axs = axs.ravel()
    for idx, label in enumerate(volume_labels):
        axs[idx].plot(stc_orig.times, subj_lab_act[label], '#00868B',
                    linewidth=0.9, label=('%s vol-%.2f'
                                          % (subject_from, indiv_spacing)))
        axs[idx].plot(stc_orig.times, temp_lab_act[label], '#CD7600', ls=':',
                    linewidth=0.9, label=('%s volume morphed vol-%.2f'
                                          % (subject_from, temp_spacing)))
        axs[idx].set_title(label, fontsize='medium', loc='right')
        axs[idx].ticklabel_format(style='sci', axis='both')
        axs[idx].set_xlabel('Time [s]')
        axs[idx].set_ylabel('Amplitude [T]')
        axs[idx].set_xlim(stc_orig.times[0], stc_orig.times[-1])
        axs[idx].get_xaxis().grid(True)

    suptitle = 'Summed activity in volume labels - %s[%.2f]' % (subject_from, indiv_spacing)
    suptitle = suptitle + ' -> %s [%.2f] | Cond.: %s' % (subject_to, temp_spacing, cond)
    suptitle = suptitle + run_title + n_iter_title

    fig.suptitle(suptitle, fontsize=16)

    if save:
        fname_save_fig = '%s_to_%s' + run_fname
        fname_save_fig = fname_save_fig + ',vol-%.2f,%s'
        fname_save_fig = fname_save_fig % (subject_from, subject_to, indiv_spacing, cond)
        fname_save_fig = fname_save_fig + n_iter_fname
        fname_save_fig = op.join(directory, fname_save_fig + ',labelwise-stc.png')

        plt.savefig(fname_save_fig, facecolor=fig.get_facecolor(),
                    format='png', edgecolor='none')
        plt.close()
    else:
        plt.show()

    orig_act_sum = np.sum(stc_orig.data.sum(axis=0))
    morphed_act_sum = np.sum(new_data.sum(axis=0))
    act_diff_perc = ((morphed_act_sum - orig_act_sum) / orig_act_sum) * 100
    act_sum_morphed_normed = np.sum(new_data.sum(axis=0))
    act_diff_perc_morphed_normed = ((act_sum_morphed_normed - orig_act_sum)
                                    / orig_act_sum) * 100

    f, (ax1) = plt.subplots(1, figsize=(16, 5))
    ax1.plot(stc_orig.times, stc_orig.data.sum(axis=0), '#00868B', linewidth=1,
             label='%s' % subject_from)
    ax1.plot(stc_orig.times, new_data.sum(axis=0), '#CD7600', linewidth=1,
             label='%s morphed' % subject_from)

    title = 'Summed Source Amplitude - %s[%.2f] ' % (subject_from, indiv_spacing)
    title = title + '-> %s [%.2f] | Cond.: %s' % (subject_to, temp_spacing, cond)
    title = title + run_title + n_iter_title

    ax1.set_title(title)
    ax1.text(stc_orig.times[0],
             np.maximum(stc_orig.data.sum(axis=0), new_data.sum(axis=0)).max(),
             """Total Amplitude Difference: %+.2f %%
             Total Amplitude Difference (norm):  %+.2f %%"""
             % (act_diff_perc, act_diff_perc_morphed_normed),
             size=12, ha="left", va="top",
             bbox=dict(boxstyle="round",
                       ec="grey",
                       fc="white",
                       )
             )

    ax1.set_ylabel('Summed Source Amplitude')
    ax1.legend(fontsize='large', facecolor="white", edgecolor="grey")
    ax1.get_xaxis().grid(True)
    plt.tight_layout()
    if save:
        fname_save_fig = '%s_to_%s' + run_fname
        fname_save_fig = fname_save_fig + ',vol-%.2f,%s'
        fname_save_fig = fname_save_fig % (subject_from, subject_to, indiv_spacing, cond)
        fname_save_fig = fname_save_fig + n_iter_fname
        fname_save_fig = op.join(directory, fname_save_fig + ',stc.png')

        plt.savefig(fname_save_fig, facecolor=fig.get_facecolor(),
                    format='png', edgecolor='none')
        plt.close()
    else:
        plt.show()

    return


def plot_vstc(vstc, vsrc, tstep, subjects_dir, time_sample=None, coords=None,
              figure=None, axes=None, cmap='gist_ncar', symmetric_cbar=False,
              threshold='min', save=False, fname_save=None):
    """ Plot a volume source space estimation.

    Parameters
    ----------
    vstc : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of SourceSpaces
        The source space of the subject equivalent to the
        subject.
    tstep : scalar
        Time step between successive samples in data.
    subjects_dir : str
        The path to the subjects directory.
    time_sample : int, float | None
        None is default for finding the time sample with the voxel with global
        maximal amplitude. If int, float the given time sample is selected and
        plotted.
    coords : arr | None
        None is default for finding the coordinates with the maximal amplitude
        for the given or automatically found time sample
    figure : matplotlib.figure | None
        Specify the figure container to plot in. If None, a new
        matplotlib.figure is created
    axes : matplotlib.figure.axes | None
        Specify the axes of the given figure to plot in. Only necessary if
        a figure is passed.
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The ccolormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    save : bool | None
        Default is False. If True the plot is forced to close and written to disk
        at fname_save location
    fname : string
        The path where to save the plot.

    Returns
    -------
    Figure : matplotlib.figure
          VolSourceEstimation plotted for given or 'auto' coordinates at given
          or 'auto' timepoint.
    """
    vstcdata = vstc.data
    img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
    subject = vsrc[0]['subject_his_id']
    if vstc == 0:
        if tstep is not None:
            img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
        else:
            print('    Please provide the tstep value !')
    img_data = img.get_data()
    aff = img.affine
    if time_sample is None:
        # global maximum amp in time
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])
    else:
        print('    Time slice', time_sample)
        t = time_sample
    t_in_ms = vstc.times[t] * 1e3
    print('    Found time slice: ', t_in_ms, 'ms')
    if coords is None:
        cut_coords = np.where(img_data == img_data[:, :, :, t].max())
        max_try = np.concatenate((np.array([cut_coords[0][0]]),
                                  np.array([cut_coords[1][0]]),
                                  np.array([cut_coords[2][0]])))
        cut_coords = apply_affine(aff, max_try)
    else:
        cut_coords = coords
    slice_x, slice_y = int(cut_coords[0]), int(cut_coords[1])
    slice_z = int(cut_coords[2])
    print(('    Coords [mri-space]:'
           + 'X: ', slice_x, 'Y: ', slice_y, 'Z: ', slice_z))
    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    if threshold == 'min':
        threshold = vstcdata.min()

    vstc_plt = plotting.plot_stat_map(index_img(img, t), temp_t1_fname,
                                      figure=figure, axes=axes,
                                      display_mode='ortho',
                                      threshold=threshold,
                                      annotate=True,
                                      title='%s | t=%.2f ms'
                                            % (subject, t_in_ms),
                                      cut_coords=(slice_x, slice_y, slice_z),
                                      cmap=cmap, symmetric_cbar=symmetric_cbar)
    if save:
        if fname_save is None:
            print('please provide an filepath to save .png')
        else:
            plt.savefig(fname_save)
            plt.close()

    return vstc_plt


def plot_vstc_sliced_grid(subjects_dir, vstc, vsrc, tstep, title, time=None,
                          display_mode=['x'], cut_coords=6,
                          cmap='nipy_spectral', threshold='min',
                          only_positive_values=False, grid=[4, 6],
                          res_save=[1920, 1080], fn_image='plot.png'):
    """

    Parameters:
    -----------
    subjects_dir : str
        The path to the subjects directory.
    vstc : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of SourceSpaces
        The source space of the subject equivalent to the
        subject.
    tstep : int
        Time step between successive samples in data.
    title : str
        Title for the plot.
    time : float
        Time point for which the image will be created.
    display_mode : 'x', 'y', 'z'
        Direction in which the brain is sliced.
    cut_coords : None, a tuple of floats, or an integer
        The MNI coordinates of the point where the cut is performed
        If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None is given, the cuts is calculated automaticaly.
        If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
        in which case it specifies the number of cuts to perform
    cmap : str
        Name of the matplotlib color map to use.
        See https://matplotlib.org/examples/color/colormaps_reference.html
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    only_positive_values : bool
        Constrain the plots to only positive values, e.g., if there
        are no negative values as in the case of MFT inverse solutions.
    grid : 2-tuple
        Specifies how many images per row and column are to be depicted.
    res_save : 2-tuple
        Resolution of the saved image.
    fn_image : str
        File name for the saved image.

    Returns:
    --------
    None
    """

    if display_mode not in {'x', 'y', 'z'}:
        raise ValueError("display_mode must be one of 'x', 'y', or 'z'.")

    if not op.exists(fn_image):

        start_time = time2.time()

        print(strftime('Start at %H:%M:%S on the %d.%m.%Y \n', localtime()))

        figure, axes = plt.subplots(grid[0], grid[1])

        axes = axes.flatten()

        params_plot_img_with_bg = get_params_for_grid_slice(vstc, vsrc, tstep, subjects_dir,
                                                            only_positive_values=only_positive_values)

        for i, (ax, z) in enumerate(zip(axes, cut_coords)):
            cut_coords_slice = [z]

            colorbar = False
            if grid[1] - 1 == i:
                colorbar = True

            vstc_plot = plot_vstc_grid_slice(vstc=vstc, params_plot_img_with_bg=params_plot_img_with_bg, time=time,
                                             cut_coords=cut_coords_slice, display_mode=display_mode,
                                             figure=figure, axes=ax, colorbar=colorbar, cmap=cmap,
                                             threshold=threshold)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            wspace=0, hspace=0)

        if title is not None:
            plt.suptitle(title)

        DPI = figure.get_dpi()
        figure.set_size_inches(res_save[0] / float(DPI), res_save[1] / float(DPI))
        # bbox_inches='tight' not usefule for images for videos, see:
        # https://github.com/matplotlib/matplotlib/issues/8543#issuecomment-400679840

        frmt = fn_image.split('.')[-1]
        print(DPI, figure.get_size_inches())
        plt.savefig(fn_image, format=frmt, dpi=DPI)

        plt.close()

        end_time = time2.time()

        print(strftime('End at %H:%M:%S on the %d.%m.%Y \n', localtime()))
        minutes = (end_time - start_time) / 60
        seconds = (end_time - start_time) % 60
        print("Calculation took %d minutes and %d seconds" % (minutes, seconds))
        print("")


def get_params_for_grid_slice(vstc, vsrc, tstep, subjects_dir, only_positive_values=False, **kwargs):
    """
    Makes calculations that would be executed repeatedly every time a slice is
    computed and saves the results in a dictionary which is then read by
    plot_vstc_grid_slice().

    Parameters:
    -----------
    vstc : mne.VolSourceEstimate
        The volume source estimate.
    vsrc : mne.SourceSpaces
        The source space of the subject equivalent to the
    tstep : int
        Time step between successive samples in data.
    subjects_dir:
        Path to the subject directory.
    only_positive_values : bool
        Constrain the plots to only positive values, e.g., if there
        are no negative values as in the case of MFT inverse solutions.

    Returns:
    --------
    params_plot_img_with_bg : dict
        Dictionary containing the parameters for plotting.
    """

    img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
    if vstc == 0:
        if tstep is not None:
            img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
        else:
            print('    Please provide the tstep value !')

    subject = vsrc[0]['subject_his_id']
    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    bg_img = temp_t1_fname
    dim = 'auto'
    black_bg = 'auto'
    vmax = None
    symmetric_cbar = False

    from nilearn.plotting.img_plotting import _load_anat, _get_colorbar_and_data_ranges
    from nilearn._utils import check_niimg_4d
    from nilearn._utils.niimg_conversions import _safe_get_data

    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim, black_bg=black_bg)

    stat_map_img = check_niimg_4d(img, dtype='auto')

    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        _safe_get_data(stat_map_img, ensure_finite=True), vmax, symmetric_cbar, kwargs)

    if only_positive_values:
        # there are no negative values, e.g., for MFT solutions
        cbar_vmin = 0.
        vmin = 0.

    params_plot_img_with_bg = dict()
    params_plot_img_with_bg['bg_img'] = bg_img
    params_plot_img_with_bg['black_bg'] = black_bg
    params_plot_img_with_bg['bg_vmin'] = bg_vmin
    params_plot_img_with_bg['bg_vmax'] = bg_vmax
    params_plot_img_with_bg['stat_map_img'] = stat_map_img
    params_plot_img_with_bg['cbar_vmin'] = cbar_vmin
    params_plot_img_with_bg['cbar_vmax'] = cbar_vmax
    params_plot_img_with_bg['vmin'] = vmin
    params_plot_img_with_bg['vmax'] = vmax

    return params_plot_img_with_bg


def plot_vstc_grid_slice(vstc, params_plot_img_with_bg, time=None, cut_coords=6, display_mode='z',
                         figure=None, axes=None, colorbar=False, cmap='nipy_spectral', threshold='min',
                         **kwargs):
    """
    Plot a volume source space estimation for one slice in the grid in
    plot_vstc_sliced_grid.

    Parameters:
    -----------
    vstc : VolSourceEstimate
        The volume source estimate.
    time : int, float | None
        None is default for finding the time sample with the voxel with global
        maximal amplitude. If int, float the given time point is selected and
        plotted.
    cut_coords : None, a tuple of floats, or an integer
        The MNI coordinates of the point where the cut is performed
        If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None is given, the cuts is calculated automaticaly.
        If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
        in which case it specifies the number of cuts to perform
    display_mode : 'x', 'y', 'z'
        Direction in which the brain is sliced.
    figure : matplotlib.figure | None
        Specify the figure container to plot in. If None, a new
        matplotlib.figure is created
    axes : matplotlib.figure.axes | None
        Specify the axes of the given figure to plot in. Only necessary if
        a figure is passed.
    colorbar : bool
        Show the colorbar.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The colormap *must* be
        symmetrical.
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.

    Returns:
    --------
    Figure : matplotlib.figure
          VolSourceEstimation plotted for given or 'auto' coordinates at given
          or 'auto' timepoint.
    """

    vstcdata = vstc.data

    if time is None:
        # global maximum amp in time
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])
        t_in_ms = vstc.times[t] * 1e3

    else:
        t = np.argmin(np.fabs(vstc.times - time))
        t_in_ms = vstc.times[t] * 1e3

    print('    Found time slice: ', t_in_ms, 'ms')

    if threshold == 'min':
        threshold = vstcdata.min()

    # jumeg_plot_stat_map starts here
    output_file = None
    title = None
    annotate = True
    draw_cross = True
    resampling_interpolation = 'continuous'

    bg_img = params_plot_img_with_bg['bg_img']
    black_bg = params_plot_img_with_bg['black_bg']
    bg_vmin = params_plot_img_with_bg['bg_vmin']
    bg_vmax = params_plot_img_with_bg['bg_vmax']
    stat_map_img = params_plot_img_with_bg['stat_map_img']
    cbar_vmin = params_plot_img_with_bg['cbar_vmin']
    cbar_vmax = params_plot_img_with_bg['cbar_vmax']
    vmin = params_plot_img_with_bg['vmin']
    vmax = params_plot_img_with_bg['vmax']

    # noqa: E501
    # dim the background
    from nilearn.plotting.img_plotting import _plot_img_with_bg
    from nilearn._utils import check_niimg_3d

    stat_map_img_at_time_t = index_img(stat_map_img, t)
    stat_map_img_at_time_t = check_niimg_3d(stat_map_img_at_time_t, dtype='auto')

    display = _plot_img_with_bg(
        img=stat_map_img_at_time_t, bg_img=bg_img, cut_coords=cut_coords,
        output_file=output_file, display_mode=display_mode,
        figure=figure, axes=axes, title=title, annotate=annotate,
        draw_cross=draw_cross, black_bg=black_bg, threshold=threshold,
        bg_vmin=bg_vmin, bg_vmax=bg_vmax, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=colorbar, cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
        resampling_interpolation=resampling_interpolation, **kwargs)

    vstc_plt = display

    return vstc_plt


def jumeg_plot_stat_map_grid(stat_map_img, t, bg_img=MNI152TEMPLATE, cut_coords=None,
                             output_file=None, display_mode='ortho', colorbar=True,
                             figure=None, axes=None, title=None, threshold=1e-6,
                             annotate=True, draw_cross=True, black_bg='auto',
                             cmap='gist_ncar', symmetric_cbar="auto", only_positive_values=False,
                             dim='auto', vmax=None, resampling_interpolation='continuous',
                             **kwargs):
    """
    Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and
    Lateral)
    This is based on nilearn.plotting.plot_stat_map
    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image
    t : int
        Plot activity at time point given by time t.
    bg_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the ROI/mask will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
    cut_coords : None, a tuple of floats, or an integer
        The MNI coordinates of the point where the cut is performed
        If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None is given, the cuts is calculated automaticaly.
        If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
        in which case it specifies the number of cuts to perform
    output_file : string, or None, optional
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.
    display_mode : {'ortho', 'x', 'y', 'z', 'yx', 'xz', 'yz'}
        Choose the direction of the cuts: 'x' - sagittal, 'y' - coronal,
        'z' - axial, 'ortho' - three cuts are performed in orthogonal
        directions.
    colorbar : boolean, optional
        If True, display a colorbar on the right of the plots.
    figure : integer or matplotlib figure, optional
        Matplotlib figure used or its number. If None is given, a
        new figure is created.
    axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
        The axes, or the coordinates, in matplotlib figure space,
        of the axes used to display the plot. If None, the complete
        figure is used.
    title : string, optional
        The title displayed on the figure.
    threshold : a number, None, or 'auto'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    annotate : boolean, optional
        If annotate is True, positions and left/right annotation
        are added to the plot.
    draw_cross : boolean, optional
        If draw_cross is True, a cross is drawn on the plot to
        indicate the cut plosition.
    black_bg : boolean, optional
        If True, the background of the image is set to be black. If
        you wish to save figures with a black background, you
        will need to pass "facecolor='k', edgecolor='k'"
        to matplotlib.pyplot.savefig.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The ccolormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    only_positive_values : bool
        Constrain the plots to only positive values, e.g., if there
        are no negative values as in the case of MFT inverse solutions.
    dim : float, 'auto' (by default), optional
        Dimming factor applied to background image. By default, automatic
        heuristics are applied based upon the background image intensity.
        Accepted float values, where a typical scan is between -2 and 2
        (-2 = increase constrast; 2 = decrease contrast), but larger values
        can be used for a more pronounced effect. 0 means no dimming.
    vmax : float
        Upper bound for plotting, passed to matplotlib.pyplot.imshow
    resampling_interpolation : str
        Interpolation to use when resampling the image to the destination
        space. Can be "continuous" (default) to use 3rd-order spline
        interpolation, or "nearest" to use nearest-neighbor mapping.
        "nearest" is faster but can be noisier in some cases.
    Notes
    -----
    Arrays should be passed in numpy convention: (x, y, z)
    ordered.
    For visualization, non-finite values found in passed 'stat_map_img' or
    'bg_img' are set to zero.
    See Also
    --------
    nilearn.plotting.plot_anat : To simply plot anatomical images
    nilearn.plotting.plot_epi : To simply plot raw EPI images
    nilearn.plotting.plot_glass_brain : To plot maps in a glass brain
    """
    # noqa: E501
    # dim the background
    from nilearn.plotting.img_plotting import _load_anat, _plot_img_with_bg, _get_colorbar_and_data_ranges
    from nilearn._utils import check_niimg_3d, check_niimg_4d
    from nilearn._utils.niimg_conversions import _safe_get_data

    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim,
                                                    black_bg=black_bg)

    stat_map_img = check_niimg_4d(stat_map_img, dtype='auto')

    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(_safe_get_data(stat_map_img, ensure_finite=True),
                                                                     vmax, symmetric_cbar, kwargs)

    if only_positive_values:
        # there are no negative values
        cbar_vmin = 0.
        vmin = 0.

    stat_map_img_at_time_t = index_img(stat_map_img, t)
    stat_map_img_at_time_t = check_niimg_3d(stat_map_img_at_time_t, dtype='auto')

    display = _plot_img_with_bg(
        img=stat_map_img_at_time_t, bg_img=bg_img, cut_coords=cut_coords,
        output_file=output_file, display_mode=display_mode,
        figure=figure, axes=axes, title=title, annotate=annotate,
        draw_cross=draw_cross, black_bg=black_bg, threshold=threshold,
        bg_vmin=bg_vmin, bg_vmax=bg_vmax, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=colorbar, cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
        resampling_interpolation=resampling_interpolation, **kwargs)

    return display


def plot_vstc_sliced_old(vstc, vsrc, tstep, subjects_dir, time=None, cut_coords=6,
                         display_mode='z', figure=None, axes=None, colorbar=False, cmap='gist_ncar',
                         symmetric_cbar=False, threshold='min', save=False, fname_save=None):
    """
    Plot a volume source space estimation.

    Parameters
    ----------
    vstc : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of SourceSpaces
        The source space of the subject equivalent to the
        subject.
    tstep : scalar
        Time step between successive samples in data.
    subjects_dir : str
        The path to the subjects directory.
    time : int, float | None
        None is default for finding the time sample with the voxel with global
        maximal amplitude. If int, float the given time point is selected and
        plotted.
    display_mode : 'x', 'y', 'z'
        Direction in which the brain is sliced.
    cut_coords : None, a tuple of floats, or an integer
        The MNI coordinates of the point where the cut is performed
        If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None is given, the cuts is calculated automaticaly.
        If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
        in which case it specifies the number of cuts to perform
    figure : matplotlib.figure | None
        Specify the figure container to plot in. If None, a new
        matplotlib.figure is created
    axes : matplotlib.figure.axes | None
        Specify the axes of the given figure to plot in. Only necessary if
        a figure is passed.
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    colorbar : bool
        Show the colorbar.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The colormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    save : bool | None
        Default is False. If True the plot is forced to close and written to disk
        at fname_save location
    fname : string
        The path where to save the plot.

    Returns
    -------
    Figure : matplotlib.figure
          VolSourceEstimation plotted for given or 'auto' coordinates at given
          or 'auto' timepoint.
    """
    vstcdata = vstc.data
    img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
    subject = vsrc[0]['subject_his_id']
    if vstc == 0:
        if tstep is not None:
            img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
        else:
            print('    Please provide the tstep value !')

    if time is None:
        # global maximum amp in time
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])
        t_in_ms = vstc.times[t] * 1e3

    else:
        t = np.argmin(np.fabs(vstc.times - time))
        t_in_ms = vstc.times[t] * 1e3
    print('    Found time slice: ', t_in_ms, 'ms')

    # img_data = img.get_data()
    # aff = img.affine
    # if cut_coords is None:
    #     cut_coords = np.where(img_data == img_data[:, :, :, t].max())
    #     max_try = np.concatenate((np.array([cut_coords[0][0]]),
    #                               np.array([cut_coords[1][0]]),
    #                               np.array([cut_coords[2][0]])))
    #     cut_coords = apply_affine(aff, max_try)

    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    if threshold == 'min':
        threshold = vstcdata.min()

    vstc_plt = plotting.plot_stat_map(stat_map_img=index_img(img, t),
                                      bg_img=temp_t1_fname,
                                      figure=figure, axes=axes,
                                      display_mode=display_mode,
                                      threshold=threshold,
                                      annotate=True, title=None,
                                      cut_coords=cut_coords,
                                      cmap=cmap, colorbar=colorbar,
                                      symmetric_cbar=symmetric_cbar)
    if save:
        if fname_save is None:
            print('please provide an filepath to save .png')
        else:
            plt.savefig(fname_save)
            plt.close()

    return vstc_plt


def plot_vstc_sliced(vstc, vsrc, tstep, subjects_dir, img=None, time=None, cut_coords=6,
                     display_mode='z', figure=None, axes=None, colorbar=False, cmap='gist_ncar',
                     symmetric_cbar=False, only_positive_values=False, threshold='min', save=False,
                     fname_save=None):
    """ Plot a volume source space estimation.

    Parameters
    ----------
    vstc : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of SourceSpaces
        The source space of the subject equivalent to the
        subject.
    tstep : scalar
        Time step between successive samples in data.
    subjects_dir : str
        The path to the subjects directory.
    img : Nifti1Image | None
        Pre-computed vstc.as_volume(vsrc, dest='mri', mri_resolution=False).
    time : int, float | None
        None is default for finding the time sample with the voxel with global
        maximal amplitude. If int, float the given time point is selected and
        plotted.
    display_mode : 'x', 'y', 'z'
        Direction in which the brain is sliced.
    cut_coords : None, a tuple of floats, or an integer
        The MNI coordinates of the point where the cut is performed
        If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None is given, the cuts is calculated automaticaly.
        If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
        in which case it specifies the number of cuts to perform
    figure : matplotlib.figure | None
        Specify the figure container to plot in. If None, a new
        matplotlib.figure is created
    axes : matplotlib.figure.axes | None
        Specify the axes of the given figure to plot in. Only necessary if
        a figure is passed.
    threshold : a number, None, 'auto', or 'min'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    colorbar : bool
        Show the colorbar.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The colormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    only_positive_values : bool
        Constrain the plots to only positive values, e.g., if there
        are no negative values as in the case of MFT inverse solutions.
    save : bool | None
        Default is False. If True the plot is forced to close and written to disk
        at fname_save location
    fname : string
        The path where to save the plot.

    Returns
    -------
    Figure : matplotlib.figure
          VolSourceEstimation plotted for given or 'auto' coordinates at given
          or 'auto' timepoint.
    """
    vstcdata = vstc.data
    if img is None:
        img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
        if vstc == 0:
            if tstep is not None:
                img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
            else:
                print('    Please provide the tstep value !')
    subject = vsrc[0]['subject_his_id']

    if time is None:
        # global maximum amp in time
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])
    else:
        t = np.argmin(np.fabs(vstc.times - time))

    t_in_ms = vstc.times[t] * 1e3
    print('    Found time slice: ', t_in_ms, 'ms')

    # img_data = img.get_data()
    # aff = img.affine
    # if cut_coords is None:
    #     cut_coords = np.where(img_data == img_data[:, :, :, t].max())
    #     max_try = np.concatenate((np.array([cut_coords[0][0]]),
    #                               np.array([cut_coords[1][0]]),
    #                               np.array([cut_coords[2][0]])))
    #     cut_coords = apply_affine(aff, max_try)

    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')

    if threshold == 'min':
        threshold = vstcdata.min()

    vstc_plt = jumeg_plot_stat_map(stat_map_img=img, t=t,
                                   bg_img=temp_t1_fname,
                                   figure=figure, axes=axes,
                                   display_mode=display_mode,
                                   threshold=threshold,
                                   annotate=True, title=None,
                                   cut_coords=cut_coords,
                                   cmap=cmap, colorbar=colorbar,
                                   symmetric_cbar=symmetric_cbar,
                                   only_positive_values=only_positive_values)
    if save:
        if fname_save is None:
            print('please provide an filepath to save .png')
        else:
            plt.savefig(fname_save)
            plt.close()

    return vstc_plt


def jumeg_plot_stat_map(stat_map_img, t, bg_img=MNI152TEMPLATE, cut_coords=None,
                        output_file=None, display_mode='ortho', colorbar=True,
                        figure=None, axes=None, title=None, threshold=1e-6,
                        annotate=True, draw_cross=True, black_bg='auto',
                        cmap='gist_ncar', symmetric_cbar="auto", only_positive_values=False,
                        dim='auto', vmax=None, resampling_interpolation='continuous',
                        **kwargs):
    """
    Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and
    Lateral)

    This is based on nilearn.plotting.plot_stat_map
    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image
    t : int
        Plot activity at time point given by time t.
    bg_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the ROI/mask will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
    cut_coords : None, a tuple of floats, or an integer
        The MNI coordinates of the point where the cut is performed
        If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
        For display_mode == 'x', 'y', or 'z', then these are the
        coordinates of each cut in the corresponding direction.
        If None is given, the cuts is calculated automaticaly.
        If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
        in which case it specifies the number of cuts to perform
    output_file : string, or None, optional
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.
    display_mode : {'ortho', 'x', 'y', 'z', 'yx', 'xz', 'yz'}
        Choose the direction of the cuts: 'x' - sagittal, 'y' - coronal,
        'z' - axial, 'ortho' - three cuts are performed in orthogonal
        directions.
    colorbar : boolean, optional
        If True, display a colorbar on the right of the plots.
    figure : integer or matplotlib figure, optional
        Matplotlib figure used or its number. If None is given, a
        new figure is created.
    axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
        The axes, or the coordinates, in matplotlib figure space,
        of the axes used to display the plot. If None, the complete
        figure is used.
    title : string, optional
        The title displayed on the figure.
    threshold : a number, None, or 'auto'
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    annotate : boolean, optional
        If annotate is True, positions and left/right annotation
        are added to the plot.
    draw_cross : boolean, optional
        If draw_cross is True, a cross is drawn on the plot to
        indicate the cut plosition.
    black_bg : boolean, optional
        If True, the background of the image is set to be black. If
        you wish to save figures with a black background, you
        will need to pass "facecolor='k', edgecolor='k'"
        to matplotlib.pyplot.savefig.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The ccolormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    only_positive_values : bool
        Constrain the plots to only positive values, e.g., if there
        are no negative values as in the case of MFT inverse solutions.
    dim : float, 'auto' (by default), optional
        Dimming factor applied to background image. By default, automatic
        heuristics are applied based upon the background image intensity.
        Accepted float values, where a typical scan is between -2 and 2
        (-2 = increase constrast; 2 = decrease contrast), but larger values
        can be used for a more pronounced effect. 0 means no dimming.
    vmax : float
        Upper bound for plotting, passed to matplotlib.pyplot.imshow
    resampling_interpolation : str
        Interpolation to use when resampling the image to the destination
        space. Can be "continuous" (default) to use 3rd-order spline
        interpolation, or "nearest" to use nearest-neighbor mapping.
        "nearest" is faster but can be noisier in some cases.

    Notes
    -----
    Arrays should be passed in numpy convention: (x, y, z)
    ordered.

    For visualization, non-finite values found in passed 'stat_map_img' or
    'bg_img' are set to zero.

    See Also
    --------

    nilearn.plotting.plot_anat : To simply plot anatomical images
    nilearn.plotting.plot_epi : To simply plot raw EPI images
    nilearn.plotting.plot_glass_brain : To plot maps in a glass brain

    """
    # noqa: E501
    # dim the background
    from nilearn.plotting.img_plotting import _load_anat, _plot_img_with_bg, _get_colorbar_and_data_ranges
    from nilearn._utils import check_niimg_3d, check_niimg_4d
    from nilearn._utils.niimg_conversions import _safe_get_data

    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim,
                                                    black_bg=black_bg)

    stat_map_img = check_niimg_4d(stat_map_img, dtype='auto')

    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(_safe_get_data(stat_map_img, ensure_finite=True),
                                                                     vmax, symmetric_cbar, kwargs)

    if only_positive_values:
        # there are no negative values
        cbar_vmin = 0.
        vmin = 0.

    stat_map_img_at_time_t = index_img(stat_map_img, t)
    stat_map_img_at_time_t = check_niimg_3d(stat_map_img_at_time_t, dtype='auto')

    display = _plot_img_with_bg(
        img=stat_map_img_at_time_t, bg_img=bg_img, cut_coords=cut_coords,
        output_file=output_file, display_mode=display_mode,
        figure=figure, axes=axes, title=title, annotate=annotate,
        draw_cross=draw_cross, black_bg=black_bg, threshold=threshold,
        bg_vmin=bg_vmin, bg_vmax=bg_vmax, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar=colorbar, cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
        resampling_interpolation=resampling_interpolation, **kwargs)

    return display


def _make_image(stc_data, vsrc, tstep, label_inds=None, dest='mri',
                mri_resolution=False):
    """Make a volume source estimation in a NIfTI file.

    Parameters
    ----------
    stc_data : VolSourceEstimate
        The volume source estimate.
    vsrc : instance of VolSourceSpaces
        The source space of the subject equivalent to the 
        subject.
    tstep : float
        The tstep value for the recorded data
    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.

    Returns
    -------
    img : instance Nifti1Image
        The image object.
    """
    n_times = stc_data.shape[1]
    shape = vsrc[0]['shape']
    shape3d = (shape[2], shape[1], shape[0])
    shape = (n_times, shape[2], shape[1], shape[0])
    vol = np.zeros(shape)

    if label_inds is not None:
        inuse_replace = np.zeros(vsrc[0]['inuse'].shape, dtype=int)
        for i, idx in enumerate(label_inds):
            inuse_replace[idx] = 1
        mask3d = inuse_replace.reshape(shape3d).astype(np.bool)
    else:
        mask3d = vsrc[0]['inuse'].reshape(shape3d).astype(np.bool)

    if mri_resolution:
        mri_shape3d = (vsrc[0]['mri_height'], vsrc[0]['mri_depth'],
                       vsrc[0]['mri_width'])
        mri_shape = (n_times, vsrc[0]['mri_height'], vsrc[0]['mri_depth'],
                     vsrc[0]['mri_width'])
        mri_vol = np.zeros(mri_shape)
        interpolator = vsrc[0]['interpolator']

    for k, v in enumerate(vol):
        v[mask3d] = stc_data[:, k]
        if mri_resolution:
            mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)

    if mri_resolution:
        vol = mri_vol

    vol = vol.T

    if mri_resolution:
        affine = vsrc[0]['vox_mri_t']['trans'].copy()
    else:
        affine = vsrc[0]['src_mri_t']['trans'].copy()
    if dest == 'mri':
        affine = np.dot(vsrc[0]['mri_ras_t']['trans'], affine)
    affine[:3] *= 1e3

    try:
        import nibabel as nib  # lazy import to avoid dependency
    except ImportError:
        raise ImportError("nibabel is required to save volume images.")

    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    header['pixdim'][4] = 1e3 * tstep
    img = nib.Nifti1Image(vol, affine, header=header)

    return img


def plot_VSTCPT(vstc, vsrc, tstep, subjects_dir, time_sample=None, coords=None,
                figure=None, axes=None, title=None, save=False, fname_save=None):
    """ Plotting the cluster permutation test results based on a
        volume source estimation.

    Parameters
    ----------
    fname_stc_orig : String
          Filename
    subject_from : list of Labels
          Filename
    Returns
    -------
    new-data : dictionary of one or more new stc
          The generated source time courses.
    """

    from jumeg.jumeg_volmorpher import _make_image
    print('\n#### Attempting to plot volume stc from file..')
    print('    Creating 3D image from stc..')
    vstcdata = vstc.data
    img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
    subject = vsrc[0]['subject_his_id']
    if vstc == 0:
        if tstep is not None:
            img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
        else:
            print('    Please provide the tstep value !')
    img_data = img.get_data()
    aff = img.affine
    if time_sample is None:
        print('    Searching for maximal Activation..')
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])  # maximum amp in time
    else:
        print('    Using Cluster No.', time_sample)
        t = time_sample
    if title is None:
        title = 'Cluster No. %i' % t
        if t == 0:
            title = 'All Cluster'  # |sig.%i'%vstc.times.shape[0]-1
    if coords is None:
        cut_coords = np.where(img_data == img_data[:, :, :, t].max())
        max_try = np.concatenate((
            np.array([cut_coords[0][0]]),
            np.array([cut_coords[1][0]]),
            np.array([cut_coords[2][0]])))
        cut_coords = apply_affine(aff, max_try)
    else:
        cut_coords = coords
    slice_x, slice_y = int(cut_coords[0]), int(cut_coords[1])
    slice_z = int(cut_coords[2])
    print('    Respective Space Coords [mri-space]:')
    print('    X: ', slice_x, '    Y: ', slice_y, '    Z: ', slice_z)
    temp_t1_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    VSTCPT_plot = plotting.plot_stat_map(index_img(img, t), temp_t1_fname,
                                         figure=figure, axes=axes,
                                         display_mode='ortho',
                                         threshold=vstcdata.min(),
                                         annotate=True,
                                         title=title,
                                         cut_coords=None,
                                         cmap='black_red')
    if save:
        if fname_save is None:
            print('please provide an filepath to save .png')
        else:
            plt.savefig(fname_save)
            plt.close()

    return VSTCPT_plot


def make_indiv_spacing(subject, ave_subject, template_spacing, subjects_dir):
    """
    Identifies the suiting grid space difference of a subject's volume
    source space to a template's volume source space, before a planned
    morphing takes place.
    
    Parameters:
    -----------
    subject : str
        Subject ID.
    ave_subject : str
        Name or ID of the template brain, e.g., fsaverage.
    template_spacing : float
        Grid spacing used for the template brain.
    subjects_dir : str
        Path to the subjects directory.

    Returns:
    --------
    trans : SourceEstimate
          The generated source time courses.
    """
    fname_surf = op.join(subjects_dir, subject, 'bem', 'watershed', '%s_inner_skull_surface' % subject)
    fname_surf_temp = op.join(subjects_dir, ave_subject, 'bem', 'watershed', '%s_inner_skull_surface' % ave_subject)
    surf = mne.read_surface(fname_surf, return_dict=True, verbose='ERROR')[-1]
    surf_temp = mne.read_surface(fname_surf_temp, return_dict=True, verbose='ERROR')[-1]
    x_sp, y_sp, z_sp = surf['rr'].T / 1e3
    x_sptemp, y_sptemp, z_sptemp = surf_temp['rr'].T / 1e3
    mins = np.min(surf['rr'], axis=0)
    maxs = np.max(surf['rr'], axis=0)
    mins_temp = np.min(surf_temp['rr'], axis=0)
    maxs_temp = np.max(surf_temp['rr'], axis=0)
    # Check which dimension (x,y,z) has greatest difference
    diff = (maxs - mins)
    diff_temp = (maxs_temp - mins_temp)
    # print additional information
    #    for c, mi, ma, md in zip('xyz', mins, maxs, diff):
    #      logger.info('    %s = %6.1f ... %6.1f mm --> Difference:  %6.1f mm'
    #                  % (c, mi, ma, md))
    #    for c, mi, ma, md in zip('xyz', mins_temp, maxs_temp, diff_temp):
    #      logger.info('    %s = %6.1f ... %6.1f mm --> Difference:  %6.1f mm'
    #                  % (c, mi, ma, md))
    prop = (diff / diff_temp).mean()
    indiv_spacing = (prop * template_spacing)
    print("    '%s' individual-spacing to '%s'[%.2f] is: %.4fmm" % (
        subject, ave_subject, template_spacing, indiv_spacing))

    return indiv_spacing


def _remove_vert_duplicates(subject, subj_src, label_dict_subject,
                            subjects_dir):
    """
    Removes all vertex duplicates from the vertex label list.
    (Those appear because of an unsuitable process of creating labelwise
    volume source spaces in mne-python)
    
    Parameters:
    -----------
    subject : str
        Subject ID.
    subj_src : mne.SourceSpaces
        Volume source space for the subject brain.
    label_dict_subject : dict
        Dictionary with the labels and their respective vertices
        for the subject.
    subjects_dir : str
        Path to the subjects directory.

    Returns:
    --------
    label_dict_subject : dict
        Dictionary with the labels and their respective vertices
        for the subject where duplicate vertices have been removed.
    """
    fname_s_aseg = op.join(subjects_dir, subject, 'mri', 'aseg.mgz')
    mgz = nib.load(fname_s_aseg)
    mgz_data = mgz.get_data()
    lut = _get_lut()
    vox2rastkr_trans = _get_mgz_header(fname_s_aseg)['vox2ras_tkr']
    vox2rastkr_trans[:3] /= 1000.
    inv_vox2rastkr_trans = linalg.inv(vox2rastkr_trans)

    all_volume_labels = mne.get_volume_labels_from_aseg(fname_s_aseg)
    all_volume_labels.remove('Unknown')

    print("""\n#### Attempting to check for vertice duplicates in labels due to
    spatial aliasing in %s's volume source creation..""" % subject)
    del_count = 0

    for p, label in enumerate(all_volume_labels):
        loadingBar(p, len(all_volume_labels), task_part=None)
        lab_arr = label_dict_subject[label]

        # get freesurfer LUT ID for the label
        lab_id = _get_lut_id(lut, label, True)[0]
        del_ver_idx_list = []
        for arr_id, i in enumerate(lab_arr, 0):
            # get the coordinates of the vertex in subject source space
            lab_vert_coord = subj_src[0]['rr'][i]
            # transform to mgz indices
            lab_vert_mgz_idx = mne.transforms.apply_trans(inv_vox2rastkr_trans, lab_vert_coord)
            # get ID from the mgt indices
            orig_idx = mgz_data[int(round(lab_vert_mgz_idx[0])),
                                int(round(lab_vert_mgz_idx[1])),
                                int(round(lab_vert_mgz_idx[2]))]

            # if ID and LUT ID do not match the vertex is removed
            if orig_idx != lab_id:
                del_ver_idx_list.append(arr_id)
                del_count += 1
        del_ver_idx = np.asarray(del_ver_idx_list)
        label_dict_subject[label] = np.delete(label_dict_subject[label], del_ver_idx)
    print('    Deleted', del_count, 'vertice duplicates.\n')

    return label_dict_subject


# %% ===========================================================================
# # Statistical Analysis Section
# =============================================================================

def sum_up_vol_cluster(clu, p_thresh=0.05, tstep=1e-3, tmin=0,
                       subject=None, vertices=None):
    """Assemble summary VolSourceEstimate from spatiotemporal cluster results.

    This helps visualizing results from spatio-temporal-clustering
    permutation tests.

    Parameters
    ----------
    clu : tuple
        the output from clustering permutation tests.
    p_thresh : float
        The significance threshold for inclusion of clusters.
    tstep : float
        The temporal difference between two time samples.
    tmin : float | int
        The time of the first sample.
    subject : str
        The name of the subject.
    vertices : list of arrays | None
        The vertex numbers associated with the source space locations.

    Returns
    -------
    out : instance of VolSourceEstimate
        A summary of the clusters. The first time point in this VolSourceEstimate
        object is the summation of all the clusters. Subsequent time points
        contain each individual cluster. The magnitude of the activity
        corresponds to the length the cluster spans in time (in samples).
    """
    T_obs, clusters, clu_pvals, _ = clu
    n_times, n_vertices = T_obs.shape
    good_cluster_inds = np.where(clu_pvals < p_thresh)[0]
    #  Build a convenient representation of each cluster, where each
    #  cluster becomes a "time point" in the VolSourceEstimate
    if len(good_cluster_inds) > 0:
        data = np.zeros((n_vertices, n_times))
        data_summary = np.zeros((n_vertices, len(good_cluster_inds) + 1))
        print('Data_summary is in shape of:', data_summary.shape)
        for ii, cluster_ind in enumerate(good_cluster_inds):
            loadingBar(ii + 1, len(good_cluster_inds), task_part='Cluster Idx %i' % cluster_ind)
            data.fill(0)
            v_inds = clusters[cluster_ind][1]
            t_inds = clusters[cluster_ind][0]
            data[v_inds, t_inds] = T_obs[t_inds, v_inds]
            # Store a nice visualization of the cluster by summing across time
            data = np.sign(data) * np.logical_not(data == 0) * tstep
            data_summary[:, ii + 1] = 1e3 * np.sum(data, axis=1)
            # Make the first "time point" a sum across all clusters for easy
            # visualization
        data_summary[:, 0] = np.sum(data_summary, axis=1)

        return VolSourceEstimate(data_summary, vertices, tmin=tmin, tstep=tstep,
                                 subject=subject)
    else:
        raise RuntimeError('No significant clusters available. Please adjust '
                           'your threshold or check your statistical '
                           'analysis.')


def plot_T_obs(T_obs, threshold, tail, save, fname_save):
    """ Visualize the Volume Source Estimate as an Nifti1 file """

    # T_obs plot code
    T_obs_flat = T_obs.flatten()
    plt.figure('T-Statistics', figsize=(8, 8))
    T_max = T_obs.max()
    T_min = T_obs.min()
    T_mean = T_obs.mean()
    str_tail = 'one tail'
    if tail is 0 or tail is None:
        plt.xlim([-20, 20])
        str_tail = 'two tail'
    elif tail is -1:
        plt.xlim([-20, 0])
    else:
        plt.xlim([0, T_obs_flat.max() * 1.05])
    y, bin_edges = np.histogram(T_obs_flat,
                               range=(0, T_obs_flat.max()),
                               bins=500)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if threshold is not None:
        plt.plot([threshold, threshold], (0, y[bin_centers >= 0.].max()), color='#CD7600',
                 linestyle=':', linewidth=2)

    legend = ('T-Statistics:\n'
              '      Mean:  %.2f\n'
              '      Minimum:  %.2f\n'
              '      Maximum:  %.2f\n'
              '      Threshold:  %.2f  \n'
              '      ') % (T_mean, T_min, T_max, threshold)
    plt.ylim(None, y[bin_centers >= 0.].max() * 1.1)
    plt.xlabel('T-scores', fontsize=12)
    plt.ylabel('T-values count', fontsize=12)
    plt.title('T statistics distribution of t-test - %s' % str_tail, fontsize=15)
    plt.plot(bin_centers, y, label=legend, color='#00868B')
    #    plt.xlim([])
    plt.tight_layout()
    legend = plt.legend(loc='upper right', shadow=True, fontsize='large', frameon=True)

    if save:
        plt.savefig(fname_save)
        plt.close()

    return


def plot_T_obs_3D(T_obs, save, fname_save):
    """ Visualize the Volume Source Estimate as an Nifti1 file """
    from matplotlib import cm as cm_mpl
    fig = plt.figure(facecolor='w', figsize=(8, 8))
    ax = fig.gca(projection='3d')
    vertc, timez = np.mgrid[0:T_obs.shape[0], 0:T_obs.shape[1]]
    Ts = T_obs
    title = 'T Obs'
    t_obs_stats = ax.plot_surface(vertc, timez, Ts, cmap=cm_mpl.hot)  # , **kwargs)
    # plt.set_xticks([])
    # plt.set_yticks([])
    ax.set_xlabel('times [ms]')
    ax.set_ylabel('Vertice No')
    ax.set_zlabel('Statistical Amplitude')
    ax.w_zaxis.set_major_locator(LinearLocator(6))
    ax.set_zlim(0, np.max(T_obs))
    ax.set_title(title)
    fig.colorbar(t_obs_stats, shrink=0.5)
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(fname_save)
        plt.close()
    return
