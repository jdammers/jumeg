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
import time
import os.path
from scipy.interpolate import griddata
from mne.source_estimate import _write_stc
from nilearn import plotting
from nilearn.image import index_img
from nibabel.affines import apply_affine
import matplotlib.pyplot as plt
from mne.source_estimate import VolSourceEstimate
from matplotlib import cm
from matplotlib.ticker import LinearLocator


# =============================================================================
# 
# =============================================================================
def convert_to_unicode(inlist):
    if type(inlist) != unicode:
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
    label_list : list
        A list containing all labels available for the subject's source space
        with the according vertice indices
    
    """
    fname_labels = fname_src[:-4] + '_vertno_labelwise.npy'
    label_list = np.load(fname_labels).item()
    subj_vert_src = mne.read_source_spaces(fname_src)
    label_list = _remove_vert_duplicates(subject, subj_vert_src, label_list,
                                         subjects_dir)
    del subj_vert_src

    return label_list


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


def auto_match_labels(fname_subj_src, label_list_subject,
                      fname_temp_src, label_list_template,
                      volume_labels, template_spacing,
                      e_func, fname_save, save_trans=False):
    """
    Matches a subject's volume source space labelwise to another volume
    source space
    
    Parameters
    ----------
    fname_subj_src : string
        Filename of the first volume source space
    fname_temp_src : string
        Filename of the second volume source space to match on
    volume_labels : list of volume Labels
        List of the volume labels of interest
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
        errfunc = _point_cloud_error_balltree
    if e_func == 'euclidean':
        err_function = 'Euclidean Error Function'
        errfunc = _point_cloud_error
    if e_func is None:
        print 'No Error Function provided, using BallTree instead'
        err_function = 'BallTree Error Function'
        errfunc = _point_cloud_error_balltree

    subj_src = mne.read_source_spaces(fname_subj_src)
    subject_from = subj_src[0]['subject_his_id']
    x, y, z = subj_src[0]['rr'].T
    # hlight: subj_p contains the coordinates of the vertices
    subj_p = np.c_[x, y, z]
    # hlight: how is subject different from subject_from?
    subject = subj_src[0]['subject_his_id']

    temp_src = mne.read_source_spaces(fname_temp_src)
    subject_to = temp_src[0]['subject_his_id']
    x1, y1, z1 = temp_src[0]['rr'].T
    # hlight: temp_p contains the coordinates of the vertices
    temp_p = np.c_[x1, y1, z1]
    # hlight: how is template different from subject_to?
    template = temp_src[0]['subject_his_id']

    print """\n#### Attempting to match %d volume source space labels from
    Subject: '%s' to Template: '%s' with
    %s...""" % (len(volume_labels), subject, template, err_function)

    # hlight: wouldn't it be easier to just sum up the shape of the label list instead of making a list?
    vert_sum = []
    vert_sum_temp = []

    for label_i in volume_labels:
        vert_sum.append(label_list_subject[label_i].shape[0])
        vert_sum_temp.append(label_list_template[label_i].shape[0])

        # check for overlapping labels
        for label_j in volume_labels:
            if label_i != label_j:
                h = np.intersect1d(label_list_subject[label_i], label_list_subject[label_j])
                if h.shape[0] > 0:
                    print 'Label %s contains %d vertices from label %s' % (label_i, h.shape[0], label_i)
                    # hlight: why break here? Couldn't there be overlap with other labels as well?
                    break

    print '    # N subject vertices:', np.sum(np.asarray(vert_sum))
    print '    # N template vertices:', np.sum(np.asarray(vert_sum_temp))

    # Prepare empty containers to store the possible transformation results
    label_trans_dic = {}
    label_trans_dic_err = {}
    label_trans_dic_var_dist = {}
    label_trans_dic_mean_dist = {}
    label_trans_dic_max_dist = {}
    start_time = time.time()
    del subj_src, temp_src

    for label_idx, label in enumerate(volume_labels):
        loadingBar(count=label_idx, total=len(volume_labels),
                   task_part='%s' % label)
        print ''

        # Select coords for label and check if they exceed the label size limit
        s_pts = subj_p[label_list_subject[label]]
        t_pts = temp_p[label_list_template[label]]

        # hlight: what's the significance of s_pts having less than 6 elements?
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
            if e_func == 'balltree':
                temp_tree = BallTree(t_pts)
            elif e_func == 'euclidean':
                continue
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

            # hlight: fix comparison of float with zero
            if t_x_diff == 0 or s_x_diff == 0:
                x_scale = 0.
            else:
                x_scale = t_x_diff / s_x_diff

            if t_y_diff == 0 or s_y_diff == 0:
                y_scale = 0.
            else:
                y_scale = t_y_diff / s_y_diff

            if t_z_diff == 0 or s_z_diff == 0:
                z_scale = 0.
            else:
                z_scale = t_z_diff / s_z_diff

            # Find center of mass
            cm_s = np.mean(s_pts, axis=0)
            cm_t = np.mean(t_pts, axis=0)
            initial_transl = (cm_t - cm_s)
            # Perform the transformation of the initial transformation matrix
            init_trans = np.zeros([4, 4])
            # hlight: what happens if the label is neither left nor right? -> init_trans is filled with zeros
            if label[0] == 'L':
                init_trans[:3, :3] = rotation3d(0., 0., 0.) * [x_scale, y_scale, z_scale]
            elif label[0] == 'R':
                init_trans[:3, :3] = rotation3d(0., 0., 0.) * [x_scale, y_scale, z_scale]

            # =============================================================================
            #         ENDING CHANGES
            # =============================================================================
            init_trans[0, 3] = initial_transl[0]
            init_trans[1, 3] = initial_transl[1]
            init_trans[2, 3] = initial_transl[2]
            init_trans[3, 3] = 1.

        # hlight: if t_pts.shape[0] == 0: errfunc, temp_tree, and init_trans are not defined or rather the previous ones are used
        # Find the min summed distance for initial transformation
        poss_trans = find_min(template_spacing, e_func, errfunc, temp_tree, t_pts, s_pts, init_trans)
        all_dist_max_l = []
        all_dist_mean_l = []
        all_dist_var_l = []
        all_dist_err_l = []
        for tra in poss_trans:
            to_match_points = s_pts
            to_match_points = apply_trans(tra, to_match_points)

            if e_func == 'balltree':
                all_dist_max_l.append(np.array(
                    np.max(errfunc(to_match_points[:, :3], temp_tree))
                ))
                all_dist_mean_l.append(np.array(
                    np.mean(errfunc(to_match_points[:, :3], temp_tree))
                ))
                all_dist_var_l.append(np.array(np.var(
                    errfunc(to_match_points[:, :3], temp_tree)
                )))
                all_dist_err_l.append(np.array(
                    errfunc(to_match_points[:, :3], temp_tree))
                )

            if e_func == 'euclidean':
                all_dist_max_l.append(np.array(
                    np.max(errfunc(to_match_points[:, :3], t_pts))
                ))
                all_dist_mean_l.append(np.array(
                    np.mean(errfunc(to_match_points[:, :3], t_pts))
                ))
                all_dist_var_l.append(np.array(np.var(
                    errfunc(to_match_points[:, :3], t_pts)
                )))
                all_dist_err_l.append(np.array(
                    errfunc(to_match_points[:, :3], t_pts))
                )
            del to_match_points

        all_dist_max = np.asarray(all_dist_max_l)
        all_dist_mean = np.asarray(all_dist_mean_l)
        all_dist_var = np.asarray(all_dist_var_l)
        # Decide for the bestg fitting Transformation-Matrix
        idx1 = np.where(all_dist_mean == np.min(all_dist_mean))[0][0]
        # Collect all the possible inidcator values
        trans = poss_trans[idx1]
        del poss_trans

        to_match_points = s_pts
        to_match_points = apply_trans(trans, to_match_points)
        if e_func == 'balltree':
            all_dist_max = np.array(
                (np.max(errfunc(to_match_points[:, :3], temp_tree))))
            all_dist_mean = np.array(
                (np.mean(errfunc(to_match_points[:, :3], temp_tree))))
            all_dist_var = np.array(
                (np.var(errfunc(to_match_points[:, :3], temp_tree))))
            all_dist_err = (errfunc(to_match_points[:, :3], temp_tree))
        if e_func == 'euclidean':
            all_dist_max = np.array(
                (np.max(errfunc(to_match_points[:, :3], t_pts))))
            all_dist_mean = np.array(
                (np.mean(errfunc(to_match_points[:, :3], t_pts))))
            all_dist_var = np.array(
                (np.var(errfunc(to_match_points[:, :3], t_pts))))
            all_dist_err = (errfunc(to_match_points[:, :3], t_pts))
        del to_match_points

        # Append the Dictionaries with the result and error values
        label_trans_dic.update({label: trans})
        label_trans_dic_mean_dist.update({label: np.min(all_dist_mean)})
        label_trans_dic_max_dist.update({label: np.min(all_dist_max)})
        label_trans_dic_var_dist.update({label: np.min(all_dist_var)})
        label_trans_dic_err.update({label: all_dist_err})

    if save_trans:
        print '\n    Writing Transformation matrices to file..'
        fname_lw_trans = fname_save
        mat_mak_trans_dict = {}
        mat_mak_trans_dict['ID'] = '%s -> %s' % (subject_from, subject_to)
        mat_mak_trans_dict['Labeltransformation'] = label_trans_dic
        mat_mak_trans_dict['Transformation Error[mm]'] = label_trans_dic_err
        mat_mak_trans_dict['Mean Distance Error [mm]'] = label_trans_dic_mean_dist
        mat_mak_trans_dict['Max Distance Error [mm]'] = label_trans_dic_max_dist
        mat_mak_trans_dict['Distance Variance Error [mm]'] = label_trans_dic_var_dist
        mat_mak_trans_dict_arr = np.array([mat_mak_trans_dict])
        np.save(fname_lw_trans, mat_mak_trans_dict_arr)
        print '    [done] -> Calculation Time: %.2f minutes.' % (
            ((time.time() - start_time) / 60))

        return

    else:
        return (label_trans_dic, label_trans_dic_err, label_trans_dic_mean_dist,
                label_trans_dic_max_dist, label_trans_dic_var_dist)


def find_min(template_spacing, e_func, errfunc, temp_tree, t_pts, s_pts, init_trans):
    """
    Aux. function for auto_match_labels.
    """

    sourr = template_spacing / 1e3
    auto_match_iters = np.array([[0., 0., 0.],
                                 [0., 0., sourr], [0., 0., sourr * 2], [0., 0., sourr * 3],
                                 [sourr, 0., 0.], [sourr * 2, 0., 0.], [sourr * 3, 0., 0.],
                                 [0., sourr, 0.], [0., sourr * 2, 0.], [0., sourr * 3, 0.],
                                 [0., 0., -sourr], [0., 0., -sourr * 2], [0., 0., -sourr * 3],
                                 [-sourr, 0., 0.], [-sourr * 2, 0., 0.], [-sourr * 3, 0., 0.],
                                 [0., -sourr, 0.], [0., -sourr * 2, 0.], [0., -sourr * 3, 0.]])

    poss_trans = []
    for p, i in enumerate(auto_match_iters):
        # initial translation value
        tx, ty, tz = init_trans[0, 3] + i[0], init_trans[1, 3] + i[1], init_trans[2, 3] + i[2]
        sx, sy, sz = init_trans[0, 0], init_trans[1, 1], init_trans[2, 2]
        rx, ry, rz = 0, 0, 0
        x0 = (tx, ty, tz, rx, ry, rz)

        # hlight: possible to take this outside of find_min?
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


def _transform_src_lw(vsrc_subject_from, label_list_subject_from,
                      volume_labels, subject_to,
                      subjects_dir, label_trans_dic=None):
    """Transformes given Labels of interest from one subjects' to another.
    
    Parameters
    ----------
    vsrc_subject_from : instance of SourceSpaces
        The source spaces that will be transformed.
    volume_labels : list
        List of the volume labels of interest
    subject_to : str | None
        The template subject.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    
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
    label_list = label_list_subject_from
    print """\n#### Attempting to transform %s source space labelwise to 
    %s source space..""" % (subject, subject_to)

    if label_trans_dic is None:
        print '\n#### Attempting to read MatchMaking Transformations from file..'
        indiv_spacing = (np.abs(subj_vol[0]['rr'][0, 0]) -
                         np.abs(subj_vol[0]['rr'][1, 0])) * 1e3
        fname_lw_trans = (subjects_dir + subject + '/' +
                          '%s_%s_vol-%.2f_lw-trans.npy' % (subject, subject_to,
                                                           indiv_spacing))

        try:
            mat_mak_trans_dict_arr = np.load(fname_lw_trans)

        except:
            print 'MatchMaking Transformations file NOT found:'
            print fname_lw_trans, '\n'
            print 'Please calculate the according transformation matrix dictionary'
            print 'by using the jumeg.jumeg_volmorpher.auto_match_labels function.'

            import sys
            sys.exit(-1)


        label_trans_ID = mat_mak_trans_dict_arr[0]['ID']
        print '    Reading MatchMaking file %s..' % label_trans_ID
        label_trans_dic = mat_mak_trans_dict_arr[0]['Labeltransformation']
    else:
        label_trans_dic = label_trans_dic

    vert_sum = []
    for i in volume_labels:
        vert_sum.append(label_list[i].shape[0])
        for j in volume_labels:
            if i != j:
                h = np.intersect1d(label_list[i], label_list[j])
                if h.shape[0] > 0:
                    print "In Label:", i, """ are vertices from
            Label:""", j, "(", h.shape[0], ")"
                    break

    transformed_p = np.array([[0, 0, 0]])
    vert_sum = []
    idx_vertices = []
    # TODO: p stands for points?
    for p, label in enumerate(volume_labels):
        loadingBar(p, len(volume_labels), task_part=label)
        vert_sum.append(label_list[label].shape[0])
        idx_vertices.append(label_list[label])
        trans_p = subj_p[label_list[label]]
        trans = label_trans_dic[label]
        # apply trans
        trans_p = apply_trans(trans, trans_p)
        del trans
        transformed_p = np.concatenate((transformed_p, trans_p))
        del trans_p
    transformed_p = transformed_p[1:]
    idx_vertices = np.concatenate(np.asarray(idx_vertices))
    print '    [done]'

    return (transformed_p, idx_vertices)


def volume_morph_stc(fname_stc_orig, subject_from, fname_vsrc_subject_from,
                     volume_labels, subject_to, fname_vsrc_subject_to,
                     cond, n_iter, interpolation_method, normalize,
                     subjects_dir, unwanted_to_zero=True, label_trans_dic=None,
                     fname_save_stc=None, save_stc=False, plot=False):
    """ Perform a volume morphing from one subject to a template.
    
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
    interpolation_method : string | None
        Either 'balltree' or 'euclidean'. Default is 'balltree'
    cond : str (Not really needed)
        Experimental condition under which the data was recorded.
    n_iter : int (Not really needed)
        Number of iterations performed during MFT.
    normalize : bool
        hlight: what to say here?
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.
    unwanted_to_zero : bool
        hlight: what to say here?
    label_trans_dic : dict
        hlight: what to say here?
    fname_save_stc : None | str
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
    print '####                  START                ####'
    print '####             Volume Morphing           ####'
    print '    Subject: %s  | Cond.: %s  | Iter.: %s' % (subject_from,
                                                         cond, n_iter)
    print '\n#### Attempting to read essential data files..'
    # STC 
    stc_orig = mne.read_source_estimate(fname_stc_orig)
    stcdata = stc_orig.data
    nvert, ntimes = stc_orig.shape
    tmin, tstep = stc_orig.times[0], stc_orig.tstep

    # Source Spaces
    subj_vol = mne.read_source_spaces(fname_vsrc_subject_from)
    temp_vol = mne.read_source_spaces(fname_vsrc_subject_to)
    fname_label_list_subject_from = (fname_vsrc_subject_from[:-4] +
                                     '_vertno_labelwise.npy')
    label_list_subject_from = np.load(fname_label_list_subject_from).item()
    fname_label_list_subject_to = (fname_vsrc_subject_to[:-4] +
                                   '_vertno_labelwise.npy')
    label_list_subject_to = np.load(fname_label_list_subject_to).item()

    # Check for vertex duplicates
    label_list_subject_from = _remove_vert_duplicates(subject_from, subj_vol,
                                                      label_list_subject_from,
                                                      subjects_dir)

    # Transform the whole subject source space labelwise
    transformed_p, idx_vertices = _transform_src_lw(subj_vol,
                                                    label_list_subject_from,
                                                    volume_labels, subject_to,
                                                    subjects_dir,
                                                    label_trans_dic)
    xn, yn, zn = transformed_p.T

    stcdata_sel = []
    for p, i in enumerate(idx_vertices):
        stcdata_sel.append(np.where(idx_vertices[p] == subj_vol[0]['vertno']))
    stcdata_sel = np.asarray(stcdata_sel).flatten()
    stcdata_ch = stcdata[stcdata_sel]

    # =========================================================================
    #        Interpolate the data   
    new_data = {}
    for inter_m in interpolation_method:

        print '\n#### Attempting to interpolate STC Data for every time sample..'
        print '    Interpolation method: ', inter_m
        st_time = time.time()
        xt, yt, zt = temp_vol[0]['rr'][temp_vol[0]['inuse'].astype(bool)].T
        inter_data = np.zeros([xt.shape[0], ntimes])
        for i in range(0, ntimes):
            loadingBar(i, ntimes, task_part='Time slice: %i' % (i + 1))
            inter_data[:, i] = griddata((xn, yn, zn),
                                        stcdata_ch[:, i], (xt, yt, zt),
                                        method=inter_m, rescale=True)
        if inter_m == 'linear':
            inter_data = np.nan_to_num(inter_data)

        if unwanted_to_zero:
            print '#### Setting all unknown vertex values to zero..'

            # vertnos_unknown = label_list_subject_to['Unknown']
            # vert_U_idx = np.array([], dtype=int)
            # for i in xrange(0, vertnos_unknown.shape[0]):
            #     vert_U_idx = np.append(vert_U_idx,
            #                            np.where(vertnos_unknown[i] == temp_vol[0]['vertno'])
            #                            )
            # inter_data[vert_U_idx, :] = 0.
            #
            # # now the original data
            # vertnos_unknown_from = label_list_subject_from['Unknown']
            # vert_U_idx = np.array([], dtype=int)
            # for i in xrange(0, vertnos_unknown_from.shape[0]):
            #     vert_U_idx = np.append(vert_U_idx,
            #                            np.where(vertnos_unknown_from[i] == subj_vol[0]['vertno'])
            #                            )
            # stc_orig.data[vert_U_idx, :] = 0.

            temp_LOI_idx = np.array([], dtype=int)
            for p, labels in enumerate(volume_labels):
                lab_verts_temp = label_list_subject_to[labels]
                for i in xrange(0, lab_verts_temp.shape[0]):
                    temp_LOI_idx = np.append(temp_LOI_idx,
                                             np.where(lab_verts_temp[i]
                                                      ==
                                                      temp_vol[0]['vertno'])
                                             )
            d2 = np.zeros(inter_data.shape)
            d2[temp_LOI_idx, :] = inter_data[temp_LOI_idx, :]
            inter_data = d2

            subj_LOI_idx = np.array([], dtype=int)
            for p, labels in enumerate(volume_labels):
                lab_verts_temp = label_list_subject_from[labels]
                for i in xrange(0, lab_verts_temp.shape[0]):
                    subj_LOI_idx = np.append(subj_LOI_idx,
                                             np.where(lab_verts_temp[i]
                                                      ==
                                                      subj_vol[0]['vertno'])
                                             )
            d2 = np.zeros(stc_orig.data.shape)
            d2[subj_LOI_idx, :] = stc_orig.data[subj_LOI_idx, :]
            # TODO: why is in stc_orig.data.flags WRITEABLE=False ?? causes crash
            if not stc_orig.data.flags["WRITEABLE"]:
                stc_orig.data.setflags(write=1)

            stc_orig.data[:, :] = d2[:, :]

        if normalize:
            print '\n#### Attempting to normalize the vol-morphed stc..'
            normalized_new_data = inter_data.copy()
            for p, labels in enumerate(volume_labels):
                lab_verts = label_list_subject_from[labels]
                lab_verts_temp = label_list_subject_to[labels]
                subj_vert_idx = np.array([], dtype=int)
                for i in xrange(0, lab_verts.shape[0]):
                    subj_vert_idx = np.append(subj_vert_idx,
                                              np.where(lab_verts[i]
                                                       ==
                                                       subj_vol[0]['vertno'])
                                              )
                temp_vert_idx = np.array([], dtype=int)
                for i in xrange(0, lab_verts_temp.shape[0]):
                    temp_vert_idx = np.append(temp_vert_idx,
                                              np.where(lab_verts_temp[i]
                                                       ==
                                                       temp_vol[0]['vertno'])
                                              )
                a = np.sum(stc_orig.data[subj_vert_idx], axis=0)
                b = np.sum(inter_data[temp_vert_idx], axis=0)
                norm_m_score = a / b
                normalized_new_data[temp_vert_idx] *= norm_m_score
            new_data.update({inter_m + '_norm': normalized_new_data})
        else:
            new_data.update({inter_m: inter_data})

        print '    [done] -> Calculation Time: %.2f minutes.' % (
                (time.time() - st_time) / 60.
        )

    if save_stc:
        print '\n#### Attempting to write interpolated STC Data to file..'
        for inter_m in interpolation_method:

            if fname_save_stc is None:
                fname_stc_orig_morphed = (fname_stc_orig[:-7] +
                                          '_morphed_to_%s_%s-vl.stc' % (subject_to,
                                                                        inter_m))
            else:
                fname_stc_orig_morphed = fname_save_stc

            print '    Destination:', fname_stc_orig_morphed
            if normalize:
                _write_stc(fname_stc_orig_morphed, tmin=tmin, tstep=tstep,
                           vertices=temp_vol[0]['vertno'],
                           data=new_data[inter_m + '_norm'])
                stc_morphed = mne.read_source_estimate(fname_stc_orig_morphed)

                if plot:
                    _volumemorphing_plot_results(stc_orig, stc_morphed,
                                                 interpolation_method,
                                                 subj_vol, label_list_subject_from,
                                                 temp_vol, label_list_subject_to,
                                                 volume_labels, subject_from,
                                                 subject_to, cond=cond, n_iter=n_iter,
                                                 subjects_dir=subjects_dir, save=True)
            else:
                _write_stc(fname_stc_orig_morphed, tmin=tmin, tstep=tstep,
                           vertices=temp_vol[0]['vertno'], data=new_data[inter_m])
                stc_morphed = mne.read_source_estimate(fname_stc_orig_morphed)

                if plot:
                    _volumemorphing_plot_results(stc_orig, stc_morphed,
                                                 interpolation_method,
                                                 subj_vol, temp_vol,
                                                 volume_labels,
                                                 subject_from, subject_to,
                                                 cond=cond, n_iter=n_iter,
                                                 subjects_dir=subjects_dir,
                                                 save=True)
        print '[done]'
        print '####             Volume Morphing           ####'
        print '####                  DONE                 ####'

        return stc_morphed

    print '#### Volume morphed stc data NOT saved..\n'
    print '####             Volume Morphing           ####'
    print '####                  DONE                 ####'

    return new_data


def _volumemorphing_plot_results(stc_orig, stc_morphed,
                                 interpolation_method,
                                 volume_orig, label_list_from,
                                 volume_temp, label_list_to,
                                 volume_labels, subject, subject_to,
                                 cond, n_iter, subjects_dir, save=False):
    """Gathering information and plot before and after morphing results.
    
    Parameters
    ----------
    stc_orig : VolSourceEstimate
        Volume source estimate for the original subject.
    stc_morphed : VolSourceEstimate
        Volume source estimate for the destination subject.
    interpolation_method : str | None
        Interpolationmethod as a string to give the plot more intel
    volume_orig : instance of SourceSpaces
        The original source space that were morphed to the current
        subject.
    label_list_from : list
        Equivalent label vertice list to the original source space
    volume_temp : instance of SourceSpaces
        The template source space that is  morphed on.
    label_list_to : list
        Equivalent label vertice list to the template source space
    volume_labels : list of volume Labels
        List of the volume labels of interest
    subject : string
        Name of the subject from which to morph as named in the SUBJECTS_DIR
    subject_to : string
        Name of the subject on which to morph as named in the SUBJECTS_DIR
    cond : str | None
        Evoked contition as a string to give the plot more intel
    n_iter : int
        Number of iterations performed during MFT.
    subjects_dir : string, or None
        Path to SUBJECTS_DIR if it is not set in the environment.

    Returns
    -------
    if save=True : None
        Automatically creates matplotlib.figure and writes it to disk.
    if save=Fales : returns matplotlib.figure
    
    """
    if subject_to is None:
        subject_to = ''
    else:
        subject_to = subject_to
    if cond is None:
        cond = ''
    else:
        cond = cond
    if n_iter is None:
        n_iter = ''
    else:
        n_iter = n_iter

    subj_vol = volume_orig
    subject_from = volume_orig[0]['subject_his_id']
    temp_vol = volume_temp
    temp_spacing = (abs(temp_vol[0]['rr'][0, 0]
                        - temp_vol[0]['rr'][1, 0]) * 1000).round()
    subject_to = volume_temp[0]['subject_his_id']
    label_list = label_list_from
    label_list_template = label_list_to
    new_data = stc_morphed.data
    indiv_spacing = make_indiv_spacing(subject_from, subject_to,
                                       temp_spacing, subjects_dir)

    print '\n#### Attempting to save the volume morphing results ..'
    directory = subjects_dir + '%s/plots/VolumeMorphing/' % (subject)
    if not os.path.exists(directory):
        os.makedirs(directory)

        # Create new figure and two subplots, sharing both axes
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True,
                                   num=999, figsize=(16, 9))
    fig.text(0.985, 0.75, 'Amplitude [T]', color='white', size='large',
             horizontalalignment='right', verticalalignment='center',
             rotation=-90, transform=ax1.transAxes)
    fig.text(0.985, 0.25, 'Amplitude [T]', color='white', size='large',
             horizontalalignment='right', verticalalignment='center',
             rotation=-90, transform=ax2.transAxes)
    plt.suptitle('VolumeMorphing from %s to %s | Cond.: %s, Iter.: %d'
                 % (subject_from, subject_to, cond, n_iter),
                 fontsize=16, color='white')
    fig.set_facecolor('black')
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.04, top=0.94,
                        left=0.0, right=0.97)
    t = int(np.where(np.sum(stc_orig.data, axis=0)
                     == np.max(np.sum(stc_orig.data, axis=0)))[0])
    plot_vstc(vstc=stc_orig, vsrc=volume_orig, tstep=stc_orig.tstep,
              subjects_dir=subjects_dir,
              time_sample=t, coords=None,
              figure=999, axes=ax1,
              save=False)

    plot_vstc(vstc=stc_morphed, vsrc=volume_temp, tstep=stc_orig.tstep,
              subjects_dir=subjects_dir,
              time_sample=t, coords=None,
              figure=999, axes=ax2,
              save=False)
    if save:
        fname_save_fig = (directory +
                          '/%s_%s_vol-%.2f_%s_%s_volmorphing-result.png'
                          % (subject_from, subject_to,
                             indiv_spacing, cond, n_iter))
        plt.savefig(fname_save_fig, facecolor=fig.get_facecolor(),
                    format='png', edgecolor='none')
        plt.close()
    else:
        plt.show()

    print """\n#### Attempting to compare subjects activity and interpolated
        activity in template for all labels.."""
    subj_lab_act = {}
    temp_lab_act = {}
    for p, label in enumerate(volume_labels):
        lab_arr = label_list[str(label)]
        lab_arr_temp = label_list_template[str(label)]
        subj_vert_idx = np.array([], dtype=int)
        temp_vert_idx = np.array([], dtype=int)
        for i in xrange(0, lab_arr.shape[0]):
            subj_vert_idx = np.append(subj_vert_idx,
                                      np.where(lab_arr[i]
                                               == subj_vol[0]['vertno']))
        for i in xrange(0, lab_arr_temp.shape[0]):
            temp_vert_idx = np.append(temp_vert_idx,
                                      np.where(lab_arr_temp[i]
                                               == temp_vol[0]['vertno']))
        lab_act_sum = np.array([])
        lab_act_sum_temp = np.array([])
        for t in xrange(0, stc_orig.times.shape[0]):
            lab_act_sum = np.append(lab_act_sum,
                                    np.sum(stc_orig.data[subj_vert_idx, t]))
            lab_act_sum_temp = np.append(lab_act_sum_temp,
                                         np.sum(stc_morphed.data[temp_vert_idx, t]))
        subj_lab_act.update({label: lab_act_sum})
        temp_lab_act.update({label: lab_act_sum_temp})
    print '    [done]'

    # Stc per label  
    # fig, axs = plt.subplots(len(volume_labels) / 5, 5, figsize=(15, 15),
    #                         facecolor='w', edgecolor='k')
    fig, axs = plt.subplots(int(np.ceil(len(volume_labels) / 5.)), 5, figsize=(15, 15),
                            facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.6, wspace=.255,
                        bottom=0.089, top=.9,
                        left=0.03, right=0.985)
    axs = axs.ravel()
    for p, label in enumerate(volume_labels):
        axs[p].plot(stc_orig.times, subj_lab_act[label], '#00868B',
                    linewidth=0.9, label=('%s vol-%.2f'
                                          % (subject_from, indiv_spacing)))
        axs[p].plot(stc_orig.times, temp_lab_act[label], '#CD7600', ls=':',
                    linewidth=0.9, label=('%s volume morphed vol-%.2f'
                                          % (subject_from, temp_spacing)))
        axs[p].set_title(label, fontsize='medium', loc='right')
        axs[p].ticklabel_format(style='sci', axis='both')
        axs[p].set_xlabel('Time [s]')
        axs[p].set_ylabel('Amplitude [T]')
        axs[p].set_xlim(stc_orig.times[0], stc_orig.times[-1])
        axs[p].get_xaxis().grid(True)

    fig.suptitle('Summed activity in volume labels - %s[%.2f]' % (subject_from, indiv_spacing)
                 + ' -> %s [%.2f] | Cond.: %s, Iter.: %d'
                 % (subject_to, temp_spacing, cond, n_iter),
                 fontsize=16)
    if save:
        fname_save_fig = os.path.join(directory, '%s_%s_vol-%.2f_%s_iter-%d_labelwise-stc.png')
        fname_save_fig = fname_save_fig % (subject_from, subject_to, indiv_spacing, cond, n_iter)
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
    ax1.set_title('Summed Source Amplitude - %s[%.2f] ' % (subject_from, indiv_spacing)
                  + '-> %s [%.2f] | Cond.: %s, Iter.: %d'
                  % (subject_to, temp_spacing, cond, n_iter))
    ax1.text(stc_orig.times[0],
             np.maximum(stc_orig.data.sum(axis=0), new_data.sum(axis=0)).max(),
             """Total Amplitude Difference: %+.2f %%
             Total Amplitude Difference (norm):  %+.2f %%"""
             % (act_diff_perc, act_diff_perc_morphed_normed),
             size=12, ha="left", va="top",
             bbox=dict(boxstyle="round",
                       ec=("grey"),
                       fc=("white"),
                       )
             )
    ax1.set_ylabel('Summed Source Amplitude')
    ax1.legend(fontsize='large', facecolor="white", edgecolor="grey")
    ax1.get_xaxis().grid(True)
    plt.tight_layout()
    if save:
        fname_save_fig = os.path.join(directory, '%s_%s_vol-%.2f_%s_iter-%d_stc.png')
        fname_save_fig = fname_save_fig % (subject_from, subject_to, indiv_spacing, cond, n_iter)
        plt.savefig(fname_save_fig, facecolor=fig.get_facecolor(),
                    format='png', edgecolor='none')
        plt.close()
    else:
        plt.show()

    return


def plot_vstc(vstc, vsrc, tstep, subjects_dir, time_sample=None, coords=None,
              figure=None, axes=None, cmap='hot', symmetric_cbar=False,
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
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
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
            print '    Please provide the tstep value !'
    img_data = img.get_data()
    aff = img.affine
    if time_sample is None:
        # global maximum amp in time
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])
    else:
        print '    Time slice', time_sample
        t = time_sample
    t_in_ms = vstc.times[t] * 1e3
    print '    Found time slice: ', t_in_ms, 'ms'
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
    print ('    Coords [mri-space]:'
           + 'X: ', slice_x, 'Y: ', slice_y, 'Z: ', slice_z)
    temp_t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')

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
        if fname_save == None:
            print 'please provide an filepath to save .png'
        else:
            plt.savefig(fname_save)
            plt.close()

    return vstc_plt


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
        for i, idx in enumerate(label_inds): inuse_replace[idx] = 1
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
    from nilearn import plotting
    from nilearn.image import index_img
    from nibabel.affines import apply_affine
    from jumeg.jumeg_volmorpher import _make_image
    print '\n#### Attempting to plot volume stc from file..'
    print '    Creating 3D image from stc..'
    vstcdata = vstc.data
    img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
    subject = vsrc[0]['subject_his_id']
    if vstc == 0:
        if tstep is not None:
            img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
        else:
            print '    Please provide the tstep value !'
    img_data = img.get_data()
    aff = img.affine
    if time_sample is None:
        print '    Searching for maximal Activation..'
        t = int(np.where(np.sum(vstcdata, axis=0) == np.max(np.sum(vstcdata, axis=0)))[0])  # maximum amp in time
    else:
        print '    Using Cluster No.', time_sample
        t = time_sample
    if title == None:
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
    print '    Respective Space Coords [mri-space]:'
    print '    X: ', slice_x, '    Y: ', slice_y, '    Z: ', slice_z
    temp_t1_fname = subjects_dir + subject + '/mri/T1.mgz'
    VSTCPT_plot = plotting.plot_stat_map(index_img(img, t), temp_t1_fname,
                                         figure=figure, axes=axes,
                                         display_mode='ortho',
                                         threshold=vstcdata.min(),
                                         annotate=True,
                                         title=title,
                                         cut_coords=None,
                                         cmap='black_red')
    if save:
        if fname_save == None:
            print 'please provide an filepath to save .png'
        else:
            plt.savefig(fname_save)
            plt.close()

    return VSTCPT_plot


def make_indiv_spacing(subject, ave_subject, standard_spacing, subjects_dir):
    """ Identifies the suiting grid space difference of a subject's volume
        source space to a template's volume source space, before a planned
        morphing takes place.
    
    Parameters
    ----------
    s_pts : String
          Filename
    t_pts : list of Labels
          Filename
  
    Returns
    -------
    trans : SourceEstimate
          The generated source time courses.
    """
    fname_surf = subjects_dir + subject + '/bem/watershed/%s_inner_skull_surface' % subject
    fname_surf_temp = subjects_dir + ave_subject + '/bem/watershed/%s_inner_skull_surface' % ave_subject
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
    indiv_spacing = (prop * standard_spacing)
    print "    '%s' individual-spacing to '%s'[%.2f] is: %.4fmm" % (
    subject, ave_subject, standard_spacing, indiv_spacing)

    return indiv_spacing


def _remove_vert_duplicates(subject, subj_src, label_list_subject,
                            subjects_dir):
    """ Removes all vertice duplicates from the vertice label list.
        (Those appear because of an unsuitable process of creating labelwise
        volume source spaces in mne-python)
    
    Parameters
    ----------
    stc_data : Data of VolSourceEstimate
        The source estimate data

    Returns
    -------
    
    """
    fname_s_aseg = subjects_dir + subject + '/mri/aseg.mgz'
    mgz = nib.load(fname_s_aseg)
    mgz_data = mgz.get_data()
    lut = _get_lut()
    vox2rastkr_trans = _get_mgz_header(fname_s_aseg)['vox2ras_tkr']
    vox2rastkr_trans[:3] /= 1000.
    inv_vox2rastkr_trans = linalg.inv(vox2rastkr_trans)
    all_volume_labels = []
    vol_lab = mne.get_volume_labels_from_aseg(fname_s_aseg)
    for lab in vol_lab: all_volume_labels.append(lab.encode())
    all_volume_labels.remove('Unknown')

    print """\n#### Attempting to check for vertice duplicates in labels due to
    spatial aliasing in %s's volume source creation..""" % subject
    del_count = 0
    for p, label in enumerate(all_volume_labels):
        loadingBar(p, len(all_volume_labels), task_part=None)
        lab_arr = label_list_subject[label]
        lab_id = _get_lut_id(lut, label, True)[0]
        del_ver_idx_list = []
        for arr_id, i in enumerate(lab_arr, 0):
            lab_vert_coord = subj_src[0]['rr'][i]
            lab_vert_mgz_idx = mne.transforms.apply_trans(inv_vox2rastkr_trans, lab_vert_coord)
            orig_idx = mgz_data[int(round(lab_vert_mgz_idx[0])),
                                int(round(lab_vert_mgz_idx[1])),
                                int(round(lab_vert_mgz_idx[2]))]
            if orig_idx != lab_id:
                del_ver_idx_list.append(arr_id)
                del_count += 1
        del_ver_idx = np.asarray(del_ver_idx_list)
        label_list_subject[label] = np.delete(label_list_subject[label], del_ver_idx)
    print '    Deleted', del_count, 'vertice duplicates.\n'

    return label_list_subject


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
        print 'Data_summary is in shape of:', data_summary.shape
        for ii, cluster_ind in enumerate(good_cluster_inds):
            loadingBar(ii + 1, len(good_cluster_inds), task_part='Cluster Idx %i' % (cluster_ind))
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
    plt.figure('T-Statistics', figsize=((8, 8)))
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
    y, binEdges = np.histogram(T_obs_flat,
                               range=(0, T_obs_flat.max()),
                               bins=500)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    if threshold is not None:
        plt.plot([threshold, threshold], (0, y[bincenters >= 0.].max()), color='#CD7600',
                 linestyle=':', linewidth=2)

    legend = """T-Statistics:
      Mean:  %.2f
      Minimum:  %.2f
      Maximum:  %.2f
      Threshold:  %.2f  
      """ % (T_mean, T_min, T_max, threshold)
    plt.ylim(None, y[bincenters >= 0.].max() * 1.1)
    plt.xlabel('T-scores', fontsize=12)
    plt.ylabel('T-values count', fontsize=12)
    plt.title('T statistics distribution of t-test - %s' % str_tail, fontsize=15)
    plt.plot(bincenters, y, label=legend, color='#00868B')
    #    plt.xlim([])
    plt.tight_layout()
    legend = plt.legend(loc='upper right', shadow=True, fontsize='large', frameon=True)

    if save:
        plt.savefig(fname_save)
        plt.close()

    return


def plot_T_obs_3D(T_obs, save, fname_save):
    """ Visualize the Volume Source Estimate as an Nifti1 file """
    fig = plt.figure(facecolor='w', figsize=((8, 8)))
    ax = fig.gca(projection='3d')
    vertc, timez = np.mgrid[0:T_obs.shape[0], 0:T_obs.shape[1]]
    Ts = T_obs
    title = 'T Obs'
    t_obs_stats = ax.plot_surface(vertc, timez, Ts, cmap=cm.hot)  # , **kwargs)
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
