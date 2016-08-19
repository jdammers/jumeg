
import os
import numpy as np
from scipy import stats as stats
import glob

import mne
from mne import (spatial_tris_connectivity,
                 grade_to_tris)
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc, spatio_temporal_cluster_test)
from jumeg.jumeg_utils import set_directory
from jumeg.jumeg_preprocessing import get_files_from_list


def apply_inverse_ave(fnevo, subjects_dir):
    ''' Make individual inverse operator.

        Parameter
        ---------
        fnevo: string or list
            The evoked file with ECG, EOG and environmental noise free.
        subjects_dir: The total bath of all the subjects.

    '''
    from mne import make_forward_solution
    from mne.minimum_norm import make_inverse_operator, write_inverse_operator
    fnlist = get_files_from_list(fnevo)

    # loop across all filenames
    for fname in fnlist:
        fn_path = os.path.split(fname)[0]
        name = os.path.basename(fname)
        subject = name.split('_')[0]
        fn_inv = fn_path + '/%s_fibp1-45,ave-inv.fif' % subject
        subject_path = subjects_dir + '/%s' % subject

        fn_trans = fn_path + '/%s-trans.fif' % subject
        fn_cov = fn_path + '/%s_empty,fibp1-45-cov.fif' % subject
        fn_src = subject_path + '/bem/%s-oct-6-src.fif' % subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' % subject
        [evoked] = mne.read_evokeds(fname)
        evoked.pick_types(meg=True, ref_meg=False)
        noise_cov = mne.read_cov(fn_cov)
        # noise_cov = mne.cov.regularize(noise_cov, evoked.info,
        #                               mag=0.05, grad=0.05, proj=True)
        fwd = make_forward_solution(evoked.info, fn_trans, fn_src, fn_bem)
        fwd['surf_ori'] = True
        inv = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2,
                                    depth=0.8, limit_depth_chs=False)
        write_inverse_operator(fn_inv, inv)


def apply_STC_ave(fnevo, method='dSPM', snr=3.0):
    ''' Inverse evoked data into the source space.
        Parameter
        ---------
        fnevo: string or list
            The evoked file with ECG, EOG and environmental noise free.
        method:string
            Inverse method, 'MNE' or 'dSPM'
        snr: float
            Signal to noise ratio for inverse solution.
    '''
    #Get the default subjects_dir
    from mne.minimum_norm import apply_inverse, read_inverse_operator
    fnlist = get_files_from_list(fnevo)
    # loop across all filenames
    for fname in fnlist:
        name = os.path.basename(fname)
        fn_path = os.path.split(fname)[0]
        fn_stc = fname[:fname.rfind('-ave.fif')]
        # fn_inv = fname[:fname.rfind('-ave.fif')] + ',ave-inv.fif'
        subject = name.split('_')[0]
        fn_inv = fn_path + '/%s_fibp1-45,ave-inv.fif' % subject
        snr = snr
        lambda2 = 1.0 / snr ** 2
        # noise_cov = mne.read_cov(fn_cov)
        [evoked] = mne.read_evokeds(fname)
        evoked.pick_types(meg=True, ref_meg=False)
        inv = read_inverse_operator(fn_inv)
        stc = apply_inverse(evoked, inv, lambda2, method,
                            pick_ori='normal')
        stc.save(fn_stc)


def morph_STC(fn_stc, grade, subjects_dir, template='fsaverage', event='LLst',
              baseline=True, btmin=-0.3, btmax=0.):
    '''
        Morph individual STC into the common brain space.

        Parameter
        ------------------------------------
        fn_stc: string or list
            The path of the individual STC.
        subjects_dir: The total bath of all the subjects.
        template: string
            The subject name as the common brain.
        event: string
            The name of event
        baseline: bool
            If true, prestimulus segment from 'btmin' to 'btmax' will be saved,
            If false, no baseline segment is saved.
        btmin, btmax: float
            If 'baseline' is True, baseline is croped using this period.

    '''
    from mne import read_source_estimate, morph_data
    fnlist = get_files_from_list(fn_stc)
    for fname in fnlist:
        name = os.path.basename(fname)
        subject = name.split('_')[0]
        stc_name = name[:name.rfind('-lh.stc')]
        min_dir = subjects_dir + '/%s' % template
        # this path used for ROI definition
        stc_path = min_dir + '/dSPM_ROIs/%s' % (subject)
        # fn_cov = meg_path + '/%s_empty,fibp1-45,nr-cov.fif' % subject
        set_directory(stc_path)
        # Morph STC
        stc = read_source_estimate(fname)
        stc_morph = morph_data(subject, template, stc, grade=grade)
        stc_morph.save(stc_path + '/%s' % (stc_name), ftype='stc')
        if baseline is True:
            stc_base = stc_morph.crop(btmin, btmax)
            stc_base.save(stc_path + '/%s_%s_baseline' % (subject, event[:2]),
                          ftype='stc')


def Ara_contr(evt_list, tmin, tmax, conf_type, out_path, n_subjects=14,
              template='fsaverage', subjects_dir=None):

    ''' Prepare arrays for the contrasts of conflicts perception
        and conflicts response.

        Parameter
        ---------
        evt_list: list
            The events list.
        tmin, tmax: float (s)
            The time period of data.
        conf_type: string
            The type of contrasts,'conf_per' or 'conf_res'
        out_path: string
            The path to store aranged arrays.
        n_subjects: int
            The amount subjects.
        subjects_dir: The total bath of all the subjects.
    '''
    con_stcs = []
    for evt in evt_list[:2]:
        fn_stc_list1 = glob.glob(subjects_dir + '/fsaverage/dSPM_ROIs/*[0-9]/*fibp1-45,evtW_%s_bc-lh.st-lh.stc' % evt)
        for fn_stc1 in fn_stc_list1[:n_subjects]:
            stc1 = mne.read_source_estimate(fn_stc1, subject=template)
            stc1.crop(tmin, tmax)
            con_stcs.append(stc1.data)
    cons = np.array(con_stcs).transpose(1, 2, 0)

    # tmin = stc1.tmin
    tstep = stc1.tstep
    fsave_vertices = stc1.vertices
    del stc1

    incon_stcs = []
    for evt in evt_list[2:]:
        fn_stc_list2 = glob.glob(subjects_dir + '/fsaverage/dSPM_ROIs/*[0-9]/*fibp1-45,evtW_%s_bc-lh.st-lh.stc' % evt)
        for fn_stc2 in fn_stc_list2[:n_subjects]:
            stc2 = mne.read_source_estimate(fn_stc2, subject=template)
            stc2.crop(tmin, tmax)
            incon_stcs.append(stc2.data)
    incons = np.array(incon_stcs).transpose(1, 2, 0)
    del stc2
    X = [cons[:, :, :], incons[:, :, :]]
    # import pdb
    # pdb.set_trace()
    # save data matrix
    X = np.array(X).transpose(1, 2, 3, 0)
    X = np.abs(X)  # only magnitude
    np.savez(out_path + '%s.npz' % conf_type, X=X, tstep=tstep,
             fsave_vertices=fsave_vertices)
    return tstep, fsave_vertices, X


def Ara_contr_base(evt_list, tmin, tmax, conf_type, out_path, n_subjects=14,
                   template='fsaverage', subjects_dir=None):

    ''' Prepare arrays for the data contrasts of prestimulus and post-stimulus.

        Parameter
        ---------
        evt_list: list
            The events list.
        tmin, tmax: float (s)
            The time period of data.
        conf_type: string
            The type of contrasts,'sti' or 'res'
        out_path: string
            The path to store aranged arrays.
        n_subjects: int
            The amount subjects.
        subjects_dir: The total bath of all the subjects.
    '''
    for evt in evt_list:
        stcs = []
        bs_stcs = []
        fn_stc_list1 = glob.glob(subjects_dir + '/fsaverage/dSPM_ROIs/*[0-9]/*fibp1-45,evtW_%s_bc-lh.st-lh.stc' % evt)
        for fn_stc1 in fn_stc_list1[:n_subjects]:
            # fn_stc2 = fn_stc1.split(evt)[0] + evt[:2] +  fn_stc1.split(evt)[1]
            name = os.path.basename(fn_stc1)
            fn_path = os.path.split(fn_stc1)[0]
            subject = name.split('_')[0]
            fn_stc2 = fn_path + '/%s_%s_baseline-lh.stc' % (subject, evt[:2])
            stc1 = mne.read_source_estimate(fn_stc1, subject=template)
            stc1.crop(tmin, tmax)
            stcs.append(stc1.data)
            stc2 = mne.read_source_estimate(fn_stc2, subject=template)
            bs_stcs.append(stc2.data)
        stcs = np.array(stcs).transpose(1, 2, 0)
        bs_stcs = np.array(bs_stcs).transpose(1, 2, 0)
        # tmin = stc1.tmin
        tstep = stc1.tstep
        fsave_vertices = stc1.vertices
        if stcs.shape[1] > bs_stcs.shape[1]:
            X = [stcs[:, :bs_stcs.shape[1], :], bs_stcs[:, :, :]]
        else:
            X = [stcs[:, :, :], bs_stcs[:, :stcs.shape[1], :]]
        del stcs, bs_stcs
        # save data matrix
        X = np.array(X).transpose(1, 2, 3, 0)
        X = np.abs(X)  # only magnitude
        np.savez(out_path + '%s_%s.npz' % (conf_type, evt), X=X, tstep=tstep,
                 fsave_vertices=fsave_vertices)
        del X
    # return tstep, fsave_vertices, X


def mv_ave(X, window, overlap, freqs=678.17):
    '''
      Resample the data on the time dimension.

      Parameter
      ---------
      X: array
        The shape of X should be (Vertices, timepoints, subjects, cases)
      window: int
        The window size for moving average
      overlap: int
        The overlap for the moving average
      freqs: float or int
        The sampling rate
    '''
    mv_wind = window * 0.001
    step_wind = (window - overlap) * 0.001
    st_point = 0
    win_id = int(mv_wind * freqs)
    ste_id = int(step_wind * freqs)
    N_X = []
    while win_id < X.shape[1]:
        N_X.append(X[:, st_point:win_id, :, :].mean(axis=1))
        win_id = win_id + ste_id
        st_point = st_point + ste_id
    N_X = np.array(N_X).transpose(1, 0, 2, 3)
    return N_X


def stat_clus(X, tstep, n_per=8192, p_threshold=0.01, p=0.05, fn_clu_out=None):
    '''
      Calculate significant clusters using 1sample ttest.

      Parameter
      ---------
      X: array
        The shape of X should be (Vertices, timepoints, subjects)
      tstep: float
        The interval between timepoints.
      n_per: int
        The permutation for ttest.
      p_threshold: float
        The significant p_values.
      p: float
        The corrected p_values for comparisons.
      fn_clu_out: string
        The fnname for saving clusters.
    '''

    print('Computing connectivity.')
    connectivity = spatial_tris_connectivity(grade_to_tris(5))

    #    Note that X needs to be a multi-dimensional array of shape
    #    samples (subjects) x time x space, so we permute dimensions
    X = np.transpose(X, [2, 1, 0])
    n_subjects = X.shape[0]
    fsave_vertices = [np.arange(X.shape[-1]/2), np.arange(X.shape[-1]/2)]

    #    Now let's actually do the clustering. This can take a long time...
    #    Here we set the threshold quite high to reduce computation.
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    print('Clustering.')
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(X, connectivity=connectivity,
                                           n_jobs=1, threshold=t_threshold,
                                           n_permutations=n_per)

    #    Now select the clusters that are sig. at p < 0.05 (note that this value
    #    is multiple-comparisons corrected).
    good_cluster_inds = np.where(cluster_p_values < p)[0]
    print 'the amount of significant clusters are: %d' %good_cluster_inds.shape

    # Save the clusters as stc file
    np.savez(fn_clu_out, clu=clu, tstep=tstep, fsave_vertices=fsave_vertices)
    assert good_cluster_inds.shape != 0, ('Current p_threshold is %f %p_thr,\
                                 maybe you need to reset a lower p_threshold')



def per2test(X1, X2, p_thr, p, tstep, n_per=8192, fn_clu_out=None):
    '''
      Calculate significant clusters using 2 sample ftest.

      Parameter
      ---------
      X1, X2: array
        The shape of X should be (Vertices, timepoints, subjects)
      tstep: float
        The interval between timepoints.
      n_per: int
        The permutation for ttest.
      p_thr: float
        The significant p_values.
      p: float
        The corrected p_values for comparisons.
      fn_clu_out: string
        The fnname for saving clusters.
    '''
    #    Note that X needs to be a multi-dimensional array of shape
    #    samples (subjects) x time x space, so we permute dimensions
    n_subjects1 = X1.shape[2]
    n_subjects2 = X2.shape[2]
    fsave_vertices = [np.arange(X1.shape[0]/2), np.arange(X1.shape[0]/2)]
    X1 = np.transpose(X1, [2, 1, 0])
    X2 = np.transpose(X2, [2, 1, 0])
    X = [X1, X2]

    #    Now let's actually do the clustering. This can take a long time...
    #    Here we set the threshold quite high to reduce computation.
    f_threshold = stats.distributions.f.ppf(1. - p_thr / 2., n_subjects1 - 1,
                                            n_subjects2 - 1)
    # t_threshold = stats.distributions.t.ppf(1. - p_thr / 2., n_subjects1 - 1,
    #                                         n_subjects2 - 1)

    print('Clustering...')
    connectivity = spatial_tris_connectivity(grade_to_tris(5))
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_test(X, n_permutations=n_per, #step_down_p=0.001,
                                     connectivity=connectivity, n_jobs=1,
                                     # threshold=t_threshold, stat_fun=stats.ttest_ind)
                                     threshold=f_threshold)

    #    Now select the clusters that are sig. at p < 0.05 (note that this value
    #    is multiple-comparisons corrected).
    good_cluster_inds = np.where(cluster_p_values < p)[0]
    print 'the amount of significant clusters are: %d' % good_cluster_inds.shape

    # Save the clusters as stc file
    np.savez(fn_clu_out, clu=clu, tstep=tstep, fsave_vertices=fsave_vertices)
    assert good_cluster_inds.shape != 0, ('Current p_threshold is %f %p_thr,\
                                 maybe you need to reset a lower p_threshold')


def clu2STC(fn_cluster, p_thre=0.05, tstep=None):
    '''
        Generate STCs from significant clusters

        Parameters
        -------
        fn_cluster: string
            The filename of significant clusters.
        p_thre: float
            The corrected p_values.
        tstep: float
            The interval between timepoints.
    '''
    fn_stc_out = fn_cluster[:fn_cluster.rfind('.npz')] + ',temp_%.3f' % (p_thre)
    npz = np.load(fn_cluster)
    if tstep is None:
        tstep = npz['tstep'].flatten()[0]
    clu = npz['clu']
    fsave_vertices = list(npz['fsave_vertices'])
    stc_all_cluster_vis = summarize_clusters_stc(clu, p_thre, tstep=tstep,
                                                 vertices=fsave_vertices,
                                                 subject='fsaverage')

    if fn_stc_out is None:
        return stc_all_cluster_vis
    stc_all_cluster_vis.save(fn_stc_out)
