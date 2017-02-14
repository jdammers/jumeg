# -*- coding: UTF-8 -*-

"""
This script is applied to estimate the significant clusters related with events
by comparing data segments of prestimulus and post-stimulus.
"""
# Authors: Dong Qunxi <dongqunxi@gmail.com>
#         Jürgen Dammers <j.dammers@fz-juelich.de>
# License: BSD (3-clause)

import os, glob, sys
from stat_cluster import Ara_contr_base, apply_inverse_ave, clu2STC,\
                         apply_STC_ave, morph_STC, mv_ave, per2test, stat_clus
print(__doc__)

###############################################################################
# Set parameters
# ------------------------------------------------

# Set the path for storing STCs of conflicts processing
subjects_dir = os.environ['SUBJECTS_DIR']
stcs_path = subjects_dir + '/fsaverage/conf_stc/'

n_subjects = 14 # The amount of subjects
st_list = ['LLst', 'RRst', 'RLst',  'LRst'] # stimulus events
res_list = ['LLrt', 'RRrt', 'LRrt', 'RLrt'] # response events
st_max, st_min = 0.3, 0. # time period of stimulus
res_max, res_min = 0.1, -0.2 # time period of response

# parameter of Morphing
grade = 5
method = 'dSPM'
snr = 2
template = 'fsaverage' # The common brain space

# Parameters of Moving average
mv_window = 10 # miliseconds
overlap = 0 # miliseconds
nfreqs = 678.17

# The parameters for clusterring test
permutation = 16384
p_th = 0.0001 # spatial p-value
p_v = 0.005 # comparisons corrected p-value

# Cluster operation
do_apply_invers_ave = False # Making inverse operator
do_apply_STC_ave = True # Inversing conduction
do_morph_STC_ave = True # STC morphing conduction
do_calc_matrix = True # Form the group matrix or load directly
do_mv_ave = False #The moving average conduction
do_ftest = False # 2sample f test conduction
do_ttest = True # 1sample t test
do_clu2STC = False
# Set the option for stimulus or response
conf_per = False
conf_res = True
#conf_per = sys.argv[1]
#conf_res = sys.argv[2]

#stimulus
if conf_per == True:
#if conf_per == 'True':
    evt_list = st_list
    tmin, tmax = st_min, st_max
    conf_type = 'sti'
    baseline = True

#response
elif conf_res == True:
#elif conf_res == 'True':
    evt_list = res_list
    tmin, tmax = res_min, res_max
    conf_type = 'res'
    baseline = False


print '>>> Calculate under the condition %s ....' %conf_type
###############################################################################
# Inverse evoked data for each condition
# ------------------------------------------------
if do_apply_invers_ave:
    print '>>> Calculate inverse solution ....'
    fn_evt_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_LLst_bc-ave.fif')
    apply_inverse_ave(fn_evt_list, subjects_dir)
    print '>>> FINISHED with inverse solution.'
    print ''



###############################################################################
# Inverse evoked data for each condition
# ------------------------------------------------
if do_apply_STC_ave:
    print '>>> Calculate STC ....'
    for evt in evt_list:
        fn_evt_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_%s_bc-ave.fif' %evt)
        apply_STC_ave(fn_evt_list, method=method, snr=snr)
    print '>>> FINISHED with STC generation.'
    print ''

###############################################################################
# Morph STC data for each condition
# ------------------------------------------------
if do_morph_STC_ave:
    print '>>> Calculate morphed STC ....'
    for evt in evt_list:
        fn_stc_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*fibp1-45,evtW_%s_bc-lh.stc' %evt)
        morph_STC(fn_stc_list, grade, subjects_dir, template, event=evt, baseline=baseline)
    print '>>> FINISHED with morphed STC generation.'
    print ''

###############################################################################
# conflicts contrasts
# -----------------
if do_calc_matrix:
    print '>>> Calculate Matrix for contrasts ....'
    Ara_contr_base(evt_list, tmin, tmax, conf_type, stcs_path, n_subjects=n_subjects,
                   template='fsaverage', subjects_dir=subjects_dir)
    print '>>> FINISHED with a group matrix generation.'
    print ''

###############################################################################
# Clustering using 1sample t-test
# -----------------
if do_ttest:
    ''' This comparison is suitable for the samples from the same entirety
    '''
    print '>>> ttest for clustering ....'
    for evt in evt_list:
        evt = '%s_%s' %(conf_type, evt)
        fnmat = stcs_path + evt + '.npz'
        permutation1 = permutation * 2
        #conf_mark = 'ttest_' + conf_type
        print '>>> load Matrix for contrasts ....'
        import numpy as np
        npz = np.load(fnmat)
        tstep = npz['tstep'].flatten()[0]
        X = npz['X']
        print '>>> FINISHED with the group matrix loaded.'
        print ''
        if do_mv_ave:
            print '>>> Moving averages with window length %dms ....' %(mv_window)
            evt = 'mv%d_' %mv_window + evt
            X = mv_ave(X, mv_window, overlap, freqs=nfreqs)
            print '>>> FINISHED with the smothed group matrix generation.'
            print ''
        X1 = X[:, :, :n_subjects, 0]
        X2 = X[:, :, :n_subjects, 1]
        fn_clu_out = stcs_path + 'Ttestpermu%d_pthr%.4f_%s.npz' %(permutation1, p_th, evt)
        Y = X1 - X2  # make paired contrast
        stat_clus(Y, tstep, n_per=permutation1, p_threshold=p_th, p=p_v,
                fn_clu_out=fn_clu_out)
        print Y.shape
        del Y
        clu2STC(fn_clu_out, p_thre=p_v, tstep=0.01)
    print '>>> FINISHED with the clusters generation.'
    print ''

###############################################################################
# Clustering using 2sample f-test
# -----------------
if do_ftest:
    ''' This comparison is suitable for the samples from different entireties
    '''
    print '>>> 2smpletest for clustering ....'
    for evt in evt_list:
        evt = '%s_%s' %(conf_type, evt)
        print '>>> load Matrix for contrasts ....'
        import numpy as np
        fnmat = stcs_path + evt + '.npz'
        npz = np.load(fnmat)
        tstep = npz['tstep'].flatten()[0]
        X = npz['X']
        print '>>> FINISHED with the group matrix loaded.'
        print ''
        if do_mv_ave:
            print '>>> Moving averages with window length %dms ....' % (mv_window)
            evt = 'mv%d_' %mv_window + evt
            X = mv_ave(X, mv_window, overlap, freqs=nfreqs)
            print '>>> FINISHED with the smothed group matrix generation.'
            print ''
        X1 = X[:, :, :n_subjects, 0]
        X2 = X[:, :, :n_subjects, 1]
        fn_clu_out = stcs_path + 'FTpermu%d_pthr%.4f_%s.npz' % (permutation, p_th, evt)
        per2test(X1, X2, p_thr=p_th, p=p_v, tstep=tstep, n_per=permutation,
                fn_clu_out=fn_clu_out)
        print X1.shape
        del X1, X2, X
        clu2STC(fn_clu_out, p_thre=p_v, tstep=0.01)
    print '>>> FINISHED with the clusters generation.'
    print ''


if do_clu2STC:
    print '>>> Transfer cluster to STC ....'
    for evt in evt_list:
        fn_cluster = stcs_path + 'Ttestpermu%d_pthr%.4f_%s.npz' %(permutation, p_th, evt)
        clu2STC(fn_cluster, p_thre=p_v, tstep=0.01)
