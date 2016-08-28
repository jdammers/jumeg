# -*- coding: UTF-8 -*-

"""
=================================================================
Causality analysis
=================================================================
Before running this script, the absolute path of each ROI label
needs to be provided in the form of a text file,
'func_list.txt'/'func_label_list.txt'.
"""

# Authors: Dong Qunxi <dongqunxi@gmail.com>
#         Jürgen Dammers <j.dammers@fz-juelich.de>
# License: BSD (3-clause)

import os
import glob
from apply_causality_whole import apply_inverse_oper, apply_STC_epo
from apply_causality_whole import (cal_labelts, normalize_data, sig_thresh,
                                   group_causality, model_order,
                                   model_estimation, causal_analysis, diff_mat)

print(__doc__)

# Set parameters
subjects_dir = os.environ['SUBJECTS_DIR']
cau_path = subjects_dir+'/fsaverage/stcs'
st_list = ['LLst', 'RRst', 'RLst',  'LRst']
ROIs = ('ROI0', 'ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 'ROI6', 'ROI7',
        'ROI8', 'ROI9', 'ROI10', 'ROI11',
        'ROI12', 'ROI13', 'ROI14', 'ROI15')

sfreq = 678.17 # Sampling rate
morder = 40 # Fixed model order
per = 99.99 # Percentile for causal surrogates
ifre = int(sfreq / (2 * morder))
freqs = [(ifre, 2*ifre), (2*ifre, 3*ifre), (3*ifre, 4*ifre), (4*ifre, 5*ifre)]

# Cluster operation
do_apply_invers_oper = False # Making inverse operator
do_apply_STC_epo = False # Making STCs
do_extract_rSTCs = True
do_norm = False
do_morder = False
do_moesti = False
do_cau = False
do_sig_thr = False
do_group = False
do_group_plot = False
do_diff = False

# Make inverse operator for each subject
if do_apply_invers_oper:
    print '>>> Calculate Inverse Operator ....'
    fn_epo_list = glob.glob(subjects_dir+'/*[0-9]/MEG/*ocarta,evtW_LLst_bc-epo.fif')
    apply_inverse_oper(fn_epo_list, subjects_dir=subjects_dir)
    print '>>> FINISHED with STC generation.'
    print ''


# Making STCs
if do_apply_STC_epo:
    print '>>> Calculate morphed STCs ....'
    for evt in st_list[2:]:
        fn_epo = glob.glob(subjects_dir+'/*[0-9]/MEG/*ocarta,evtW_%s_bc-epo.fif' %evt)
        apply_STC_epo(fn_epo, event=evt, subjects_dir=subjects_dir)
    print '>>> FINISHED with morphed STC generation.'
    print ''


# Extract representative STCs from ROIs
if do_extract_rSTCs:
    print '>>> Calculate representative STCs ....'
    func_list_file = subjects_dir+'/fsaverage/conf_stc/STC_ROI/func_list.txt'
    for evt_st in st_list:
        # Calculate the representative STCs(rSTCs) for each ROI.
        stcs_path = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/%s/' % evt_st)
        cal_labelts(stcs_path, func_list_file, condition=evt_st, min_subject='fsaverage')
    print '>>> FINISHED with rSTC generation.'
    print ''


# Normalization STCs
if do_norm:
    print '>>> Calculate normalized rSTCs ....'
    ts_path = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/*_labels_ts.npy')
    normalize_data(ts_path)
    print '>>> FINISHED with normalized rSTC generation.'
    print ''


# 1) Model construction and estimation
# 2) Causality analysis
if do_morder:
    print '>>> Calculate the optimized Model order....'
    fn_norm = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/*_labels_ts,norm.npy')
    # Get optimized model order using BIC
    model_order(fn_norm, p_max=100)
    print '>>> FINISHED with optimized model order generation.'
    print ''

if do_moesti:
    print '>>> Envaluate the cosistency, whiteness, and stable features of the Model....'
    fn_monorm = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/*_labels_ts,norm.npy')
    model_estimation(fn_monorm, morder=40)
    print '>>> FINISHED with the results of statistical tests generation.'
    print ''

if do_cau:
    print '>>> Make the causality analysis....'
    fn_monorm = glob.glob(subjects_dir+'/fsaverage/stcs/*[0-9]/*_labels_ts,norm.npy')
    # fn_monorm = glob.glob(cau_path + '/*[0-9]/*_labels_ts,norm.npy')
    causal_analysis(fn_monorm[1:16], repeats=1000, morder=40, per=per, method='GPDC')
    print '>>> FINISHED with causal matrices and surr-causal matrices generation.'
    print ''

if do_sig_thr:
    print '>>> Calculate the significance of the causality matrices....'
    fn_cau = glob.glob(cau_path + '/*[0-9]/sig_cau_40/*,cau.npy')
    sig_thresh(cau_list=fn_cau, per=per)
    print '>>> FINISHED with significant causal matrices generation.'
    print ''

if do_group:
    print '>>> Generate the group causal matrices....'
    for evt_st in st_list:
        out_path = cau_path + '/causality'
        fnsig_list = glob.glob(cau_path + '/*[0-9]/sig_cau_40/%s_sig_con_band.npy' %evt_st)
        group_causality(fnsig_list, evt_st, ROI_labels=ROIs, submount=10, out_path=out_path)
    print '>>> FINISHED with group causal matrices generation.'
    print ''

if do_diff:
    # Difference between incongruent and congruent tasks
    for ifreq in freqs:
        fmin = ifreq[0]
        fmax = ifreq[1]
        mat_dir = cau_path + '/causality'
        diff_mat(fmin=fmin, fmax=fmax, mat_dir=mat_dir, ROI_labels=ROIs)
