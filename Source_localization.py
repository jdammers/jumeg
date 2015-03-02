# -*- coding: utf-8 -*-
#######################################################
#                                                     #
# small utility function to handle file lists         #
#                                                     #
#######################################################
def get_files_from_list(fin):
    ''' Return files as iterables lists '''
    if isinstance(fin, list):
        fout = fin
    else:
        if isinstance(fin, str):
            fout = list([fin])
        else:
            fout = list(fin)
    return fout
    
subjects_dir = '/home/qdong/freesurfer/subjects/'
MNI_dir = subjects_dir + 'fsaverage/'#the subject as the template brain space
# the source file of the template brain space
fn_inv = MNI_dir + 'bem/fsaverage-ico-4-src.fif' 
subject_id = 'fsaverage'
#######################################################################
# ROIs definition using filtered raw data
# 1) 'make_inverse_operator': evoked and filtered data are inversed 
#    and morphed into the MNI brain space
# 2) 'ROIs_definition': ROIs clustering based on 'trigger' and 'response'
#     events separately
# 3) 'ROIs_Merging': Merge the two kinds of labels: Trigger and Response
# 4) 'ROIs_standardlization': standardize the size of these labels
# 5) 'group_ROI': merger ROIs across subjects
# 6) 'com_ROI': select common ROIs in more than 3 subjects
#######################################################################
def make_inverse_operator(fname_evoked):
    from mne.minimum_norm import (apply_inverse)
    import mne, os
    fnlist = get_files_from_list(fname_evoked)
    # loop across all filenames
    for fn_evoked in fnlist:
        #extract the subject infromation from the file name
        name = os.path.basename(fn_evoked)
        subject = name.split('_')[0]
        
        fn_inv = fn_evoked.split('.fif')[0] + '-inv.fif'
        fn_stc = fn_evoked.split('.fif')[0] 
        fn_morph = fn_evoked.split('.fif')[0] + ',morph'
        subject_path = subjects_dir + subject
        fn_cov = subject_path + '/MEG/%s,bp1-45Hz,empty-cov.fif' %subject
        fn_trans = subject_path + '/MEG/%s-trans.fif' %subject
        fn_src = subject_path + '/bem/%s-ico-4-src.fif' %subject
        fn_bem = subject_path + '/bem/%s-5120-5120-5120-bem-sol.fif' %subject
        snr = 3.0
        lambda2 = 1.0 / snr ** 2
        # Load data
        evoked = mne.read_evokeds(fn_evoked, condition=0, baseline=(None, 0))
        fwd = mne.make_forward_solution(evoked.info, mri=fn_trans, src=fn_src, 
                                    bem=fn_bem,fname=None, meg=True, eeg=False, 
                                    mindist=5.0,n_jobs=2, overwrite=True)
        fwd = mne.convert_forward_solution(fwd, surf_ori=True)
        noise_cov = mne.read_cov(fn_cov)
        noise_cov = mne.cov.regularize(noise_cov, evoked.info,
                                    mag=0.05, grad=0.05, proj=True)
        forward_meg = mne.pick_types_forward(fwd, meg=True, eeg=False)
        inverse_operator = mne.minimum_norm.make_inverse_operator(evoked.info, 
                                    forward_meg, noise_cov, loose=0.2, depth=0.8)
        mne.minimum_norm.write_inverse_operator(fn_inv, inverse_operator)
        stcs = dict()
        # Compute inverse solution
        stcs[subject] = apply_inverse(evoked, inverse_operator, lambda2, "dSPM",
                                pick_ori=None)
        # Morph STC
        subject_id = 'fsaverage'
        #vertices_to = mne.grade_to_vertices(subject_to, grade=5)
        #stcs['morph'] = mne.morph_data(subject, subject_to, stcs[subject], n_jobs=1,
        #                              grade=vertices_to)
        stcs[subject].save(fn_stc)
        stcs['morph'] = mne.morph_data(subject, subject_id, stcs[subject], 4, smooth=4)
        stcs['morph'].save(fn_morph)
        fig_out = fn_morph + '.png'
        plot_evoked_stc(subject,stcs, fig_out)
    
        
#plot the comparasions of STCs before and after morphing    
def plot_evoked_stc(subject, stcs,fig_out):
    import matplotlib.pyplot as plt
    import numpy as np
    names = [subject, 'morph']
    plt.close('all')
    plt.figure(figsize=(8, 6))
    for ii in range(len(stcs)):
        name = names[ii]
        stc = stcs[name]
        plt.subplot(len(stcs), 1, ii + 1)
        src_pow = np.sum(stc.data ** 2, axis=1)
        plt.plot(1e3 * stc.times, stc.data[src_pow > np.percentile(src_pow, 90)].T)
        plt.ylabel('%s\ndSPM value' % str.upper(name))
    plt.xlabel('time (ms)')
    plt.show()
    plt.savefig(fig_out, dpi=100)
    plt.close()
    


def ROIs_definition(fname_stc, tri='STI 014'):
    import mne, os
    import numpy as np
    fnlist = get_files_from_list(fname_stc)
    # loop across all filenames
    for fn_stc in fnlist:
        #extract the subject infromation from the file name
        name = os.path.basename(fn_stc)
        subject = name.split('_')[0]
        
        subject_path = subjects_dir + subject
        src_inv = mne.read_source_spaces(fn_inv, add_geom=True) 
        if tri == 'STI 014':
            #stc_thr = 85 
            stc_thr = 95
            tri = 'tri'
        elif tri == 'STI 013':
            stc_thr = 95
            tri = 'res'
        stc_morph = mne.read_source_estimate(fn_stc, subject=subject_id)
        src_pow = np.sum(stc_morph.data ** 2, axis=1)
        stc_morph.data[src_pow < np.percentile(src_pow, stc_thr)] = 0. 
        func_labels_lh, func_labels_rh = mne.stc_to_label(stc_morph, src=src_inv, smooth=5,
                                    subjects_dir=subjects_dir, connected=True)    
        # Left hemisphere definition                                                                
        i = 0
        while i < len(func_labels_lh):
            func_label = func_labels_lh[i]
            func_label.save(subject_path+'/func_labels/%s' %(tri)+str(i))
            i = i + 1
        # right hemisphere definition      
        j = 0
        while j < len(func_labels_rh):
            func_label = func_labels_rh[j]
            func_label.save(subject_path+'/func_labels/%s' %(tri)+str(j))
            j = j + 1
                
###################################################################################
#  Merge overlaped ROIs
###################################################################################
def ROIs_Merging(subject):
    import os,mne
    import numpy as np
    subject_path = subjects_dir + subject
    list_dirs = os.walk(subject_path + '/func_labels/')
    tri_list = ['']
    res_list = ['']
    for root, dirs, files in list_dirs: 
        for f in files:
            label_fname = os.path.join(root, f) 
            if f[0:3]=='tri':
                tri_list.append(label_fname)
            elif f[0:3]=='res': 
                res_list.append(label_fname)
    
    tri_list=tri_list[1:]
    res_list=res_list[1:]
    
    mer_path = subject_path+'/func_labels/merged/'
    isExists=os.path.exists(mer_path)
    if not isExists:
        os.makedirs(mer_path) 
        
    com_list=['']        
    for fn_tri in tri_list:
        tri_label = mne.read_label(fn_tri) 
        com_label = tri_label.copy()
        for fn_res in res_list:
            res_label = mne.read_label(fn_res) 
            if tri_label.hemi != res_label.hemi:
                continue
            if len(np.intersect1d(tri_label.vertices, res_label.vertices)) > 0:
                com_label = tri_label + res_label
                tri_label.name += ',%s' %res_label.name
                com_list.append(fn_res)#Keep the overlapped ROIs related with res 
        mne.write_label(mer_path + '%s' %tri_label.name, com_label)

    # save the independent res ROIs
    com_list=com_list[1:]
    ind_list = list(set(res_list)-set(com_list))
    for fn_res in ind_list:
        res_label = mne.read_label(fn_res) 
        res_label.save(mer_path + '%s' %res_label.name)
####################################################################
# For the special subject, to standardize the size of ROIs
####################################################################
def ROIs_standardlization(fname_stc, size=8.0):
    import mne,os
    import numpy as np
    fnlist = get_files_from_list(fname_stc)
    # loop across all filenames
    for fn_stc in fnlist:
        stc_morph = mne.read_source_estimate(fn_stc, subject=subject_id)
        
        #extract the subject infromation from the file name
        name = os.path.basename(fn_stc)
        subject = name.split('_')[0]
        subject_path = subjects_dir + subject
        sta_path = MNI_dir+'func_labels/standard/'
        list_dirs = os.walk(subject_path + '/func_labels/merged/') 
        for root, dirs, files in list_dirs: 
            for f in files:
                label_fname = os.path.join(root, f) 
                label = mne.read_label(label_fname)
                stc_label = stc_morph.in_label(label)
                src_pow = np.sum(stc_label.data ** 2, axis=1)
                if label.hemi == 'lh':
                    seed_vertno = stc_label.vertno[0][np.argmax(src_pow)]#Get the max MNE value within each ROI
                    func_label = mne.grow_labels(subject_id, seed_vertno, extents=size, 
                                                hemis=0, subjects_dir=subjects_dir, 
                                                n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(sta_path+'%s_%s' %(subject,f))
                elif label.hemi == 'rh':
                    seed_vertno = stc_label.vertno[1][np.argmax(src_pow)]
                    func_label = mne.grow_labels(subject_id, seed_vertno, extents=size, 
                                                hemis=1, subjects_dir=subjects_dir, 
                                                n_jobs=1)
                    func_label = func_label[0]
                    func_label.save(sta_path+'%s_%s' %(subject,f))
                    
##################################################################################
# Evaluate the group ROIs across subjects:
#  1) merge the overlapped labels across subjects
#  2) select the ROIs coming out in at least am_sub subjects
#################################################################################
def cluster_ROI(mer_path, label_list):
    import mne, os
    import numpy as np
    class_list = []
    class_list.append(label_list[0]) 
    for test_fn in label_list[1:]:
        test_label = mne.read_label(test_fn)
        i = 0
        belong = False
        while (i < len(class_list)) and (belong == False):
            class_label = mne.read_label(class_list[i])
            label_name = class_label.name
            if test_label.hemi != class_label.hemi:
                i = i + 1
                continue
            if len(np.intersect1d(test_label.vertices, class_label.vertices)) > 0:
                com_label = test_label + class_label
                pre_test = test_label.name.split('_')[0]
                pre_class = class_label.name.split('_')[0]
                if pre_test != pre_class:
                    pre_class += ',%s' %pre_test
                    pre_class = list(set(pre_class.split(',')))
                    new_pre = ''
                    for pre in pre_class[:-1]:
                        new_pre += '%s,' %pre
                    new_pre += pre_class[-1]
                    label_name = '%s_' %new_pre + class_label.name.split('_')[1]
                if os.path.dirname(class_list[i]) == mer_path:
                    os.remove(class_list[i])
                if os.path.dirname(test_fn) == mer_path:
                    os.remove(test_fn)
                mne.write_label(mer_path + '/%s.label' %label_name, com_label)
                print label_name
                class_list[i] = mer_path + '/%s.label' %label_name 
                belong = True
            i = i + 1
        if belong == False:
            class_list.append(test_fn)
    return len(class_list)
    
def group_ROI():
    import os, shutil
    import numpy as np
    subject_path = subjects_dir + 'fsaverage'
    
    #Merge the individual subject's ROIs
    list_dirs = os.walk(subject_path + '/func_labels/standard/')
    label_list = ['']
    for root, dirs, files in list_dirs: 
        for f in files:
            label_fname = os.path.join(root, f) 
            label_list.append(label_fname)
    label_list=label_list[1:]
    mer_path = subject_path+'/func_labels/merged'
    isExists=os.path.exists(mer_path)
    if isExists:
        shutil.rmtree(mer_path)
    os.makedirs(mer_path) 
    cluster_ROI(mer_path, label_list)
    
    #Merge the overlabpped class
    list_dirs = os.walk(mer_path)
    label_list = ['']
    for root, dirs, files in list_dirs: 
        for f in files:
            label_fname = os.path.join(root, f) 
            label_list.append(label_fname)
    label_list=label_list[1:]
    len_class = 0
    while len_class < len(label_list):
        len_class = cluster_ROI(mer_path, label_list)
        list_dirs = os.walk(mer_path)
        label_list = ['']
        for root, dirs, files in list_dirs: 
            for f in files:
                label_fname = os.path.join(root, f) 
                label_list.append(label_fname)
        label_list = label_list[1:]
        print len_class, len(label_list)
        
def com_ROI(am_sub):
#Select the ROIs more than am_sub subjects        
    import shutil, os
    subject_path = subjects_dir + 'fsaverage'
    com_path = subject_path+'/func_labels/common/'
    mer_path = subject_path+'/func_labels/merged/'
    isExists=os.path.exists(com_path)
    if isExists:
        shutil.rmtree(com_path)
    os.makedirs(com_path) 
    list_dirs = os.walk(mer_path)
    label_list = ['']
    for root, dirs, files in list_dirs: 
        for f in files:
            label_fname = os.path.join(root, f) 
            label_list.append(label_fname)
    label_list=label_list[1:]    
    for fn_label in label_list:
        fn_name = os.path.basename(fn_label)
        subjects = (fn_name.split('_')[0]).split(',')
        if len(subjects) >= am_sub:          
            shutil.copy(fn_label, com_path)
            
            
    
    
