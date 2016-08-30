'''
Once we get ROIs, we need to make a file named 'func_label_list.txt'
with absolute paths manually, which includs the path of each ROI.
This file as the indices of ROIs for causality analysis.
'''

import os
import mne
import numpy as np
from jumeg.jumeg_utils import reset_directory


def _merge_rois(mer_path, label_list):
    """
    Function to merge a list of given labels.

    Parameters
    ----------
    mer_path: str
        The directory for storing merged ROIs.
    label_list: list
        Labels to be merged
    """
    class_list = []
    class_list.append(label_list[0])
    for test_fn in label_list[1:]:
        test_label = mne.read_label(test_fn)
        i = 0
        belong = False
        while (i < len(class_list)) and (belong is False):
            class_label = mne.read_label(class_list[i])
            label_name = class_label.name
            if test_label.hemi != class_label.hemi:
                i = i + 1
                continue
            overlapped = len(np.intersect1d(test_label.vertices,
                                            class_label.vertices))
            if overlapped > 0:
                com_label = test_label + class_label
                pre_test = test_label.name.split('_')[0]
                pre_class = class_label.name.split('_')[0]
                # label_name = pre_class + '_%s-%s' %(pre_test,class_label.name.split('-')[-1])
                if pre_test != pre_class:
                    pre_class += ',%s' % pre_test
                    pre_class = list(set(pre_class.split(',')))
                    new_pre = ''
                    for pre in pre_class[:-1]:
                        new_pre += '%s,' % pre
                    new_pre = pre_class[-1]
                    label_name = '%s_' % (new_pre) + \
                        class_label.name.split('_')[-1]
                os.remove(class_list[i])
                os.remove(test_fn)
                fn_newlabel = mer_path + '%s.label' %label_name
                if os.path.isfile(fn_newlabel):
                    fn_newlabel = fn_newlabel[:fn_newlabel.rfind('_')] + '_new, %s' % fn_newlabel.split('_')[-1]
                mne.write_label(fn_newlabel, com_label)
                class_list[i] = fn_newlabel
                belong = True
            i = i + 1
        if belong is False:
            class_list.append(test_fn)
    return len(class_list)

def redu_small(red_path, vert_size, fn_src):
    '''
    red_path: str
        The directory for storing merged ROIs.
    '''
    import glob
    # vert = []
    # list_dirs = os.walk(mer_path)
    # label_pieces = []
    # for root, dirs, files in list_dirs:
    #    for f in files:
    #        label_fname = os.path.join(root, f)
    #        label = mne.read_label(label_fname)
    #        label_pieces.append(label)
    # vert += [lab.vertices.size for lab in label_pieces]
    # vert_mean = np.round(np.array(vert).mean())
    # import pdb
    # pdb.set_trace()
    # print np.round(np.array(vert).max()), np.round(np.array(vert).min()), vert_mean
    files = glob.glob(red_path + '*')
    src = mne.read_source_spaces(fn_src)
    for f in files:
        label = mne.read_label(f)
        if label.hemi == 'lh':
            hemi = 0
        elif label.hemi == 'rh':
            hemi = 1
        dist = src[hemi]['dist']
        label_dist = dist[label.vertices, :][:, label.vertices]
        max_dist = round(label_dist.max() * 1000)
        if max_dist > vert_size:
            print('Size of the %s:' %label.name, max_dist, 'mm')
        elif max_dist < vert_size:
        # if label.vertices.size < vert_mean:
            print('Size of the %s:' %label.name, max_dist, 'mm lower than %d mm' %vert_size)
            os.remove(f)


def apply_merge(labels_path):
    '''
    Merge the concentrated ROIs.
    Parameter
    ---------
    labels_path: string.
        The path of concentrated labels.
    vert_num: int
        The least amount of vertices.
    red_little: bool
       If true, small ROIs will be removed.
    '''
    import glob
    import shutil

    mer_path = labels_path + 'merge/'
    reset_directory(mer_path)
    source = []
    source_path = labels_path + 'ini/'
    source = glob.glob(os.path.join(source_path, '*.*'))
    for filename in source:
        shutil.copy(filename, mer_path)
    reducer = True
    while reducer:
        list_dirs = os.walk(mer_path)
        label_list = ['']
        for root, dirs, files in list_dirs:
            for f in files:
                label_fname = os.path.join(root, f)
                label_list.append(label_fname)
        label_list = label_list[1:]
        len_class = _merge_rois(mer_path, label_list)
        if len_class == len(label_list):
            reducer = False
