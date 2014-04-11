#!/usr/bin/env python

# Wrapper script for the mne_brain_vision2fiff 

# Author: praveen.sripad@rwth-aachen.de
# License: BSD 3 clause

import sys

batch_mode = 0

def wrapper_brain_vision2fiff(header_fname):
    import os, sys
    mne_brain_vision2fiff_path = os.environ['MNE_BIN_PATH']+'/mne_brain_vision2fiff'
    if header_fname != "" :
        if header_fname.endswith('fif'):
            print "Usage: .py <header_file>"
            print "Please use the original binary to pass other arguments."
            sys.exit()
        else:
            print "The header file name provided is %s" %(header_fname)
    
    os.system(mne_brain_vision2fiff_path + ' --header ' + header_fname + ' --out ' + header_fname.split('.')[0] + '-eeg')

if batch_mode:
    # If you want to use wrapper in batch mode provide a file with list of .vhdr filenames.
    with open(sys.argv[1]) as temp_file:
        header_file_list = [line.rstrip('\n') for line in temp_file]
    for i in header_file_list:
        wrapper_brain_vision2fiff(i)
else:
    # If you want to use wrapper directly, provide the .vhdr file as argument. 
    wrapper_brain_vision2fiff(sys.argv[1])
