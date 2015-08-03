#!/usr/bin/env python

"""
    JuMEG TSV Time Series Viewer
    FB 26.02.2015
    last updae FB 26.02.2015
"""


import sys, getopt, os, os.path

import mne


#fout="/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1/110058_MEG94T_121001_1331_1_c,rfDC-float32,30sec-raw.fif'
#raw.save( fname,format='single',tmax=30.0)

# jumeg_tsv.py --fname=110058_MEG94T_121001_1331_1_c,rfDC-raw.fif --path=/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1

# jumeg_tsv.py --fname=110058_MEG94T_121001_1331_1_c,rfDC_30sec-raw.fif --path=/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1


#--- jumegs functions

from jumeg.tsv.jumeg_tsv_utils import jumeg_tsv_utils_args

from jumeg.tsv.jumeg_tsv_gui import jumeg_tsv_gui
#from jumeg.tsv.jumeg_tsv_gui_orig import jumeg_tsv_gui

def run():

    opt = jumeg_tsv_utils_args()
    
    if opt.debug:
       opt.fname="110058_MEG94T_121001_1331_1_c,rfDC_30sec-raw.fif"
       opt.path ="/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1"


    jumeg_tsv_gui(fname=opt.fname,path=opt.path,verbose=opt.verbose,debug=opt.debug,experiment=opt.experiment,
                  duration=opt.duration,start=opt.start,n_channels=opt.n_channels,bads=opt.bads)


is_main = (__name__ == '__main__')

if is_main:
   run()


#p='/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1'
#f='110058_MEG94T_121001_1331_1_c,rfDC_30sec-raw.fif'
#ffif=p+'/'+f
#raw=mne.io.Raw(ffif,preload=True)
