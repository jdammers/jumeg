#!/usr/bin/env python

"""
    JuMEG TSV Time Series Viewer
    FB 26.02.2015
    last updae FB 26.02.2015
"""


import sys, getopt, os, os.path

import mne


#fout="/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1/110058_MEG94T_121001_1331_1_c,rfDC-float32,30sec-raw.fif'
#raw.save( fname,format='single',tmax=30.0)import wx

# jumeg_tsv.py --fname=110058_MEG94T_121001_1331_1_c,rfDC-raw.fif --path=/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1

# jumeg_tsv.py --fname=110058_MEG94T_121001_1331_1_c,rfDC_30sec-raw.fif --path=/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1


#--- jumegs functions

from jumeg.tsv.utils.jumeg_tsv_utils import jumeg_tsv_utils_get_args
from jumeg.tsv.jumeg_tsv_gui import jumeg_tsv_gui

#from jumeg.tsv.jumeg_tsv_gui_orig import jumeg_tsv_gui

def run():

    opt,parser  = jumeg_tsv_utils_get_args()
    
    if opt.debug:
       opt.verbose = True 
       opt.fname= "205382_MEG94T_120904_1310_2_c,rfDC-raw.fif"
       opt.path =  os.environ['HOME'] + "/MEGBoers/data/exp/MEG94T/mne/205382/MEG94T/120904_1310/2/"
 
    elif opt.dd:
       opt.fname='200098_leda_test_10_raw.fif' 
       opt.verbose = True 
       
    elif opt.ddd:
       opt.fname='200098_leda_test_60_raw.fif' 
       opt.verbose = True 
   
   
    if opt.verbose:
       for k,v in vars(opt).iteritems():
           if v:
              print "---> " + k   +" : "+ str( v )
           else:
              print "---> " + k + " : None"
       print"\n"
      
   
    jumeg_tsv_gui(fname=opt.fname,path=opt.path,verbose=opt.verbose,debug=opt.debug,experiment=opt.experiment,
                  duration=opt.duration,start=opt.start,n_channels=opt.n_channels,n_cols=opt.n_cols,bads=opt.bads)


is_main = (__name__ == '__main__')

if is_main:
   run()


#p='/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1'
#f='110058_MEG94T_121001_1331_1_c,rfDC_30sec-raw.fif'
#ffif=p+'/'+f
#raw=mne.io.Raw(ffif,preload=True)
