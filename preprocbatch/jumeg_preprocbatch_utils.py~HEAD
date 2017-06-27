# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:38:32 2015

@author: fboers
"""

import os
import argparse

def msg_on_attribute_error(info):
    print "\n ===> FYI: NO parameters for <" +info+ "> defined\n\n"
    

def get_args():
    
    info_global="""
     JuMEG Preprocessing for FIF Files & FIF-RAW-Obj-Data 
    
     ---> porcess fif file for experiment MEG94T   
      jumeg_preprocessing_4raw_data.py -exp MEG94T -pfif /localdata/frank/data/MEG94T/mne/205386/MEG94T/120906_1401/1 -fif 205386_MEG94T_120906_1401_1_c,rfDC-raw.fif -v -r

     ---> process fif file list for experiment MEG94T
      jumeg_preprocessing_4raw_data.py -exp MEG94T -s /data/meg_store2/exp/MEG94T/mne -plist=/data/meg_store2/exp/MEG94T/doc -flist=meg94t_Gr_Static_0T_bads.txt -v -r
    
     ---> process fif file list for experiment InKomp
      jumeg_preprocessing_4raw_data.py -exp InKomp -s /data/meg_store1/exp/Chrono/mne -plist=/data/meg_store1/exp/Chrono/doc -flist=inkomp_mne_bads.txt -v -r

      jumeg_preprocessing_4raw_data.py -exp InKompFibp1-45 -s /data/meg_store1/exp/Chrono/mne -plist=/data/meg_store1/exp/Chrono/doc -flist=inkomp_mne_bads.txt -v -r 
    """
    
    
    info_stage="""
     MNE stage path, additional start path for files from list
     e.g. start path to mne fif file directory structure
    """

#--- parser     
    
    parser = argparse.ArgumentParser(info_global)
  
  #---exp/condi
    parser.add_argument("-exp","--experiment",  help="experiment name for <jumeg experiment template>: -exp TEST", default='default')
    parser.add_argument("-c","--conditions",help="condition list for <jumeg epocher template>: -c HAPPY SAD",nargs='*',default=())
  
  #---files  
    parser.add_argument("-s","--stage", help=info_stage,default=()) 
    parser.add_argument("-pfif","--pathfif",  help="path to single fif file",default=os.getcwd() )
    
    #fifname = parser.add_mutually_exclusive_group()
    parser.add_argument("-fif","--fifname", help="fif filename")
   
    parser.add_argument("-plist","--pathlist",help="path to text file with fif files",default=os.getcwd() )
    parser.add_argument("-flist","--fnamelist",help="text file with fif files",default=() )
    
    parser.add_argument("-b","--bads",nargs='*', help="list of channels to mark as bad: --bads A1 A2",default=() ) 

  #---flags:
  
    parser.add_argument("-v","--verbose",action="store_true",help="verbose mode")
    parser.add_argument("-d","--debug",  action="store_true",help="!!! DEBUG MODE used by FB") 
    parser.add_argument("-r","--run",    action="store_true",help="!!! EXECUTE & RUN this program !!!") 
    
    return parser.parse_args(),parser



