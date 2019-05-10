#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 18.04.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------
"""
JuMEG preprcessing script frame work

Head
- setup defaults

-------------

preprocessing function

0_preprocessing.py -s $JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/FV -subj 211747 -c fbconfig_file.yaml -log -v -d -r

0_preprocessing.py -s $JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/mne -subj 211890,211747 -c fbconfig_file.yaml -log -v -d -r -rec

0_preprocessing.py -s $JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/mne -subj 211747 -c fbconfig_file.yaml -log -v -d -r -rec --logoverwrite

0_preprocessing.py -s $JUMEG_PATH_LOCAL_DATA/exp/MEG94T/mne -lname=meg94t_meeg.txt -lpath=$JUMEG_PATH_LOCAL_DATA/exp/MEG94T/mne -c fbconfig_file.yaml -log -v -d -r -rec --logoverwrite
-------------

MAIN

"""

import os,sys,logging


from jumeg.base import jumeg_logger
import jumeg.base.pipelines.jumeg_pipelines_utils0 as utils
from jumeg.base.pipelines.jumeg_pipeline_looper import JuMEG_PipelineLooper

logger = logging.getLogger("root")

__version__= "2019.05.10.001"

defaults={
          "stage"          : ".",
          "fif_extention"  : ["meeg-raw.fif","rfDC-empty.fif"],
          "config"         : "config_file.yaml",
          "subjects"       : None,
          "log2file"       : True,
          "logprefix"      : "preproc0",
          "logoverwrite"   : False,
          "overwrite"      : False,
          "verbose"        : False,
          "debug"          : False,
          "recursive"      : True
         }

#-----------------------------------------------------------
#--- apply
#-----------------------------------------------------------
def apply(name=None,opt=None,defaults=None,logprefix="preproc"):
    """
     jumeg preprocessing step 1
      noise reducer
      interpolate bads (bad channels & suggested_bads)
      filter
      resample
     
    :param opt: arparser option obj
    :param defaults: default dict
    :param logprefix: prefix for logfile e.g. name of script
    :return:
    """
   #---
    raw = None
    
   #--- init/update logger
    jumeg_logger.setup_script_logging(name=name,opt=opt,logger=logger)
 
    jpl = JuMEG_PipelineLooper(options=opt,defaults=defaults)
    
    for raw_fname,subject_id,dir in jpl.file_list():
      
       #--- call noise reduction
        raw_fname,raw = utils.apply_noise_reducer(raw_fname,raw=raw,**jpl.config.get("noise_reducer"))
    
       #--- call suggest_bads
        raw_fname,raw = utils.apply_suggest_bads(raw_fname,raw=raw,**jpl.config.get("suggest_bads"))
       
       #--- call interploate_bads
        raw_fname,raw = utils.apply_interpolate_bads(raw_fname,raw=raw,**jpl.config.get("interpolate_bads") )
        
       #--- call filter
       # raw_fname,raw = utils.apply_filter(raw_fname,raw=raw,**jpl.config.get("filtering") )

       #--- call resample
       # raw_fname,raw = utils.apply_resample(raw_fname,raw=raw,**jpl.config.get("resampling"))


#=========================================================================================
#==== MAIN
#=========================================================================================
def main(argv):
   
    opt, parser = utils.get_args(argv,defaults=defaults)
    if len(argv) < 2:
       parser.print_help()
       sys.exit(-1)
       
    if opt.run: apply(name=argv[0],opt=opt,defaults=defaults)
    
if __name__ == "__main__":
   main(sys.argv)

