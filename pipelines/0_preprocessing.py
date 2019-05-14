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

- setup defaults

- apply; do your stuff

Main
  get args, set up parameter, call apply
-------------

Examples:
----------
call script with parameter or -h for help

#--- run for id(s)
0_preprocessing.py -s $JUMEG_TEST_DATA/mne -subj 211747 -c config0.yaml -log -v -d -r

#--- run for id, recursive looking into subdirs, overwrite logfile
0_preprocessing.py -s $JUMEG_TEST_DATA/mne -subj 211747 -c config0.yaml -log -v -d -r -rec --logoverwrite

#--- run for ids, recursive looking into subdirs, overwrite logfile
0_preprocessing.py -s $JUMEG_TEST_DATA/mne -subj 211747,211890 config0.yaml -log -v -d -r -rec --logoverwrite

#--- run for files in list, overwrite logfile
0_preprocessing.py -s $JUMEG_TEST_DATA/mne -lname=list_test.txt -lpath=$JUMEG_TEST_DATA/mne -c config0.yaml -log -v -d -r --logoverwrite

#--- run for one file, overwrite logfile
0_preprocessing.py -s $JUMEG_TEST_DATA/mne -fpath $JUMEG_TEST_DATA/mne/211747/FREEVIEW01/180109_0955/1 -fname 211747_FREEVIEW01_180109_0955_1_c,rfDC,meeg-raw.fif -c config0.yaml -log -v -d -r --logoverwrite

#--- run for MEG94T
0_preprocessing.py -s $JUMEG_PATH_LOCAL_DATA/exp/MEG94T/mne -lpath $JUMEG_LOCAL_DATA/exp/MEG94T/mne -fname test01.txt -log -v -d -r --logoverwrite


"""

import os,sys,logging

from jumeg.base import jumeg_logger
from jumeg.base.pipelines.jumeg_pipeline_looper import JuMEG_PipelineLooper

import jumeg.base.pipelines.jumeg_pipelines_utils0 as utils

logger = logging.getLogger("jumeg")

__version__= "2019.05.13.001"

#--- parameter / argparser defaults
defaults={
          "stage"          : ".",
          "fif_extention"  : ["meeg-raw.fif","rfDC-empty.fif"],
          "config"         : "config_file.yaml",
    
          "subjects"       : None,
          "list_path"      : None,
          "list_file"      : None,
          "fpath"          : None,
          "fname"          : None,
          "overwrite"      : False,
          "recursive"      : True,
      
          "log2file"       : True,
          "logprefix"      : "preproc0",
          "logoverwrite"   : False,
      
          "verbose"        : False,
          "debug"          : False
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
    jpl.ExitOnError=True
    
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

        logger.info(" --> DONE raw file output: {}\n".format(raw_fname))
        
#=========================================================================================
#==== MAIN
#=========================================================================================
def main(argv):
   
    opt, parser = utils.get_args(argv,defaults=defaults,version=__version__)
    if len(argv) < 2:
       parser.print_help()
       sys.exit(-1)
       
    if opt.run: apply(name=argv[0],opt=opt,defaults=defaults)
    
if __name__ == "__main__":
   main(sys.argv)

