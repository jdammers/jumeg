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
1_preprocessing.py -s $JUMEG_TEST_DATA/mne -subj 211747 -c config0.yaml -log -v -d -r

#--- run for id, recursive looking into subdirs, overwrite logfile
1_preprocessing.py -s $JUMEG_TEST_DATA/mne -subj 211747 -c config0.yaml -log -v -d -r -rec --logoverwrite

#--- run for ids, recursive looking into subdirs, overwrite logfile
1_preprocessing.py -s $JUMEG_TEST_DATA/mne -subj 211747,211890 -c config0.yaml -log -v -d -r -rec --logoverwrite

List Example:
-------------

cd ${MNE_DATA_PATH}
find | grep meeg-raw.fif > meeg_filelist.txt
find | grep empty.fif > empty_filelist.txt

#--- call preproc 1 for meeg data
1_preprocessing.py -c preproc_config01.yaml -lname meeg_filelist.txt -lpath ${MNE_DATA_PATH} -v -r

#--- call preproc 1 for empty-room data
1_preprocessing.py -c preproc_config01.yaml -lname empty_filelist.txt -lpath ${MNE_DATA_PATH} -v -r


#--- INTEXT
1_preprocessing.py -c intext_config01.yaml -subj 201772,203404 -v -r
"""

import os,sys,logging

from jumeg.base import jumeg_logger
from jumeg.base.pipelines.jumeg_pipeline_looper import JuMEG_PipelineLooper

import jumeg.base.pipelines.jumeg_pipelines_utils1 as utils

logger = logging.getLogger("jumeg")

__version__= "2019.08.07.001"

#--- parameter / argparser defaults
defaults={
          "stage"         : None,
          "file_extention": None, # ["meeg-raw.fif","c,rfDC-raw.fif","rfDC-empty.fif"],
          "config"        : "1_preprocessing_config.yaml",
          #"subjects"      : None,
          "list_path"     : None, #"$JUMEG_PATH_MNE_IMPORT2/MEG94T/source/jumeg/pipelines",
          "list_name"     : None,
          "fpath"         : None,
          "fname"         : None,
          "recursive"     : True,
          "log2file"      : True,
          "logprefix"     : "preproc_1",
          "logoverwrite"  : True,
          "verbose"       : False,
          "debug"         : False
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
    
    for fname,subject_id,raw_dir in jpl.file_list():
      
        if not opt.run: continue
           
        raw = None # !!! important  !!!
       #--- call noise reduction
        raw_fname,raw = utils.apply_noise_reducer(raw_fname=fname,raw=None,config=jpl.config.get("noise_reducer"))

       #--- call suggest_bads
        raw_fname,raw = utils.apply_suggest_bads(raw_fname=raw_fname,raw=raw,config=jpl.config.get("suggest_bads"))
       
       #--- call interploate_bads
        raw_fname,raw = utils.apply_interpolate_bads(raw_fname=raw_fname,raw=raw,config=jpl.config.get("interpolate_bads") )
        
       #--- call filter
       # raw_fname,raw = utils.apply_filter(raw_fname,raw=raw,config=jpl.config.get("filtering") )

       #--- call resample
       # raw_fname,raw = utils.apply_resample(raw_fname,raw=raw,config=jpl.config.get("resampling"))

       #--- call interploate_bads
        raw_fname,raw = utils.apply_ica(raw_fname=raw_fname,raw=raw,config=jpl.config.get("ica") )
      

        logger.info(" --> DONE preproc subject id: {}\n".format(subject_id)+
                    "  -> input  file: {}\n".format(fname)+
                    "  -> output file: {}\n".format(raw_fname))
        
#=========================================================================================
#==== MAIN
#=========================================================================================
def main(argv):
   
    opt, parser = utils.get_args(argv,defaults=defaults,version=__version__)
    if len(argv) < 2:
       parser.print_help()
       sys.exit(-1)
       
    apply(name=argv[0],opt=opt,defaults=defaults)
    
if __name__ == "__main__":
   main(sys.argv)

