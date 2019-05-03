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

-------------

MAIN

"""

import os,sys,logging

import fb_preproc_utils as utils
from jumeg.base import jumeg_logger

# from jumeg.base.jumeg_base import jumeg_base as jb

logger = logging.getLogger("root")

__version__= "2019.04.30.001"

defaults={
          "stage"          : ".",
          "fif_extention"  : ["meeg-raw.fif","rfDC-empty.fif"],
          "config"         : "config_file.yaml",
          "subjects"       : None,
          "logfile"        : True,
          "overwrite"      : False,
          "verbose"        : False,
          "debug"          : False,
          "recursive"      : False
         }

#-----------------------------------------------------------
#--- apply
#-----------------------------------------------------------
def apply(opt=None,defaults=None,logprefix="preproc"):
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
   #--- init globals
    config,subjects,stage,file_extention,recursive = utils.init_globals(opt=opt,defaults=defaults)
    raw_fname = None
    raw       = None
    file_idx  = 1
    
   #--- main for loop do preprocessing
    for raw_fname,recordings_dir,subject_id in utils.subject_file_list(subjects,stage=stage,file_extention=file_extention,recursive=recursive):
       #--- setup logfile
        if opt.logfile:
           log_name = os.path.splitext(raw_fname)[0]
           Hlog = jumeg_logger.update_filehandler(logger=logger,path=os.path.join(recordings_dir,"log"),prefix=logprefix,name=log_name)
           logger.info("  -> writing log to: {}".format(Hlog.filename))

        logger.info("---> Start PreProc Ids: {}\n".format(len(subjects))+
                    " --> subject id       : {} file number: {}\n".format(subject_id,file_idx)+
                    "  -> raw file name    : {}\n".format(raw_fname)+
                    "  -> stage            : {}\n".format(stage)+
                    "  -> basedir          : {}\n".format( os.getcwd())+
                    "  -> recordings dir   : {}".format(recordings_dir))
        
       #--- call noise reduction
        cfg = config.get("noise_reducer")
        if cfg.get("run"):
           logger.info(" --> preproc noise_reducer: {} / {} raw file: {}".format(file_idx,len(subjects),raw_fname))
           if utils.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention",file_extention) ):
              raw_fname,raw = utils.apply_noise_reducer(os.path.join(recordings_dir,raw_fname),raw=raw,**cfg)
   
       #--- call interploate_bads
        """
        cfg = config.get("interpolate_bads")
        if cfg.get("run"):
           if utils.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention") ):
              raw_fname,raw = utils.apply_interpolate_bads(raw_fname,raw=raw,**cfg)
           
       
   
       #--- call filter
        cfg = config.get("filter")
        if cfg.get("run"):
           if check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention") ):
              apply_filter(raw_fname,raw=raw,**cfg)
      
       #--- call resample
        cfg = config.get("resample")
        if cfg.get("run"):
           if check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention") ):
              apply_resample(raw_fname,raw=raw,**cfg)
        """
        file_idx += 1
        
        if Hlog:
           Hlog.close()
   
#=========================================================================================
#==== MAIN
#=========================================================================================
def main(argv):
   
    opt, parser = utils.get_args(argv,defaults=defaults)
    if len(argv) < 2:
       parser.print_help()
       sys.exit(-1)
       
  #--- init/update logger
    jumeg_logger.setup_script_logging(name=argv[0],opt=opt,logger=logger)
    
  #--- logfile  prefix
    pgr_name = os.path.splitext( os.path.basename(argv[0]) )[0]
    
    if opt.run: apply(opt=opt,defaults=defaults,logprefix=pgr_name)
    
if __name__ == "__main__":
   main(sys.argv)

