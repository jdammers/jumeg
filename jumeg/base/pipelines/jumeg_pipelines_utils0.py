#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 30.04.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import sys,os,logging,yaml,argparse,glob
#from contextlib import redirect_stdout

import mne

from jumeg.base.jumeg_base         import jumeg_base as jb
from jumeg.base                    import jumeg_logger  #     import STDStreamLogger # capture stout,stderr
from jumeg.plot.jumeg_plot_preproc import JuMEG_PLOT_PSD

#--- preproc
from jumeg.jumeg_noise_reducer     import noise_reducer
from jumeg.jumeg_suggest_bads      import suggest_bads
from jumeg.jumeg_interpolate_bads  import interpolate_bads as jumeg_interpolate_bads

logger = logging.getLogger("root")

__version__= "2019.05.10.001"


#---
def get_args(argv,parser=None,defaults=None):
    """
    get args using argparse.ArgumentParser ArgumentParser
    e.g: argparse  https://docs.python.org/3/library/argparse.html

    :param argv:   the arguments, parameter e.g.: sys.argv
    :param parser: argparser obj, the base/default obj like --verbose. --debug
    :return:
    
    Results:
    --------
     parser.parse_args(), parser
    """
    description = """
                  JuMEG Preprocessing Script
                  version: {}
                  used python version: {}
                  """.format(__version__,sys.version.replace("\n"," "))
    h_stage     = """
                  stage/base dir: start path for ids from list
                  -> start path to directory structure
                  e.g. /data/megstore1/exp/M100/mne/
                  """
    h_lpath    = "path for list file, list of file to process containing list of full filenames"
    h_lname    = "list file name, list of file to process containing list of full filenames"
    
    h_fpath    = "path for file to process"
    h_fname    = "file name to process"
    
    h_fextention ="fif file extention or list of extentions, looking for files to process with these extention e.g. meeg-raw.fif or [meeg-raw.fif,rfDC-empty.fif]"
    
    h_subjects  = "subject id or ids  e.g.: 123 or 234,456"
    h_config    = "script config file, full filename"
    h_verbose   = "bool, str, int, or None"
    h_overwrite = "overwrite existing files"
    
   #--- parser
    if not parser:
       parser = argparse.ArgumentParser(description=description)
    else:
       parser.description = description
    
    if not defaults:
       defaults = {}
       
   #---  parameter settings  if opt  elif config else use defaults
    parser.add_argument("-s",   "--stage",    help=h_stage)#,default=defaults.get("stage",".") )
    parser.add_argument("-subj","--subjects", help=h_subjects)#,default=defaults.get("subjects"))
    parser.add_argument("-fif_ext","--fif_extention",help=h_fextention)#,default=defaults.get("fif_extention") )
   #---
    parser.add_argument("-lpath","--list_path",help=h_lpath)
    parser.add_argument("-lname","--list_name",help=h_lname)
   #---
    parser.add_argument("-fpath","--fpath",help=h_fpath)
    parser.add_argument("-fname","--fname",help=h_fname)
   #---
    parser.add_argument("-c","--config",help=h_config,default=defaults.get("config") )
   #--- flags
    parser.add_argument("-o",  "--overwrite",action="store_true",help=h_overwrite, default=defaults.get("overwrite") )
    parser.add_argument("-v",  "--verbose",  action="store_true",help=h_verbose,   default=defaults.get("verbose"))
    parser.add_argument("-d",  "--debug",    action="store_true",help="debug mode",default=defaults.get("debug"))
    parser.add_argument("-rec","--recursive",action="store_true",help="search recursive find files in subfolders",default=defaults.get("recursive",False))

    parser.add_argument("-r",  "--run",      action="store_true",help="!!! EXECUTE & RUN this program !!!",default=defaults.get("run",True))
    parser.add_argument("-log","--log2file", action="store_true",help="generate logfile",default=defaults.get("log2file"))
    parser.add_argument("-logoverwrite","--logoverwrite", action="store_true",help="overwrite existing logfile",default=defaults.get("logoverwrite"))
    parser.add_argument("-logprefix","--logprefix", help="logfile prefix",default= os.path.splitext( os.path.basename(argv[0]) )[0])
  
  #--- init flags
  # ck if flag is set in argv as True
  # problem can not switch on/off flag via cmd call
    opt = parser.parse_args()
    for g in parser._action_groups:
        for obj in g._group_actions:
            if str( type(obj) ).endswith('_StoreTrueAction\'>'):
               if vars( opt ).get(obj.dest):
                  opt.__dict__[obj.dest] = False
                  for flg in argv:
                      if flg in obj.option_strings:
                         opt.__dict__[obj.dest] = True
                         break
  
    return opt, parser
 

#===========================================================
#=== preproc part
#===========================================================
def print_to_logger(raw_fname,raw=None,**cfg):
  
   #--- log stdout,stderr
   jumeg_logger.log_stdout(label=" LOGTEST")
   jumeg_logger.log_stderr()
    
   print("  -> TEST1 print to logger: {}".format(raw_fname) )

  #--- return back stdout/stderr from logger
   jumeg_logger.log_stdout(reset=True)
   jumeg_logger.log_stderr(reset=True)
   
   return raw_fname,raw

#---------------------------------------------------
#--- apply_noise_reducer
#---------------------------------------------------
def apply_noise_reducer(raw_fname,raw=None,**cfg):
    '''
    apply <magic ee noise reducer> thrice to reference channels with different freq parameters
    save PSD plot in subfolder /plots
   
    !!! overwrite raw-obj, works inplace !!!
  
    0) reset bads and check for dead channels
    1) apply nr low pass filter for freq below e.g.: 5Hz  <reflp>
    2) apply nr high pass filter if defined               <reflp>
    3) apply nr notch filter to remove power line noise   <refnotch>
    4) save PSD plot
    
    Parameter:
    -----------
    <noise_reducer> parameter used in this function :
     fname_raw  : input raw filename
     raw        : <None>\n
     cfg        : dict, part of config file <None>
        from config file <noise_reducer> part\n
        reflp      : <None>\n
        refhp      : <None>\n
        refnotch   : <None>\n
        
        plot: True
        plot_show : True
        plot_dir   : subdir to save plots
       
        postfix       : "nr"
        file_extention: ["meeg-raw.fif","rfDC-empty.fif"]
        
        run      : True
        save     : True
        overwrite: True
        
    ToDo add parameter extended
    <noise_reducer> parameter extended
       signals=[], noiseref=[], detrending=None,
       tmin=None, tmax=None,
       exclude_artifacts=True, checkresults=True, return_raw=False,
       complementary_signal=False, fnout=None, verbose=False

    Return:
    --------
     filename,raw-obj
    '''
    
   #--- init plot
    jplt = None
    logger.debug("  ->noise reducer config parameter:\n{}".format( jb.pp_list2str(cfg) ))

    if not jb.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention") ):
       return

    if not cfg.get("run"):
     #--- return raw_fname,raw
       return jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=False,
                                     postfix = cfg.get("postfix","nr"),overwrite = cfg.get("overwrite",True))
    
    
    logger.info(" --> preproc noise_reducer for raw file: {}".format(raw_fname))
    
   #--- catch stdout,stderr
    jumeg_logger.log_stdout(label="noise_reducer")
    jumeg_logger.log_stderr(label="noise_reducer")
    
   #--- noise reduction
    # apply noise reducer thrice to reference channels with different freq parameters
    # !!! overwrite raw-obj !!!
    save        = False
    raw_changed = False
    
    jb.verbose = cfg.get("verbose")
   
   #--- load raw, reset bads
    raw,raw_fname = jb.get_raw_obj(raw_fname,raw=raw,reset_bads=True)
    
   #--- check dead channes and mark them as bad
    jb.picks.check_dead_channels(raw=raw)
  
   #--- start plot denoising orig raw psd, avoid reloading raw data
    if cfg.get("plot"):
       jplt = JuMEG_PLOT_PSD(n_plots=2,name="denoising",verbose=True)
       jplt.plot(raw,title="orig: " + os.path.basename(raw_fname),check_dead_channels=False)
    
    #--- 1 nr low pass filter for freq below 5 hz
    if cfg.get("reflp"):
       #with redirect_stdout(logger):
       raw = noise_reducer(None,raw=raw,reflp=cfg.get("reflp"),return_raw=True,verbose=cfg.get("verbose"),exclude_artifacts=False)
       raw_changed = True
    #--- 2 nr high pass filter
    if cfg.get("refhp"):
       #with redirect_stdout(logger):
       raw = noise_reducer(None,raw=raw,reflp=cfg.get("refhp"),return_raw=True,verbose=cfg.get("verbose"),exclude_artifacts=False)
       raw_changed = True
    #--- 3  nr notch filter to remove power line noise
    if cfg.get("refnotch"):
       #with redirect_stdout(logger):
       raw = noise_reducer(None,raw=raw,refnotch=cfg.get("refnotch"),fnout=None,return_raw=True,
                           verbose=cfg.get("verbose"),exclude_artifacts=False)
       raw_changed = True

   #--- save and update filename in raw
    if cfg.get("save"):
       save = raw_changed
    fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=save,
                                           postfix=cfg.get("postfix","nr"),overwrite=cfg.get("overwrite",True))
    
    #--- plot results, avoid reloading raw data
    if cfg.get("plot"):
       jplt.plot(raw,title="denoised: "+os.path.basename(fname_out),check_dead_channels=False)
       if cfg.get("plot_show"):
          jplt.show()
       jplt.save(fname=fname_out,plot_dir=cfg.get("plor_dir","plots"))
  
   #--- return back stdout/stderr from logger
    jumeg_logger.log_stdout(reset=True)
    jumeg_logger.log_stderr(reset=True)

    return fname_out,raw

#---------------------------------------------------
#--- apply_suggest_bads
#---------------------------------------------------
def apply_suggest_bads(raw_fname,raw=None,**cfg):
    """

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj
    """
    logger.info("  ->config parameter:\n{}".format(cfg))

    if not jb.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention")):
        return
    
        #--- return raw_fname,raw
    if not cfg.get("run"):
        return jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=False,
                                      postfix=cfg.get("postfix","bcc"),overwrite=cfg.get("overwrite",True))

    #--- catch stdout,stderr
    jumeg_logger.log_stdout(label="suggest_bads")
    jumeg_logger.log_stderr(label="suggest_bads")
 
    raw_changed   = True
    jb.verbose    = cfg.get("verbose")
    raw,raw_fname = jb.get_raw_obj(raw_fname,raw=raw)
   #---
    
    marked,raw = suggest_bads(raw) #,**cfg["parameter"]) #show_raw=cfg.get("show_raw") )

   #--- save and update filename in raw
    if raw_changed:
        fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),
                                               postfix=cfg.get("postfix","bcc"),overwrite=cfg.get("overwrite",True))

   #--- return back stdout/stderr from logger
    jumeg_logger.log_stdout(reset=True)
    jumeg_logger.log_stderr(reset=True)

    return fname_out,raw

#---------------------------------------------------
#--- apply_interpolate_bads
#---------------------------------------------------
def apply_interpolate_bads(raw_fname,raw=None,**cfg):
    """

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj
    """

    logger.info("  ->config parameter:\n{}".format(cfg))

    if not jb.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention")):
        return
    #--- return raw_fname,raw
    if not cfg.get("run"):
        return jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=False,
                                      postfix=cfg.get("postfix","bcc"),overwrite=cfg.get("overwrite",True))

    #--- catch stdout,stderr
    jumeg_logger.log_stdout(label="suggest_bads")
    jumeg_logger.log_stderr(label="suggest_bads")

    jb.verbose = cfg.get("verbose")
    raw,raw_fname = jb.get_raw_obj(raw_fname,raw=raw)
    
    #--- Interpolate bad channels using jumeg
    raw = jumeg_interpolate_bads(raw) #,**cfg.get("parameter"))  #,origin=cfg.get("origin",None),reset_bads=cfg.get("reset_bads",True) )
    
    #-- check results
    if cfg.get("plot_block"):
       raw.plot(block=cfg.get("plot_block"))
    
    #--- save and update filename in raw
    if raw:
       fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),
                                              postfix=cfg.get("postfix","int"),overwrite=cfg.get("overwrite",True))

   #--- return back stdout/stderr from logger
    jumeg_logger.log_stdout(reset=True)
    jumeg_logger.log_stderr(reset=True)

    
    return fname_out,raw

#---------------------------------------------------
#--- apply_filter
#---------------------------------------------------
def apply_filter(raw_fname,raw=None,**cfg):
    """

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj
    """
    return
    logger.info("  ->config parameter:\n{}".format(cfg))

    if not jb.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention")):
        return
        #--- return raw_fname,raw
    if not cfg.get("run"):
        return jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=False,
                                      postfix=cfg.get("postfix","fi"),overwrite=cfg.get("overwrite",True))

    #--- catch stdout,stderr
    #jumeg_logger.log_stdout(label="filter")
    #jumeg_logger.log_stderr(label="filter")

    jb.verbose = cfg.get("verbose")
    raw,raw_fname = jb.get_raw_obj(raw_fname,raw=raw)
    
    #--- ToDo setup  mne filter as jumeg CLS
    #raw,raw_fname = jumeg_mne_fileter(raw)
    raw_changed=True
    
    #--- save and update filename in raw
    #if raw_changed:
    #    fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),
    #                                           postfix=cfg.get("postfix","bcc"),overwrite=cfg.get("overwrite",True))

   #--- return back stdout/stderr from logger
    #jumeg_logger.log_stdout(reset=True)
    #jumeg_logger.log_stderr(reset=True)

    return fname_out,raw


#---------------------------------------------------
#--- apply_resample
#---------------------------------------------------
def apply_resample(raw_fname,raw=None,**cfg):
    """

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj
    """
    return
    logger.info("  ->config parameter:\n{}".format(cfg))
    
    if not jb.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention")):
       return
        #--- return raw_fname,raw
    if not cfg.get("run"):
       return jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=False,
                                      postfix=cfg.get("postfix","res"),overwrite=cfg.get("overwrite",True))

   #--- catch stdout,stderr
    #jumeg_logger.log_stdout(label="filter")
    #jumeg_logger.log_stderr(label="filter")

    jb.verbose = cfg.get("verbose")
    raw,raw_fname = jb.get_raw_obj(raw_fname,raw=raw)
    
    #--- ToDo setup  resampling
    #raw,raw_fname = jumeg_mne_fileter(raw)
    raw_changed=True
    
    #--- save and update filename in raw
    #if raw_changed:
    #    fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),
    #                                          postfix=cfg.get("postfix","bcc"),overwrite=cfg.get("overwrite",True))

   #--- return back stdout/stderr from logger
    #jumeg_logger.log_stdout(reset=True)
    #jumeg_logger.log_stderr(reset=True)

    return fname_out,raw


