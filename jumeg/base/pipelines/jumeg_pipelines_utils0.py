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
"""
preproc functions:
 apply_noise_reducer    call noise_reducer
 apply_suggest_bads     call suggest_bads
 apply_interpolate_bads call interpolate_bads

"""
import sys,os,logging,yaml,argparse,glob
#from contextlib import redirect_stdout

import mne

from jumeg.base.jumeg_base         import jumeg_base as jb
from jumeg.base                    import jumeg_logger  #     import STDStreamLogger # capture stout,stderr
from jumeg.plot.jumeg_plot_preproc import JuMEG_PLOT_PSD
from jumeg.base.pipelines.jumeg_pipelines_utils_base import get_args
#--- preproc
from jumeg.jumeg_noise_reducer     import noise_reducer
from jumeg.jumeg_suggest_bads      import suggest_bads
from jumeg.jumeg_interpolate_bads  import interpolate_bads as jumeg_interpolate_bads

logger = logging.getLogger("jumeg")

__version__= "2019.05.10.001"

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
    fname_out = None
    logger.info("  -> apply_noise_reducer file name: {}".format(raw_fname))
    logger.debug("  -> config parameter:\n{}".format( cfg ))
    
    if not jb.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention") ):
       return

    if not cfg.get("run"):
     #--- return raw_fname,raw
       return jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=False,
                                     postfix = cfg.get("postfix","nr"),overwrite = cfg.get("overwrite",True))
    
    
    logger.info(" --> preproc noise_reducer for raw file: {}".format(raw_fname))
    
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
    
   #--- with redirect stdout/err
    with jumeg_logger.StreamLoggerSTD(label="noise_reducer"):
        #--- 1 nr low pass filter for freq below 5 hz
         if cfg.get("reflp"):
            raw = noise_reducer(None,raw=raw,reflp=cfg.get("reflp"),return_raw=True,verbose=cfg.get("verbose"),exclude_artifacts=False)
            raw_changed = True
        #--- 2 nr high pass filter
         if cfg.get("refhp"):
            raw = noise_reducer(None,raw=raw,reflp=cfg.get("refhp"),return_raw=True,verbose=cfg.get("verbose"),exclude_artifacts=False)
            raw_changed = True
        #--- 3  nr notch filter to remove power line noise
         if cfg.get("refnotch"):
            raw = noise_reducer(None,raw=raw,refnotch=cfg.get("refnotch"),fnout=None,return_raw=True,verbose=cfg.get("verbose"),exclude_artifacts=False)
            raw_changed = True
    
            
   #--- save and update filename in raw
    if cfg.get("save"):
       save = raw_changed
       
    #--- update filename in raw and save if save
    fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=save,update_raw_filenname=True,
                                           postfix=cfg.get("postfix","nr"),overwrite=cfg.get("overwrite",True))
    
    #--- plot results, avoid reloading raw data
    if cfg.get("plot"):
       jplt.plot(raw,title="denoised: "+os.path.basename(fname_out),check_dead_channels=False)
       if cfg.get("plot_show"):
          jplt.show()
       jplt.save(fname=fname_out,plot_dir=cfg.get("plor_dir","plots"))
  
    if fname_out:
        return fname_out,raw
    else:
        raise Exception("---> ERROR file name not defined !!!")

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
    fname_out = None
    logger.info("  -> apply_suggest_bads file name: {}".format(raw_fname))
    logger.debug("  -> config parameter:\n{}".format(cfg))
    
  
    if not jb.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention")):
       return
    
        #--- return raw_fname,raw
    if not cfg.get("run"):
       return jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=False,
                                     postfix=cfg.get("postfix","bcc"),overwrite=cfg.get("overwrite",True))
    
    raw_changed   = True
    jb.verbose    = cfg.get("verbose")
    raw,raw_fname = jb.get_raw_obj(raw_fname,raw=raw)
    
    if raw:
       with jumeg_logger.StreamLoggerSTD(label="suggest_bads"):
            marked,raw = suggest_bads(raw) #,**cfg["parameter"]) #show_raw=cfg.get("show_raw") )

    fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),update_raw_filenname=True,
                                           postfix=cfg.get("postfix","bcc"),overwrite=cfg.get("overwrite",True))
 
    if fname_out:
       return fname_out,raw
    else:
       raise Exception( "---> ERROR file name not defined !!!" )
      

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
    fname_out = None
    logger.info("  -> apply_interpolate_bad file name: {}".format(raw_fname))
    logger.debug("  -> config parameter:\n{}".format(cfg))
    jb.verbose = cfg.get("verbose")
    
    if not jb.check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention")):
       return
    #--- return raw_fname,raw
    if not cfg.get("run"):
       return jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=False,update_raw_filenname=True,
                                     postfix=cfg.get("postfix","int"),overwrite=cfg.get("overwrite",True))

    raw,raw_fname = jb.get_raw_obj(raw_fname,raw=raw)
    
    if raw:
       logger.info("fname: {}".format(raw_fname) )
      #--- Interpolate bad channels using jumeg
       with jumeg_logger.StreamLoggerSTD(label="interpolate_bads"):
            raw = jumeg_interpolate_bads(raw) #,**cfg.get("parameter"))  #,origin=cfg.get("origin",None),reset_bads=cfg.get("reset_bads",True) )
    
    #-- check results
       if cfg.get("plot_block"):
          raw.plot(block=cfg.get("plot_block"))
    
    #--- update filename in raw and save if save
    fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),
                                           postfix=cfg.get("postfix","int"),overwrite=cfg.get("overwrite",True))
   
    if fname_out:
       return fname_out,raw
    else:
       raise Exception( "---> ERROR file name not defined !!!" )
       
    

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
    logger.info("  -> apply_filter file name: {}".format(raw_fname))
    logger.debug("  -> config parameter:\n{}".format(cfg))

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
    #    fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),update_raw_filenname=True,
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
    logger.info("  -> apply_resample file name: {}".format(raw_fname))
    logger.debug("  -> config parameter:\n{}".format(cfg))
    
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
    #    fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),update_raw_filenname=True,
    #                                          postfix=cfg.get("postfix","bcc"),overwrite=cfg.get("overwrite",True))

   #--- return back stdout/stderr from logger
    #jumeg_logger.log_stdout(reset=True)
    #jumeg_logger.log_stderr(reset=True)

    return fname_out,raw


