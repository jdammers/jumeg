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

import mne

from jumeg.base                                      import jumeg_logger
from jumeg.base.jumeg_base                           import jumeg_base as jb
from jumeg.base.jumeg_badchannel_table               import update_bads_in_hdf
from jumeg.base.pipelines.jumeg_pipelines_utils_base import get_args,JuMEG_PipelineFrame
from jumeg.base.pipelines.jumeg_pipelines_ica        import JuMEG_PIPELINES_ICA
from jumeg.plot.jumeg_plot_preproc                   import JuMEG_PLOT_PSD

#--- preproc
from jumeg.jumeg_noise_reducer     import noise_reducer
from jumeg.jumeg_suggest_bads      import suggest_bads
from jumeg.jumeg_interpolate_bads  import interpolate_bads as jumeg_interpolate_bads

logger = logging.getLogger("jumeg")

__version__= "2019.08.07.001"


#--- init Class JuMEG_PipelineFrame as instance
#JPL=JuMEG_PipelineFrame()

#---------------------------------------------------
#--- apply_noise_reducer
#---------------------------------------------------
@JuMEG_PipelineFrame # first  call execute __init__; later calls  execute __call__
def apply_noise_reducer(raw_fname=None,raw=None,config=None,label="noise reducer",fname_out=None):
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
    #with JuMEG_PipelineFrame(raw_fname=raw_fname,raw=raw,name="noise reducer",config=cfg) as JPF:

    RawIsChanged = False
   #--- check dead channes and mark them as bad
    jb.picks.check_dead_channels(raw=raw)
        
   #--- start plot denoising orig raw psd, avoid reloading raw data
    if config.get("plot"):
       jplt = JuMEG_PLOT_PSD(n_plots=2,name="denoising",verbose=True)
       jplt.plot(raw,title="orig: " + os.path.basename(raw_fname),check_dead_channels=False,fmax=config.get("fmax"))
        
   #--- with redirect stdout/err
    with jumeg_logger.StreamLoggerSTD(label=label):
        #--- 1 nr low pass filter for freq below 5 hz
         if config.get("reflp"):
            raw = noise_reducer(None,raw=raw,reflp=config.get("reflp"),return_raw=True,verbose=config.get("verbose"),exclude_artifacts=False)
            RawIsChanged = True
        #--- 2 nr high pass filter
         if config.get("refhp"):
            raw = noise_reducer(None,raw=raw,reflp=config.get("refhp"),return_raw=True,verbose=config.get("verbose"),exclude_artifacts=False)
            RawIsChanged = True
        #--- 3  nr notch filter to remove power line noise
         if config.get("refnotch"):
            raw = noise_reducer(None,raw=raw,refnotch=config.get("refnotch"),fnout=None,return_raw=True,verbose=config.get("verbose"),exclude_artifacts=False)
            RawIsChanged = True
        
   #--- plot results, avoid reloading raw data
    if config.get("plot"):
       jplt.plot(raw,title="denoised: " + os.path.basename(fname_out),check_dead_channels=False,fmax=config.get("fmax"))
       if config.get("plot_show"):
          jplt.show()
       jplt.save(fname=fname_out,plot_dir=config.get("plor_dir","plots"))
    
    return fname_out,raw,RawIsChanged

#---------------------------------------------------
#--- apply_suggest_bads
#---------------------------------------------------
@JuMEG_PipelineFrame
def apply_suggest_bads(raw_fname=None,raw=None,config=None,label="suggest_bads",fname_out=None):
    """

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
   
     filename,raw-obj
    """
    with jumeg_logger.StreamLoggerSTD(label=label):
         marked,raw = suggest_bads(raw) #,**cfg["parameter"]) #show_raw=config.get("show_raw") )
         
   #--- set bads in global HDF
    fhdf = os.path.join( config.get("stage"),config.get("hdfname","badchannels.hdf5"))
    update_bads_in_hdf(fhdf=fhdf,bads=marked,fname=raw_fname,verbose=config.get("verbose"))
    
    return fname_out,raw,True

#---------------------------------------------------
#--- apply_interpolate_bads
#---------------------------------------------------
@JuMEG_PipelineFrame
def apply_interpolate_bads(raw_fname=None,raw=None,config=None,label="interpolate bads",fname_out=None):
    """

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj
    """
   #--- Interpolate bad channels using jumeg
    with jumeg_logger.StreamLoggerSTD(label=label):
         raw = jumeg_interpolate_bads(raw)
       #-- check results
         if config.get("plot_block"):
            raw.plot(block=config.get("plot_block"))

    return fname_out,raw,True


#---------------------------------------------------
#--- apply_ica
#---------------------------------------------------
# @JuMEG_PipelineFrame
def apply_ica(raw_fname=None,raw=None,config=None,label="ica",fname_out=None):
    """

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj,True
    """
 
    if not config.get("run"): return fname_out,raw
   
    jICA =JuMEG_PIPELINES_ICA()
    path = os.path.dirname(raw_fname)
    if config.get("do_fit",False):
       jICA.apply_fit(raw=raw,raw_fname=raw_fname,path=path,config=config)
   
   #--- here needs visual inspection
   
    if config.get("do_transform",False):
        jICA.apply_transfort(raw=raw,raw_fname=raw_fname,path=path,config=config)

    return fname_out,raw



#---------------------------------------------------
#--- apply_filter
#---------------------------------------------------
@JuMEG_PipelineFrame
def apply_filter(raw_fname,raw=None,config=None,label="filter",fname_out=None):
    """
    ToDo implement call to mne filter
    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj,True
    """
    
    return fname_out,raw,True

#---------------------------------------------------
#--- apply_resample
#---------------------------------------------------
@JuMEG_PipelineFrame
def apply_resample(raw_fname,raw=None,config=None,label="resample",fname_out=None):
    """
    ToDo implement call to mne resample
    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj
    """
    return fname_out,raw,True


