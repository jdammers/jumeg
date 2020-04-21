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
import sys,os,yaml,argparse,glob

import mne
#---
from jumeg.base                                      import jumeg_logger
from jumeg.base.jumeg_base                           import jumeg_base as jb
from jumeg.base.jumeg_badchannel_table               import update_bads_in_hdf
from jumeg.base.jumeg_base_config                    import JuMEG_CONFIG as jCFG
#---
from jumeg.base.pipelines.jumeg_pipelines_utils_base import get_args,JuMEG_PipelineFrame
from jumeg.base.pipelines.jumeg_pipelines_ica        import JuMEG_PIPELINES_ICA
from jumeg.base.pipelines.jumeg_pipelines_report     import JuMEG_REPORT
#---
from jumeg.base.plot.jumeg_base_plot_preproc         import JuMEG_PLOT_PSD
from jumeg.filter.jumeg_mne_filter                   import JuMEG_MNE_FILTER
#--- preproc
from jumeg.jumeg_noise_reducer     import noise_reducer
from jumeg.jumeg_suggest_bads      import suggest_bads
from jumeg.jumeg_interpolate_bads  import interpolate_bads as jumeg_interpolate_bads

logger = jumeg_logger.get_logger()

__version__= "2019.08.07.001"

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


    IN config
        noise_reducer:
          file_extention:
          - meeg-raw.fif
          - rfDC-empty.fif
          fmax: 300
          noiseref_hp:
          - RFG ...
          overwrite: false
          plot: true
          plot_dir: report
          plot_show: false
          postfix: nr
          refhp: 0.1
          reflp: 5.0
          refnotch:
          - 50.0
          - 100.0
          - 150.0
          - 200.0
          - 250.0
          - 300.0
          - 350.0
          - 400.0
          run: true
          save: true
     
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
       jplt = JuMEG_PLOT_PSD(n_plots=3,name="denoising",verbose=True) #,pick_types=["meg","ref"])
       jplt.plot(raw,color="green",title="REF: " + os.path.basename(raw_fname),check_dead_channels=False,fmax=config.get("fmax"),picks=jb.picks.ref_nobads(raw))
       jplt.plot(raw,color="blue",title="MEG orig: " + os.path.basename(raw_fname),check_dead_channels=False,fmax=config.get("fmax"),picks=jb.picks.meg_nobads(raw))
           #self.picks = jb.picks.meg_nobads(raw))
         
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
       jplt.plot(raw,title="MEG denoised: " + os.path.basename(fname_out),check_dead_channels=False,fmax=config.get("fmax"),picks=jb.picks.meg_nobads(raw))
       if config.get("plot_show"):
          jplt.show()
       fout = jplt.save(fname=fname_out,plot_dir=config.get("plot_dir","report"))
   
     #--- update image list in report-config for later update MNE Report
       CFG  = jCFG()
       data = None
       report_path   = os.path.dirname(fout)
       report_config = os.path.join(report_path,raw_fname.rsplit("_",1)[0]+"-report.yaml")

       if not CFG.load_cfg( fname=report_config ):
          data = {"noise_reducer":{ "files": os.path.basename(fout) } }
       else:
          CFG.config["noise_reducer"] = { "files": os.path.basename(fout) }
       CFG.save_cfg(fname=report_config,data=data)
   
    return fname_out,raw,RawIsChanged,None

#---------------------------------------------------
#--- apply_suggest_bads
#---------------------------------------------------
@JuMEG_PipelineFrame
def apply_suggest_bads(raw_fname=None,raw=None,config=None,label="suggest_bads",fname_out=None):
    """
    in config:
    suggest_bads:
      file_extention:
      - ',nr-raw.fif'
      - rfDC,nr-empty.fif
      fmax: 100
      overwrite: false
      parameter:
        epoch_length: None
        fraction: 0.001
        sensitivity_psd: 95
        sensitivity_steps: 97
        show_raw: false
        summary_plot: false
        validation: true
      plot: false
      plot_show: false
      postfix: bcc
      run: true
      save: true

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
   
     filename,raw-obj
    """
    with jumeg_logger.StreamLoggerSTD(label=label):
         marked,raw = suggest_bads(raw) #,**cfg["parameter"]) #show_raw=config.get("show_raw") )
  
   #--- set bads in global HDF
   # fhdf = os.path.join(config.get("stage"),config.get("hdfname","badchannels.hdf5"))
   # update_bads_in_hdf(fhdf=fhdf,bads=marked,fname=raw_fname,verbose=config.get("verbose"))
   
    return fname_out,raw,True,None

#---------------------------------------------------
#--- apply_bads_to_hdf
#---------------------------------------------------
@JuMEG_PipelineFrame
def apply_bads_to_hdf(raw_fname=None,raw=None,config=None,label="bads_to_HDF"):
   """
    update ads in global HDF: e.g.:  $STAGE/<jumeg_badchannels.hdf5>
    in config
       bads_to_hdf:
         run: true
         hdfname: jumeg_badchannel_info.hdf

   :param raw_fname:
   :param raw:
   :param config:
   :param label:
   :param fname_out:
   :return:
     raw_fname,raw,True,None
   """
   fhdf = os.path.join(config.get("stage"),config.get("hdfname","badchannels.hdf5"))
   update_bads_in_hdf(fhdf=fhdf,bads=raw.info.get("bads"),fname=raw_fname,verbose=config.get("verbose"))
   
   return raw_fname,raw,True,None

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

    return fname_out,raw,True,None

#---------------------------------------------------
#--- apply_ica
#---------------------------------------------------
@JuMEG_PipelineFrame
def apply_ica(raw_fname=None,raw=None,path=None,config=None,label="ica",fname_out=None):
    """

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj,True
    """
 
    if not config.get("run"): return fname_out,raw
   
    jICA = JuMEG_PIPELINES_ICA()
    
    if not path:
       path = os.path.dirname(raw_fname)

    raw,raw_filtered_clean = jICA.run(raw=raw,raw_fname=raw_fname,path=path,config=config)
    raw_filtered_clean.close()
    
    fname_out = jb.get_raw_filename(raw)
  
    return fname_out,raw,True,None

#---------------------------------------------------
#--- apply_report
#---------------------------------------------------
def apply_report(stage=None,subject_id=None,experiment=None,path=None,fname=None,config=None):
    """
        
    :param stage:
    :param subject_id:
    :param fname:
    :param path
    
    :param config:  dict e.g.:  
    
        overwrite: true
        path: report
        run: true
        save: true
   
        ica:
           extention: ar.png
           run: true
        noise_reducer:
           extention: nr-raw.png
           run: true
    
    :return:
    """
    if config.get("run"):
       jReport = JuMEG_REPORT()
       jReport.run(stage=stage,subject_id=subject_id,experiment=experiment,path=path,fname=fname,config=config)
 
#---------------------------------------------------
#--- apply_filter
#---------------------------------------------------
@JuMEG_PipelineFrame
def apply_filter(raw_fname,raw=None,config=None,label="filter",fname_out=None):
    """
    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj,True
    """
    #--- ini MNE_Filter class
    jfi = JuMEG_MNE_FILTER( flow=config.get("flow"),fhigh=config.get("fhigh") )

    if not config.get("run"):
       return fname_out,raw,False,jfi.postfix
    
   
   #--- filter inplace ; update file name in raw
    jfi.apply(raw=raw,flow=config.get("flow"),fhigh=config.get("fhigh"),picks=None,save=False,
              verbose=config.get("verbose"),overwrite=config.get("overwrite"))
    
    fname_out = jb.get_raw_filename(raw)
    
   #--- add new postfix from filter e.g: fibp01.0-45.0
    return fname_out,raw,True,jfi.postfix

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
    return fname_out,raw,True,None


