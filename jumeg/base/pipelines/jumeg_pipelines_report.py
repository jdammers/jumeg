#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 16.12.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import os
from distutils.dir_util import mkpath

#import pandas as pd
#from PIL import Image

import warnings
import mne
from mne.report import Report

from jumeg.base.jumeg_base         import JUMEG_SLOTS
from jumeg.base.jumeg_base         import jumeg_base as jb

from jumeg.base.jumeg_base_config  import JuMEG_CONFIG as jCFG
from jumeg.base                    import jumeg_logger


__version__= "2020.05.05.001"

logger = jumeg_logger.get_logger()

class MNE_REPORT(JUMEG_SLOTS):
    """
    saving
     noise_reducer fig
     ica figs in HDF
     epocher figs

    and show in MNE Reports
    """
    __slots__ = { "postfix","html_extention","h5_extention","info_name","experiment","subject_id",
                  "title","open_browser","write_hdf5","raw_psd","image_format","overwrite","verbose","debug",
                  "_isOpen","_path","_fhdf","_MNE_REPORT"}
    
    def __init__(self,**kwargs):
        super().__init__()
        self._init()
        
        self._path        = "."
       #---
        self.image_format   = "png"
        self.postfix        = "report"
        self.html_extention = ".html"
        self.h5_extention   = ".h5"
        self.raw_psd        = False
        self.title          = "JuMEG Preprocessing"
        self.subject_id     = "0815"
        
        self._update_from_kwargs(**kwargs)
        
    @property
    def path(self):
        return self._path

    @path.setter
    def path(self,v):
        if v:
           self._path = jb.expandvars(v)
           mkpath(self._path,mode=0o770)
        else:
            self._path=v
    
    @property
    def fname(self): return self.experiment+"_"+self.subject_id+"-"+self.postfix
    
    @property
    def fullname(self):
        return os.path.join(self.path,self.fname)

    @property
    def html_name(self): return self.fullname + self.html_extention
    @property
    def hdf_name(self): return self.fullname + self.h5_extention
    
    @property
    def image_extention(self): return "."+self.image_format
    
    @property
    def isOpen(self): return self._isOpen

    @property
    def MNEreport(self): return self._MNE_REPORT
    
    @property
    def fhdf(self): return self._fhdf
    @fhdf.setter
    def fhdf(self,v):
        self._fhdf = jb.expandvars(v)
        mkpath(os.path.dirname(self._fhdf),mode=0o770)
    
    def _update_from_kwargs(self,**kwargs):
        super()._update_from_kwargs(**kwargs)
        if "path" in kwargs:
           self.path = kwargs.get("path",self._path)
        
    def open(self,**kwargs):
        """

        :param kwargs:
        :return:
        """
        self._update_from_kwargs(**kwargs)
        self._isOpen = False
        # logger.info("report path (stage): {}".format(self.fullname))
       
        if self.overwrite:
           try:
              for fext in [ self.h5_extention,self.html_extention ]:
                  fname = self.fullname+fext
                  if os.path.isfile(fname):
                     os.remove(self.fname)
           except:
              logger.exception("ERROR: can not overwrite report ile: {}".format(self.fullname))
              return False
       
       # https://mne.tools/dev/auto_tutorials/misc/plot_report.html
        try:
            if os.path.isfile(self.hdf_name):
               self._MNE_REPORT = mne.open_report(self.hdf_name)
            else:
               self._MNE_REPORT = mne.Report(info_fname=self.info_name,title=self.title,image_format=self.image_format,
                                             raw_psd=self.raw_psd,verbose=self.verbose)
               logger.info("open Report (h5): \n   -> {}".format(self._MNE_REPORT))
               
            self._isOpen = True
        except:
            logger.exception("ERROR: can not open or create MNE Report {}".format(self.hdf_name))
        return self._isOpen

    def save(self,overwrite=True):
        if not self.isOpen:
            logger.exception("ERROR in saving JuMEG MNE report: {}\n ---> Report not open\n".format(self.fullname))
            return self.isOpen
        
        # mkpath( self.path,mode=0o770)
      
        if overwrite:
           if os.path.isfile(self.html_name):
              os.remove(self.html_name)
           if os.path.isfile(self.hdf_name):
              os.remove(self.hdf_name)
       #--- html
        self.MNEreport.save(self.html_name,overwrite=overwrite,open_browser=self.open_browser)
        logger.info("DONE saving JuMEG MNE report [overwrite: {}] : HTML: {}\n".format(overwrite,self.html_name))
       #--- h5
        self.MNEreport.save( self.hdf_name, overwrite=overwrite,open_browser=False)
        logger.info("DONE saving JuMEG MNE report [overwrite: {}] : HDF5: {}\n".format(overwrite,self.hdf_name))
     
        return self.isOpen
    
    def  update_report(self,path=None,data=None,section=None,prefix=None,replace=True):
        """
        load img from list
        add to report
        :param path   : report image path
        :param data   : dict or list of pngs
        :param section: section in report e.g.: Noise reducer, ICA
        :param prefix : prefix for caption e.g.: ICA-0815_
        :return:
        """
        if not data: return False
 
        fimgs    = []
        captions = []
        section  = section
    
        if isinstance(data,(dict)):
           for k in data.keys():
               files    = []
               fimgs    = []
               captions = []
               section  = k
               
               if data.get(k):
                  files.extend( data.get(k) )
               for f in files:
                   fimgs.append( os.path.join(path,f.rstrip()) )
                   captions.append( prefix+"-" + os.path.basename(f.rstrip().rsplit(self.image_extention,1)[0]) )
               if self.debug:
                  logger.debug("update MNE report: {}\n counts: {} ->\n {}".format(section,len(fimgs),fimgs) )
               self.MNEreport.add_images_to_section(fimgs,captions=captions,section=section,replace=replace)
                   
           return True
        
        if isinstance(data,(list)):#
           for f in data:
               fimgs.append( os.path.join(path,f.rstrip()) )
               captions.append( prefix+"-" + os.path.basename(f.rstrip().rsplit(self.image_extention,1)[0]) )
        else:
           fimgs.append(os.path.join(path,data.rstrip()))
           captions.append(prefix + "-" + os.path.basename(data.rstrip().rsplit(self.image_extention,1)[0]))
        if self.debug:
           logger.debug("update MNE report: {}\n counts: {} ->\n {}".format(section,len(fimgs),fimgs))
        self.MNEreport.add_images_to_section(fimgs,captions=captions,section=section,replace=replace)

        if self.verbose:
           logger.info("update MNE report: {}\n counts: {} ->\n {}".format(section,len(fimgs),fimgs) )

        return True

class JuMEG_REPORT(JUMEG_SLOTS):
    """
    saving
     noise_reducer fig
     ica figs in HDF
     epocher figs
    
    and show in MNE Reports
    """
    __slots__= {"stage","_path","fname","report_cfg_extention","_REPORT_CFG","_REPORT","_jumeg_cfg"}
    
    def __init__(self,**kwargs):
        super().__init__()
        self.init()
        self._jumeg_cfg = {"run":True,"save":True,"overwrite":False,"path":"report",
                           "noise_reducer":{"run":True,"extention":"nr-raw.png"},
                           "ica":{"run":True,"extention":"ar.png"}}
        self._path = "."
        self.report_cfg_extention = "-report.yaml"
        self._REPORT = MNE_REPORT(**kwargs)
        self._REPORT_CFG = jCFG()
        
    @property
    def jumeg_cfg(self): return self._jumeg_cfg

    @property
    def report_cfg(self): return self._REPORT_CFG

    @property
    def Report(self):
        return self._REPORT

    @property
    def verbose(self): return self._REPORT.verbose
    @verbose.setter
    def verbose(self,v):
        self._REPORT.verbose=v
    
    @property
    def path(self): return self._path
    @path.setter
    def path(self,v):
        pext = self._jumeg_cfg.get("path","report")
        if not v.endswith(pext):
           self._path = jb.expandvars( os.path.join(v,pext) )
        else:
           self._path = jb.expandvars(v)
      
    def update_report_cfg(self):
       #--- ['0815_TEST_20200412_1001_3', 'c']
        f    = self.fname.split(",",1)[0].rsplit("_",1)[0] + self.report_cfg_extention
        fcfg = os.path.join(self.path,f)
        self._REPORT_CFG.update(config=fcfg)
        
    def run(self,**kwargs):
        """
        :param kwargs:
        :return:
        stage=jpl.stage,subject_id=subject_id,fname=raw_fname,config=jpl.config.get("report")
        jReport.run(path=report_path,report_config_file=report_config_file,
                    experiment="QUARTERS",subject_id=210857,config=config)
            
        open/read cfg /reprt/fname-report.yaml
        MNEreport.open
         update NR
         update ICa
         save HDf5
         save htlm
         
        report config as dict
        report_path=report_path,path=path,fname=raw_fname,subject_id=210857,config=config
       
        report:
         run: True
         save: True
         overwrite: False
         noise_reducer:
          run: True
         ica:
          run: True
        """
        # logger.info(kwargs)
        self._update_from_kwargs(**kwargs)
        self.Report._update_from_kwargs(**kwargs)
        
       #--- try from jumeg config <report>  config=config.get("report")
        if "config" in kwargs:
           self._jumeg_cfg = kwargs.get("config")
        
        #logger.info("jumeg config report: {}".format( self.jumeg_cfg))
        
       #--- update from kwargs
        self.stage = jb.expandvars(self.stage)
        self.path  = kwargs.get("path",self.path) #--- report image path / image yaml file
        
        if self.stage.endswith("mne"):
           rp = self.stage.replace("mne","report")
        else:
           rp = self.stage
       #--- setup & open MNE report
        if not self.Report.open(path=rp):  return False
      
       #--- report image config
        self.update_report_cfg()
        
       #--- noise reducer
        if self.jumeg_cfg.get("noise_reducer",False):
           cfg = self.report_cfg.GetDataDict("noise_reducer")
           if cfg:
              self.Report.update_report(data=cfg.get("files"), path=self.path,section="Noise Reducer",prefix="NR")
       #--- ica
        if self.jumeg_cfg.get("ica",False):
           cfg = self.report_cfg.GetDataDict("ica")
           if cfg:
              self.Report.update_report(data=cfg,path=self.path,section="ICA",prefix="ICA")
       #--- save
        if self.jumeg_cfg.get("save",False):
           self.Report.save(overwrite=True)
     
if __name__ == "__main__":
  # jumeg_logger.setup_script_logging(logger=logger)
   pass