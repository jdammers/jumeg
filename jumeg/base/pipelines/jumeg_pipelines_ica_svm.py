#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 13:12:26 2020
@author: nkampel
"""

import os,sys,argparse

import numpy as np
import pickle

import mne
#from mne.preprocessing import find_ecg_events, find_eog_events

from jumeg.base.jumeg_base   import jumeg_base as jb
from jumeg.base              import jumeg_logger

logger = jumeg_logger.get_logger()

__version__= "2020.04.22.001"


#--- SVM ICA
class JuMEG_ICA_SVM():
    def __init__(self,**kwargs):
        self._raw        = None
        self._fname      = None
        self.picks       = None
        
        self.ICA            = None
        self._ics_found     = None
        self._ics_found_svm = None
        
        self._models     = None
        self._systemtype = "4d"
        self._isModelLoaded = False
        self.predict_proba = None
        self.model_name  = "all_included_model.pckl"
        self.model_path  = os.path.dirname(__file__)
        self.systemtypes = ['4d','nmag','ctf']
       #---
        self.do_copy = True
       #---
        self.do_crop = False
        self.tmin    = 0.
        self.tmax    = None
       #---
        self.sfreq  = 250
        self.n_jobs = 2
        self.l_freq = 2.
        self.h_freq = 50
        self.threshold    = 0.3
        self.n_components = 40
        self.method ='fastica'
      
    def update_from_kwargs(self,**kwargs):
        
        if kwargs.get("raw",None):
            self.raw = kwargs.get("raw")
        elif kwargs.get("fname",None): # full filepath
            self.raw,self._fname = jb.get_raw_obj(kwargs.get("fname"),raw=None)

        self.do_crop = kwargs.get("do_crop",self.do_crop)
        self.do_copy = kwargs.get("do_copy",self.do_copy)
        self.tmin    = kwargs.get("tmin",   self.tmin)
        self.tmax    = kwargs.get("tmax",   self.tmax)
       #---
        self.sfreq        = kwargs.get("sfreq", self.sfreq)
        self.n_jobs       = kwargs.get("n_jobs",self.n_jobs)
        self.l_freq       = kwargs.get("l_freq",self.l_freq)
        self.h_freq       = kwargs.get("h_freq",self.h_freq)
        self.threshold    = kwargs.get("threshold",self.threshold)
        self.n_components = kwargs.get("n_components",self.n_components)
        self.method       = kwargs.get("method",self.method)
        self.picks        = kwargs.get("picks",self.picks)
        self.ICA          = kwargs.get("ICA",self.ICA)

    @property
    def ICsMNE(self):
        return self._ics_found

    @property
    def ICsSVM(self):
        return self._ics_found_svm

    @property
    def systemtype(self): return self._systemtype
    @systemtype.setter
    def systemtype(self,type):
        if type in self.systemtypes:
           self._systemtype=type
           
    @property
    def isModelLoaded(self): return self._isModelLoaded
    
    @property
    def modelfile(self):
        return os.path.join(self.model_path,self.model_name)
    
    @property
    def classifier(self):
        if not self._isModelLoaded:
           self.load_model()
        return self._models[self.systemtype]

    def load_model(self):
        """
        load the SVM model only once
        :return:
        """
        self._isModelLoaded = False
        self._models        = pickle.load(open(self.modelfile,"rb"))
        self._isModelLoaded = True

    def _crop_raw(self,**kwargs):
        if self.do_crop:
            raw_c = self.raw.copy().crop(tmin=self.tmin,tmax=self.tmax)
        elif self.do_copy:
            raw_c = self.raw.copy()
        else:
            raw_c = self.raw
    
        return raw_c
    
    def _get_ica_artifacts_compatible(self,**kwargs):
        
        """Artifact removal using the mne ICA-decomposition
        for common MEG systems (neuromag, ctf, 4d).
        Artifactual components are being indentified by a support vector
        machine. Model trained on a 4d dataset composed on 48 Subjects.

        Compatibility to ctf and neuromag systems is established via
        3d interpolation of the
        magnetometer sensordata to the  4d system

        Parameters
        ----------
        raw : .fif
            a raw .fif file
        modelfile :  object
            an skit-learn classifier object 'download from jumeg...'
        systemtype : str
            the system type of the raw file (4d, nmag, ctf)
        thres : float (from 0. to 1.)
            higher is less sensitive to artifact components,
            lower is more sensitive to artifact components,
            defaults to 0.8.
        ica_parameters:
            standard parameters for the mne ica function

        """
        self.update_from_kwargs(**kwargs)
        self._ics_found     = None
        self._ics_found_svm = None
        
        self.predict_proba = None
        
      #--- we are fitting the ica on filtered copy of the original
        raw_c = self._crop_raw()
        raw_c.resample(sfreq=self.sfreq,n_jobs=self.n_jobs)
        
        if raw_c.info["bads"]:
           logger.info("SVM => interpolating bads")
           raw_c.interpolate_bads()  # to get the dimesions right
        
        logger.info("SVM => filtering RAW data")
        raw_c.filter(l_freq=self.l_freq,h_freq=self.h_freq)
        
        if not self.ICA:
           if not self.picks:
              self.picks = jb.picks.meg_nobads(raw_c)
           logger.info("SVM => calculating & fitting ICA")
           self.ICA = mne.preprocessing.ICA(n_components=self.n_components,method=self.method)
           self.ICA.fit(raw_c,picks=self.picks)  # ,start=0.0,stop=20.0)
        
        self._ics_found = list(set( self.ICA.exclude ))
        self._ics_found.sort()
        
       #--- get topo-maps
        logger.info("SVM => start classifier with topos")
        topos = np.dot(self.ICA.mixing_matrix_[:,:].T,self.ICA.pca_components_[:self.ICA.n_components_])
       #--- compatibility section -- 3d interpolation to 4d sensorlayout
        self.predict_proba = self.classifier.predict_proba(topos)
        art_inds = np.where( np.amax( self.predict_proba[:,0:2],axis=1 ) > self.threshold)[0]
       
       #--- artifact annotation
        self.ICA.exclude = list(set(art_inds))
        # self._ics_found_svm = [item for item in self.ICA.exclude if item not in self._ics_found ]
        self._ics_found_svm = self.ICA.exclude
        self._ics_found_svm.sort()
        logger.debug("SVM ICs found:\n  -> ICs before: {}\n  -> ICs SVM: {}\  -> ICs excluded: {}".
                    format(self._ics_found,self._ics_found_svm,self.ICA.exclude) )
        logger.info("done SVM")
        
        return self.ICA,self.predict_proba
    
    def clear(self):
        self.raw           = None
        self.ICA           = None
        self.predict_proba = None
        self.picks         = None
        self._ics_found     = None
        self._ics_found_svm = None
        
    def run(self,**kwargs):
        """
        raw
        ICA
        picks
        :param kwargs:
        :return:
        """
        self._get_ica_artifacts_compatible(**kwargs)
       #--- plot results
       # ica.plot_sources(self.raw)
      
        return self.ICA,self.predict_proba

def run(opt):
    jumeg_logger.setup_script_logging(logger=logger)
    logger.info("JuMEG SVM ICA mne-version: {}".format(mne.__version__))
    
    from jumeg.base.pipelines.jumeg_pipelines_ica_perfromance import JuMEG_ICA_PERFORMANCE
    jIP = JuMEG_ICA_PERFORMANCE()

  #--- init raw
    raw = None
    if opt.stage:    
       path = opt.stage
    if opt.path:
       path = os.path.join(path,opt.path)
   #--- init SVM class & run : load raw obj & apply SVM for ICs
    jSVM = JuMEG_ICA_SVM()
    fname= os.path.join(path,opt.fraw) 
    ica, predict_proba = jSVM.run( fname= fname )
    raw = jSVM.raw
     
   #--- raw cleaned
    raw_clean = ica.apply(raw.copy() )
  
    #raw.plot(block=True)
  
   #--- prepare performance plot
   #--- find ECG
    jIP.ECG.find_events(raw=raw)
   #--- find EOG in raw
    annotations = jIP.EOG.find_events(raw=raw)
    raw.set_annotations(annotations)
   #---  
    fout = fname.rsplit("-",1)[0] + "-ar"
    jIP.plot(raw=raw,raw_clean=raw_clean,verbose=opt.verbose,plot_path=path,fout=fout)

    jIP.Plot.figure.show()
    
    logger.info("DONE JuMEG SVM ICA")


def get_args(argv):
    """
    get args using argparse.ArgumentParser ArgumentParser
    e.g: argparse  https://docs.python.org/3/library/argparse.html
            
    Results:
    --------
    parser.parse_args(), parser
        
    """    
    info_global = """
                  JuMEG SVM

                  ---> merge eeg data with meg fif file 
                  jumeg_svm -f <xyz.fif> -p <path to fif file.> -v -d
                  """
            
   #--- parser
    parser = argparse.ArgumentParser(info_global)
 
   # ---meg input files
    parser.add_argument("-f", "--fraw",help="fif file")
    parser.add_argument("-p", "--path",help="path to fif file", default=".")
    parser.add_argument("-s", "--stage",help="stage path to fif file", default=".")
    parser.add_argument("-v", "--verbose", action="store_true",help="verbose mode")
    parser.add_argument("-d", "--debug",   action="store_true",help="debug mode")
  
   #---- ck if flag is set in argv as True
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


if __name__ == "__main__":

     
   if (len(sys.argv) < 1):
      parser.print_help()
      sys.exit()
   
   opt, parser = get_args(sys.argv)

   if opt.debug:
      opt.stage   = "/data/MEG/meg_store2/exp/JUMEGTest/mne/"
      opt.fraw    = "206720_MEG94T0T2_130820_1335_1_c,rfDC,meeg,nr,bcc,int-raw.fif"
      opt.path    = "206720/MEG94T0T2/130820_1335/1"
      opt.verbose = True

   run(opt)
  
