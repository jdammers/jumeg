#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 13.12.18
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import os,sys,copy
from pubsub                             import pub
from jumeg.base.template.jumeg_template import JuMEG_Template

import logging
logger = logging.getLogger("jumeg")

__version__="2019.05.14.001"

__DEFAULT_EXPERIMENT_TEMPLATE__={
"info":{
          "time":None,
          "user": None,
          "version": None
         },
"experiment":{
              "name" : "default",
              #"ids"  : [], not used jet
              "scans": [],
              "stages":["${JUMEG_PATH_MNE_IMPORT}/exp","${JUMEG_PATH_MNE_IMPORT2}/exp","${JUMEG_PATH_LOCAL_DATA}/exp"],
              "bads_list":["MEG 010","MEG 142","MEG 156","RFM 011"], # MEG 007,MEG 042
              "segmentation":{
                              "path":{
                                      "mrdata"     : "mrdata",
                                      "dicom"      : "mrdata/dicom",
                                      "freesurfer" : "mrdata/freesurfer"
                                    }
                             },
              "path":{
                       "mne"        : "mne",
                       "eeg"        : "eeg",
                       "mft"        : "mft",
                       "doc"        : "doc",
                       "source"     : "source",
                       "stimuli"    : "stimuli"
                      }
 },

"bti_export": {
              "bti_path"          : ["${JUMEG_PATH_BTI_EXPORT}","${JUMEG_PATH_LOCAL_DATA}/megdaw_data21"],
              "pdf_name"          : "c,rfDC",
              "config_fname"      :"config",
              "head_shape_fname"  :"hs_file",
              "rotation_x"        : None,
              "translation"       : None,
              "ecg_ch"           : None,
              "eog_ch"           : None,
              "fif_extention"     :"-raw.fif",
              "emptyroom"         : "-empty.fif",
              "overwrite"         : False,
              "fakesHS"           : False
             },
}

'''
ToDo : setup BIDS and PrePorc
  "preprocessing": {
                    "meeg_merger": {},
                    "epocher": {},
                    "suggest_bads": {},
                    "noise_reducer": {},
                    "artifact_rejection": {},
                    "events": {}
                    }
'''


class JuMEG_Template_Experiments(JuMEG_Template):
    """
    class to work with <jumeg experiment templates>
    overwrite _init(**kwargs) for you settings

    Example
    -------
     from jumeg.template.jumeg_template import JuMEG_Template_Experiments

     class JuMEG_ExpTemplate(JuMEG_Template_Experiments):
        def __init__(self,**kwargs):
            super().__init__()

        def update_from_kwargs(self,**kwargs):
           self.template_path = kwargs.get("template_path",self.template_path)

        def _init(self,**kwargs):
            self.update_from_kwargs(**kwargs)

     TMP = JuMEG_ExpTemplate()
     print(TMP.template_path)

    """
    def __init__ (self,**kwargs):
        super().__init__()
        self.template_path    = os.getenv('JUMEG_PATH_TEMPLATE_EXPERIMENTS',self.template_path_default + '/jumeg_experiments')
        self.template_name    = 'default'
        self.template_postfix = 'jumeg_experiment_template'
        self._init(**kwargs)

    def _init(self,**kwargs):
        pass
    
class JuMEG_ExpTemplate(JuMEG_Template_Experiments):
   def __init__(self,**kwargs):
       super().__init__()
       
   @property
   def data(self): return self.template_data['experiment']

 #--- tmp path default
   @property
   def template_path_default(self):
       return os.getenv("JUMEG_PATH_TEMPLATE_EXPERIMENTS",os.getenv("JUMEG_PATH") + '/data/templates/jumeg_experiments')

   @property
   def bti_data(self):
       return self.template_data['bti_export']

   @property
   def name(self ): return self.data.get("name")
   @property
   def scans(self): return self.data.get("scans")
   @property
   def bads(self):  return self.data.get("bads_list")

   @property
   def stages(self):return self.data.get("stages")
   @stages.setter
   def stages(self,v):
       if isinstance(v,(list)):
          self.data['stages']=v
       else:
          self.data['stages'].append(v)
   
   @property
   def paths(self):return self.data.get('paths')

   @property
   def segmentation_paths(self):
       return self.data.get('paths',[])

   def update_from_kwargs( self, **kwargs ):
       self.template_path = kwargs.get("template_path", self.template_path)
       self._pubsub_error_msg = kwargs.get("pubsub_error_msg", "MAIN_FRAME.MSG.ERROR")

   def _init( self, **kwargs ):
       self.update_from_kwargs(**kwargs)

   def get_experiments(self,**kwargs):
       """
       find experimnet template files in  ${JUMEG_PATH}/templates/jumeg_experiment
       if no template file found:
          reset template data wit default parameter dict
          init experimnet name list with <default name>
       
       :param issorted: <True>
       :return: experiment template name list
       """
       exp = self.get_sorted_experiments(**kwargs)
       if exp:
          return exp
       self.template_data_reset()
       # print(self.template_name)
       return [ self.template_name ]
       
   def get_sorted_experiments(self,issorted=True,default_on_top=True):
       """
       :param issorted sort the list <True>
       :param default_on_top  list <default> first <True>
      
       Result
       -------
        sorted list of scans
       """
       exps = self.template_update_name_list()
       
       if issorted:
          exps = sorted( exps )
       if default_on_top:
          dname= __DEFAULT_EXPERIMENT_TEMPLATE__["experiment"]["name"]
          try:
              i = exps.index( dname )
          except:
              exps.append(dname)
              i = len(exps)-1
          
          a      = exps[0]
          exps[0]= exps[i]
          exps[i]= a
          
       return exps

   def get_sorted_scans(self,issorted=True):
       """
       :param issorted sort the list <True>
       Result
       -------
        sorted list of scans
       """
       try:
           if isinstance( self.scans, (list)):
              if issorted:
                 return sorted( self.scans )
              return self.scans
           else:
              return [ self.scans ]

       except:
           return []

   def template_check_experiment_data(self):
        """
        check's template for <experiment> structure e.g.:
        "experiment":{
              "name"  : experiment name,
              "scans" :[ list of scans],
              "stages":[ list of start dirs]
              }
        Result:
        -------
        True/False
        """
        error_msg=[]
        if not self.template_data:
           error_msg.append("No template data found : " + self.template_full_filename)
        elif not self.template_data.get('experiment',None):
           error_msg.append("No <experiment> structure found : "+self.template_name)
        else:
           exp = self.template_data.get('experiment',None)
           for k in["scans","stages"]:
               if not exp.get(k):
                  error_msg.append("No <{}>  found".format(k))
           if error_msg:
              error_msg.insert(0,"Checking Experiment Template")
              error_msg.append("Module  : "+sys._getframe(1).f_code.co_name )
              error_msg.append("Function: check_experiment_template_data")
              logger.error("\n".join(error_msg))
              pub.sendMessage(self._pubsub_error_msg,data=error_msg)
              return False
        return True

   def template_update(self,exp,path=None,verbose=False):
        """ update a JuMEG template

        Parameters
        ----------
         exp    : name of experiment
         path   : <None>
         verbose:<false>

        Result
        ------
         template data dict
        """
        
        self.template_name = exp
        if path:
           self.template_path = path
        self.verbose = verbose
        if not exp == "default":
           if not self.template_update_file(exit_on_error=False):
              return False
        if self.template_check_experiment_data():
           return self.template_data
        return False

   def template_data_reset(self):
       """
       reset <default> template to parameter defined on to of this python script
       eg if no experiment template exists
       :return:
       """
       self.template_data = {}
       self.template_data = copy.deepcopy(__DEFAULT_EXPERIMENT_TEMPLATE__)
       self.template_name = self.template_data["experiment"]["name"]
    
   

experiment = JuMEG_Template_Experiments()