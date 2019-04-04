#!/usr/bin/env python3
# -+-coding: utf-8 -+-

'''
Class JuMEG_Epocher
-> read template file
-> extract event/epoch information and save to hdf5

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de>
#
#--------------------------------------------
# Date: 21.11.18
#--------------------------------------------
# License: BSD (3-clause)
#--------------------------------------------
# Updates
# 18.12.2018 use python3 logging instead of print()
#--------------------------------------------
extract mne-events per condition, save to HDF5 file

Example:
--------
   
from jumeg.epocher.jumeg_epocher import jumeg_epocher

template_name = 'M100'
template_path = "${JUMEG_PATH_TEMPLATE_EPOCHER}"
evt_param = { "condition_list":condition_list,
              "template_path": template_path, 
              "template_name": template_name,
              "verbose"      : verbose
            }
           
(_,raw,epocher_hdf_fname) = jumeg_epocher.apply_events(fname,raw=raw,**evt_param)


ep_param={
          "condition_list": condition_list,
          "template_path" : template_path, 
          "template_name" : template_name,
          "verbose"       : verbose,
          "parameter":{
                       "event_extention": ".eve",
                       "save_condition":{"events":True,"epochs":True,"evoked":True}
        }}  

event_ids = jumeg_epocher.apply_epochs(fname=fname,raw=raw,**ep_param)

'''

import os,sys,argparse
#import numpy as np
#import pandas as pd

import logging
logger = logging.getLogger("root")

from jumeg.epocher.jumeg_epocher_epochs import JuMEG_Epocher_Epochs

__version__= "2019.04.04.001"

class JuMEG_Epocher(JuMEG_Epocher_Epochs):
    def __init__ (self,template_path=None,template_name="DEFAULT",verbose=False):

        super(JuMEG_Epocher, self).__init__()
        
        if template_path:
           self.template_path = template_path
        if template_name:   
           self.template_name = template_name
    
       #--- flags from base-basic cls
        self.verbose  = verbose
    
#---
    def apply_events(self,fname,raw=None,**kwargs):
        """
        find stimulus and/or response events for each condition; save to hdf5 format
        
        Parameter:
        ----------    
        string : fif file name <None>
        raw obj: <None>
        list   : list of conditions defined in epocher template file
        verbose: <False>
        template_path : path to jumeg epocher template
        template_name : template name 
        
        Results:
        --------    
        raw obj 
        string: file name 
        string: HDF name
        """
        #print("LOGlevel")
       # print( logger.getEffectiveLevel() )
        logger.info( "---> START  apply events to HDF  fname: {}\n".format(fname))
        try:
            raw,fname = self.events_store_to_hdf(fname=fname,raw=raw,**kwargs)
        except:
            logger.exception( "---> apply_events:\n  -> parameter:\n" +self.pp_list2str(kwargs))
            sys.exit()
        logger.info( "---> DONE  apply events to HDF: {}\n".format(self.hdf_filename))
        
        return (raw,fname)
    
#---
    def apply_epochs(self,fname,raw=None,**kwargs):
        """
        extract events,epochs,averages with or without baseline correction from HDF obj
       
        Parameter:
        ----------    
        string : fif file name <None>
        raw obj: <None>
        list   : list of conditions defined in epocher template file
        verbose: <False>
        template_path : path to jumeg epocher template
        template_name : template name 
        
        Results:
        --------    
        raw obj
        string: file name       
        string: HDF file name       
        """
        
        logger.info("---> START apply epocher => fname   : {}\n".format(fname))
        
        try:
            raw,fname = self.apply_hdf_to_epochs(fname=fname,raw=raw,**kwargs)
        except:
            logger.exception( "---> apply_epochs:\n  -> parameter:\n" +self.pp_list2str(kwargs))
            sys.exit()
        logger.info("---> DONE apply epocher  => hdf name: {}\n".format(self.hdf_filename))
        return (raw,fname)
     
        
    
 #---ToDo update for ecg eog onsets   
    
#---
    def apply_update_ecg_eog(self,fname,raw=None,ecg_events=None,ecg_parameter=None,eog_events=None,eog_parameter=None,template_name=None):
        """
        store ecg and eog parameter (e.g.output from ocarta) in HDF epocher-object/file

        input:
        ecg_events = ocarta.idx_R_peak[:,0]
        ecg_parameter={ 'num_events': ocarta.idx_R_peak[:,0].size,
                       'ch_name':  ocarta.name_ecg,'thresh' : ocarta.thresh_ecg,'explvar':ocarta.explVar,
                       'freq_correlation':None,'performanse':None}

        eog_events= ocarta.idx_eye_peak[:,0
        eog_parameter={ 'num_events': ocarta.idx_eye_peak[:,0].size,
                       'ch_name':  ocarta.name_eog,'thresh' : ocarta.thresh_eog,'explvar':ocarta.explVar,
                       'freq_correlation':None,'performanse':None}


        return:
               fname,raw,fhdf
        """
       #--- TODO: ck for HDFobj open / close; may use self.HDFobj

        if template_name:
           self.template_name = template_name
           
       #--- open HDFobj
        HDFobj = self.hdf_obj_open(fname=fname,raw=raw)
        fhdf   = HDFobj.filename


       #--- update ecg events
        if ecg_events.size:
           self.hdf_obj_update_dataframe(pd.DataFrame( {'onset' : ecg_events}).astype(np.int32),key='/ocarta/ecg_events',param=ecg_parameter)

       #--- update eog events
        if eog_events.size:        
           self.hdf_obj_update_dataframe(pd.DataFrame( {'onset' : eog_events}).astype(np.int32),key='/ocarta/eog_events',param=eog_parameter)
        
        HDFobj.close()

        return (fname,raw,fhdf)


jumeg_epocher = JuMEG_Epocher()

'''
{
"default":{
           "version"          :"2018-06-19-001",
           "experiment"       : "FreeView",
           "postfix"          : "test",
           "time_pre"         : -0.2,
           "time_post"        :  0.4,
        
           "baseline" :{"method":"mean","type_input":"iod_onset","baseline": [null,0]},
           
           "marker"   :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"sac_onset","prefix":"iod","type_result":"hit"},
           "response" :{"matching":true,"channel":"ETevents","type_input":"sac_onset","type_offset":"sac_offset","prefix":"sac"},
           
           "iod"      :{"marker"  :{"channel":"StimImageOnset","type_input":"img_onset","prefix":"img"},
                        "response":{"matching":true,"channel":"IOD","type_input":"iod_onset","type_offset":"iod_onset","prefix":"iod"}},

           "reject"   : {"mag": 5e-9},
          
           "ETevents":{
                       "events":{
                                  "stim_channel"   : "ET_events",
                                  "output"         : "onset",
                                  "consecutive"    : true,
                                  "min_duration"   : 0.0005,
                                  "shortest_event" : 1,
                                  "mask"           : null,
                                  "initial_event"  : true
                                 },
                        "and_mask"          : null,
                        "event_id"          : null,
                        "window"            : [0.02,6.0],
                        "counts"            : "all",
                        "system_delay_ms"   : 0.0,
                        "early_ids_to_ignore" : "all"
                        
                       },
                     
           "StimImageOnset":{
                       "events":{
                                  "stim_channel"   : "STI 014",
                                  "output"         : "onset",
                                  "consecutive"    : true,
                                  "min_duration"   : 0.0005,
                                  "shortest_event" : 1,
                                  "mask"           : null,
                                  "initial_event"  : true

                                 },
                        
                        "event_id"           : 84,
                        "and_mask"           : 255,
                        "system_delay_ms"    : 0.0,
                        "early_ids_to_ignore" : null
                        },

            "IOD":{
                        "events":{
                                  "stim_channel"   : "STI 013",
                                  "output"         : "onset",
                                  "consecutive"    : true,
                                  "min_duration"   : 0.0005,
                                  "shortest_event" : 1,
                                  "mask"           : null,
                                  "initial_event"  : true

                                 },
                        
                        "window"               : [0.0,0.2],
                        "counts"               : "first",
                        "system_delay_ms"      : 0.0,
                        "early_ids_to_ignore"  : null,
                        "event_id"             : 128,
                        "and_mask"             : 255
                       }
              },

"ImoIODBc":{
         "postfix"   : "ImoIOD",
         "info"      : "all, image onset FV,ME,SE, iod onset, with baseline correction",
        
         "marker"    : {"channel":"StimImageOnset","type_input":"iod_onset","type_output":"iod_onset","prefix":"iod","type_result":"hit"},
         "response"  : {"matching":false},
       
         "StimImageOnset": {"event_id":"74,84,94"},
         "IOD"           : {"event_id":128}
         },
       
"FVImoBc":{
         "postfix"        : "FVimo",
         "info"           : "freeviewing, image onset, iod onset, baseline correction",
         "marker"         : {"channel":"StimImageOnset","type_input":"iod_onset","type_output":"iod_onset","prefix":"iod","type_result":"hit"},
       
         "response"       : {"matching":false},
         "StimImageOnset" : {"event_id":74},
         "IOD"            : {"event_id":128}
         },

"FVsaccadeBc":{
         "postfix"   :"FVsac",
         "info"      :"freeviewing, saccade onset, baseline correction",
      
         "marker"   :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"sac_onset","prefix":"iod","type_result":"hit"},
         "response" :{"matching":true,"channel":"ETevents","type_input":"sac_onset","type_offset":"sac_offset","prefix":"sac"},
         
         "StimImageOnset"   : {"event_id":74},
         "ETevents"         : {"event_id":"250, 251, 252, 253, 254"}
         },
        
"FVfixationBc":{
         "postfix"   :"FVfix",
         "info"      :"freeviewing, fixation onset, baseline correction via response_channel=>StimImageOnset",
         
         "marker"   :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"fix_onset","prefix":"iod","type_result":"hit"},
         "response" :{"matching":true,"channel":"ETevents","type_input":"fix_onset","type_offset":"fix_offset","prefix":"fix"},
         
         "StimImageOnset"   : {"event_id":74},
         "ETevents"         : {"event_id":"260, 261"}
         },
      
         
"MEImoBc":{
         "postfix"  :"MEimo",
         "info"     :"memory, image onset, iod onset, baseline correction",
         "marker"   :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"iod_onset","prefix":"iod","type_result":"hit"},
         "response" :{"matching":false},
         
         "StimImageOnset"   : {"event_id":94},
         "IOD"              : {"event_id":128}
         },

"MEsaccadeBc":{
         "postfix"   :"MEsac",
         "info"      :"memory, saccade onset, baseline correction via response_channel=>StimImageOnset",
         "marker"   :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"sac_onset","prefix":"iod","type_result":"hit"},
         "response" :{"matching":true,"channel":"ETevents","type_input":"sac_onset","type_offset":"sac_offset","prefix":"sac"},
         
         "StimImageOnset"   : {"event_id":94},
         "ETevents"         : {"event_id":"250, 251, 252, 253, 254"}
         },
         
"MEfixationBc":{
         "postfix"  :"MEfix",
         "info"     :"memory, fixation onset, baseline correction via response_channel=>StimImageOnset",

         "marker"   : {"type_input":"iod_onset","type_output":"fix_onset"},
         "response" :{"matching":true,"channel":"ETevents","type_input":"fix_onset","type_offset":"fix_offset","prefix":"fix"},
      
         "StimImageOnset"   : {"event_id":94},
         "ETevents"         : {"event_id":"260, 261"}
         },

         
"SEImoBc":{
         "postfix"   :"SEimo",
         "info"      :"search, image onset, iod onset, baseline correction",
         
         "marker"   :{"type_input":"iod_onset","type_output":"iod_onset"},
         "response" :{"matching":false},
        
         "StimImageOnset"   : {"event_id":84},
         "IOD"              : {"event_id":128}
         },

"SEsaccadeBc":{
         "postfix"   :"SEsac",
         "info"      :"search, saccade onset, baseline correction via StimImageOnset",
        
         "marker"   :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"sac_onset","prefix":"iod","type_result":"hit"},
         "response" :{"matching":true,"channel":"ETevents","type_input":"sac_onset","type_offset":"sac_offset","prefix":"sac"},
         
         "StimImageOnset"   : {"event_id":84},
         "ETevents"         : {"event_id":"250, 251, 252, 253, 254"}
         },
         
"SEfixationBc":{
         "postfix"   :"SEfix",
         "info"      :"search, fixation onset, baseline correction via StimImageOnset",

         "marker"   : {"type_input":"iod_onset","type_output":"fix_onset"},
         "response" :{"matching":true,"channel":"ETevents","type_input":"fix_onset","type_offset":"fix_offset","prefix":"fix"},
      
         "StimImageOnset"   : {"event_id":84},
         "ETevents"         : {"event_id":"260, 261"}
         },
         

"SACBc":{
         "postfix"   :"sac",
         "info"      :"saccade onset, baseline correction via StimImageOnset",
        
         "marker"    :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"sac_onset","prefix":"iod","type_result":"hit"},
         "response"  :{"matching":true,"channel":"ETevents","type_input":"sac_onset","type_offset":"sac_offset","prefix":"sac"},
 
         "StimImageOnset"   : {"event_id":"74,84,94"},
         "ETevents"         : {"event_id":"250, 251, 252, 253, 254"}
         },

"SACBcmne":{
         "postfix"   :"sacmne",
         "info"      :"fixation onset, baseline correction via StimImageOnset",
         "baseline"  :{"method":"mean","type_input":"sac_onset","baseline": [null,0]},
         "marker"    :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"sac_onset","prefix":"iod","type_result":"hit"},
         "response"  :{"matching":true,"channel":"ETevents","type_input":"sac_onset","type_offset":"sac_offset","prefix":"sac"},

         "StimImageOnset"   : {"event_id":"74,84,94"},
         "ETevents"         : {"event_id":"250, 251, 252, 253, 254"}
         },

"SAC":{
         "postfix"   :"sac",
         "info"      :"saccade onset, baseline correction via StimImageOnset",
         "baseline"  : null,
         "marker"    :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"sac_onset","prefix":"iod","type_result":"hit"},
         "response"  :{"matching":true,"channel":"ETevents","type_input":"sac_onset","type_offset":"sac_offset","prefix":"sac"},

         "StimImageOnset"   : {"event_id":"74,84,94"},
         "ETevents"         : {"event_id":"250, 251, 252, 253, 254"}
         },

"FIXBc":{
         "postfix"   :"fix",
         "info"      :"fixation onset, baseline correction via StimImageOnset",
        
         "marker"   :{"type_output":"fix_onset"},
         "response" :{"matching":true,"channel":"ETevents","type_input":"fix_onset","type_offset":"fix_offset","prefix":"fix"},
     
         "StimImageOnset": {"event_id":"74,84,94"},
         "ETevents"      : {"event_id":"260, 261"}
         },

"BlnkBc":{
         "postfix"   :"blnk",
         "info"      :"eye blink onset, baseline correction via StimImageOnset",

         "marker"    :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"blnk_onset","prefix":"iod","type_result":"hit"},
         "response" :{"matching":true,"channel":"ETevents","type_input":"blnk_onset","type_offset":"blnk_offset","prefix":"blnk"},

         "StimImageOnset": {"event_id":"74,84,94"},
         "ETevents"      : {"event_id":"280"}
         }

}

'''

