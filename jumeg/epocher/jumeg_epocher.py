'''Class JuMEG_Epocher

-> read template file 
-> extract event/epoch information and save to hdf5
-> 
Author:
         Frank Boers     <f.boers@fz-juelich.de>
----------------------------------------------------------------
extract mne-events per condition, save to HDF5 file

Example:
--------
   
from jumeg.epocher.jumeg_epocher import jumeg_epocher

template_name = 'M100'
template_path = "/data/templates/epocher/"
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

# import os,sys,argparse
import numpy as np
import pandas as pd

#from jumeg.jumeg_base import jumeg_base
from jumeg.epocher.jumeg_epocher_epochs import JuMEG_Epocher_Epochs

__version__="2018.06.19.001"

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
        raw,fname = self.events_store_to_hdf(fname=fname,raw=raw,**kwargs)

        print "===> DONE  apply events to HDF: " + self.hdf_filename +"\n"
        self.line()
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

        raw,fname = self.apply_hdf_to_epochs(fname=fname,raw=raw,**kwargs)

        print "===> DONE apply epocher: " + self.hdf_filename +"\n"
        self.line()
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


