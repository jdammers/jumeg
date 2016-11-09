#import os
import numpy as np
#import pandas as pd

from jumeg.jumeg_base import jumeg_base
from jumeg.epocher.jumeg_epocher_ctps import JuMEG_Epocher_CTPS

class JuMEG_Epocher(JuMEG_Epocher_CTPS):
    def __init__ (self,template_name="DEFAULT",do_run=False,do_average=False,verbose=False,save=False):

        super(JuMEG_Epocher, self).__init__()

       #--- flags from base-basic cls
        self.do_run        = do_run
        self.do_average    = do_average
        self.do_save       = save
        self.verbose       = verbose

        self.template_name = template_name

       #--- init CLS Events->HDF file name & extention
        self.hdf_postfix      = '-epocher.hdf5'
        self.hdf_stat_postfix = '-epocher-stats.csv'

#---
    def apply_events_to_hdf(self, fname,raw=None,condition_list=None,picks=None,**kwargv):
        """
         find stimulus and/or response events for each condition; save to hdf5 format
        """

        if kwargv['template_name']:
           self.template_name = kwargv['template_name']

        if kwargv['verbose']:
           self.verbose = kwargv['verbose']

        self.template_update_file()

        fhdf = None
        raw,fname = jumeg_base.get_raw_obj(fname,raw=raw)
        
        fhdf = self.events_store_to_hdf(raw,condition_list=condition_list)

        print "===> DONE  apply epoches to HDF: " + fhdf +"\n"

        return (fname,raw,fhdf)
        
#---
    def apply_events_export_events(self,fname,raw=None,condition_list=None,picks=None,**kwargv):
        """
         export events and epochs for condition into mne fif data
         
        """

        if kwargv['template_name']:
           self.template_name = kwargv['template_name']

        if kwargv['verbose']:
           self.verbose = kwargv['verbose']

        # self.template_update_file()

        fhdf      = None
        raw,fname = jumeg_base.get_raw_obj(fname,raw=raw)        
        evt_ids   = self.events_export_events(raw=raw,fhdf=fhdf,condition_list=condition_list,**kwargv['parameter'])
        
        print "===> DONE apply events export events: " + fname +"\n"

        return (fname,raw,evt_ids)

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

        import pandas as pd

        if template_name:
           self.template_name = template_name


       #--- open HDFobj
        HDFobj = self.hdf_obj_open(fname=fname,raw=raw)
        fhdf   = HDFobj.filename


       #--- update ecg events
        self.hdf_obj_update_dataframe(pd.DataFrame( {'onset' : ecg_events}).astype(np.int32),key='/ocarta/ecg_events',param=ecg_parameter)

       #--- update eog events
        self.hdf_obj_update_dataframe(pd.DataFrame( {'onset' : eog_events}).astype(np.int32),key='/ocarta/eog_events',param=eog_parameter)
        HDFobj.close()

        return (fname,raw,fhdf)


    def apply_update_brain_responses(self,fname,raw=None,fname_ica=None,ica_raw=None,condition_list=None,**kwargv):
          """

          :param fname:
          :param kwargv:
          :return:
          """

          fhdr = None
         # fhdr = self.ctps_ica_brain_responses_update(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,condition_list=condition_list,**kwargv)


          # TODO: select ICs via ctps-surogates / threshold
          #self.ctps_select_brain_response_ics(fhdr=fhdr)

         # if kwargv['do_update']:
          fhdr = self.ctps_ica_brain_responses_update(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,condition_list=condition_list,**kwargv)

          #if kwargv['do_select']:
          #   fhdr = self.ctps_ica_brain_responses_select(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,condition_list=condition_list,**kwargv)

          #if kwargv['do_clean']:
          #   fhdr = self.ctps_ica_brain_responses_clean(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,condition_list=condition_list,**kwargv)

          #self.ctps_brain_responses_average

          print"===> Done apply_update_brain_responses"

          return fname,raw,fhdr



jumeg_epocher = JuMEG_Epocher()
