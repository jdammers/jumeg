#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 22:18:07 2016

@author: fboers
"""

from jumeg.jumeg_base                   import jumeg_base
from jumeg.epocher.jumeg_epocher        import JuMEG_Epocher  #jumeg_epocher
#from jumeg.epocher.jumeg_epocher_hdf    import JuMEG_Epocher_HDF
#from jumeg.epocher.jumeg_epocher_events import JuMEG_Epocher_Events

'''
Epocher CLS
read raw
find events      -> store in hdf
find eog and eog -> store in hdf
create epochs, avg for mne
plot epochs,avg


'''
class JuMEG_EpocherX(JuMEG_Epocher):
    def __init__(self):

        super(JuMEG_EpocherX, self).__init__()

        self.fif   = '203867_Chrono01_110615_1516_1_c,rfDC,bcc,nr,ar,m1-raw.fif'
        self.pfif  ='/home/fboers/MEGBoers/data/exp/Chrono/mne'
        self.fname = None #self.pfif.'/'.self.fif        
        self.raw   = None
        self.fnout = None
        self.fhdf  = None
        self.template_name = 'InKomp'
        self.experiment    = 'InKomp'
        
        self.verbose    = True        
        self.do_apply_events_to_hdf         = True
        self.do_apply_update_ecg_eog_to_hdf = True
        self.do_apply_events_export         = True
        
        self.epocher = {
          'template_name': 'InKomp',
          'fif_extention': '.fif',
          'do_run': True,
          'verbose': False,
          'save': True
        }

        self.events = {
                           'parameter':{
                                        'event_extention':'.eve',
                                        'weights':{'mode':'equal','method':'median','skipp_first':None},
                                        'save_condition':{'events':True,'epochs':True,'evoked':True},
                                        'time':{'time_pre':None,'time_post':None,'baseline':None},
                                        'exclude_events':{'eog_events':{'tmin':-0.4,'tmax':0.6} } },
                           'fif_postfix': 'evt',
                           'template_name':'InKomp',
                           'verbose':self.verbose
                           }
                           
      #--- artefacts from ECG EOG 
        self.ecg_events = None  # np.array([],dtype=np.int32)
        self.eog_events = None
                         
        self.ecg_parameter = {'num_events': 0,
                              'ch_name': 'ecg','thresh' :None,'explvar':None,
                              'freq_correlation':None,'performance':None}

        self.eog_parameter  = {'num_events': 0,
                               'ch_name':'eog','thresh' : None,'explvar':None,
                               'freq_correlation':None,'performance':None}
                   
 
 #----------------------------------------------------------------------------                          
 #--- run
 #----------------------------------------------------------------------------
    def run(self):
        print "---> Start JuMEG Epocher"
        self.raw,self.fname = jumeg_base.get_raw_obj(self.pfif+'/'+self.fif)
        print "---> FIF: " +self.fname
    #---
        if self.do_apply_events_to_hdf:
           self.apply_events_to_hdf(self.fname,**self.events)
    #---
          
    #--- save ecg & eog onsets in HDFobj
        if self.do_apply_update_ecg_eog_to_hdf:
   
    # TODO need ecg and eog onsets index points as numpy array 

           (self.fnout,self.raw,self.fhdf) = self.apply_update_ecg_eog(self.fnout,raw=self.raw,ecg_events=self.ecg_events,ecg_parameter=self.ecg_parameter,
                                                                eog_events=self.eog_events,eog_parameter=self.eog_parameter,template_name=self.template_name)
    #--- export events    
        if self.do_apply_events_export:

           (self.fnout,self.raw,self.fhdf) = self.apply_events_export_events(self.fname,raw=self.raw,**self.events)

           print "===> done JuMEG apply epocher export events data: " + str(self.fhdf) + "\n"

        
        print"\n---> Done JuMEg Epocher\n"
   
def jumeg_epocher_run():
    JEV = JuMEG_EpocherX()
    JEV.run()


    
if __name__ == "__main__":
   jumeg_epocher_run()
    