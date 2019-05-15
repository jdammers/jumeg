# -*- coding: utf-8 -*-
"""
Created on 08.06.2018

@author: fboers
"""

import os,os.path,logging
import numpy as np
import matplotlib.pyplot as pl
from   matplotlib.backends.backend_pdf import PdfPages

import mne
from jumeg.base.jumeg_base import JuMEG_Base_IO

logger = logging.getLogger('jumeg')
__version__="2019.05.14.001"

#--- A4 landscape
pl.rc('figure', figsize=(11.69,8.27))
pl.rcParams.update({'font.size': 8})


class JuMEG_Epocher_Plot(JuMEG_Base_IO):
    def __init__ (self,raw=None):
        super(JuMEG_Epocher_Plot, self).__init__()
        self.raw = raw
        self.dpi = 100
        self.file_extention = '.png'
    
    def minmax(self,d):
        ymin, ymax = d.min(),d.max()
        ymin -= np.abs(ymin) * 0.1 #factor * 1.1
        ymax += np.abs(ymax) * 0.1 #factor * 1.1
      
        return ymin,ymax
        
    def plot_evoked(self,evt,fname=None,save_plot=True,show_plot=False,condition=None,plot_dir=None,
                    info={'meg':{'scale':1e15,'unit':'fT'},'eeg':{'scale':1e3,'unit':'mV'}}):
        '''
        
        :param evt:
        
         event dictionary
         evt['events']  : <np.array([])>  from mne.find_events
         evt['event_id']: <None> list of event ids
         evt['baseline_corrected']: True/False
         baseline:
         evt['bc']['events']   = np.array([])
         evt['bc']['event_id'] = None
  
        :param fname:
        :param save_plot:
        :param show_plot:
        :param condition:
        :param plot_dir:
        :param info:
        
        plot subplots evoked/average
        MEG
        ECG/EOG + performance
        STIM Trigger/Response
        events, rt mean median min max
        
        :return:
        '''
      
        if not evt: return
        ep   = evt["epochs"]
        name = 'test'
        subject_id = name
        if fname:
           fout_path = os.path.dirname(fname)
           name      = os.path.splitext( os.path.basename(fname) )[0]
           subject_id = name.split('_')[0]
        else:
           name      = "test.png"
           fout_path = "."
       
        if plot_dir:
           fout_path += "/" + plot_dir
           try:
               os.makedirs(fout_path,exist_ok=True)
           except:
               logger.exception("---> can not create epocher plot\n"+
                                "  -> directory: {}\n".format(fout_path)+
                                "  -> filename : {}".format(fname) )
               return
            
          # mkpath( fout_path )
        fout =fout_path +'/'+ name   
           
        #pl.ioff()  # switch  off (interactive) plot visualisation
        pl.figure(name)
        pl.clf()
        #fig = pl.figure(name,figsize=(10, 8), dpi=100))
        
        pl.title(name)
        
       #---ck if channels exist
        nplt  = 3
        t0,t1 = ep.tmin,ep.tmax
        
      #--- meg  
        pl.subplot(nplt,1,1)
        picks = self.picks.meg_nobads(ep)
        if picks.any():
           avg   = ep.average(picks=picks)
           avg.data *= info['meg']['scale']
           pl.plot(avg.times, avg.data.T,color='black')
           pl.ylim(self.minmax(avg.data))
           pl.ylabel('['+ info['meg']['unit']+ ']')
      
        pl.xlim(t0,t1)
        pl.grid(True)
       
        t = subject_id +' Evoked '
        if condition:
           t +=' '+condition
        if ep.info['bads']:         
           s = ','. join( ep.info['bads'] )
           pl.title(t +' bads: ' + s)
        else:
           pl.title(t)
                
       #--- ecg eog        
        pl.subplot(nplt,1,2)
        picks  = self.picks.ecg_eog(ep)
        
        if picks.any():
           labels =[ ep.info['ch_names'][x] for x in picks]
           avg    = ep.average(picks=picks) 
           avg.data *= info['eeg']['scale']
           d = pl.plot(avg.times, avg.data.T)
           pl.legend(d, labels, loc=2,prop={'size':8})
           pl.ylim(self.minmax(avg.data))
           pl.ylabel('['+ info['eeg']['unit']+ ']')
        
        pl.xlim(t0,t1)
        pl.grid(True)
     
       #--- stim        
        pl.subplot(nplt,1,3)
        picks = self.picks.stim_response(ep) 
        if picks.any():
           labels =[ ep.info['ch_names'][x] for x in picks]    
           labels[0] += '  Evts: %d Id: %d' %(ep.events.shape[0],ep.events[0,2]) 
           avg   = ep.average(picks=picks) 
           pl.ylim(self.minmax(avg.data))
           d = pl.plot(avg.times, avg.data.T)
           pl.legend(d, labels, loc=2,prop={'size':8},)            
      
        pl.xlim(t0,t1)
        pl.xlabel('[s]')
        pl.grid(True)
        
      #--- plt event_id table
        cols = ('EvtId', 'Counts')
      #--- get ids and counts
        ids,cnts = np.unique( evt["events"][:,-1],return_counts=True)
        
        #ids=np.arange(80)
        #cnts=ids+11
        data = np.zeros((len(ids),2),dtype=np.int)
        data[:,0] += ids
        data[:,1] += cnts
        
        yend= len(ids)*0.12  #[left, bottom, width, height]
        if yend > 4.0:
           yend= 4.0
        tab = pl.table(cellText=data,colLabels=cols,loc='top',
                       colWidths=[0.04 for x in cols],
                       bbox=[-0.15, -0.40, 0.1, yend], cellLoc='left')
        
        #cellDict = tab.get_celld()
        #for i in range(0,len(cols)):
        #    cellDict[(0,i)].set_height(.02)
        #    for j in range(1,len(ids)+1):
        #        cellDict[(j,i)].set_height(.02)
        tab.set_fontsize(9)
      #---
        if save_plot:
           fout += self.file_extention              
           pl.savefig(fout, dpi=self.dpi)
           if self.verbose:
              logger.info("---> done saving plot: " +fout)
        else:
           fout= "no plot saved"
       #---
        if show_plot:
           pl.show()
        else:   
           pl.close()            
         
        return fout

jumeg_epocher_plot = JuMEG_Epocher_Plot()