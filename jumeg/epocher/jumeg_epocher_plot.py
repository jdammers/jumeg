# -*- coding: utf-8 -*-
"""
Created on 08.06.2018

@author: fboers
"""
import os.path
from distutils.dir_util import mkpath

import numpy as np
#import matplotlib.pylab as pl
import mne

from jumeg.jumeg_base import JuMEG_Base_IO

import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages


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
        
    def plot_evoked(self,ep,fname=None,save_plot=True,show_plot=False,condition=None,plot_dir=None,
                    info={'meg':{'scale':1e15,'unit':'fT'},'eeg':{'scale':1e3,'unit':'mV'}}):
        '''
        plot subplots evoked/average 
        MEG
        ECG/EOG + performance 
        STIM Trigger/Response
        events, rt mean median min max        
        '''
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
           mkpath( fout_path )        
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
              
       #---
        if save_plot:
           fout += self.file_extention              
           pl.savefig(fout, dpi=self.dpi)
           if self.verbose:
              print("---> done saving plot: " +fout) 
        else:
           fout= "no plot saved"
       #---
        if show_plot:
           pl.show()
        else:   
           pl.close()            
         
        return fout

#TODO
# ck units ep.info['chs']['unit']  -> 107 
# call mne.io.constant.FIFF 
# returns 'FIFF_UNIT_V': 107,

jumeg_epocher_plot = JuMEG_Epocher_Plot()