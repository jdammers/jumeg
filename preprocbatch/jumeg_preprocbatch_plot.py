# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:10:28 2015

@author: fboers
"""
import os.path
from distutils.dir_util import mkpath

import numpy as np
import matplotlib.pylab as pl
import mne

from jumeg.jumeg_base import JuMEG_Base_IO


class JuMEG_PPB_Plot(JuMEG_Base_IO):
    def __init__ (self,raw=None):
        super(JuMEG_PPB_Plot, self).__init__()
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
       #--- meg
        pl.subplot(311)
        picks = self.picks.meg_nobads(ep)
        avg   = ep.average(picks=picks) 
        avg.data *= info['meg']['scale']
        t0,t1=avg.times.min(), avg.times.max() 
        
        pl.ylim(self.minmax(avg.data))
        pl.xlim(t0,t1)
         
        #pl.xlabel('[s]')
        pl.ylabel('['+ info['meg']['unit']+ ']')
        
        t= subject_id +' Evoked '
        if condition:
           t +=' '+condition
        if ep.info['bads']:         
           s = ','. join( ep.info['bads'] )
           pl.title(t +' bads: ' + s)
        else:
           pl.title(t)
       
        pl.grid(True)
        pl.plot(avg.times, avg.data.T,color='black')
         
       #--- ecg eog        
        pl.subplot(312)
        picks  = self.picks.ecg_eog(ep)
        labels =[ ep.info['ch_names'][x] for x in picks]
        avg    = ep.average(picks=picks) 
        avg.data *= info['eeg']['scale']
        pl.ylim(self.minmax(avg.data))
        pl.xlim(t0,t1)
        pl.ylabel('['+ info['eeg']['unit']+ ']')
        
        pl.grid(True)
        d = pl.plot(avg.times, avg.data.T)
        pl.legend(d, labels, loc=2,prop={'size':8})
    
       #--- stim        
        pl.subplot(313)
        picks = self.picks.stim_response(ep) 
        labels =[ ep.info['ch_names'][x] for x in picks]    
        labels[0] += '  Evts: %d Id: %d' %(ep.events.shape[0],ep.events[0,2]) 
        avg   = ep.average(picks=picks) 
        pl.ylim(self.minmax(avg.data))
        pl.xlim(t0,t1)
        pl.xlabel('[s]')
      
        pl.grid(True)
        d = pl.plot(avg.times, avg.data.T)
        pl.legend(d, labels, loc=2,prop={'size':8},)            
              
       #---
        if save_plot:
           fout += self.file_extention              
           pl.savefig(fout, dpi=self.dpi)
           if self.verbose:
              print"---> done saving plot: " +fout 
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

jumeg_preprocbatch_plot = JuMEG_PPB_Plot()