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
        self.colors = ['r','b','g','c','m','y','k']
        
    def minmax(self,d):
        ymin, ymax = d.min(),d.max()
        ymin -= np.abs(ymin) * 0.1 #factor * 1.1
        ymax += np.abs(ymax) * 0.1 #factor * 1.1
      
        return ymin,ymax
    
    def _set_colors(self):
        for i,j in enumerate(pl.gca().lines):
            j.set_color(self.colors[i % len(self.colors)])
       
    def plot_group(self,ep,group="meg",picks=None,info=None,show_evt=False,show_labels=True):
        """
        
        :param ep:
        :param group:
        :param picks:
        :param info:
        :param show_evt:
        :param show_labels:
        :return:
        """
        if picks.any():
           labels = [ ep.info['ch_names'][x] for x in picks ]
           avg    = ep.average(picks=picks)
           if info:
              avg.data *= info.get('scale',1.0)
              pl.ylabel('[' + info.get('unit','au') + ']')
           d = pl.plot(avg.times, avg.data.T)
           self._set_colors()
           
           if show_evt:
            #--- change legend
               idx0 = np.where(avg.times == 0)
               labels = [ep.info['ch_names'][x] for x in picks]
               if idx0:
                  for idx in range(len(labels)):
                      labels[idx] += " evt: {} ".format(int(avg.data[idx,idx0].flatten()))
           
           if show_labels:
              pl.legend(d, labels, loc=2,prop={'size':8})
           pl.ylim(self.minmax(avg.data))
           

        pl.xlim(ep.tmin,ep.tmax)
        pl.xlabel('[s]')
        pl.grid(True)
        return avg.data
    
    def plot_stim(self,ep,group="stim",picks=None,info=None,show_evt=False,show_labels=True):
        """
        
        :param ep:
        :param group:
        :param picks:
        :param info:
        :param show_evt:
        :param show_labels:
        :return:
        """
        if picks.any():
           labels = [ ep.info['ch_names'][x] for x in picks ]
           avg    = ep.average(picks=picks)
           if info:
              avg.data *= info.get('scale',1.0)
              pl.ylabel('[' + info.get('unit','au') + ']')
           d = pl.plot(avg.times, avg.data.T)
           self._set_colors()
           
           if show_evt:
            #--- change legend
               idx0 = np.where(avg.times == 0)
               # labels = [ep.info['ch_names'][x] for x in picks]
               if idx0:
                  for idx in range(len(labels)):
                      labels[idx] += " evt: {} ".format(int(avg.data[idx,idx0].flatten()))
           
           if show_labels:
              pl.legend(d, labels, loc=2,prop={'size':8})
           pl.ylim(self.minmax(avg.data))
           

        pl.xlim(ep.tmin,ep.tmax)
        pl.xlabel('[s]')
        pl.grid(True)
 
    def plot_evoked(self,evt,fname=None,save_plot=True,show_plot=False,condition=None,plot_dir=None,
                    info={'meg':{'scale':1e15,'unit':'fT'},'eeg':{'scale':1e3,'unit':'mV'},'emg':{'scale':1e3,'unit':'mV'},}):
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
        fout = fout_path +'/'+ name
        
        #pl.ioff()  # switch  off (interactive) plot visualisation
        pl.figure(name)
        pl.clf()
        #fig = pl.figure(name,figsize=(10, 8), dpi=100))
        
        pl.title(name)
    
       #--- make title
        t = subject_id + ' Evoked '
        if condition:
           t += ' ' + condition
        t += ' Id: {} counts: {}'.format(ep.events[0,2],ep.events.shape[0])
        if ep.info['bads']:
           t = t + "  bads: " + ','.join(ep.info['bads'])
           
       #---ck if emg channels exist
        picks = self.picks.emg_nobads(ep)
        if picks.any():
           nplt  = 4
        else:
           nplt = 3
        
      #--- meg
        pl.subplot(nplt,1,1)
        pl.title(t)
        self.plot_group(ep,group="meg",picks=self.picks.meg_nobads(ep),info=info.get('meg'),show_labels=False)
        
      #--- ecg eog
        pl.subplot(nplt,1,2)
        self.plot_group(ep,group="ecg eog",picks=self.picks.ecg_eog(ep),info=info.get('eeg'))
   
      #--- stim
        pl.subplot(nplt,1,3)
        self.plot_group(ep,group="stim",picks=self.picks.stim_response(ep),info=info.get('stim'),show_evt=True)
       
        '''
        ax = pl.gca()
        
        ax.set_ylabel('Stim', color=self.colors[0])
        ax.tick_params(axis='y', labelcolor=self.colors[0])
        ax.set_ylim(0,data[0].max() +10)
        
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('RES', color=self.colors[1])
        ax2.tick_params(axis='y', labelcolor=self.colors[1])
        ax2.set_ylim( 0,data[1].max()+10 )
        
        #fig.tight_layout()  # otherwise the right y-label is slightly clipped
        '''
        
      #--- emg
        if nplt > 3:
           pl.subplot(nplt,1,4)
           self.plot_group(ep,group="emg",picks=self.picks.emg_nobads(ep),info=info.get('emg'))
      
      #--- plt event_id table
        cols = ('EvtId', 'Counts')
       #--- get ids and counts
        ids,cnts = np.unique( evt["events"][:,-1],return_counts=True)
        
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