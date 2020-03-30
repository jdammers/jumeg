#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 20.01.20
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

#import copy
import os,os.path as op
import warnings
import logging,time,datetime

import numpy as np
from distutils.dir_util import mkpath

import matplotlib.pyplot as plt

import mne
from mne.preprocessing import find_ecg_events, find_eog_events

from jumeg.base.jumeg_base            import jumeg_base as jb
from jumeg.base.jumeg_base            import JUMEG_SLOTS
from jumeg.base.jumeg_base_config     import JuMEG_CONFIG as jCFG
from jumeg.base                       import jumeg_logger

logger = logging.getLogger("jumeg")

__version__= "2020.03.06.001"

class ARTEFACT_EVENTS(JUMEG_SLOTS):
    """
     artefact event dict:
       ch_name:  str or list of str
       event_id: int or list of int
       tmin: float tmin is s
       tmax: float tmax in s
     
     Example:
     --------
      ecg:
        ch_name: "ECG"
        event_id: 999
        tmin: -0.4
        tmax: 0.4
    
      eog:
       ch_name: ['EOG ver','EOG hor']
       event_id: [997,998]
       tmin: -0.4
       tmax: 0.4
       
       
       import mne,logging
       from mne.preprocessing import find_ecg_events, find_eog_events
       
       logger = logging.getLogger("jumeg")
      
      #--- find ECG
       ECG = ARTEFACT_EVENTS(raw=raw,ch_name="ECG",event_id=999,tmin=-0.4,tmax=0.4,_call = find_ecg_events)
       ECG.find_events(raw=raw,**config.get("ecg"))
       EOG.GetInfo(debug=True)
     
      #--- find EOG
       EOG = ARTEFACT_EVENTS(raw=raw,ch_name=['EOG ver','EOG hor'],event_id=[997,998],tmin=-0.4,tmax=0.4,
                                    _call = find_eog_events)
       EOG.find_events(raw=raw,**config.get("eog"))
       EOG.GetInfo(debug=True)
       
    """
    __slots__ = ["raw","ch_name","set_annotations","event_id","events","tmin","tmax","verbose","debug","_call"]
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.init(**kwargs)
       #--- default for set annotations = True
        self.set_annotations = kwargs.get("set_annotations",True)

    def find_events(self,**kwargs):
        """
         raw:
         ch_name:
         set_annotations:
         event_id:
         tmin:
         tmax:
         verbose:
         debug:
         
         
         parameter for mne.preprocessing  find_ecg_events, find_eog_events
         event_id=999, ch_name=None, tstart=0.0,
                    l_freq=5, h_freq=35, qrs_threshold='auto',
                    filter_length='10s', return_ecg=False,
                    reject_by_annotation=None, verbose=None
         
         
         events={ ch_name: { events: <onsets> or  <mne.events>, # [onsets,offsets,id]
                             pulse: <event counts>}
         :return:
         if set annotations
            raw.annotations
            
        """
        self.update(**kwargs)
        self.events = dict()
        
        if isinstance(self.ch_name,(list)):
           channels = [ *self.ch_name ]
           evt_id   = [ *self.event_id ]
        else:
           channels = [ self.ch_name ]
           evt_id   = [ self.event_id ]
           
        while len(channels):
            ch_name  = channels.pop()
            event_id = evt_id.pop()
            
            if ch_name not in self.raw.info['ch_names']:
               continue
               
            self.events[ch_name]= dict()
            self.events[ch_name]["index"] = self.raw.ch_names.index(ch_name)
            res = self._call(self.raw,event_id,ch_name=ch_name,verbose=self.verbose)
           
            if isinstance(res[1],(np.ndarray)):
               self.events[ch_name]["events"] = res
               self.events[ch_name]["pulse"]  = self.events[ch_name]["events"].shape[0]
            else:
               self.events[ch_name]["events"] = res[0]
               self.events[ch_name]["pulse"] = res[2]
        
        if self.set_annotations:
           return self.set_anotations()
        return None

    def set_anotations(self,save=False):
        """
        update raw.anotattions with artefact events e.g.: ECG,EOG
        save: save raw with annotations Talse
        return annotations
        
        """

        raw_annot = None
        evt_annot = None
        try:
            raw_annot = self.raw.annotations
        except:
            pass

        #--- store event info into raw.anotations
        time_format = '%Y-%m-%d %H:%M:%S.%f'
        orig_time   = self.raw.info.get("meas_date",self.raw.times[0])
        
        for k in self.events.keys():
            msg = ["update raw.annotations: {}".format(k)]
    
            onset  = self.events[k]['events'][:,0] / self.raw.info["sfreq"]
            #onset -= self.tmin
            duration = np.ones( onset.shape[0] ) / self.raw.info["sfreq"]  # one line in raw.plot
            #duration+= abs(-self.tmin) + self.tmax
    
            evt_annot = mne.Annotations(onset=onset.tolist(),
                                        duration=duration.tolist(),
                                        description=k, # [condition for x in range(evt["events"].shape[0])],
                                        orig_time=orig_time)
            if raw_annot:
               msg.append(" --> found mne.annotations in RAW:\n  -> {}".format(raw_annot))
             #--- clear old annotations
               kidx = np.where( raw_annot.description == k)[0] # get index
               if kidx.any():
                  msg.append("  -> delete existing annotation {} counts: {}".format(k, kidx.shape[0]) )
                  raw_annot.delete(kidx)
                  
               self.raw.set_annotations( raw_annot + evt_annot)
               raw_annot = self.raw.annotations
            else:
               self.raw.set_annotations(evt_annot)
               raw_annot = self.raw.annotations
        
        if save:
           f      = jb.get_raw_filename(raw)
           fanato = f.replace( "-raw.fif","-anato.csv")
           self.raw.annotations.save( fanato )
           
        msg.append("storing mne.annotations in RAW obj:\n  -> {}".format(self.raw.annotations))
        logger.info("\n".join(msg))
        
        return self.raw.annotations

    def GetInfo(self,debug=False):
        if debug:
           self.debug=True
           
        if not isinstance(self.events,(dict)):
           logger.warning( "!!! --> Artefact Events: not events found !!!" )
           return
        
        msg = [" --> Artefact Events:"]
        for k in self.events.keys():
            msg.extend( [" --> {}".format(k),"  -> pulse: {}".format( self.events[k]["pulse"] ) ] )
            if self.debug:
               msg.append( "  -> events:\n{}".format( self.events[k]["events"] ) )
               msg.append( "-"*25)
        logger.info( "\n".join(msg) )
        

class CalcSignal(JUMEG_SLOTS):
  
  def __init__(self,**kwargs):
      super().__init__(**kwargs)
      self.init(**kwargs)
      
  def calc_rms(self,data,average=None,rmsmean=None):
    ''' Calculate the rms value of the signal.
        Ported from Dr. J. Dammers IDL code.
    '''
    # check input
    sz = np.shape(data)
    nchan = np.size(sz)
    #  calc RMS
    rmsmean = 0
    if nchan == 1:
        ntsl = sz[0]
        return np.sqrt(np.sum(data ** 2) / ntsl)
    elif nchan == 2:
        ntsl = sz[1]
        powe = data ** 2
        if average:
            return np.sqrt(np.sum(np.sum(powe,1) / nchan) / ntsl)
        return np.sqrt(sum(powe,2) / ntsl)
        
    return -1
 

  def calc_performance(self,evoked_raw,evoked_clean):
      ''' Gives a measure of the performance of the artifact reduction.
            Percentage value returned as output.
      '''
      diff = evoked_raw.data - evoked_clean.data # ??
      rms_diff = self.calc_rms(diff,average=1)
      rms_meg = self.calc_rms(evoked_raw.data,average=1)
      arp = (rms_diff / rms_meg) * 100.0
      return np.round(arp)

  def _calc_signal(self,raw,events,event_id=None,tmin=None,tmax=None,picks=None):
      """
       calc signal from raw -> get epochs -> average
      :param raw:
      :param events : mne.events
      :param event_id:
      :param tmin:
      :param tmax:
      :param picks:
      :return:

      signal, min/max-range, times
      """
      signal = None
      range = None
      times = None
      if not isinstance(picks,(list,np.ndarray)):
          picks = jb.picks.meg_nobads(raw)
    
      #--- RAW mk epochs + average
      ep = mne.Epochs(raw,events,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
      if len(picks) > 1:
          avg = ep.average()
          times = avg.times
          data = avg._data
          range = [data.min(axis=0),data.max(axis=0)]
          signal = np.average(data,axis=0).flatten()
    
      else:  # ref channel e.g. ECG, EOG  as np.array
          signal = np.average(ep.get_data(),axis=0).flatten()
          range = [signal.min(),signal.max()]
    
      return signal,range,times

  def _calc_avg(self,raw,events,event_id=None,tmin=None,tmax=None,picks=None):
      """
       calc signal from raw -> get epochs -> average
      :param raw:
      :param events : mne.events
      :param event_id:
      :param tmin:
      :param tmax:
      :param picks:
      :return:

      signal, min/max-range, times
      """
    
      signal = None
      range = None
      times = None
      if not isinstance(picks,(list,np.ndarray)):
          picks = jb.picks.meg_nobads(raw)
    
      #--- RAW mk epochs + average
      ep = mne.Epochs(raw,events,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
      if len(picks) > 1:
          avg = ep.average()
          times = avg.times
          data = avg._data
          signal = data.T
      else:  # ref channel e.g. ECG, EOG  as np.array
          signal = np.average(ep.get_data(),axis=0).flatten()
    
      range = [signal.min(),signal.max()]
    
      return signal,range,times

  def _calc_gfp(self,raw,events,event_id=None,tmin=None,tmax=None,picks=None):
      """
       calc signal from raw -> get epochs -> average
      :param raw:
      :param events : mne.events
      :param event_id:
      :param tmin:
      :param tmax:
      :param picks:
      :return:

      signal, min/max-range, times
      """
    
      signal = None
      range = None
      times = None
      if not isinstance(picks,(list,np.ndarray)):
          picks = jb.picks.meg_nobads(raw)
    
      #--- RAW mk epochs + average
      ep = mne.Epochs(raw,events,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
      if len(picks) > 1:
          avg = ep.average()
          times = avg.times
          signal = np.sum(avg._data ** 2,axis=0)
      else:  # ref channel e.g. ECG, EOG  as np.array
          signal = np.average(ep.get_data(),axis=0).flatten()
    
      range = [signal.min(),signal.max()]
      return signal,range,times

  def _calc_ylimits(self,ranges=None,factor=1.0,offset=0.1):
      """
      ranges: list of min max np.arrays
      :param factor:
      :param offset: e.g 0.1 => +-10%
      :return:
      min,max

      """
      r = np.concatenate(ranges)
      min = r.min() * factor
      max = r.max() * factor
     # return min - (abs(min) * offset), max + (abs(max) * offset)
      return min - offset, max + offset
  
  def _calc_data(self,raw,raw_clean,evt,event_id=999,tmin=-0.4,tmax=0.4,picks=None,type="avg"):
      """

      :param raw:
      :param raw_clean:
      :param evt: events from annotation
      :param event_id:
      :param tmin:
      :param tmax:
      :param picks:
      :param type: avg,gfp,signal
      :return:

      sig_raw,sig_clean,range,t

      """
    
      if type == "gfp":
          #--- RAW mk epochs + average
          sig_raw,range_raw,t = self._calc_gfp(raw,evt,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
          #--- RAW clean mk epochs + average
          sig_cln,range_cln,_ = self._calc_gfp(raw_clean,evt,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
      if type == "avg":
          #--- RAW mk epochs + average
          sig_raw,range_raw,t = self._calc_avg(raw,evt,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
          #--- RAW clean mk epochs + average
          sig_cln,range_cln,_ = self._calc_avg(raw_clean,evt,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
      else:
          #--- RAW mk epochs + average
          sig_raw,range_raw,t = self._calc_signal(raw,evt,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
          #--- RAW clean mk epochs + average
          sig_cln,range_cln,_ = self._calc_signal(raw_clean,evt,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
    
      range = [range_raw,range_cln]
    
      return sig_raw,sig_cln,range,t
  
  

class JuMEG_ICA_PERFORMANCE_PLOT(CalcSignal):
    __slots__ = ["raw","plot_path","raw_clean","ch_name","event_id","picks","tmin","tmax","title","colors","alpha","grid","show",
                 "scale","offset","fontsize","_n_cols","n_rows","idx","plot_ypos","_figure","figsize","type","dpi","orientation","fout",
                 "plot_extention","verbose","set_title","text","save_as_png","save_as_fig"]
    """
      plotting ica performance plots as png
    """
    
    def __init__(self,**kwargs):
        super().__init__()

        self.init(**kwargs)

        self.type   = "avg"  # type = "avg" # gfp,avg,sig
        self.n_rows = 2
        self.n_cols = 1
        self.idx    = 1
        self.set_title = True
        self.tmin      = -0.4
        self.tmax      = 0.4
        self.alpha     = 0.33
        self.offset    = 0.15
        self.fontsize  = 12

        #self.figsize   = (11.69,8.27)
        self.figsize     = (16.0,9.0)
        self.dpi         = 300
        self.orientation    = 'portrait' #"landscape"
        self.plot_extention = ".png"

        self.grid      = True
        self.show      = False
        self.save      = True
        self.colors    = ["black","yellow","red","magenta","green"]
        self.scale     = { "raw":{ "factor":10.0 ** 15,"unit":"fT" },"ref":{ "factor":10.0 ** 3,"unit":"mV" } }
        
        self._update_from_kwargs(**kwargs)
       #--- A4 landscape
        plt.rc('figure',figsize=self.figsize,autolayout=True)
        plt.rcParams.update({ 'font.size':self.fontsize })
        plt.subplots_adjust(left=0.1,right=0.95,bottom=0.05,top=0.95,hspace=0.35)
        plt.rcParams['savefig.facecolor'] = "0.9"

    @property
    def figure(self): return self._figure
    
    def _plot(self,ax,t,data,ylabel,color,range=None,range_color="cyan"):
        ax.plot(t,data,color=color)
        
        if range:
          ax.fill_between(t,range[0],y2=range[1],color=range_colors,alpha=alpha)
  
        ax.set_xlabel("[s]")
        ax.set_xlim(t[0],t[-1])
        ax.set_ylabel(ylabel)
        ax.grid(True)

    def clear(self):
        try:
          if self.figure:
             plt.close('all')
             self._figure = None
        except:
          pass
        
        #s = super()
        #if hasattr(s, "clear"):
        #   s.clear()
   
    def plot(self,**kwargs):
        
        #raw=None,raw_clean=None,ch_name="ECG",event_id=999,picks=None,tmin=-0.4,tmax=0.4,title=None,
        #                 colors=["black","yellow","red","magenta","green"],alpha=0.33,grid=True,show=False,
        #                 scale={"raw":{"factor":10.0**15,"unit":"fT"},"ref":{"factor":10.0**3,"unit":"mV"}},
        #                 offset=0.1,fontsize=12):
        """
        
        :param raw:
        :param raw_clean:
        :param ch_name:
        :param event_id:
        :param picks:
        :param tmin:
        :param tmax:
        :param colors:
        :param alpha:
        :param grid:
        :param show:
        :param scale:  {"raw":{"factor":10**12,"unit":"pT"},"ref":{"factor":10**3,"unit":"mV"}},
        :param offset: 0.1
        :return:
        """
        
        self._update_from_kwargs(**kwargs)
        
        
        logger.info("RAW annotations: {}".format(self.raw.annotations))
        
   #--- get epochs  calc avgs + ref
        annotat = mne.events_from_annotations(self.raw,event_id={ self.ch_name:self.event_id },use_rounding=True,chunk_duration=None)

        if not annotat:
            logger.error("!!! ERROR No MNE Annotations found: {}\n".format(jb.get_raw_filename(self.raw)))
            return None

        evt    = annotat[0]
        counts = evt.shape[0]
      
        sig_raw,sig_clean,range,t = self._calc_data(self.raw,self.raw_clean,evt,event_id=self.event_id,tmin=self.tmin,tmax=self.tmax,picks=self.picks)
       
       #--- ref channel e.g.: ECG
        sig_ref,_,_ = self._calc_signal(self.raw,evt,event_id=self.event_id,tmin=self.tmin,tmax=self.tmax,picks=jb.picks.labels2picks(self.raw,self.ch_name))
    
        if not self.figure:
           self._figure = plt.figure()
           #self.figure.suptitle(os.path.basename(jb.get_raw_filename(self.raw)),fontsize=12)
           
       #--- subplot(nrows,ncols,idx)
        ax1 = plt.subplot(self.n_rows,self.n_cols,self.idx)
       #--- sig raw
        scl = self.scale.get("raw")
        ylim = self._calc_ylimits(ranges=range,factor=scl.get("factor"),offset=self.offset)
        self._plot(ax1,t,sig_raw * scl.get("factor"),scl.get("unit"),"black")
       #--- sig clean
        ax2 = plt.subplot(self.n_rows,self.n_cols,self.idx + self.n_cols)
        self._plot(ax2,t,sig_clean * scl.get("factor"),scl.get("unit"),"black")
        
       #---
        scl = self.scale.get("ref")
        ax3 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        self._plot(ax3,t,sig_ref * scl.get("factor"),scl.get("unit"),"red")
        ax3.tick_params(axis='y',labelcolor=color)
        ax3.legend([self.ch_name +" cnts {}".format(counts)], loc=2,prop={'size':8})

        ax4 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        self._plot(ax4,t,sig_ref * scl.get("factor"),scl.get("unit"),"green")
        ax4.tick_params(axis='y',labelcolor=color)
        ax4.legend(["Clean "+self.ch_name + " cnts {}".format(counts)],loc=2,prop={ 'size':8 })

        ax1.set_ylim(ylim[0],ylim[1])
        ax2.set_ylim(ylim[0],ylim[1])

        #if self.save:
        #   self.save_figure()

        if self.show:
           ion()
           self.figure.tight_layout()
           plt.show()

        return self.figure
    
    def save_figure(self,**kwargs):
        """
        
        :param kwargs:
        :return: fig
        """
        self._update_from_kwargs(**kwargs)
        
        self.figure.tight_layout()
        
        #plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=1.0)
        
        if self.fout:
           fout = self.fout
           if not fout.endswith(self.plot_extention):
              fout += self.plot_extention
        else:
          fout = "test"+self.plot_extention
        
        if self.plot_path:
           path = jb.isPath(self.plot_path,mkdir=True)
           fout = os.path.basename(fout)
           fout = os.path.join(self.plot_path,fout)
        
        if self.set_title:
           txt = os.path.basename(fout).rsplit(".",1)[0]
           if self.text:
              txt+= "   " +self.text
           self.figure.suptitle(txt,fontsize=10,y=0.02,x=0.05,ha="left")
        elif self.text:
           self.figure.suptitle(self.text,fontsize=10,y=0.02,x=0.05,ha="left")
       
       #--- save img
        if self.save:
           self.figure.savefig(fout,dpi=self.dpi,orientation=self.orientation)
           logger.info("done saving plot: " +fout)

        self.fout=fout
        

class JuMEG_ICA_PERFORMANCE(JUMEG_SLOTS): 
    """
    find ecg,eog artifacts in raw
     ->use jumeg or mne

    make mne.anotations
    prefromance check
     init array of figs : overview, n chops for ECg,EOG performance
     for each chop :
         avg epochs  => ECG plot raw, raw_cleaned evoked ,ECG signal, performance
                     => EOG plot raw, raw_cleaned evoked ,EOG signal, performance
                     
                     
    plot performance to mne.report
    jIP = JuMEG_ICA_PERFORMANCE(raw=raw,raw_clean=raw_ar)
    
    """
    #  raw=raw,path=path,fname=raw_fname,config=CFG.GetDataDict("ica")
    __slots__ = ["raw","path","fname","config","n_figs","_EOG","_ECG","_PLOT","picks","use_jumeg","ecg","eog","verbose"]
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._init(**kwargs)
        self._ECG = ARTEFACT_EVENTS(raw=self.raw,ch_name="ECG",event_id=999,tmin=-0.4,tmax=0.4,_call=find_ecg_events)
        self._EOG = ARTEFACT_EVENTS(raw=self.raw,ch_name=['EOG ver','EOG hor'],event_id=[997,998],tmin=-0.4,tmax=0.4,
                                    _call=find_eog_events)
        
        self._PLOT = JuMEG_ICA_PERFORMANCE_PLOT(**kwargs)
    
    @property
    def ECG(self): return self._ECG
    
    @property
    def EOG(self): return self._EOG
    
    @property
    def Plot(self): return self._PLOT

    def _mklists(self,obj,channels=list(),event_ids=list()):
        """
        
        :param obj:
        :param channels:
        :param event_ids:
        :return:
        """
       
        if isinstance(obj.ch_name,(list)):
            channels.extend(obj.ch_name)
            event_ids.extend(obj.event_id)
        else:
            channels.append(obj.ch_name)
            event_ids.append(obj.event_id)
        
        return channels,event_ids

    def plot(self,**kwargs):
        """
        plotting all ref channels e.g. ECG,EOGver,EOGhor
        
        |EOG  |EOGver|EOGhor|
        -------------------
        |clean|clean |clean|
        
        :param kwargs:
        :return:
        """
        self._update_from_kwargs(**kwargs)
        self._PLOT._update_from_kwargs(**kwargs)
        
       #--- init performance plot
        self.Plot.idx = 1
        idx = 1
        ch_names = []
        ids      = []

        self._mklists(self.ECG,channels=ch_names,event_ids=ids)
        self._mklists(self.EOG,channels=ch_names,event_ids=ids)
        
        #self.EOG.GetInfo(debug=True)
        
        self.Plot.n_cols = len(ch_names)
        self.Plot.n_rows = 2
        
        for obj in [self.ECG,self.EOG]:
            ch_names = []
            ids      = []
            self._mklists(obj,channels=ch_names,event_ids=ids)
            for i in range(len(ch_names)):
                self.Plot.plot(ch_name=ch_names[i],event_id=ids[i],picks=self.picks,tmin=obj.tmin,tmax=obj.tmax)
                idx += 1
                self.Plot.idx = idx
            
        #self.Plot.figure.show()
        self.Plot.save_figure(save=True)
        self.Plot.clear()
        return self.Plot.fout

def test1():
    #--- init/update logger
    jumeg_logger.setup_script_logging(logger=logger)
    
    raw = None
    stage = "$JUMEG_PATH_LOCAL_DATA/exp/MEG94T/mne"
    fcfg  = os.path.join(stage,"meg94t_config01.yaml")
    fpath = "206720/MEG94T0T2/130820_1335/1/"
    
    path = os.path.join(stage,fpath)
    raw_fname = "206720_MEG94T0T2_130820_1335_1_c,rfDC,meeg,nr,bcc,int-raw.fif"
    
    logger.info("JuMEG Pipeline ICA Performance ICA mne-version: {}".format(mne.__version__))
    
    f = os.path.join(path,raw_fname)
    raw,raw_fname = jb.get_raw_obj(f,raw=None)

    raw_path = os.path.dirname(raw_fname)
   #--- get picks from raw
    picks = jb.picks.meg_nobads(raw)
  
   #---
    CFG = jCFG()
    CFG.update(config=fcfg)
    config = CFG.GetDataDict("ica")
   #--
    ICAPerformance = JuMEG_ICA_PERFORMANCE(raw=raw,path=path,fname=raw_fname,)
   
   #--- find ECG
    ICAPerformance.ECG.find_events(raw=raw,**config.get("ecg"))
    ICAPerformance.ECG.GetInfo(debug=True)
   #--- find EOG
    ICAPerformance.EOG.find_events(raw=raw,**config.get("eog"))
    ICAPerformance.EOG.GetInfo(debug=True)
   
   #---
   # raw.plot(block=True)

   #--- save raw
   #fout=f.replace("-raw.fif","test-raw.fif")
   #jb.update_and_save_raw(raw,f,f)
   
def test2():
    #--- init/update logger
    jumeg_logger.setup_script_logging(logger=logger)
    
    raw = None
    stage = "$JUMEG_PATH_LOCAL_DATA/exp/MEG94T/mne"
    fcfg  = os.path.join(stage,"meg94t_config01.yaml")
    fpath = "206720/MEG94T0T2/130820_1335/1/"
    path = os.path.join(stage,fpath)

    #fraw   =  "206720_MEG94T0T2_130820_1335_1_c,rfDC,meeg,nr,bcc,int,000516-000645-raw.fif"
    #fraw_ar = "206720_MEG94T0T2_130820_1335_1_c,rfDC,meeg,nr,bcc,int,000516-000645,ar-raw.fif"
    
    fraw    = "206720_MEG94T0T2_130820_1335_1_c,rfDC,meeg,nr,bcc,int,fibp0.10-45.0-raw.fif"
    fraw_ar = "206720_MEG94T0T2_130820_1335_1_c,rfDC,meeg,nr,bcc,int,fibp0.10-45.0,ar-raw.fif"
    
    logger.info("JuMEG Pipeline ICA Performance ICA mne-version: {}".format(mne.__version__))
   #---
    f = os.path.join(path,fraw)
    raw,raw_fname = jb.get_raw_obj(f,raw=None)
    raw_path      = os.path.dirname(raw_fname)
    picks         = jb.picks.meg_nobads(raw)
   #---
    f = os.path.join(path,fraw_ar)
    raw_ar,raw_ar_fname = jb.get_raw_obj(f,raw=None)
    
   #--- read config
    CFG = jCFG()
    CFG.update(config=fcfg)
    config = CFG.GetDataDict("ica")
    
   #
    jIP = JuMEG_ICA_PERFORMANCE(raw=raw,raw_clean=raw_ar,picks=picks)
    #jIP.report()
    
    fout = raw_fname.rsplit("-",1)[0] + "-ar"
    jIP.plot(verbose=True,fout=fout)
    
    
if __name__ == "__main__":
    # test1()
    test2()
