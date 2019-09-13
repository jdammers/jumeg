# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 08:42:19 2015
@author: fboers
"""

import numpy as np
import logging
logger = logging.getLogger("jumeg")


__version__="2019-09-13-001"

class OPT_BASE(object):
      def __init__ (self):
          self._do_scroll          = False
          self._idx_start          =  0
          self._idx_delta          =  1
          self._counts             =  1
         # self._counts_selected    = 1
          
          self.__idx_start_save     = -1
          self.__idx_end_range_save = -1
          self.__tg_idx_start_save  = -1
          self.__tg_idx_delta_save  = -1
          self.__isToggleAll        = False
          
      @property
      def idx_end(self):
          return self._idx_start + self._idx_delta -1
      
      @property
      def idx_end_range(self):
          return self._idx_start + self._idx_delta
      
      @property
      def do_scroll(self): return self._do_scroll
      @do_scroll.setter
      def do_scroll(self,v):
          self._do_scroll = v
      
      @property
      def counts(self): return self._counts
      @counts.setter
      def counts(self,v):
          self._last_start_idx  = -1.0
          self._last_end_range  = -1.0
          self._counts          = v
          self._counts_selected = v
          self.idx_start        = 0 # update done in idx_start
      
      #@property
      #def counts_selected(self): return self._counts_selected
      #@counts_selected.setter
      #def counts_selected(self,v):
      #    self._counts_selected = v
          
      @property
      def idx_start(self): return self._idx_start
      @idx_start.setter
      def idx_start(self,v):
          self.__idx_start_save = self._idx_start
          self._idx_start = v
          self._update()
          
      def _update(self,n_counts=None):
          self._ck_index_range(n_counts=n_counts)
          return self._ck_do_scroll()
          
      def _ck_index_range(self,n_counts=None):
          if n_counts:
             self._counts = n_counts
             
          if self._idx_start < 0:
             self._idx_start = 0
          #logger.info("---> CK IDX RANGE start : {} delta {}  cnt {}".format(self._idx_start,self._idx_delta,self._counts))
          
          if (self._idx_start + self._idx_delta > self._counts):
             self._idx_start = self._counts - self._idx_delta
             if self._idx_start < 0:
                self._idx_start = 0       
                self._idx_delta = self._counts -1
          #logger.info("---> DONE IDX RANGE  start : {} delta {}  cnt {}".format(self._idx_start,self._idx_delta,self._counts))
          
      def _ck_do_scroll(self):
          """  calc start & end index range
               set do_scroll flag
               return: idx start, idx end range (end idx +1)
          """
          if (self.__idx_start_save != self._idx_start):
              # self.__idx_start_save = self._idx_start
              self._do_scroll = True
    
          elif (self.__idx_end_range_save != self.idx_end):
              self.__idx_end_range_save = self.idx_end
              self._do_scroll = True
          else:
              self._do_scroll = False
    
          return self._idx_start,self.idx_end_range

      def index_range(self,n_counts=None):
          return self._update(n_counts=n_counts)

      def toggle_all(self):
          if self.__isToggleAll: # untoggle
             self._idx_delta    = self.__tg_idx_delta_save
             self.__idx_start_save = -1
             self._idx_start    = self.__tg_idx_start_save
             self.__isToggleAll = False
          else:  # toggle
             self.__tg_idx_start_save = self._idx_start
             self.__tg_idx_delta_save = self._idx_delta
             self.__idx_end_range_save = -1
             self._idx_delta = self.counts
             self._idx_start = 0
             self.__isToggleAll = True
          self._do_scroll=True
    
      def _get_info(self):
          msg = [
                "idx start:     {}".format(self.idx_start),
                "counts:        {}".format(self.counts),
                "idx_delta:     {}".format(self._idx_delta),
                "idx_end:       {}".format(self.idx_end),
                "idx_end_range: {}".format(self.idx_end_range),
                "do scroll:     {}".format(self._do_scroll)]
            
          logger.info("---> base option info:\n  -> " + "\n  -> ".join(msg))

#----------------------------------------
class OPT_CHANNELS(OPT_BASE):

      def __init__ (self,start=0,counts=0):
          super().__init__()
          
          self.action_list = ['UP','DOWN','PAGEUP','PAGEDOWN','TOP','BOTTOM','CHANNELS_DISPLAY_ALL']
          
      @property
      def last_channel_idx(self): return self.counts -1
      
      @property
      def last_channel(self): return self.idx_end +1
      
      @property
      def channels_to_display(self): return self._idx_delta
      @channels_to_display.setter
      def channels_to_display(self,v):
          self._idx_delta = v
          self._update()
      
      def action(self,act):
          if act == "TOP":
             self.idx_start = 0
          elif act == "UP":
               self.idx_start -= 1  #A1 on top
          elif act == "DOWN":
               self.idx_start += 1  # A248 on bottom
          elif act == "PAGEUP":
               self.idx_start -= self._idx_delta -1
          elif act == "PAGEDOWN":
               self.idx_start += self._idx_delta -1
          elif act == "BOTTOM":
               self.idx_start = self.counts # - self.plot.plots +1
          elif act == "CHANNELS_DISPLAY_ALL":
               self.toggle_all()
               
              
         # logger.info("---> action : {}\n  --> Channels  idx start: {} channels to display: {} channel_counts: {}".
         #             format(act,self.idx_start,self.channels_to_display,self.counts))
         # self.GetInfo()
         
      def GetInfo(self):
          msg=[
          "start channel:       {}".format(self.idx_start),
          "counts:              {}".format(self.counts),
          "last channel:        {}".format(self.last_channel),
          "channels to display: {}".format(self.channels_to_display),
          "idx_end:             {}".format(self.idx_end),
          "idx_end_range:       {}".format(self.idx_end_range)]
         
          logger.info("---> Channels option info:\n" + "\n  -> ".join(msg))
          
          self._get_info()
          
class OPT_TIME(OPT_BASE):
   
     
     #--- time
     # start
     # inc_factor
     # time.window
     # time.scroll_step
     # inc_factor_list
   
      def __init__(self): #,timepoints=1,sfreq=1.0,start=0.0,pretime=0.0,window=10.0,scroll_step=5.0):
          super().__init__()
          
          self.action_list    = ['FORWARD','FAST_FORWARD','REWIND','FAST_REWIND','START','END','TIME_DISPLAY_ALL']
          
          self.sfreq            = 1.0
          self.pretime          = 0.0
          self._timepoints      = None
          self._idx_scroll_step = 1
         #--- wx parameter floatspin increment
          self.inc_factor         = 1.000
      
    #--- alias
      @property
      def timepoints(self): return self._timepoints
      @timepoints.setter
      def timepoints(self,v):
          self._timepoints= v
        
      @property
      def end(self):
          return self.tsl2time(self.counts-1)
     
      @property
      def start(self):
          return self.tsl2time(self.idx_start)
      @start.setter
      def start(self,v):
          self.idx_start = self.time2tsl(v)
     #---
      @property
      def window(self):
          return self.tsl2time(self._idx_delta)
      @window.setter
      def window(self,v):
          self._idx_delta = self.time2tsl(v)
          self._update()
     #---
      @property
      def window_end(self):
          return self.tsl2time(self.idx_end)
    #---
      @property
      def scroll_step(self):
          return self.tsl2time(self._idx_scroll_step)
      @scroll_step.setter
      def scroll_step(self,v):
          self._idx_scroll_step = self.time2tsl(v)
    #---
      def time2tsl(self,t):
         # try:
         #     return np.where( t <= self._timepoints)[0][0]
         # except:
         #     logger.exception("---> ERROR timepoint not defiend or index out of range: tp: {} iddx: {}".format(self.timepoints.shape[0],t))
          return int( t * self.sfreq) # faster
    
      def tsl2time(self,t):
          #try:
          # return self._timepoints[t]
          #except:
          #    logger.exception("---> ERROR timepoint not defiend or index out of index: {}  < range: tp: {} ".format(t,self.timepoints.shape[0]))
          return ( t / self.sfreq )

       
      def action(self,act):
          if act == "START":
               self.idx_start=0
          elif act == "REWIND":
               self.idx_start = self._idx_start - self._idx_scroll_step
          elif act == "FAST_REWIND":
              self.idx_start -= self._idx_delta
          elif act == "FORWARD":
              self.idx_start += self._idx_scroll_step
          elif act == "FAST_FORWARD":
              self.idx_start += self._idx_delta
          elif act == "END":
              self.idx_start = self.counts
          elif act == "TIME_DISPLAY_ALL":
              self.toggle_all()
           
          #logger.info(" --> Time Action: {}\n  -> start: {} end: {} scroll step: {} window: {}".
          #        format(act,self.start,self.end,self.scroll_step,self.window))
          #self.GetInfo()
          
      def GetInfo(self):
          msg=[
          "timepoints:  {}".format(self.timepoints),
          "counts:      {}".format(self.counts),
          "start:       {}".format(self.start),
          "end:         {}".format(self.end),
          "scroll_step: {}".format(self.scroll_step),
         # "pretime:     {}".format(self.pretime),
          "window:      {}".format(self.window),
          "window_end:  {}".format(self.window_end),
          "idx_end:              {}".format(self.idx_end),
          "idx_end_range:        {}".format(self.idx_end_range),
          "idx_scroll_step:      {}".format(self._idx_scroll_step)]
         
          logger.info("---> Time option info:\n  -> " + "\n  -> ".join(msg))
          
          self._get_info()

class JuMEG_TSV_PLOT2D_OPTIONS(object):
      """
      Helper class for plot options
      """
      __slots__=["time","channels","n_cols","verbose","_is_init" ]
      def __init__ (self):        
          self.time        = OPT_TIME()
          self.channels    = OPT_CHANNELS()
          self.n_cols      = 1
          self.verbose     = False
          self._is_init    = False
          
      @property
      def isInit(self):
          return self._is_init
      @isInit.setter
      def isInit(self,v):
          self._is_init=v
    #--plots
      @property
      def n_plots(self):
          return self.channels.channels_to_display      
      @n_plots.setter
      def n_plots(self,v):
          self.channels.channels_to_display = v
    #-- rows
      @property
      def n_rows(self):
          return int(np.ceil(self.n_plots * 1.0 /self.n_cols))
    #--
      @property
      def plot_start(self):
          return self.channels.idx_start +1     
      @plot_start.setter
      def plot_start(self,v):
          self.channels.idx_start = v -1
     #---
      @property
      def do_scroll(self):
          return (self.channels.do_scroll or self.time.do_scroll)
      @do_scroll.setter
      def do_scroll(self,v):
          self.channels.do_scroll,self.time.do_scroll = v
          
      def GotoChannel(self,ch):
          self.plot_start = ch

      def action(self,act):
          
      #--- scroll channels
          if act in self.channels.action_list:
             self.channels.action(act)
             
      #--- scroll in time
          elif act in self.time.action_list :
             self.time.action(act)
             
      def GetInfo(self):
          return
          msg=[" plots: {} cols: {} rows: {}".format(self.n_plots,self.n_cols,self.n_rows),
                "start: {} doscroll: {}".format(self.plot_start,self.do_scroll) ]
               
          logger.info("---> Plot option info:\n"+ "\n  -> ".join(msg) )
          self.channels.GetInfo()
          self.time.GetInfo()
      
      def SetOptions(self,**kwargs):
          if kwargs.get("plot",False):
             d = kwargs.get("plot")
             self.channels.counts = d.get("counts",self.channels.counts)
             self.plot_start      = d.get("start",self.plot_start)
             self.n_plots         = d.get("n_plots",self.n_plots)
             self.n_cols          = d.get("n_cols",self.n_cols)
          
          if kwargs.get("time",False):
             d = kwargs.get("time")
             self.time.start       = d.get("start",self.time.start)
             self.time.pretime     = d.get("pretime",self.time.pretime)
             self.time.window      = d.get("window",self.time.window)
             self.time.inc_factor  = d.get("inc_factor",self.time.inc_factor)
             self.time.scroll_step = d.get("scroll_step",self.time.scroll_step)
           
          #logger.info("---> SetOption info")
          #self.GetInfo()

