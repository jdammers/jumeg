# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 08:42:19 2015

@author: fboers
"""
"""
class OPT_PLOT(object):
      
      def __init__ (self,channels=10,cols=1,type='subplot') :
          self.channels_to_display = channels
          self.cols      = cols
          self.type      = type
          self.do_scroll = False
         
      def __number_of_plots(self):
          return self.channels  # * self.cols 
      number_of_plots = property(__number_of_plots)

      def __get_rows(self):
          return np.ceil(self.channels*1.0 / self.cols*1.0)
      rows = property(__get_rows)
"""
       
import numpy as np


class OPT_BASE(object):
      def __init__ (self):
          self.do_scroll      = False
          self.last_start_idx = -1.0
          self.last_end_range = -1.0
          self._idx_start     = 0
          self._idx_delta     = 1
          self._counts        = 1
          self.counts         = 1
      
      
      def _get_counts(self):
          return self._counts
          
      def _set_counts(self,v):
          self.last_start_idx = -1.0
          self.last_end_range = -1.0
          self._idx_start     = 0
          self._counts        = v  
          self._ck_idx_range()         
      counts = property(_get_counts,_set_counts)         
      
      def __get_idx_start(self):
          return self._idx_start
      def __set_idx_start(self,v): 
          self._idx_start = v
          self._ck_idx_range()           
      idx_start=property(__get_idx_start,__set_idx_start)    
     
     
      def index_range(self): 
          """  calc start & end index range
               return: idx start, idx end range (end idx +1)          
          """
          if ( self.last_start_idx != self.idx_start):
             self.last_start_idx = self.idx_start             
             self.do_scroll      = True
              
          elif( self.last_end_range != self.idx_end):
              self.last_end_range  = self.idx_end             
              self.do_scroll       = True
          else:
              self.do_scroll = False              
                  
          return self.idx_start,self.idx_end_range
          
      def _ck_idx_range(self):
          if self._idx_start < 0:
             self._idx_start = 0
          if (self._idx_start + self._idx_delta > self._counts):
             self._idx_start = self._counts - self._idx_delta
             if self._idx_start < 0:
                self._idx_start = 0       
                self._idx_delta = self._counts -1
                
      def _ck_ch_range(self,v):
          if (v > self.counts ):
              v = self.counts
          if (v < 1):
              v = 1     
          return v  
  
#----------------------------------------
class OPT_CHANNELS(OPT_BASE):
      
      def __init__ (self,start=0,counts=0):
          super(OPT_CHANNELS,self).__init__()
          self.action_list    = ['UP','DOWN','PAGEUP','PAGEDOWN','TOP','BOTTOM','CHANNELS_DISPLAY_ALL']          
          
          self.__idx_start_save = -1
          self.__channels_to_display_save=-1
          
      def __get_last_channel_idx(self):
          return self.counts -1
      last_channel_idx = property(__get_last_channel_idx)              
      
      def __get_idx_end(self):
          self._ck_idx_range()
          return self._idx_start + self._idx_delta -1 
      idx_end = property(__get_idx_end)          
      
      def __get_idx_end_range(self):
          """ calc +1 for range fct"""          
          return self.idx_end +1
      idx_end_range= property(__get_idx_end_range)       
      last_channel = property(__get_idx_end_range)         
      
      def __get_channels_to_display(self):
          return self._idx_delta         
      def __set_channels_to_display(self,v):    
          self._idx_delta = v
          self._ck_idx_range()        
      channels_to_display = property(__get_channels_to_display,__set_channels_to_display)    

      def toggle_all_channels(self):
          if ( self.idx_start > 0 ):
             self.__idx_start_save = self.idx_start
             self.idx_start = 0 
          else:
             self.idx_start = self.__idx_start_save
          if (self.channels_to_display < self.counts) :
             self.__channels_to_display_save = self.channels_to_display
             self.channels_to_display = self.counts
          else:
             self.channels_to_display = self.__channels_to_display_save
                 
class OPT_TIME(OPT_BASE):
      
      def __init__(self): #,timepoints=1,sfreq=1.0,start=0.0,pretime=0.0,window=10.0,scroll_step=5.0):
          super(OPT_TIME,self).__init__()
          
          self.action_list    = ['FORWARD','FAST_FORWARD','REWIND','FAST_REWIND','START','END','TIME_DISPLAY_ALL']
          
          self.sfreq          = 1.0
          self.scroll_step    = 1.0
          
          self.__pretime      = 0.0
          self._idx_scroll_speed = 1
          
          self.timepoints         = 1       
          #self.start              = start
          self.window             = 1.0 #window
          self.scroll_step        = 1.0 #scroll_step
         #--- wx parameter floatspin increment  
          self.inc_factor         = 1.000

          self.__start_save  = -1
          self.__window_save = -1        
      
      
      def _get_counts(self):
         return super(OPT_TIME,self)._get_counts()
      def _set_counts(self,v):
         return super(OPT_TIME,self)._set_counts(v)
      timepoints = property(_get_counts,_set_counts) 
     
      def __get_last_idx(self):
          return self.counts -1
      idx_last = property(__get_last_idx)              
     
      def __get_end(self):
          return self.tsl2time(self.counts)
      end = property(__get_end)      
   
    #---
      def __get_pretime(self):   
          return self.__pretime
      def __set_pretime(self,v):    
          self.__pretime = v
      pretime = property(__get_pretime,__set_pretime)
   
   #---   
      def __get_start(self):
          return self.tsl2time(self.idx_start)
      def __set_start(self,v):
          self.idx_start = self.time2tsl(v)        
      start = property(__get_start,__set_start)
   #---   
      def __get_idx_window(self):
          self._ck_idx_range()
          return self._idx_delta
      def __set_idx_window(self,v):
          self._idx_delta = v
          self._ck_idx_range()    
      idx_window = property(__get_idx_window,__set_idx_window)
    #--- 
      def __get_window(self):
          return self.tsl2time(self.idx_window)
      def __set_window(self,v):
          self.idx_window = self.time2tsl(v)        
      window = property(__get_window,__set_window)
    #---  
      def __window_end_idx(self):
          return self.idx_start + self.idx_window
      window_end_idx = property(__window_end_idx)
    #---        
      def __window_end(self):
          return self.tsl2time(self.window_end_idx)
      window_end = property(__window_end)
       
      def __get_window_end_idx_range(self):
          """ calc +1 for range fct"""
          return self.window_end_idx +1
      window_end_idx_range = property(__get_window_end_idx_range)
         
    #---
      def __get_idx_end(self):
          return self.idx_start + self.idx_window -1
      idx_end = property(__get_idx_end)          
      
      def __get_idx_end_range(self):
          """ calc +1 for range fct"""          
          return self.idx_end +1
      idx_end_range= property(__get_idx_end_range)       
    #---
      def __get_idx_scroll_step(self):
          return self.__idx_scroll_step
      def __set_idx_scroll_step(self,v):
          self.__idx_scroll_step = self.time2tsl(v)   
      idx_scroll_step = property(__get_idx_scroll_step,__set_idx_scroll_step)          
    #---       
      def __get_scroll_step(self):
          return self.time2tsl(self.idx_scroll_step)
      def __set_scroll_step(self,v):
          self.idx_scroll_step = self.time2tsl(v)   
      scroll_step = property(__get_scroll_step,__set_scroll_step)   
  
    #---
      def time2tsl(self,t):
          return int( t * self.sfreq ) 
    
      def tsl2time(self,t):
          return  t / self.sfreq

      def toggle_all_times(self):
         if ( self.start > 0 ):
              self.__start_save = self.start
              self.start = 0 
         else:
              self.start = self.__start_save
         if (self.window < self.end) :
            self.__window_save = self.window
            self.window = self.end
         else:
            self.window = self.__window_save
     
class JuMEG_TSV_PLOT2D_OPTIONS(object):
      """
      Helper class for plot options
      """

      def __init__ (self):        
          self.time        = OPT_TIME()
          self.channels    = OPT_CHANNELS()
          self.__plot_cols = 1
          
          self.verbose     = False
          
    #--plots 
      def __get_plots(self):
          return self.channels.channels_to_display      
      def __set_plots(self,v):
          self.channels.channels_to_display = v
      plots = property(__get_plots,__set_plots)
    #-- cols      
      def __get_cols(self):
          return self.__plot_cols
      def __set_cols(self,v):
          self.__plot_cols = v
      plot_cols = property(__get_cols,__set_cols)  
   
   #-- rows      
      def __get_rows(self):
          return int(np.ceil(self.plots * 1.0 /self.plot_cols))
      plot_rows = property(__get_rows)  
           
   #--       
      def __get_plot_start(self):
          return self.channels.idx_start +1     
      def __set_plot_start(self,v):
          self.channels.idx_start = v -1
      plot_start = property(__get_plot_start,__set_plot_start)
      goto_channel=property(__get_plot_start,__set_plot_start)    

   #---
      def __get_do_scroll(self):
          return (self.channels.do_scroll or self.time.do_scroll)
      def __set_do_scroll(self,v):
          self.channels.do_scroll,self.time.do_scroll = v
      do_scroll = property(__get_do_scroll,__set_do_scroll)          
          
                     
      
      def action(self,act):
          
      #--- scroll channels
          if act in self.channels.action_list:    
             
             if   act == "TOP":
                  self.channels.idx_start = 0
             elif act == "UP":
                  self.channels.idx_start -= 1  #A1 on top
             elif act == "DOWN":
                  self.channels.idx_start += 1  # A248 on bottom
             elif act == "PAGEUP":
                  self.channels.idx_start -= self.plots -1
             elif act == "PAGEDOWN":
                  self.channels.idx_start += self.plots -1
             elif act == "BOTTOM":
                  self.channels.idx_start = self.channels.counts # - self.plot.plots +1
             elif act == "CHANNELS_DISPLAY_ALL":
                  self.channels.toggle_all_channels()                                
                 
               
      #--- scroll in time
          elif act in self.time.action_list : 
               
               if   act == "START":
                    self.time.start = 0.0
               elif act == "REWIND":
                    self.time.start -= self.time.scroll_speed
               elif act == "FAST_REWIND":
                    self.time.start -= self.time.window
               elif act == "FORWARD":
                    self.time.start += self.time.scroll_speed
               elif act == "FAST_FORWARD":
                    self.time.start += self.time.window
               elif act == "END":
                    self.time.start = self.time.end
               elif act == "TIME_DISPLAY_ALL":
                    self.time.toggle_all_times()
                    