# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:19:26 2015

@author: fboers
"""

import numpy as np
import mne # groups channel info

import wx.lib.colourdb as WX_CDB

class INFO_COLOR( object ):
      """
      plot color cls      
      
      get color from wx color db e.g.:  ('LIGHTGREEN', 144, 238, 144)
      
      """
      def __init__(self):
         #  self.__wxcdb    = WX_CDB
          #self.label_list = ['BLACK','RED','GREEN','MAGENTA','BLUE','BROWN','CYAN','DARKGREEN','DARKORANGE','TURQUOISE','VIOLET','GREY40','GREY50','GREY60','GREY70']
          self.label_list = ['BLACK','RED','AQUAMARINE','BLUE','MEDIUMBLUE','MIDNIGHTBLUE','ROYALBLUE','NAVYBLUE','CYAN',
                            'GREEN','DARKGREEN','YELLOW','MAGENTA','VIOLET','PURPLE1','GREY40','GREY50','GREY60','GREY70',
                            'GOLD','PERU','BROWN','ORANGE','DARKORANGE','PINK','HOTPINK','MAROON','ORCHID1']
            
          self.cdb_label_list = WX_CDB.getColourList()      # list ['red',blue'..]
          self.cdb_info_list  = WX_CDB.getColourInfoList()  # list [ ... ('LIGHTGREEN', 144, 238, 144), ...]

          self.__default_color = None
          self.default_color = 'GREY50'
        

      def label2index(self,label):
          # return self.label_list.index( label.upper() )    
          return self.cdb_label_list.index( label)  #.upper() )    
    
      #def index2colour2numpy(self,index):
      #    # return np.array(self.info_list[index][1:],dtype=np.uint32)    
      #    return np.array(self.cdb_info_list[index][1:],dtype=np.uint32)    
           
      def index2color(self,index):
          return np.array( self.cdb_info_list[index][1:],dtype=np.float32) / 255.0    
    
      def label2color(self,label):
          return self.index2color( self.cdb_label_list.index( label.upper() ) )   

      def iscolor(self,label):
          return (label in self.cdb_label_list)         
           
      def __get_default_color(self):
          return self.__default_color
      
      def __set_default_color(self,c):
          if self.iscolor(c):
             self.__default_color = c
          else:
             self.__default_color = 'GREY50'            
      default_color= property(__get_default_color,__set_default_color)
              
#----------------------------------------
class INFO_CHANNEL(object):
      
      def __init__ (self,nchan=None) :
          
          #self.color     = np.array([],dtype=np.float32)
          self.color_index = np.array([],dtype=np.uint32)
          self.color       = np.array([],dtype=np.float32)
         
          self.selected    = np.array([],dtype=np.bool)
          self.scale       = np.array([],dtype=np.float32)
          self.group_index = np.array([],dtype=np.uint32)
          self.bads        = np.zeros([],dtype=np.bool)
          self.label       = []
          #self.__ch_unit      = 
          if nchan:         
             self.init()
             
      def init(self,nchan):
         # self.color     = np.zeros((nchan,4),dtype=np.float32)
      
          self.color_index = np.zeros((nchan),dtype=np.uint32)
          self.color       = np.zeros([nchan,3],dtype=np.float32)
       
          self.selected    = np.zeros( nchan,dtype=np.bool)
          self.bads        = np.zeros( nchan,dtype=np.bool)
       
          self.scale       = np.zeros( nchan,dtype=np.float32)
          self.group_index = np.zeros( nchan,dtype=np.uint32)
          #self.__ch_unit     
        
   #---   
      def get_bads(self):
          return np.where(self.bads)
      def get_no_bads(self,idx):
          return np.where(self.bads==False)[0]              
   #---               
      def get_selected(self):
          return np.where(self.selected)[0]  
      def get_deselected(self):
          return np.where(self.selected == False)[0]
         
#----------------------------------------
class INFO_PLOT(object):
      def __init__(self) :
          self.channel= INFO_CHANNEL()
          self.color  = INFO_COLOR()



#----------------------------------------
class INFO_UNIT(object):
      def __init__(self,si="T",prefix=" "):
         self.__si          = si
         self.__prefix      = prefix
         # self.__label_list  = [" ","mili","micro","nano","pico","femto","atto"]
         self.__prefix_list = [" ","m","u","n","p","f","a"] 
         self.__choices     = [" ","m","u","n","p","f","a"]
        # self.__factors     = np.array( [ 1.0/( 1000 **np.arange(len(self.__label_list)))],dtype= np.uint32)
         self.__factors     = np.array( [ 1.0/( 1000 **np.arange(len(self.__prefix_list)))],dtype= np.uint32)
        
      def __get_unit(self):
         return self.__prefix+self.__si
      def __set_unit(self,s,prefix=None,si=None):
         if s:
            if len(s) > 1:             
               self.__prefix = s[0]
               self.__si     = s[1]
            else:
               self.__si     = s
               self.__prefix = " "
         elif prefix:
             self.__prefix = prefix
         elif si:
             self.__si = si
         
         self.__choices = []
         for s in self.__prefix_list:   
             self.__choices.append(s+self.__si)   
         self.__choices[0] = self.__si
         
      unit = property(__get_unit,__set_unit)       
      
      def __get_factor(self):
          return self.__factors[ self.__prefix_list.index( self.__prefix ) ]
          
      factor = property( __get_factor)
 
      def __get_label_list(self):
          return self.__label_list
      labels= property(__get_label_list)   
          
      def __get_prefix_list(self):
          return self.__prefix_list
      prefixes = property(__get_prefix_list)   
      
      def __get_choices(self):
          return self.__choices
      choices = property(__get_choices)   

      
#----------------------------------------
class INFO_GROUP(object):
     
      def __init__(self,label=None,group_index=None,picks=np.array([],dtype= np.uint32),bads=np.array([],dtype= np.uint32),color=None,unit=None,scale=1.0):
          
         self.label        = label
         self.index_list   = []
         self.channel_list = []
         self.unit         = INFO_UNIT()
         self.enabled      = True
         self.index        = np.array([picks],    dtype = np.uint32)
         self.bads         = np.array([bads],     dtype = np.bool)
         #self.selected_channesl = np.ones([picks.size],dtype = np.bool)
      
         self.color       = color
         self.unit.unit   = unit
         self.scale       = scale
         self.group_index = group_index
      
      def is_enabled(self):
          return self.enabled
          

class JuMEG_TSV_PLOT2D_DATA_INFO(object):
      def __init__ (self,raw=None,verbose=False,debug=False) :
          
          self.raw          = raw
          self.verbose      = verbose
          self.debug        = debug
          
          self.plt          = INFO_PLOT()
                   
          self.group      = {}
          self.group_list = []
          self.channels   = {}
          self.group_label_list = ['grad','mag','eeg','stim','eog','emg','ecg','ref_meg','resp','exci','ias','syst','misc','seeg','chpi']
          self.group_color_list = ['BLACK','RED','GREEN','MAGENTA','BLUE','BROWN','CYAN','DARKGREEN','DARKORANGE','TURQUOISE','VIOLET','GREY40','GREY50','GREY60','GREY70']
         #self.group_color_list = ['BLACK','RED','GREEN','MAGENTA','BLUE','BROWN','CYAN','DARKGREEN','DARKORANGE','TURQUOISE','VIOLET','GREY40','GREY50','GREY60','GREY70']
          
          self.bad_channel_color='GREY20'
          
          self.init_info()
    
    #-- TDOD exclude UTl channel    
      #   'UTL 001'
          
      def init_info(self,raw=None):
          print "update group channels"
          
          if raw:
             self.raw = raw
          
          if self.raw: 
             self.plt.channel.init(self.raw.info['nchan'])
             self.init_group_info()
          
         # else:
         #    assert"No RAW object defined"
          
          
   
      def index2channel_color(self,idx):
          """
          return numpy array f32
          """            
          return self.plt.channel.color[idx]
     
      def index2channel_label(self,idx):
          """
          return numpy array f32
          """            
          return self.plt.channel.label[idx]
          
#todo if update update ch color fron wxcdb          
          
          #self.
          #print "INFO CLTEST : %d  " %(idx)         
          #col= self.plt_color.index2color( self.plt_channels.color_index[idx] )
          #print col          
          #print "\n"          
          #return self.plt_color.index2color( self.plt_channels.color_index[idx] )
       # ToDo make color array for plot channel save time for index to color r.g.b values fron wxcdb           
       
      def init_group_info(self):
          
          self.group      = {}
          self.group_list = []
          
          for idx in range( self.raw.info['nchan'] ):
             #--- get group for idx meg eeg ... 
              g = mne.io.pick.channel_type(self.raw.info, idx)
              if g not in self.group_list:
                  
                 if g not in self.group_label_list:
                    self.group_label_list.append(g) 
                    self.group_color_list.append(self.default_color)
                    
                 self.group_list.append(g)
                 self.group[g]             = INFO_GROUP(label=g)
                 self.group[g].group_index = self.group_label_list.index(g)
                 self.group[g].unit.unit   = "X"       # raw.info['chs'][0]['unit'] -> number 112
                 
              self.group[g].index_list.append(idx)
              self.group[g].channel_list.append( self.raw.info['ch_names'][idx] )
             # ToDo load group settings from json fle
              self.group[g].group_index = self.group_label_list.index(g)
              self.group[g].color       = self.group_color_list[self.group[g].group_index]
              self.group[g].color_index = self.plt.color.label2index(self.group[g].color)
              self.group[g].selected    = True
           
           #--- working with numpy arrays in pl_channels class
              self.plt.channel.color_index[idx] = self.group[g].color_index # get color label/index from wx color  db!!!
              self.plt.channel.selected[idx]    = self.group[g].selected
              self.plt.channel.label.append( self.raw.info['ch_names'][idx] )
              self.plt.channel.scale[idx]       = 1.0
              self.plt.channel.group_index[idx] = self.group[g].group_index
                    
          for g in self.group_list:
              self.group[g].index = np.array( self.group[g].index_list,dtype=np.long )
              self.plt.channel.color[ self.group[g].index ]  = self.plt.color.index2color( self.group[g].color_index )
              
          if self.raw.info['bads']:   
             print self.raw.info['bads']       
             # self.plt.channel.bads[ self.raw.info['bads'] ] = False
             #self.plt_channels.selected[ raw.info['bads'] ] = False
  
         # self.data_index =   find selected idx  # np.arange(self.raw._data.shape[0],dtype=np.uint32)
        
      def update(self):
          for g in self.group_list:
              self.group[g].color_index = self.plt.color.label2index(self.group[g].color)
           
              
              self.plt.channel.color_index[self.group[g].index] = self.group[g].color_index
              self.plt.channel.color[ self.group[g].index ]     = self.plt.color.index2color( self.group[g].color_index )
     
              self.plt.channel.selected[self.group[g].index]    = self.group[g].selected
              
              if self.verbose:             
                 print "Info Group: "+ g              
                 print "---> group color: " + self.group[g].color
                 print " ---> selected channels:"
                 print self.plt.channel.selected[self.group[g].index]
                 print " ---> channel color index:"
                 print self.plt.channel.color_index[self.group[g].index]
                 print "\n"
        
              #self.plt_channels.scale[self.group[g].index]       = 
              #self.group[g].unit.unit
         
    
      def info(self):
          
          for g in self.group_list:
              print'--->'+ g
              print self.group[g].unit.unit
              print self.group[g].color
              print self.group[g].selected
              #self.plt_channels.selected[idx]
              #print self.plt_channels.scale[idx]
              #print self.plt_channels.group_index[idx]
              
      def selected_channel_index(self):
          return np.array( np.where(self.plt.channel.selected == True),dtype=np.uint32).flatten()
 
      def selected_channel_label(self):
          return self.plt.channel.label[ self.selected_channel_index() ]
              
""""


           

# sel = []
#for k, name in enumerate(ch_names):
#if (len(include) == 0 or name in include) and name not in exclude:
#sel.append(k)
#sel = np.unique(sel)
#np.sort(sel)

     
      def update_channels(self,idx=None,label=None,color=np.array(np.zeros(4),dtype=float32)):
          print "update channels"
          #self.__dinfo = np.zeros( self.raw._data.shape,
          #                        [ ('position', [ ('x', float, 1),('y', float, 1),('z', float, 1)]),
          #                          ('color',    [ ('r', float, 1),('g', float, 1),('b', float, 1)]),
          #                          ('range',    [ ('min', float, 1),('max', float, 1),('scale', float, 1)]),
          #                        ])
          
          # color rgb
          # position x,y,z          
          # range min,max,scale
          
          self.channels[idx]=
          INFO_CHANNELS
          label=None,index=None,group_index=None,is_bad=False,color=np.array(np.zeros(4),dtype=np.float32),unit=None,scale=1.0
          self._meg_idx       = idx
          self.__color        = np.zeros( (self.raw._data.shape[0],3) ,dtype=np.float32)
          self.__color[self._meg_idx,:] = np.array((1.0,0.5,0.4,0.0),dtype=np.float32)
          
          #self.__meg_position = np.zeros( (self._meg_idx.size,12),dtype=np.float32)
          # self.__dinfo[self.pick_meg_nobads]
          
          
          
      def __set_channel_color(self,idx=None,color=None):
          if color in self.color_tabel 
          
          if idx:
             self.__color[idx,:] = c  #np.array((1.0,0.5,0.4,0.0),dtype=np.float32)
          
      def __get_channel_color(self,idx):
          return self.__color[idx]
      
      channel_color = property(__set_channel_color,__get_channel_color)
 

class __INFO_CHANNELS(object):
      
      def __init__ (self,label=None,index=None,group_index=None,is_bad=False,color=np.array(np.zeros(4),dtype=np.float32),unit=None,scale=1.0) :
          
          self.index       = index          
          self.label       = label
          self.group_index = group_index
          self.is_bad      = is_bad
          self.color       = color
          self.unit        = unit          
          self.scale       = scale


          self.__ch_color     = np.zeros(self.raw.info['nchan'],4),dtype=np.float32)
          self.__ch_selected  = np.zeros(self.raw.info['nchan'],dtype=np.bool)
          self.__ch_scale     = np.zeros(self.raw.info['nchan'],dtype=np.float32)
          self.__ch_group_idx = np.zeros(self.raw.info['nchan'],4),dtype=np.float32)
          self.__ch_unit      = ()
          
          
          self.plt_channels.update(nchan=self.raw.info['nchan'])

class CHANNEL(object):
  __slots__=['group','unit']
# __slots__=['label','index','group','unit']

  def __init__(self,g,u):
      #self.label= l
      #self.index= i
      self.group= g
      self.unit = u
     #---




      
class __PLT_COLOR(object):
   
   __slots__=['label','r','g','b','a']   
   
   def __init( self,l,r,g,b,a):
        
       self.label = l
       self.r  = r
       self.g  = g
       self.b  = b
       self.a  = a

#    def __set_r(self,v):
#        self.__r=v
#    def __get_r(self,v):
#      return self.__r
#    r = property(__set_r,__get_r)#
#
#    def __set_g(self,v):
#        self.__g=v
#    def __get_g(self,v):
#      return self.__g
#    g = property(__set_g,__get_g)#
#
#    def __set_b(self,v):
#        self.__b=v
#    def __get_b(self,v):
#      return self.__b
#    b = property(__set_b,__get_b)#
#
#    def __set_a(self,v):
#        self.__a=v
#    def __get_a(self,v):
#      return self.__a
#    a = property(__set_a,__get_a)




from wx import wx.ColourDatabase


import wx.lib.colourdb
 
colour_db= wx.lib.colourdb

c_label_list= colour_db.getColourList

c_info_list =colour_db.getColourListInfo


#--- label to index 
c_label_list.index('RED')
184

#--- index to colour
c_info_list[index][1:]
(248, 248, 255) # tuple




ci[1][1:]
Out[70]: (248, 248, 255)


# clist.getColourInfoList  clist.getColourList      clist.updateColourDB
ci=clist.getColourInfoList() 

ci[1]
Out[66]: ('GHOST WHITE', 248, 248, 255)
ci[1][1:]
Out[70]: (248, 248, 255)

colour_db.getColourList

# TODO check for wx ColourDatabaase
# wx.ColourDatabase()




#---
#--- https://github.com/mne-tools/mne-python/blob/master/mne/io/pick.py  lines 20ff
#---  type : 'grad' | 'mag' | 'eeg' | 'stim' | 'eog' | 'emg' | 'ecg' |'ref_meg' | 'resp' | 'exci' | 'ias' | 'syst' | 'misc'|'seeg' | 'chpi'

#--- gnuplot color table ->  >>gnuplot show palette 
#--- ToDo may use WX or read from jason 
color_tab={"antiquewhite": [205, 192, 176, 0], "aquamarine": [127, 255, 212, 0], "beige": [245, 245, 220, 0], "bisque": [205, 183, 158, 0], 
           "black": [0, 0, 0, 0], "blue": [0, 0, 255, 0], "brown": [165, 42, 42, 0], "brown4": [128, 20, 20, 0], "chartreuse": [124, 255, 64, 0], 
           "coral": [255, 127, 80, 0], "cyan": [0, 255, 255, 0], "dark-blue": [0, 0, 139, 0], "dark-chartreuse": [64, 128, 0, 0], "dark-cyan": [0, 238, 238, 0],
           "dark-goldenrod": [184, 134, 11, 0], "dark-gray": [160, 160, 160, 0], "dark-green": [0, 100, 0, 0], "dark-grey": [160, 160, 160, 0], 
           "dark-khaki": [189, 183, 107, 0], "dark-magenta": [192, 0, 255, 0], "dark-olivegreen": [85, 107, 47, 0], "dark-orange": [192, 64, 0, 0], 
           "dark-pink": [255, 20, 147, 0], "dark-plum": [144, 80, 64, 0], "dark-red": [139, 0, 0, 0], "dark-salmon": [233, 150, 122, 0], 
           "dark-spring-green": [0, 128, 64, 0], "dark-turquoise": [0, 206, 209, 0], "dark-violet": [148, 0, 211, 0], "dark-yellow": [200, 200, 0, 0],
           "forest-green": [34, 139, 34, 0], "gold": [255, 215, 0, 0], "goldenrod": [255, 192, 32, 0], "gray": [190, 190, 190, 0], "gray0": [0, 0, 0, 0], 
           "gray10": [26, 26, 26, 0], "gray100": [255, 255, 255, 0], "gray20": [51, 51, 51, 0], "gray30": [77, 77, 77, 0], "gray40": [102, 102, 102, 0],
           "gray50": [127, 127, 127, 0], "gray60": [153, 153, 153, 0], "gray70": [179, 179, 179, 0], "gray80": [204, 204, 204, 0], "gray90": [229, 229, 229, 0], 
           "green": [0, 255, 0, 0], "greenyellow": [160, 255, 32, 0], "grey": [192, 192, 192, 0], "grey0": [0, 0, 0, 0], "grey10": [26, 26, 26, 0], 
           "grey100": [255, 255, 255, 0], "grey20": [51, 51, 51, 0], "grey30": [77, 77, 77, 0], "grey40": [102, 102, 102, 0], "grey50": [127, 127, 127, 0], 
           "grey60": [153, 153, 153, 0], "grey70": [179, 179, 179, 0], "grey80": [204, 204, 204, 0], "grey90": [229, 229, 229, 0], 
           "honeydew": [240, 255, 240, 0], "khaki": [240, 230, 140, 0], "khaki1": [255, 255, 128, 0], "lemonchiffon": [255, 255, 192, 0], 
           "light-blue": [173, 216, 230, 0], "light-coral": [240, 128, 128, 0], "light-cyan": [224, 255, 255, 0], "light-goldenrod": [238, 221, 130, 0], 
           "light-gray": [211, 211, 211, 0], "light-green": [144, 238, 144, 0], "light-grey": [211, 211, 211, 0], "light-magenta": [240, 85, 240, 0], 
           "light-pink": [255, 182, 193, 0], "light-red": [240, 50, 50, 0], "light-salmon": [255, 160, 112, 0], "light-turquoise": [175, 238, 238, 0], 
           "magenta": [255, 0, 255, 0], "medium-blue": [0, 0, 205, 0], "mediumpurple3": [128, 96, 192, 0], "midnight-blue": [25, 25, 112, 0], 
           "navy": [0, 0, 128, 0], "olive": [160, 128, 32, 0], "orange": [255, 165, 0, 0], "orange-red": [255, 69, 0, 0], "orangered4": [128, 20, 0, 0],
           "orchid": [255, 128, 255, 0], "orchid4": [128, 64, 128, 0], "pink": [255, 192, 192, 0], "plum": [221, 160, 221, 0], "purple": [192, 128, 255, 0], 
           "red": [255, 0, 0, 0], "royalblue": [65, 105, 225, 0], "salmon": [250, 128, 114, 0], "sandybrown": [255, 160, 96, 0], "sea-green": [46, 139, 87, 0], 
           "seagreen": [193, 255, 193, 0], "sienna1": [255, 128, 64, 0], "sienna4": [128, 64, 20, 0], "skyblue": [135, 206, 235, 0],
           "slateblue1": [128, 96, 255, 0], "slategray": [160, 182, 205, 0], "slategrey": [160, 182, 205, 0], "spring-green": [0, 255, 127, 0], 
           "steelblue": [48, 96, 128, 0], "tan1": [255, 160, 64, 0], "turquoise": [64, 224, 208, 0], "violet": [238, 130, 238, 0], 
           "web-blue": [0, 128, 255, 0], "web-green": [0, 192, 0, 0], "white": [255, 255, 255, 0], "yellow": [255, 255, 0, 0], "yellow4": [128, 128, 0, 0]}

#http://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html
#Z = np.zeros(10, [ ('position', [ ('x', float, 1),
#                                  ('y', float, 1)]),
#                    ('color',    [ ('r', float, 1),
#                                   ('g', float, 1),
#                                   ('b', float, 1)])])
#


#  tstep = 0.2
#    itmin = int(floor(tmin * raw.info['sfreq']))
#    itmax = int(ceil(tmax * raw.info['sfreq']))
#    itstep = int(ceil(tstep * raw.info['sfreq']))
#    print ">>> Set time-range to [%7.3f,%7.3f]" % (tmin, tmax)

#from mne.io.pick import pick_types, channel_indices_by_type


##################################################
#
# Get indices of matching channel names from list
# EE noise reducer
##################################################
def channel_indices_from_list(fulllist, findlist, excllist=None):
    Get indices of matching channel names from list

    Parameters
    ----------
    fulllist: list of channel names
    findlist: list of (regexp) names to find
              regexp are resolved using mne.pick_channels_regexp()
    excllist: list of channel names to exclude,
              e.g., raw.info.get('bads')

    Returns
    -------
    chnpick: array with indices
    
    chnpick = []
    for ir in xrange(len(findlist)):
        if findlist[ir].translate(None, ' ').isalnum():
            try:
                chnpicktmp = ([fulllist.index(findlist[ir])])
                chnpick = np.array(np.concatenate((chnpick, chnpicktmp), axis=1), dtype=int)
            except:
                print ">>>>> Channel '%s' not found." % findlist[ir]
        else:
            chnpicktmp = (mne.pick_channels_regexp(fulllist,findlist[ir]))
            if len(chnpicktmp) == 0:
                print ">>>>> '%s' does not match any channel name." % findlist[ir]
            else:
                chnpick = np.array(np.concatenate((chnpick, chnpicktmp), axis=1), dtype=int)
    if len(chnpick) > 1:
        # Remove duplicates:
        chnpick = np.sort(np.array(list(set(np.sort(chnpick)))))

    if excllist is not None and len(excllist) > 0:
        exclinds = [fulllist.index(excllist[ie]) for ie in xrange(len(excllist))]
        chnpick = list(np.setdiff1d(chnpick, exclinds))
    return chnpick




"""        
