import os,sys

import numpy as np
import mne
from jumeg.jumeg_base import JuMEG_Base


#print "########## Refchan geo data:"
# This is just for info to locate special 4D-refs.
#for iref in refpick:
#    print raw.info['chs'][iref]['ch_name'],
#raw.info['chs'][iref]['loc'][0:3]


#----------------------------------------------------------------------------------------
class JuMEG_TSV_DATA(JuMEG_Base):

      def __init__(self, fname=None,path=None,raw=None,experiment=None,verbose=False,debug=False,duration=None,start=None,n_channels=None,bads=None):
          super(JuMEG_TSV_DATA, self).__init__()

          self.verbose    = verbose
          self.fname      = fname
          self.path       = path
          self.raw        = raw
          self.experiment = experiment
          self.debug      = debug
          self.duration   = duration
          #self.start      = start
          #self.n_channels = n_channels
          self.bads       = bads
          self.append_bads= True

          self.dtype_original = None
          self.dtype_plot     = np.float32

          self.raw_is_loaded  = False

          #print self.fname

          #self.__color= np.array([])

      def update(self,fname=None):
          if fname:
             self.path  = os.path.dirname(  fname )
             self.fname = os.path.basename( fname )
             self.load_raw()
            # self.update_channel_info()
             
      def load_raw(self):

          self.raw_is_loaded = False
          self.raw = None

          if self.fname is None:
             print"ERROR no file foumd!!\n"
          else:

             if self.path:
                self.raw = mne.io.Raw(self.path+"/"+self.fname,preload=True)
             else:
                self.raw = mne.io.Raw(self.fname,preload=True)

             self.raw_is_loaded  = True
             self.dtype_original = self.raw._data.dtype

             if self.bads:
                self.raw,self.bads = self.update_bad_channels(raw=self.raw,bads=self.bads,append=self.append_bads,save=False)

             # TODO set _data .astype(np.float32)
             #if self.raw._data.dtype != self.dtype_plot :
             #   self.raw._data = self.raw._data.astype( self.dtype_plot )


          return self.raw,self.bads


      def save_bads(self):
          return self.update_bad_channels(raw=self.raw,bads=self.bads,append=self.append_bads,save=True)

     