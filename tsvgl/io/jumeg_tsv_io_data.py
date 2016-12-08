import os,sys
import numpy as np

import mne
from jumeg.jumeg_base import JuMEG_Base_IO


#print "########## Refchan geo data:"
# This is just for info to locate special 4D-refs.
#for iref in refpick:
#    print raw.info['chs'][iref]['ch_name'],
#raw.info['chs'][iref]['loc'][0:3]

#fname=opt.fname,path=opt.path,verbose=opt.v,debug=opt.d,experiment=opt.exp,
# duration=opt.duration,start=opt.start,n_channels=opt.n_channels,bads=opt.bads

#----------------------------------------------------------------------------------------
class JuMEG_TSV_IO_DATA(JuMEG_Base_IO):

      def __init__(self, fname=None,path=None,raw=None,experiment=None,verbose=False,bads=None):
          
          super(JuMEG_TSV_IO_DATA, self).__init__()

          self.verbose    = verbose
          self.fname      = fname
          self.path       = path
          self.raw        = raw
          self.experiment = experiment
          self.bads       = bads
          self.append_bads= True

          self.dtype_original = None
          self.dtype_plot     = np.float32

          self.raw_is_loaded  = False
        
      def update(self,path=None,fname=None,raw=None,reload=False):
          
          if (reload and self.raw_is_loaded):
              
             fname    = self.raw.info.get('filename')
             self.raw = None
         #---
          self.raw_is_loaded = False
          
          if raw:          
             self.dtype_original  = self.raw._data.dtype
             self.path,self.fname = os.path.split(raw.info.get('filename')) 
             self.raw  = raw
             self.bads = raw.info.get('bads')
             self.raw_is_loaded   = True
             return self.raw,self.bads

          if fname:
             if path:
                self.path = path
             else:
                self.path = os.path.dirname(  fname )
             
             self.fname = os.path.basename( fname )
          if not self.path:
                 self.path ="."+ os.path.sep
          elif not os.path.exists(self.path):
                 self.path ="."+ os.path.sep
                 print "JuMEG TSV IO error: path not exist: " + self.path
          if self.fname:   
             self.load_raw()
            # self.update_channel_info()
             
      def load_raw(self):

          self.raw_is_loaded = False
          self.raw = None
          if self.verbose:
             print "---> JuMEG TSV IO  loading data" 
             print"      path: "+ self.path
             print"      file: "+ self.fname +"\n"
          
          if self.fname is None:
             print"ERROR no file found!!\n"
          else:

             if os.path.exists(self.path):
                self.raw = mne.io.Raw(self.path+"/"+self.fname,preload=True)
             else:
                self.raw = mne.io.Raw(self.fname,preload=True)

             self.raw_is_loaded  = True
             self.dtype_original = self.raw._data.dtype

             if self.bads:
                self.raw,self.bads = self.update_bad_channels(raw=self.raw,bads=self.bads,append=self.append_bads,save=False)
          
          if self.verbose:
             print "---> JuMEG TSV IO  done loading data" 
             print self.raw_is_loaded 
             print "\n"
             
          return self.raw,self.bads


      def save_bads(self):
          return self.update_bad_channels(raw=self.raw,bads=self.bads,append=self.append_bads,save=True)

     