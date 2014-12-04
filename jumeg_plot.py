import os
import numpy as np
import matplotlib.pylab as pl
import mne

from jumeg.jumeg_base import jumeg_base

#################################################################
#
# plot_powerspectrum
#
#################################################################
def plot_powerspectrum(fname,raw=None,picks=None,dir_plots="plots",tmin=None,tmax=None,fmin=0.0,fmax=450.0,n_fft=4096):
         ''' 
       
         '''
         import os
         import matplotlib.pyplot as pl
         import mne
         from distutils.dir_util import mkpath
         
         if raw is None:
            assert os.path.isfile(fname), 'ERROR: file not found: ' + fname
            raw = mne.io.Raw(fname,preload=True)
    
         if picks is None :
            picks = jumeg_base.pick_meg_nobads(raw)
        
         dir_plots  = os.path.join( os.path.dirname(fname),dir_plots )
         base_fname = os.path.basename(fname).strip('.fif')
      
         mkpath(dir_plots)
        
         file_name = fname.split('/')[-1]
         fnfig = dir_plots +'/'+ base_fname + '-psds.png'
         
         pl.figure()
         pl.title('PSDS ' + file_name)
         ax = pl.axes()
         fig = raw.plot_psds(fmin=fmin,fmax=fmax,n_fft=n_fft,n_jobs=1,proj=False,ax=ax,color=(0, 0, 1),picks=picks, area_mode='range')
         pl.ioff()
         #pl.ion()
         fig.savefig(fnfig)
         pl.close()
         
         return fname
