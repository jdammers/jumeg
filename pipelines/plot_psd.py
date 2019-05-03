#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 30.04.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import os,logging
import numpy as np


#from matplotlib import gridspec as grd
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages

from mne.time_frequency import psd_welch
import mne

from jumeg.base import jumeg_logger
from jumeg.base.jumeg_base import jumeg_base as jb

logger = logging.getLogger('root')

__version__="2019.04.18.001"

class JUMEG_PLOT_BASE(object):


class JUMEG_PLOT_PSD(object):
    __slots__ = ["picks","fmin","fmax","tmin","tmax","proj","n_fft","color","area_mode","area_alpha","n_jobs","title1","title2",
                 "info","show","fnout","n_plots","file_extention","name","_fig","_plot_index","ylim"]
    
    def __init__(self,**kwargs):
        super().__init__()
        self._init(**kwargs)
    
    @property
    def fig(self): return self._fig
    
    @property
    def plot_index(self): return self._plot_index
    
    def _init_defaults(self):
        self.picks      = None
        self.fmin       =  0.0
        self.fmax       = 60.0
        self.tmin       =  0.0
        self.tmax       = None
        self.n_fft      = 4096
        self.color      = 'blue'
        self.area_mode  ='range'
        self.area_alpha = 0.33
        self.title1     = 'before denoising'
        self.title2     ='after denoising'
        self.info       = None
        self.show       = True
        self.proj       = False
        self.n_jobs     = 1
        self.n_plots    = 2
        self.file_extention = ".pdf"
        self.name           = "denoising"
        self._plot_index    = 1
        
    def get_parameter(self,key=None):
       """
       get  host parameter
       :param key:)
       :return: parameter dict or value of  parameter[key]
       """
       if key: return self.__getattribute__(key)
       return {slot: self.__getattribute__(slot) for slot in self.__slots__}
  
    def _init(self,**kwargs):
       #--- init slots
        for k in self.__slots__:
            self.__setattr__(k,None)
      
        self._init_defaults()
        self._update_from_kwargs(**kwargs)
        self._init_figure()
        
    def _update_from_kwargs(self,**kwargs):
        for k in self.__slots__:
            self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
   
    def _init_figure(self):
       #--- A4 landscape
        plt.rc('figure',figsize=(11.69,8.27))
        plt.rcParams.update({ 'font.size':8 })
    
        self._fig = plt.figure(self.name)
        plt.clf()
        #fig = pl.figure(name,figsize=(10, 8), dpi=100))
    
    def calc_psd_welch(self,raw,**kwargs):
        """
        
        :param raw:
        :return:
         psds,freq
        """
        self._update_from_kwargs(**kwargs)
        self.picks = jb.picks.check_dead_channels(raw,picks=self.picks)
      
        psds, freqs = psd_welch(raw, picks=self.picks, fmin=self.fmin, fmax=self.fmax,
                                tmin=self.tmin, tmax=self.tmax, n_fft=self.n_fft,
                                n_jobs=self.n_jobs, proj=self.proj)
           
        return psds,freqs
      
   
    def save_plot(self,fname=None,plot_dir="plots"):
        """
        
        :param fname:
        :param plot_dir:
        :return:
        """
        if fname:
           fout_path = os.path.dirname(fname)
           fnout     = os.path.splitext( os.path.basename(fname) )[0] + self.file_extention
        else:
           fnout     = "test"+self.file_extention
           fout_path = "."
      
        if plot_dir:
           fout_path = os.path.join(fout_path,plot_dir)
           try:
               os.makedirs(fout_path,exist_ok=True)
           except:
               logger.exception("---> can not create plot\n" +
                                "  -> directory: {}\n".format(fout_path) +
                                "  -> filename : {}".format(fname))
               return
            
        if fnout:
           #self.fig.savefig( os.path.join(fout_path,fnout, format='png')
           self.fig.savefig(os.path.join(fout_path,fnout),dpi=self.dpi)
           if self.verbose:
              logger.info("---> done saving plot: {}".format(os.path.join(fout_path,fnout)) )

    def show_plot(self):
        #self.fig.show()
        plt.show()
    def close(self):
        self.fig.close() #self.name)
        #self.fig.ion()
    
    def plot_power_spectrum(self,psds=None,freqs=None):
        if self.plot_index > self.n_plots:
           return
        
        p1 = plt.subplot(self.n_plots,1,self.plot_index)
      #--- Convert PSDs to dB
      
        psds = 10 * np.log10(psds)
        psd_mean = np.mean(psds,axis=0)
        
        if self.area_mode == 'std':
            psd_std = np.std(psds,axis=0)
            hyp_limits = (psd_mean - psd_std,psd_mean + psd_std)
        elif self.area_mode == 'range':
            hyp_limits = (np.min(psds,axis=0),np.max(psds,axis=0))
        else:  # area_mode is None
            hyp_limits = None
    
        p1.plot(freqs,psd_mean,color=self.color)
        
        if hyp_limits is not None:
            p1.fill_between(freqs,hyp_limits[0],y2=hyp_limits[1],
                            color=self.color,alpha=self.area_alpha)
        if self.plot_index == 1:
            p1.set_title(self.title1)
            self.ylim = [np.min(psd_mean) - 10,np.max(psd_mean) + 10]
        else:
            p1.set_title(self.title2)
    
        p1.set_xlabel('Freq (Hz)')
        p1.set_ylabel('Power Spectral Density (dB/Hz)')
        p1.set_xlim(freqs[0],freqs[-1])
        p1.set_ylim(self.ylim[0],self.ylim[1])

        self._plot_index += 1

    def plot(self,raw,**kwargs):
        psds,freqs = self.calc_psd_welch(raw,**kwargs)
        self.plot_power_spectrum(psds=psds,freqs=freqs)
        
#=========================================================================================
#==== MAIN
#=========================================================================================
def main():
  
    #from plot_psd import JUMEG_PLOT_PSD
    
   
   #--- init/update logger
   
    jumeg_logger.setup_script_logging(logger=logger,level="DEBUG")
    jplt = JUMEG_PLOT_PSD()
    
  #--- logfile  prefix
    p = "~/MEGBoers/data/exp/JUMEGTest/FV/211747"
    fraw1 = p + "/" +"211747_FREEVIEW01_180109_1049_1_c,rfDC,meeg-raw.fif"
    fraw2 = p + "/" + "211747_FREEVIEW01_180109_1049_1_c,rfDC,meeg,nr-raw.fif"

    raw,fnraw = jb.get_raw_obj(fraw1)
    jplt.plot(raw)
    jplt.show_plot()
    
    
    
if __name__ == "__main__":
   main()



'''

##################################################
#
# generate plot of power spectrum before and
# after noise reduction
#
##################################################
def plot_denoising(fname_raw, fmin=0, fmax=300, tmin=0.0, tmax=60.0,
                   proj=False, n_fft=4096, color='blue',
                   stim_name=None, event_id=1,
                   tmin_stim=-0.2, tmax_stim=0.5,
                   area_mode='range', area_alpha=0.33, n_jobs=1,
                   title1='before denoising', title2='after denoising',
                   info=None, show=True, fnout=None):
    """Plot the power spectral density across channels to show denoising.

    Parameters
    ----------
    fname_raw : list or str
        List of raw files, without denoising and with for comparison.
    tmin : float
        Start time for calculations.
    tmax : float
        End time for calculations.
    fmin : float
        Start frequency to consider.
    fmax : float
        End frequency to consider.
    proj : bool
        Apply projection.
    n_fft : int
        Number of points to use in Welch FFT calculations.
    color : str | tuple
        A matplotlib-compatible color to use.
    area_mode : str | None
        Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
        will be plotted. If 'range', the min and max (across channels) will be
        plotted. Bad channels will be excluded from these calculations.
        If None, no area will be plotted.
    area_alpha : float
        Alpha for the area.
    info : bool
        Display information in the figure.
    show : bool
        Show figure.
    fnout : str
        Name of the saved output figure. If none, no figure will be saved.
    title1, title2 : str
        Title for two psd plots.
    n_jobs : int
        Number of jobs to use for parallel computation.
    stim_name : str
        Name of the stim channel. If stim_name is set, the plot of epochs
        average is also shown alongside the PSD plots.
    event_id : int
        ID of the stim event. (only when stim_name is set)

    Example Usage
    -------------
    plot_denoising(['orig-raw.fif', 'orig,nr-raw.fif', fnout='example')
    """

    from matplotlib import gridspec as grd
    import matplotlib.pyplot as plt
    from mne.time_frequency import psd_welch

    fnraw = get_files_from_list(fname_raw)

    # ---------------------------------
    # estimate power spectrum
    # ---------------------------------
    psds_all = []
    freqs_all = []

    # loop across all filenames
    for fname in fnraw:

        # read in data
        raw = mne.io.Raw(fname, preload=True)
        picks = mne.pick_types(raw.info, meg='mag', eeg=False,
                               stim=False, eog=False, exclude='bads')

        if area_mode not in [None, 'std', 'range']:
            raise ValueError('"area_mode" must be "std", "range", or None')

        psds, freqs = psd_welch(raw, picks=picks, fmin=fmin, fmax=fmax,
                                tmin=tmin, tmax=tmax, n_fft=n_fft,
                                n_jobs=n_jobs, proj=proj)
        psds_all.append(psds)
        freqs_all.append(freqs)

    if stim_name:
        n_xplots = 2

        # get some infos
        events = mne.find_events(raw, stim_channel=stim_name, consecutive=True)

    else:
        n_xplots = 1

    fig = plt.figure('denoising', figsize=(16, 6 * n_xplots))
    gs = grd.GridSpec(n_xplots, int(len(psds_all)))

    # loop across all filenames
    for idx in range(int(len(psds_all))):

        # ---------------------------------
        # plot power spectrum
        # ---------------------------------
        p1 = plt.subplot(gs[0, idx])

        # Convert PSDs to dB
        psds = 10 * np.log10(psds_all[idx])
        psd_mean = np.mean(psds, axis=0)
        if area_mode == 'std':
            psd_std = np.std(psds, axis=0)
            hyp_limits = (psd_mean - psd_std, psd_mean + psd_std)
        elif area_mode == 'range':
            hyp_limits = (np.min(psds, axis=0), np.max(psds, axis=0))
        else:  # area_mode is None
            hyp_limits = None

        p1.plot(freqs_all[idx], psd_mean, color=color)
        if hyp_limits is not None:
            p1.fill_between(freqs_all[idx], hyp_limits[0], y2=hyp_limits[1],
                            color=color, alpha=area_alpha)

        if idx == 0:
            p1.set_title(title1)
            ylim = [np.min(psd_mean) - 10, np.max(psd_mean) + 10]
        else:
            p1.set_title(title2)

        p1.set_xlabel('Freq (Hz)')
        p1.set_ylabel('Power Spectral Density (dB/Hz)')
        p1.set_xlim(freqs_all[idx][0], freqs_all[idx][-1])
        p1.set_ylim(ylim[0], ylim[1])

        # ---------------------------------
        # plot signal around stimulus
        # onset
        # ---------------------------------
        if stim_name:
            raw = mne.io.Raw(fnraw[idx], preload=True)
            epochs = mne.Epochs(raw, events, event_id, proj=False,
                                tmin=tmin_stim, tmax=tmax_stim, picks=picks,
                                preload=True, baseline=(None, None))
            evoked = epochs.average()
            if idx == 0:
                ymin = np.min(evoked.data)
                ymax = np.max(evoked.data)

            times = evoked.times * 1e3
            p2 = plt.subplot(gs[1, idx])
            p2.plot(times, evoked.data.T, 'blue', linewidth=0.5)
            p2.set_xlim(times[0], times[len(times) - 1])
            p2.set_ylim(1.1 * ymin, 1.1 * ymax)

            if (idx == 1) and info:
                plt.text(times[0], 0.9 * ymax, '  ICs: ' + str(info))

    # save image
    if fnout:
        fig.savefig(fnout + '.png', format='png')

    # show image if requested
    if show:
        plt.show()

    plt.close('denoising')
    plt.ion()



'''