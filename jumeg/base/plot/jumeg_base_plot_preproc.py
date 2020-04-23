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

import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages

from mne.time_frequency import psd_welch

from jumeg.base.jumeg_base import jumeg_base as jb

logger = logging.getLogger('jumeg')

__version__="2020.03.30.001"

class JuMEG_PLOT_BASE(object):
    __slots__ = ["picks","fmin","fmax","tmin","tmax","proj","n_fft","color","fill_color","area_mode","area_alpha","pick_types","n_jobs","dpi","verbose",
                 "check_dead_channels","info","fnout","n_plots","file_extention","name",
                 "_fig","_plot_index","_axes","_yoffset","_ylim"]

    def __init__(self,**kwargs):
        super().__init__()
        self._init(**kwargs)
        
    @property
    def fig(self):
        return self._fig
    
    @property
    def axes(self): return self._axes

    def GetAxes(self,idx):
        return self._axes[idx]

    @property
    def plot_index(self):
        return self._plot_index

    def _init_defaults(self):
        self.fmin       = 0.0
        self.tmin       = 0.0
        self.n_fft      = 4096
        self.color      = 'blue'
        self.fill_color = "blue"
        self.dpi        = 100
        self.area_mode  = 'range'
        self.area_alpha = 0.33
        self.proj       = False
        self.pick_type  = None
        self.n_jobs     = 1
        self.n_plots    = 1
        self.name       = "PLOT"
        self.file_extention = ".png" #".pdf"
        self.check_dead_channels = True
        self._plot_index = 1
        self._axes       = []
        self._yoffset    = 10
        
    def get_parameter(self,key=None):
        """
        get  host parameter
        :param key:)
        :return: parameter dict or value of  parameter[key]
        """
        if key: return self.__getattribute__(key)
        return { slot:self.__getattribute__(slot) for slot in self.__slots__ }

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
        # plt.rc('figure',figsize=(11.69,8.27))
        plt.rc('figure',figsize=(16.0,9.0))
        plt.rcParams.update({ 'font.size':8 })
        '''
        https: // matplotlib.org / 3.1.0 / api / _as_gen / matplotlib.pyplot.subplots_adjust.html
        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
        hspace = 0.2  # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
        '''
        plt.subplots_adjust(left=0.1,right=0.95,bottom=0.05,top=0.95,hspace=0.35)

        self._fig = plt.figure(self.name)
        plt.clf()
        plt.title(self.name)
       
        #fig = pl.figure(name,figsize=(10, 8), dpi=100))
   
    def save(self,fname=None,plot_dir="report"):
        """

        :param fname:
        :param plot_dir: <report>
        :return:
          full filename e.g.  .../report/xyz.png
        """
        if fname:
           fname     = os.path.expandvars( os.path.expanduser( fname ) )
           fout_path = os.path.dirname(fname)
           fnout     = os.path.splitext(os.path.basename(fname))[0] + self.file_extention
        else:
           fnout = "test_plot" + self.file_extention
           fout_path = "."
    
        if plot_dir:
           fout_path = os.path.join(fout_path,plot_dir)
           try:
               os.makedirs(fout_path,exist_ok=True)
           except:
               logger.exception("ERROR: can not create plot\n" +
                                "  -> directory: {}\n".format(fout_path) +
                                "  -> filename : {}".format(fname))
               return False
    
        if fnout:
           fout = os.path.join(fout_path,fnout)
           
           if fout.endswith("png"):
              self.fig.savefig(fout,format="png")
           elif fout.endswith("svg"):
               self.fig.savefig(fout,format="svg")
           else: # pdf ?
               self.fig.savefig(fout,dpi=self.dpi)
           if self.verbose:
              logger.info("done saving plot: {}".format(fout))
            
           return fout
            
    def set_ylim(self,ylim=None):
        
        self.update_global_ylim(ylim)
        
        for ax in self._axes:
            ax.set_ylim(self._ylim[0],self._ylim[1])
    
    def update_global_ylim(self,ylim):
        
        if not ylim: return
        
        if not self._ylim:
           self._ylim = ylim
        else:
           if self._ylim[0] > ylim[0]:
              self._ylim[0] = ylim[0]
           if self._ylim[1] < ylim[1]:
              self._ylim[1] = ylim[1]
        
    def show(self):
        #self.fig.show()
        plt.show()

    def close(self):
        self.fig.close()  #self.name)
        #self.fig.ion()
        
    def plot(self,raw,**kwargs):
        pass
 
class JuMEG_PLOT_PSD(JuMEG_PLOT_BASE):
    """
    copy from jumeg_noise_reducer.plot_denoising
    plot power spectrum density (PSD) from fif files
    
    :param picks: channels to process and plot <meg,ref,no bads>
    :param fmin:  Start frequency to consider. <0.0 Hz>
    :param fmax:  End frequency to consider.   <sfreq/2.0 Hz>
    :param tmin:  Start time for calculations. <0.0>
    :param tmax:  End time for calculations.   <the end in sec>
    :param proj:  Apply projection.            <False>
    :param n_fft: Number of points to use in Welch FFT calculations. <4096.0>
    :param color: str | tuple                  <blue>
                  A matplotlib-compatible color to use.
    :param area_mode: str | None               <range>
           Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
           will be plotted. If 'range', the min and max (across channels) will be
           plotted. Bad channels will be excluded from these calculations.
           If None, no area will be plotted.
    :param area_alpha: Alpha for the area.     <0.33>
    :param n_jobs:                             <1>
    :param dpi: plot resolution                <100>
    :param verbose:                            <False>
    :param fnout: full filenameof the saved output figure.
    :param n_plots: Number of plots to plot, only one page is used. <1>
    :param file_extention: output fileextention <.pdf>
    :param name: main title
    :return:
    
    Example
    --------
    from jumeg.base.jumeg_base                   import jumeg_base as jb
    from jumeg.base.plot.jumeg_base_plot_preproc import JuMEG_PLOT_PSD
   
    #--- init plot
    jplt = JuMEG_PLOT_PSD(n_plots=2,fmax=300.0,name="denoising",verbose=True)

    #--- load first fif, init raw & generate plot
    p = "$JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/FV/211747"
    fraw1 = "211747_FREEVIEW01_180109_1049_1_c,rfDC,meeg-raw.fif"
    
    raw,fnraw1 = jb.get_raw_obj( os.path.join(p,fraw1) )
    jplt.plot(raw,title=fraw1)

    #--- load second fif, overwrite raw & generate plot
    fraw2 = "211747_FREEVIEW01_180109_1049_1_c,rfDC,meeg,nr-raw.fif"
    raw,fnraw2 = jb.get_raw_obj( os.path.join(p,fraw2) )
    jplt.plot(raw,title=fraw2)

    jplt.show()
    
    #--- save plot as pdf
    plot_name = fraw2.rsplit('-raw.fif')[0] + '-plot'
    jplt.save(fname=os.path.join(p,plot_name))
    
    jplt.close()
    """
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
    def _calc_psd_welch(self,raw,**kwargs):
        """
        
        :param raw:
        :return:
         psd,freq
        """
        self._update_from_kwargs(**kwargs)
        
        if self.check_dead_channels:
           self.picks = jb.picks.check_dead_channels(raw,picks=self.picks,verbose=self.verbose)
        elif not self.picks.shape:
           self.picks = jb.picks.meg_and_ref_nobads(raw)
           #self.picks = jb.picks.meg_nobads(raw)
           
        self.tmax = self.tmax if self.tmax else raw.times[-1]
        self.fmax = self.fmax if self.fmax else raw.info.get("sfreq",1000.0)/2.0
        
        return psd_welch(raw, picks=self.picks, fmin=self.fmin, fmax=self.fmax,
                              tmin=self.tmin, tmax=self.tmax, n_fft=self.n_fft,
                              n_jobs=self.n_jobs, proj=self.proj)
           
    def _psd2db(self,psd):
        return 10 * np.log10(psd)

    def _calc_hyp_limits(self,psd,psd_mean=None):
        """
        
        :param psd:
        :param psd_mean:
        :return: hyp_limit
        """
        if psd_mean is None:
           psd_mean = np.mean(psd,axis=0)
        if self.area_mode == 'std':
            psd_std = np.std(psd,axis=0)
            return (psd_mean - psd_std,psd_mean + psd_std)
        elif self.area_mode == 'range':
            return (np.min(psd,axis=0),np.max(psd,axis=0))
        else:  # area_mode is None
            return None

    def plot_power_spectrum(self,psd=None,freqs=None,title=None,grid=True):
        """
        plot powerspectrum
        :param psds:
        :param freqs:
        :param title:
        :param grid:  <True>
        :return:
        """
        
        if self.plot_index > self.n_plots:
           self._plot_index = 1
           self._axes = []
          
       #--- subplot(nrows,ncols,idx)
        ax = plt.subplot(self.n_plots,1,self._plot_index)
      # ax = plt.subplot(1,self.n_plots,self.plot_index)

      #--- Convert PSDs to dB
        psd        = self._psd2db(psd)
        psd_mean   = np.mean(psd,axis=0)
        hyp_limits = self._calc_hyp_limits(psd,psd_mean=psd_mean)
        
        ax.plot(freqs,psd_mean,color=self.color)
        
        if hyp_limits is not None:
           ax.fill_between(freqs,hyp_limits[0],y2=hyp_limits[1],
                           color=self.fill_color,alpha=self.area_alpha)
            
        self.update_global_ylim( [np.min(psd_mean) - self._yoffset,np.max(psd_mean) + self._yoffset] )
        
        if title:
           ax.set_title(title,loc="left")

        ax.set_xlabel('Freq (Hz)')
        ax.set_ylabel('Power Spectral Density (dB/Hz)')
        ax.set_xlim(freqs[0],freqs[-1])
        #ax.set_ylim(self.ylim[0],self.ylim[1])
        ax.grid(grid)
        
        self._axes.append(ax)
        
        self._plot_index += 1

        if self.plot_index > self.n_plots:
           self.set_ylim()
       
    def plot(self,raw,**kwargs):
        """
        
        :param raw:
        :param kwargs:
        :return:
        """
        psd,freqs = self._calc_psd_welch(raw,**kwargs)
        self.plot_power_spectrum(psd=psd,freqs=freqs,title=kwargs.get("title"))

