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
import time,datetime

import numpy as np
from distutils.dir_util import mkpath

import matplotlib
import matplotlib.pylab as plt
# import matplotlib.pyplot as plt

import mne
from mne.report import Report

from dcnn_utils  import logger,transform_ica2data
from dcnn_logger import setup_script_logging
from dcnn_base   import _SLOTS


__version__= "2020.08.10.001"

class CalcSignal(_SLOTS):

    def calc_rms(self,data,average=None,rmsmean=None):
        ''' Calculate the rms value of the signal.
            Ported from Dr. J. Dammers IDL code.
        '''
        # check input
        sz = np.shape(data)
        nchan = np.size(sz)
        #  calc RMS
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

       # -- RAW mk epochs + average
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

       # -- RAW mk epochs + average
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

       # -- RAW mk epochs + average
        ep = mne.Epochs(raw,events,event_id=event_id,tmin=tmin,tmax=tmax,picks=picks)
        if len(picks) > 1:
            avg = ep.average()
            times = avg.times
            signal = np.sum(avg._data**2,axis=0)
        else:  # ref channel e.g. ECG, EOG  as np.array
            signal = np.average(ep.get_data(),axis=0).flatten()
            signal = signal**2

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


class PERFORMANCE_PLOT(CalcSignal):
    __slots__ = ["raw","raw_clean","verbose","ch_name","event_id","picks","tmin","tmax","title","colors",
                 "alpha","grid","type","scale","offset","fontsize","n_rows","idx","plot_ypos","figsize","fig_nr",
                 "_fig","_figs","_n_cols","_suptitle"]

    def __init__( self ):
        super().__init__()
        self.init()

    @property
    def figure(self): return self._fig
    @property
    def figures(self): return self._figs

    @figures.setter
    def figures( self,v ):
        self.clear_figures()
        self.init_plot_parameter()
        self._figs = v

    @property
    def suptitle( self ): return self._suptitle
    @suptitle.setter
    def suptitle( self,v):
        if self.figure:
            self._suptitle = v
            self.figure.suptitle(v,fontsize=10,y=0.02,x=0.05,ha="left")

    def init(self):
        """
        init defaults
        Returns
        -------

        """
        self.type   = "avg"  # type = "avg" # gfp,avg,sig
        self.tmin   = -0.4
        self.tmax   =  0.4

        self.idx    = 1
        self.n_rows = 2
        self.n_cols = 1

        self.alpha     = 0.33
        self.offset    = 0.15
        self.fontsize  = 12
        self.figsize   = (16.0,9.0)
        self.grid      = True
        self.colors    = ["black","yellow","red","magenta","green"]
        self.scale     = { "raw":{ "factor":1e15,"unit":"fT" },"ref":{ "factor":1e3,"unit":"mV" } }

        # TODO - JD: depending on your backend a figure window will pop-up due to
        #            plt.subplots_adjust(left=0.1,right=0.95,bottom=0.05,top=0.95,hspace=0.35)
        #            in init_plot_parameter()  => do init just before plotting
        # self.init_plot_parameter()

    def clear_figures( self ):
        self._fig = None
        while self._figs:
              plt.close( self._figs.pop() )

        self._subtitle = None
        self._figs     = None
        self.fig_nr    = None

        # plt.close('all')

    def init_plot_parameter(self):
        #--- A4 landscape
        plt.rc('figure',figsize=self.figsize,autolayout=True)
        plt.rcParams.update({ 'font.size':self.fontsize })
        plt.subplots_adjust(left=0.1,right=0.95,bottom=0.05,top=0.95,hspace=0.35)
        plt.rcParams['savefig.facecolor'] = "0.9"
        if not self.fig_nr:
           self.fig_nr = 1

    def clear(self,init=True):
        """
        clear all to None
        clear fig list and init defaults

        Parameters
        ----------
        init : True

        Returns
        -------

        """
        try:
           self.clear_figures()
        except:
           pass
        super().clear()

        if init:
           self.init()

    def labels2picks(self,raw=None,labels=None):
            """
            ToDo move to dcnn_utils
            get picks from channel labels
            call to < mne.pick_channels >
            picks = mne.pick_channels(raw.info['ch_names'], include=[labels])

            Parameter
            ---------
             raw obj
             channel label or list of labels

            Result
            -------
            picks as numpy array int64
            """
            if not raw:
               raw = self.raw
            if isinstance(labels,(list)):
               return  mne.pick_channels(raw.info['ch_names'],include=labels)
            else:
               return mne.pick_channels(raw.info['ch_names'],include=[labels])

    def _plot(self,ax,t,data,ylabel,color,range=None,range_color="cyan",alpha=0.3):
        """
        plot helper function
        Parameters
        ----------
        ax :
        t :
        data :
        ylabel :
        color :
        range :
        range_color :
        alpha :

        Returns
        -------

        """
        ax.plot(t,data,color=color)

        if range:
            ax.fill_between(t,range[0],y2=range[1],color=range_color,alpha=alpha)

        ax.set_xlabel("[s]")
        ax.set_xlim(t[0],t[-1])
        ax.set_ylabel(ylabel)
        ax.grid(True)

    def plot(self,**kwargs):
        """
        plot artifact rejection  averaged signals for ECG and EOG onsets

        Parameters
        ----------
        raw :
        raw_clean :
        ch_name :
        events :
        event_id :
        picks :
        tmin :
        tmax :
        title :
        colors :
        alpha :
        grid :
        scale :
        offset :
        fontsize :

        Returns
        -------

        """

        self._update_from_kwargs(**kwargs)
        title  = kwargs.get("title")
        evt    = kwargs.get("events")
        counts = evt.shape[0]

        # init figure

        sig_raw,sig_clean,range,t = self._calc_data(self.raw,self.raw_clean,evt,event_id=self.event_id,tmin=self.tmin,
                                                    tmax=self.tmax,picks=self.picks)
       #--- ref channel e.g.: ECG
        sig_picks   = self.labels2picks(labels=self.ch_name)
        sig_ref,_,_ = self._calc_signal(self.raw,evt,event_id=self.event_id,tmin=self.tmin,tmax=self.tmax,picks=sig_picks)

        if not isinstance(self.figures,(list)):
           self.figures = []

        # if self.idx == 1:
        if not self._fig:
           self.figures.append( plt.figure(self.fig_nr) )
           self._fig = self.figures[-1]
           if title:
              self.figure.suptitle(title ,fontsize=12)
           suptitle  = kwargs.get("suptitle")
           self.suptitle = suptitle
        else:
           self._fig = plt.figure( self.fig_nr )

       #--- subplot(nrows,ncols,idx)
        ax1 = plt.subplot(self.n_rows, self.n_cols, self.idx)
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
        ax3.legend([self.ch_name + " cnts {}".format(counts)],loc=2,prop={ 'size':8 })

        ax4 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        self._plot(ax4,t,sig_ref * scl.get("factor"),scl.get("unit"),"green")
        ax4.tick_params(axis='y',labelcolor=color)
        ax4.legend(["Clean " + self.ch_name + " cnts {}".format(counts)],loc=2,prop={ 'size':8 })
        try:
           ax1.set_ylim(ylim[0],ylim[1])
           ax2.set_ylim(ylim[0],ylim[1])
        except:
           ax1.set_ylim(-1.0,1.0)
           ax2.set_ylim(-1.0,1.0)
           logger.error("ERROR in performance plot : can not set ylim : {}".format(title))

        plt.tight_layout()


   
