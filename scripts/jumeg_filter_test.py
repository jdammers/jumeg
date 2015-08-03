#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:11:25 2015

@author: fboers
jumeg_filter_test.py --notch 50 100 150 --methods bw mne
                     -f 110058_MEG94T_121001_1331_1_c,rfDC_30sec-raw.fif 
                     -p /localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1
 
"""


import sys, os, os.path
import matplotlib.pyplot as plt
import numpy as np
import mne

from jumeg.jumeg_base     import jumeg_base
from jumeg.filter         import jumeg_filter

import argparse


def get_args():
    info="""
    JuMEG Filter Test Start Parameter
    jumeg_filter_test.py --notch 50 100 150 --methods bw mne -picks 1 2 45 247 -fname fif-file -path path-to-fif -t bp -fcut1 10.0 -fcut2 70
 
    """
    parser = argparse.ArgumentParser(info)
#--- methods  
    method_list = ['bw','mne','mne_orig']
    parser.add_argument("-t","--type",help="filter type of lp | hp | bp | br | notch",default='bp',choices={'lp','hp','bp','notch','br'})
    parser.add_argument("--methods",nargs='*', help="filter methods list --methods bw mne mne_orig",choices=method_list,default=('bw','mne','mne_orig') ) 
    
#--- fif file  
    parser.add_argument("-p","--path",  help="start path to data files",default=os.path.abspath('.') )
    parser.add_argument("-f","--fname",help="Input raw FIF file")
    parser.add_argument("-picks","--picks",nargs='*',type=int,help="pick meg channels index: --picks 245 246 247",default= ())
 
#--- filter opt
    parser.add_argument("-fcut1","--fcut1",   type=float,help="fcut1",default=1.0)
    parser.add_argument("-fcut2","--fcut2",   type=float,help="fcut2 used for bp|br",default=200.0)
    parser.add_argument("-notch","--notch",nargs='*',  type=float,help="list off notches: --notch 50 100",default= () )
    parser.add_argument("-nw","--notch_width",type=float,help="notch width",default=1.0)
    parser.add_argument("-or","--order",      type=int,  help="order for butterworth",  default= 4)
    parser.add_argument("-nj","--njobs",      type=int,  help="number of parallel jobs",default= 4)
    parser.add_argument("-dc","--remove_dcoffset",action="store_true",help="remove DC offset")

#--- psd plot opt    
    parser.add_argument("-fmin","--fmin",  type=float,help="psds polt fmin",default=0.0)
    parser.add_argument("-fmax","--fmax",  type=float,help="psds plot fmax",default=200.0)
    parser.add_argument("-nfft","--nfft",  type=int,  help="psds plot nfft points",default=4096)

#--- flags    
    parser.add_argument("-v","--verbose",  action="store_true",help="verbose mode")
    parser.add_argument("-d","--debug",    action="store_true",help="!!! DEBUG MODE used by FB") 
   
    return parser.parse_args(),parser


#---- main

opt,parser = get_args()


if any( opt.notch ):
   opt.notch = np.array(opt.notch)
else:
   opt.notch = np.array( [] )
   
#--- FB debug options
if opt.debug:
   opt.fname = '110058_MEG94T_121001_1331_1_c,rfDC_30sec-raw.fif'
   opt.path  = '/localdata/frank/data/MEG94T/mne/110058/MEG94T/121001_1331/1'
   opt.notch = np.array([50,100,150,200])


if opt.fname:
   fname = opt.path + '/' + opt.fname
   raw   = mne.io.Raw(fname,preload=True)
else: 
   print"\n!!! ERROR fname not defined !!!\n"
   print parser.print_help()
   exit() 
  
if opt.debug:
   opt.picks = np.arange( 10 ) + 238
elif any( opt.picks ):
   opt.picks = np.array(opt.picks)
else:
   opt.picks = mne.pick_types(raw.info,meg=True,exclude='bads')

sfreq = raw.info['sfreq']

#--- filter picture name
fnfig = 'jumeg_fitest_'+ opt.type+'-'
if opt.type == 'bp' :
   fnfig += "%0.3f-%0.1fHz" % (opt.fcut1,opt.fcut2)
else:
   fnfig += "%0.3fHz" % (opt.fcut1)
   
if opt.order:
   fnfig += "_or%d" % ( opt.order )
   
if opt.notch.any() :
   fnfig += "n"
if ( opt.remove_dcoffset ):
   fnfig +="DC"
    
fnfig += '_psd.png'


#--- !!! save oriinal meg data, filter works inplace and will overwrite original data
data_meg_save = raw._data[opt.picks,:].copy()

plt.ion()
fig = plt.figure()
ax  = plt.axes()

# filter_method_list = ['mne','mne_orig']

color      = {'orig'    :(0,0,0),
              'bw'      :(0,0,1),
              'mne'     :(0,1,0),
              'mne_orig':(1,0,0),
              'ws'      :(1,0,1) }
              
plt_legends = []
fi          = {}

for fi_method in opt.methods:
    fi_method_str = str(fi_method)
    if fi_method_str == 'mne_orig' :
    
       raw.filter(opt.fcut1, opt.fcut2, n_jobs=opt.njobs, method='fft',picks=opt.picks)
       plt_legends.append('mne_orig')
       
    else:
#--- create your filter type object
       fi[ fi_method_str ] = jumeg_filter(filter_method=fi_method_str,filter_type=opt.type,fcut1=opt.fcut1,fcut2=opt.fcut2,remove_dcoffset=opt.remove_dcoffset,
                                    notch=opt.notch, 
                                    notch_width=opt.notch_width,
                                    sampling_frequency=sfreq,
                                    order=opt.order)
                                    #sampling_frequency=sfreq)
    
       fi[fi_method_str].apply_filter(raw._data,opt.picks)
       plt_legends.append( fi[fi_method_str].filter_info_short)
    
       print fi[fi_method_str].filter_info
     
        
    raw.plot_psds(picks=opt.picks,fmin=opt.fmin,fmax=opt.fmax,ax=ax,n_fft=opt.nfft,color=color[fi_method]) 

  #--- copy data back for next filter method
    raw._data[opt.picks,:] = data_meg_save

#--- plot orig raw data    
raw.plot_psds(picks=opt.picks,fmin=opt.fmin,fmax=opt.fmax,ax=ax,n_fft=opt.nfft,color=color['orig'])    
plt_legends.append('raw')  

p,pdf = os.path.split(raw.info['filename']) 
 
ax.set_title('JuMEG Filter Test ' + pdf)
plt.legend(plt_legends)

plt.ioff()
# pl.ion()
fig.savefig(fnfig)
plt.show()
plt.close()
