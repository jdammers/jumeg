#!/usr/bin/env python2 
#-W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 08:29:12 2017

@author: fboers
"""
import sys,os
#sys.path.append('../../')

import numpy as np
import warnings
with warnings.catch_warnings():
     warnings.filterwarnings("ignore",category=DeprecationWarning)
     import mne


#--- jumeg clases
from jumeg.filter.jumeg_filter import jumeg_filter 
from jumeg.jumeg_base import jumeg_base as jb
from jumeg.epocher.jumeg_epocher import jumeg_epocher

#---
template       = 'MEG94T'
stage = "/data/exp/MEG94T/mne/"

#-- MEG94T
#
fpath = stage +"/205720/MEG94T0T2/131016_1326/1/"
ffif  ='205720_MEG94T0T2_131016_1326_1_c,rfDC-raw.fif'

fname = fpath + ffif
raw   = None

os.environ['JUMEG_PATH_TEMPLATE'] = fpath
          
verbose    = True
DO_EVENTS  = True
DO_EPOCHER = True
DO_FILTER  = False

#condition_list=["URon","URoff""ULon","ULoff","LRon","LRoff","LLon","LLoff"]
condition_list=["LLon","LLoff"]

 #--- set template path
template_name = 'MEG94T'
template_path = "epocher/"        
jumeg_epocher.template_path = template_path
jumeg_epocher.verbose       = verbose


#--- Epocher
if DO_EVENTS:
   print"-"*50 
   print "---> EPOCHER Events"
   print "  -> FIF File: "+ fname
   print "  -> Template: "+ template_name+"\n"
    
   evt_param = { "condition_list":condition_list,
                 "template_path": template_path, 
                 "template_name": template_name,
                 "verbose"      : verbose
               }
         
   raw,fname = jumeg_epocher.apply_events(fname,raw=raw,**evt_param)


   
#--- EPOCHER
if DO_EPOCHER:
   ep_param={
          "condition_list": condition_list,
          "template_path" : template_path, 
          "template_name" : template_name,
          "verbose"       : verbose,
          "parameter":{
                       "event_extention": ".eve",
                       "save_condition":{"events":True,"epochs":True,"evoked":True},
                       "weights":{"mode":"equal","method":"median","skipp_first":None}
                      # "time":{"time_pre":null,"time_post":null,"baseline":null},
                      # "exclude_events":{"eog_events":{"tmin":-0.4,"tmax":0.6} } },
                      }}  
#---
   if DO_FILTER:   
      from jumeg.filter.jumeg_filter import jumeg_filter
      jf = jumeg_filter()
      jb.picks.exclude_trigger
      jf.vebose = verbose
      jf.apply_filter(raw._data,picks=jb.picks.exclude_trigger(raw) ) # inplace

#--- 
   print "---> EPOCHER Epochs"
   print "  -> File            : "+ fname
   print "  -> Epocher Template: "+ template_name+"\n"   
   raw,fname = jumeg_epocher.apply_epochs(fname=fname,raw=raw,**ep_param)



  
   
