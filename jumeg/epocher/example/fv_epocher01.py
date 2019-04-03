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
#from jumeg.filter.jumeg_filter import jumeg_filter 
from jumeg.jumeg_base import jumeg_base as jb
from jumeg.epocher.jumeg_epocher import jumeg_epocher

         
verbose    = True
DO_EVENTS  = True
DO_EPOCHER = True
DO_FILTER  = False

#--- file to process
fpath = '/data/exp/FREEVIEWING/epocher/'
ffif  = '211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr-raw.fif'
ffif  = '211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar-raw.fif'

fname = fpath + ffif
raw   = None   

#--- set template path
template_name = 'FreeView'
template_path = "/epocher/"

imo_list=["ImoIODBc"]
fv_list=["FVImoBc","FVsaccardeBc","FVfixationBc"]
me_list=["MEImoBc","MEsaccardeBc","MEfixationBc"]
se_list=["SEImoBc","SEsaccardeBc","SEfixationBc"]

condition_list= imo_list  + fv_list + me_list + se_list +["SACBc","FIXBc"]
#condition_list= ["SACBc","FIXBc"]
#condition_list= fv_list

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
          "event_extention": ".eve",
          "save_condition":{"events":True,"epochs":True,"evoked":True},
          #"weights"       :{"mode":"equal","method":"median","skip_first":null}
         # "exclude_events":{"eog_events":{"tmin":-0.4,"tmax":0.6} } },
           }  
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

#---
#--- check   
# import pandas as pd
# fhdf=epocher_hdf_fname 
# HDF=pd.HDFStore(fhdf)
# HDF['/epocher/ET_sacc_onset'] 
# HDF.close() 

#d=np.ones([10,4])
#a=np.arange(4)+11
#d.T+a
#df=pd.DataFrame(dd,index=range(10), columns=list("abcd"))



   
   # jee = JuMEG_Epocher_Epochs()
   # jee.epocher_update(fname=fname,raw=raw,**events)
   # jee.marker.type_result='hit'
   # jee.epocher_run(fname=fname,raw=raw,**events)
 
    
#fpath = '/home/fboers/MEGBoers/data/exp/FREEVIEWING/epocher/'
#fep  = fpath + '211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar,ImoIOD_evt-epo.fif'
#fepbc= fpath + '211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,nr,ar,ImoIOD_evt_bc-epo.fif'  
#ep=mne.read_epochs(fep)
#epbc=mne.read_epochs(fepbc)
#
#ep[0]._data.shape

"""
import pandas as pd
fpath = '/home/fboers/MEGBoers/data/exp/FREEVIEWING/epocher/'     
fhdf  = '211776_FREEVIEW01_180115_1414_1_FreeView-epocher.hdf5'     
HDFobj=pd.HDFStore(fpath+fhdf)
HDFobj.keys()

key='/epocher/FVsaccardeBc'

df=HDFobj[key]
print df.columns

tsl0=1000
tsl1=50000
resp_type_input ='iod_onset'
resp_type_offset='iod_offset'


early_idx_in = df[ ( tsl0 <= df[resp_type_input] ) & ( df[ resp_type_input ] < tsl1 ) ].index

early_idx_off = df[ ( tsl0 <= df[resp_type_offset] ) & ( df[ resp_type_offset ] < tsl1 ) ].index
early_idx     = np.unique( np.concatenate((early_idx_in,early_idx_off), axis=0) )
      



In [46]: c=np.arange(10)

In [47]: df['a']+=c

In [48]: df
Out[48]:
      a     b     c     d
0  12.0  13.0  14.0  15.0
1  13.0  13.0  14.0  15.0
2  14.0  13.0  14.0  15.0
3  15.0  13.0  14.0  15.0
4  16.0  13.0  14.0  15.0
5  17.0  13.0  14.0  15.0
6  18.0  13.0  14.0  15.0
7  19.0  13.0  14.0  15.0
8  20.0  13.0  14.0  15.0
9  21.0  13.0  14.0  15.0

In [49]: r=df[ ( 12 <= df['a'] ) & ( df[ 'a' ] <= 18 )].index

In [50]: r
Out[50]: Int64Index([0, 1, 2, 3, 4, 5, 6], dtype='int64')

In [51]:

In [51]: r.any()
Out[51]: True

In [52]: df['b'][r]
Out[52]:
0    13.0
1    13.0
2    13.0
3    13.0
4    13.0
5    13.0
6    13.0
Name: b, dtype: float64

In [53]: df['a'][r].isin([14,15,18])
Out[53]:
0    False
1    False
2     True
3     True
4    False
5    False
6     True
Name: a, dtype: bool

In [54]: found=df['a'][r].isin([14,15,18])

In [55]: np.where(found==False)
Out[55]: (array([0, 1, 4, 5]),)

In [56]: np.where(found==False)[0]
Out[56]: array([0, 1, 4, 5])
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""  
   
