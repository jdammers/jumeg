#!/usr/bin/env python3
# -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
"""
Created 2020.01.08

@author: fboers
"""

import os,sys,logging
import os.path as op

from jumeg.base.jumeg_base import jumeg_base as jb
from jumeg.epocher.jumeg_epocher import jumeg_epocher
from jumeg.filter.jumeg_mne_filter import JuMEG_MNE_FILTER
from jumeg.base import jumeg_logger

logger = logging.getLogger("jumeg")
logger = jumeg_logger.setup_script_logging(logger=logger,fname="quaters_epocher.log",level="INFO",logfile=True)#,mode="w" )

#---
DO_EVENTS    = True
DO_FILTER    = True
DO_EPOCHS    = True
verbose      = True
debug        = False
stage        = jb.expandvars("$JUMEG_PATH_LOCAL_DATA/exp/QUATERS/mne")

epocher_path = None
hdf_path     = None
raw          = None


jumeg_epocher.use_yaml = True


template_name  = "QUATERS"
template_path  = os.path.dirname(__file__)
condition_list = ["RCC_ST","RCC_FB"]#,"RCI_ST","RCI_FB","RII_ST","RII_FB","RIC_ST","RIC_FB",
                 # "PCC_ST","PCC_FB","PCI_ST","PCI_FB","PII_ST","PII_FB","PIC_ST","PIC_FB"]
                  
fraws=[
#"207048/QUATERS01/191218_1447/2/207049_QUATERS01_191218_1447_2_c,rfDC,meeg-raw.fif",
#"207049/QUATERS01/191218_1447/1/207049_QUATERS01_191218_1447_1_c,rfDC,meeg-raw.fif",
"207049/QUATERS01/191218_1447/3/207049_QUATERS01_191218_1447_3_c,rfDC,meeg-raw.fif",
#"210857/QUATERS01/191210_1325/1/210857_QUATERS01_191210_1325_1_c,rfDC,meeg,nr,bcc,int,ar-raw.fif",
#"210857/QUATERS01/191210_1325/2/210857_QUATERS01_191210_1325_2_c,rfDC,meeg,nr,bcc,int,ar-raw.fif",
#"210857/QUATERS01/191210_1325/3/210857_QUATERS01_191210_1325_3_c,rfDC,meeg,nr,bcc,int,ar-raw.fif",
#"210857/QUATERS01/191210_1325/4/210857_QUATERS01_191210_1325_4_c,rfDC,meeg,nr,bcc,int,ar-raw.fif"
]
fi45=[
"210857/QUATERS01/191210_1325/1/210857_QUATERS01_191210_1325_1_c,rfDC,meeg,nr,bcc,int,fibp0.10-45.0,ar-raw.fif",
#"210857/QUATERS01/191210_1325/2/210857_QUATERS01_191210_1325_2_c,rfDC,meeg,nr,bcc,int,fibp0.10-45.0,ar-raw.fif",
#"210857/QUATERS01/191210_1325/3/210857_QUATERS01_191210_1325_3_c,rfDC,meeg,nr,bcc,int,fibp0.10-45.0,ar-raw.fif",
#"210857/QUATERS01/191210_1325/4/210857_QUATERS01_191210_1325_4_c,rfDC,meeg,nr,bcc,int,fibp0.10-45.0,ar-raw.fif"
]
fi120=["210857/QUATERS01/191210_1325/1/210857_QUATERS01_191210_1325_1_c,rfDC,meeg,nr,bcc,int,ar,fibp0.10-120.0-raw.fif"]

flist=fraws

MNEFilter = JuMEG_MNE_FILTER()

for f in flist:

    fraw  = os.path.join(stage,f)
    fname = os.path.basename(fraw)
    fpath = os.path.dirname(fraw)

    flog = os.path.splitext(fraw)[0]+".epocher.log"
    # jumeg_logger.update_filehandler(logger=logger,fname=flog,mode="a")
        
    #logger.info(fname)
      
    raw = None
        
    epocher_path = os.path.join(os.path.dirname(fraw),"epocher")
    hdf_path     = epocher_path
        
   #--- Epocher events
    if DO_EVENTS:
            logger.info("---> EPOCHER Events\n"+
                        "  -> FIF File      : {}\n".format(fname)+
                        "  -> FIF Path      : {}\n".format(fpath)+
                        "  -> Template      : {}\n".format(template_name)+
                        "  -> Template path : {}\n".format(template_path)+
                        "  -> HDF path      : {}\n".format(hdf_path)+
                        "  -> Epocher path  : {}\n".format(epocher_path)
                        )

            evt_param = {"condition_list": condition_list,
                         "template_path": template_path,
                         "template_name": template_name,
                         "hdf_path"     : hdf_path,
                         "verbose"      : verbose,
                         "debug"        : debug
                         }
            try:
                raw, fname = jumeg_epocher.apply_events(fraw,raw=raw, **evt_param)
            except:
                logger.exception(" error in calling jumeg_epocher.apply_events")

    if DO_FILTER:
       picks = jb.picks.exclude_trigger(raw)
       fname = MNEFilter.apply(raw=raw,fname=fraw,flow=0.01,fhigh=45.0,picks=picks)

   #--- EPOCHER epochs
    if DO_EPOCHS:
       ep_param = {
                "condition_list": condition_list,
                "template_path": template_path,
                "template_name": template_name,
                "hdf_path"     : hdf_path,
                "verbose"      : verbose,
                "debug"        : debug,
                "event_extention": ".eve",
                "save_raw" : True, # mne .annotations
                "output_mode":{ "events":True,"epochs":True,"evoked":True,"annotations":True,"stage":epocher_path,"use_condition_in_path":True}
               
                # "weights"       :{"mode":"equal","method":"median","skip_first":null}
                # "exclude_events":{"eog_events":{"tmin":-0.4,"tmax":0.6} } },
            }
          # ---
            
            #logger.info("---> EPOCHER Epochs\n   -> File  :{}\n   -> Epocher Template: {}".format(fname,template_name))
       try:
           raw, fname = jumeg_epocher.apply_epochs(fname=fraw, raw=raw, **ep_param)
           #logger.info(raw.annotations)
           #raw.plot(butterfly=True,show=True,block=True,show_options=True)
       except:
           logger.exception(" error in calling jumeg_epocher.apply_epochs")
     
    logger.info("---> DONE EPOCHER Epochs\n   -> File  :{}\n   -> Epocher Template: {}".format(fname,template_name))
           
