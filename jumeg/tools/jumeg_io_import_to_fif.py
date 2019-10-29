#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
JuMEG interface to export 4D/BTi data into fif format using mne-python
 call to: mne.io.read_raw_bti
 https://martinos.org/mne/stable/generated/mne.io.read_raw_bti.html#mne.io.read_raw_bti
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 08.01.2019
#--------------------------------------------
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import os,sys,argparse,re
import numpy as np
import logging
import mne

from jumeg.base.jumeg_base  import jumeg_base as jb
from jumeg.base import jumeg_logger
logger = logging.getLogger('jumeg') # init a logger

__version__= "2019.05.14.001"

#=========================================================================================
#==== script part
#=========================================================================================

def apply_import_to_fif(opt):
    """
    apply jumeg import 4D/BTi data to FIF
    convert 4D-file to FIF format using mne
    save raw-obj to new file
    
    jumeg wrapper for <mne.io.read_raw_bti>
    https://martinos.org/mne/stable/generated/mne.io.read_raw_bti.html#mne.io.read_raw_bti
    
    Parameter
    ---------
     opt
     
    Outputfile
    ----------
    <wawa>, -raw.fif

    Returns
    -------
    return mne.io.Raw instance
    """
   # --- ck file fullpath
    fpdf = jb.isFile(opt.pdf_fname,path=opt.pdf_stage,head="apply_import_to_fif => file check: 4D/BTI <raw>",exit_on_error=True)
    if not fpdf: return
    fcfg = jb.isFile(opt.config_fname,path=opt.pdf_stage,head="apply_import_to_fif:=> file check: 4D/BTI <config>",exit_on_error=True)
    if not fcfg: return
    
    # ToDo ck if headshape file must exist
    #  pt.headshape_fake and touch hs
    #---
    #fhs  = jb.isFile(opt.head_shape_fname,path=opt.pdf_stage,head="apply_import_to_fif:=> file check: 4D/BTI <head shape>",exit_on_error=True)
    #if not fhs : return

   #--- mk fif ouput file name
   # fpdf="/mnt/meg_store2/megdaw_data21/211063/INTEXT01//18-11-15@11:20/1/c,rfDC"
   # 211063_INTEXT01_181115_1120_1_c,rfDC-raw.fif  size [Mb]:  703.534
   #---> FIF / MNE path        : /mnt/meg_store1/exp/INTEXT/mne/211063/INTEXT01/181115_1120/1/
    
    fif_out  = os.path.expandvars( os.path.expanduser( opt.fif_stage) ) +"/"
    #if opt.fif_path_prefix:
    #   fif_out += opt.fif_path_prefix+"/"
    fif_out += "/".join(fpdf.split("/")[-5:-1]) +"/"
    fif_out += "_".join(fpdf.split("/")[-5:])
    fif_out  = re.sub('/+','/',fif_out).replace('@',"_").replace(":","").replace("-","")
    fif_out +=  opt.fif_extention
   
    if not opt.overwrite:
       if os.path.isfile(fif_out):
          if opt.verbose:
             logger.info("---> 4D/BTI file   : {}\n".format(fpdf)+
                         " --> FIF / MNE file: {} size [Mb]: {}\n".format(os.path.basename(fif_out), os.path.getsize(fif_out)/1024.0**2 )+
                         " --> FIF / MNE path: {}\n".format(os.path.dirname(fif_out))+
                         "  -> !!! FIF File exist will not overwrite [overwrite={}\n".format(opt.overwrite)+
                         "===> Done")
          return
          
   # --- check and set opt parameter
    kwargs = dict()

    if jb.isNotEmpty(opt.rotation_x):   kwargs["rotation_x"] = float(opt.rotation_x)
    if jb.isNotEmpty(opt.translation) : kwargs["translation"]= float(opt.translation)
    if jb.isNotEmpty(opt.ecg_ch):       kwargs["ecg_ch"]     = opt.ecg_ch
    if jb.isNotEmpty(opt.eog_ch):       kwargs["eog_ch"]     = opt.eog_ch
   
   #--- run
    if opt.run:
       #print("RUN")
       # defaults mne017
       # pdf_fname, config_fname='config', head_shape_fname='hs_file', rotation_x=0.0, translation=(0.0, 0.02, 0.11), convert=True,
       # rename_channels=True, sort_by_ch_name=True, ecg_ch='E31', eog_ch=('E63', 'E64'), preload=False, verbose=None
       try:
           raw = mne.io.read_raw_bti(fpdf,config_fname=opt.config_fname,head_shape_fname=opt.head_shape_fname,preload=opt.preload,
                                     convert=opt.convert,rename_channels=opt.rename_channels,sort_by_ch_name=opt.sort_by_ch_name,
                                     verbose=opt.verbose,**kwargs)
       except:
           logger.exception("---> error in mne.io.read_raw_bti:\n   -> file: {}".format(fpdf))

      #--- make output filename and save
       if opt.save:
          fif_out = jb.apply_save_mne_data(raw,fname=fif_out,overwrite=opt.overwrite)
          
          if opt.verbose:
             logger.info(
                "===> 4D/BTI file   : {}\n".format(fpdf)+
                " --> FIF / MNE file: {} size [Mb]: {}\n".format(os.path.basename(fif_out), os.path.getsize(fif_out)/1024.0**2)+
                " --> FIF / MNE path: {}\n".format(os.path.dirname(fif_out)))
       return raw
    
    
def get_args(argv):
    """
    get args using argparse.ArgumentParser ArgumentParser
    e.g: argparse  https://docs.python.org/3/library/argparse.html

    jumeg wrapper for <mne.io.read_raw_bti>
    https://martinos.org/mne/stable/generated/mne.io.read_raw_bti.html#mne.io.read_raw_bti
    
    Results:
    --------
    parser.parse_args(), parser
    """
    info_global = """
                  JuMEG import 4D/BTi data to FIF
                  call <mne.io.read_raw_bti>
                  used python version: {}
                  """.format(sys.version.replace("\n"," "))
    h_fif_stage= """
                 fif stage: start path for fif files from list
                 -> start path to fif file directory structure
                 e.g. /data/megstore1/exp/INTEXT/mne/
                 """
    #h_fif_path_prefix="path prefix append to <fif stage>"
    
    h_pdf_stage="""
                pdf stage: start path for fif files from list
                -> start path to fif file directory structure
                e.g. /data/megstore1/exp/INTEXT/mne/"""
     
    h_pdf_fname         = "Path to the processed data file (PDF)"
    h_config_fname      = "Path to system config file"
    h_head_shape_fname  = "Path to the head shape file"
    h_rotation_x        = "Degrees to tilt x-axis for sensor frame misalignment. Ignored if convert is True."
    h_translation       = "array-like, shape (3,)\nThe translation to place the origin of coordinate system to the center of the head.\nIgnored if convert is True."
    h_convert           = "Convert to Neuromag coordinates or not."
    h_rename_channels   = "Whether to keep original 4D channel labels or not. Defaults to True."
    h_sort_by_ch_name   = "Reorder channels according to channel label.\n4D channels don’t have monotonically increasing numbers in their labels.\nDefaults to True."
    h_ecg_ch            = "The 4D name of the ECG channel.\nIf None, the channel will be treated as regular EEG channel."
    h_eog_ch            = "The 4D names of the EOG channels.\nIf None, the channels will be treated as regular EEG channels."
    h_preload           ="""Preload data into memory for data manipulation and faster indexing.
                            If True, the data will be preloaded into memory (fast, requires large amount of memory).
                            If preload is a string, preload is the file name of a memory-mapped file which is used
                            to store the data on the hard drive (slower, requires less memory)."""
    h_verbose           ="bool, str, int, or None"
    h_overwrite         ="overwrite existing fif files"
    
   #--- parser
    parser = argparse.ArgumentParser(info_global)

   #--- bti input files
    parser.add_argument("-spdf",   "--pdf_stage",         help=h_pdf_stage,metavar="PDF_STAGE",default=os.getenv("JUMEG_PATH_BTI_EXPORT","/data/MEG/meg_store2/megdaw_data21"))
  #  parser.add_argument("-spdf","--pdf_stage",help=h_pdf_stage,metavar="PDF_STAGE", default="${HOME}/MEGBoers/data/megdaw_data21")
   #--- fif output
    #parser.add_argument("-sfif",   "--fif_stage",         help=h_fif_stage,metavar="FIF_STAGE",default=os.getenv("JUMEG_PATH_MNE_IMPORT","/data/MEG/meg_strore1/exp"))
    parser.add_argument("-sfif","--fif_stage",help=h_fif_stage,metavar="FIF_STAGE",default="${HOME}/MEGBoers/data/exp/INTEXT/mne")
    #parser.add_argument("-fif_path_prefix","--fif_path_prefix",help=h_fif_path_prefix,default="mne")

    parser.add_argument("-fif_ext","--fif_extention",help="fif file extention",default="-raw.fif")

    #--- parameter
    parser.add_argument("-pdf_fname","--pdf_fname",help=h_pdf_fname,default="c,rfDC")
    parser.add_argument("-config_fname","--config_fname",        help=h_config_fname,    default="config")
    parser.add_argument("-head_shape_fname","--head_shape_fname",help=h_head_shape_fname,default="hs_file")

    parser.add_argument("-rot",            "--rotation_x",     help=h_rotation_x,     default=None)
    parser.add_argument("-translation",    "--translation",    help=h_translation,    default=None)
    parser.add_argument("-ecg_ch",         "--ecg_ch",         help=h_ecg_ch)
    parser.add_argument("-eog_ch",         "--eog_ch",         help=h_eog_ch)
  
  #--- flags
    parser.add_argument("-prel",  "--preload",        action="store_true",default=True,help=h_preload)
    parser.add_argument("-sort",  "--sort_by_ch_name",action="store_true",default=True,help=h_sort_by_ch_name)
    parser.add_argument("-rename","--rename_channels",action="store_true",default=True,help=h_rename_channels)
    parser.add_argument("-conv",  "--convert",        action="store_true",default=True,help=h_convert)
    parser.add_argument("-overwrite","--overwrite",   action="store_true",help=h_overwrite)
    parser.add_argument("-save",  "--save",           action="store_true",default=True,help="save as fif file")
    parser.add_argument("-v",     "--verbose",        action="store_true",help=h_verbose)
    parser.add_argument("-r",     "--run",            action="store_true",help="!!! EXECUTE & RUN this program !!!")
    parser.add_argument("-log",   "--logfile",        action="store_true",help="generate logfile")
  
  
  #--- init flags
  # ck if flag is set in argv as True
  # problem can not switch on/off flag via cmd call
    opt = parser.parse_args()
    for g in parser._action_groups:
        for obj in g._group_actions:
            if str( type(obj) ).endswith('_StoreTrueAction\'>'):
               if vars( opt ).get(obj.dest):
                  opt.__dict__[obj.dest] = False
                  for flg in argv:
                      if flg in obj.option_strings:
                         opt.__dict__[obj.dest] = True
                         break
  
    return opt, parser
  
#=========================================================================================
#==== MAIN
#=========================================================================================
def main(argv):
   
    opt, parser = get_args(argv)
    if len(argv) < 2:
       parser.print_help()
       sys.exit(-1)
       
  #--- init/update logger
    jumeg_logger.setup_script_logging(name=argv[0],opt=opt,logger=logger)
    
    if opt.run: apply_import_to_fif(opt)
    
   
if __name__ == "__main__":
   main(sys.argv)

