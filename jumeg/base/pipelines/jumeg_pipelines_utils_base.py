#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 13.05.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import sys,os,logging,yaml,argparse,glob
#from contextlib import redirect_stdout

import mne

from jumeg.base.jumeg_base         import jumeg_base as jb
from jumeg.base                    import jumeg_logger

logger = logging.getLogger("jumeg")

__version__= "2019.05.10.001"

#---
def get_args(argv,parser=None,defaults=None,version=None):
    """
    get args using argparse.ArgumentParser ArgumentParser
    e.g: argparse  https://docs.python.org/3/library/argparse.html

    :param argv:   the arguments, parameter e.g.: sys.argv
    :param parser: argparser obj, the base/default obj like --verbose. --debug
    :return:
    
    Results:
    --------
     parser.parse_args(), parser
    """
    
    description = """
                  JuMEG Preprocessing Script
                  script version : {}
                  python version : {}
                  """.format(version,sys.version.replace("\n"," "))
    h_stage     = """
                  stage/base dir: start path for ids from list
                  -> start path to directory structure
                  e.g. /data/megstore1/exp/M100/mne/
                  """
    h_lpath    = "path for list file, list of file to process containing list of full filenames"
    h_lname    = "list file name, list of file to process containing list of full filenames"
    
    h_fpath    = "path for file to process"
    h_fname    = "file name to process"
    
    h_fextention ="fif file extention or list of extentions, looking for files to process with these extention e.g. meeg-raw.fif or [meeg-raw.fif,rfDC-empty.fif]"
    
    h_subjects  = "subject id or ids  e.g.: 123 or 234,456"
    h_config    = "script config file, full filename"
    h_verbose   = "bool, str, int, or None"
    h_overwrite = "overwrite existing files"
    
   #--- parser
    if not parser:
       parser = argparse.ArgumentParser(description=description)
    else:
       parser.description = description
    
    if not defaults:
       defaults = {}
       
   #---  parameter settings  if opt  elif config else use defaults
    parser.add_argument("-s",   "--stage",    help=h_stage)#,default=defaults.get("stage",".") )
    parser.add_argument("-subj","--subjects", help=h_subjects)#,default=defaults.get("subjects"))
    parser.add_argument("-fif_ext","--fif_extention",help=h_fextention)#,default=defaults.get("fif_extention") )
   #---
    parser.add_argument("-lpath","--list_path",help=h_lpath)
    parser.add_argument("-lname","--list_name",help=h_lname)
   #---
    parser.add_argument("-fpath","--fpath",help=h_fpath)
    parser.add_argument("-fname","--fname",help=h_fname)
   #---
    parser.add_argument("-c","--config",help=h_config,default=defaults.get("config") )
   #--- flags
    parser.add_argument("-o",  "--overwrite",action="store_true",help=h_overwrite, default=defaults.get("overwrite") )
    parser.add_argument("-v",  "--verbose",  action="store_true",help=h_verbose,   default=defaults.get("verbose"))
    parser.add_argument("-d",  "--debug",    action="store_true",help="debug mode",default=defaults.get("debug"))
    parser.add_argument("-rec","--recursive",action="store_true",help="search recursive find files in subfolders",default=defaults.get("recursive",False))

    parser.add_argument("-r",  "--run",      action="store_true",help="!!! EXECUTE & RUN this program !!!",default=defaults.get("run",True))
    parser.add_argument("-log","--log2file", action="store_true",help="generate logfile",default=defaults.get("log2file"))
    parser.add_argument("-logoverwrite","--logoverwrite", action="store_true",help="overwrite existing logfile",default=defaults.get("logoverwrite"))
    parser.add_argument("-logprefix","--logprefix", help="logfile prefix",default= os.path.splitext( os.path.basename(argv[0]) )[0])

    return parser_update_flags(argv=argv,parser=parser)
    
def parser_update_flags(argv=None,parser=None):
    """
    init flags
    check if flag is set in argv as True
    if not set flag to False
    problem can not switch on/off flag via cmd call
 
    :param argv:
    :param parser:
    :return:
    opt  e.g.: parser.parse_args(), parser
    """
    opt = parser.parse_args()
    for g in parser._action_groups:
        for obj in g._group_actions:
            if str(type(obj)).endswith('_StoreTrueAction\'>'):
                if vars(opt).get(obj.dest):
                    opt.__dict__[obj.dest] = False
                    for flg in argv:
                        if flg in obj.option_strings:
                            opt.__dict__[obj.dest] = True
                            break
    return opt, parser

#===========================================================
#=== preproc part
#===========================================================
def print_to_logger(raw_fname,raw=None,**cfg):
  
   #--- log stdout,stderr
   jumeg_logger.log_stdout(label=" LOGTEST")
   jumeg_logger.log_stderr()
    
   print("  -> TEST1 print to logger: {}".format(raw_fname) )

  #--- return back stdout/stderr from logger
   jumeg_logger.log_stdout(reset=True)
   jumeg_logger.log_stderr(reset=True)
   
   return raw_fname,raw
