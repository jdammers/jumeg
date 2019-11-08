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

from contextlib import contextmanager,ContextDecorator
#from contextlib import redirect_stdout

import mne

from jumeg.base.jumeg_base         import jumeg_base as jb
from jumeg.base                    import jumeg_logger

logger = logging.getLogger("jumeg")

__version__= "2019.10.11.001"

#---
def get_args(argv,parser=None,defaults=None,version=None):
    """
    get args using argparse.ArgumentParser ArgumentParser
    e.g: argparse  https://docs.python.org/3/library/argparse.html

    :param argv:   the arguments, parameter e.g.: sys.argv
    :param parser: argparser obj, the base/default obj like --verbose. --debug
    :param version: adds version to description
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
    
    h_fpath    = "full path to file to process e.g.: must include <stage>"
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
    parser.add_argument("-fext","--file_extention",help=h_fextention)#,default=defaults.get("fif_extention") )
   #---
    parser.add_argument("-lpath","--list_path",help=h_lpath)
    parser.add_argument("-lname","--list_name",help=h_lname)
   #---
    parser.add_argument("-fpath","--fpath",help=h_fpath)
    parser.add_argument("-fname","--fname",help=h_fname)
   #---
    parser.add_argument("-c","--config",help=h_config,default=defaults.get("config") )
   #--- flags
    # parser.add_argument("-o",  "--overwrite",action="store_true",help=h_overwrite, default=defaults.get("overwrite") )
    parser.add_argument("-v",  "--verbose",  action="store_true",help=h_verbose,   default=defaults.get("verbose"))
    parser.add_argument("-d",  "--debug",    action="store_true",help="debug mode",default=defaults.get("debug"))
    parser.add_argument("-rec","--recursive",action="store_true",help="search recursive find files in subfolders",default=defaults.get("recursive",False))

    parser.add_argument("-r",  "--run",      action="store_true",help="!!! EXECUTE & RUN this program !!!",default=defaults.get("run",True))
    parser.add_argument("-log","--log2file", action="store_true",help="generate logfile",default=defaults.get("log2file"))
    parser.add_argument("-logoverwrite","--logoverwrite", action="store_true",help="overwrite existing logfile",default=defaults.get("logoverwrite"))
    parser.add_argument("-logprefix","--logprefix", help="logfile prefix",default= defaults.get("logprefix",os.path.splitext( os.path.basename(argv[0]) )[0]) )

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



#===========================================================
#=== Jumeg_PipelineFrame
#===========================================================
class JuMEG_PipelineFrame(object):
    """"
     CLS like a contextmanager
     https://stackoverflow.com/questions/9663562/what-is-the-difference-between-init-and-call
    :param kargs:
           kargs[0]: function to call
    :param kwargs:
           raw_fname=None,raw=None,config=None,fname_out=None,verbose=False,debug=False
    Return
    -------
    fname_out,raw obj
    
    Example
    -------
    from jumeg.base.pipelines.jumeg_pipelines_utils_base import JuMEG_PipelineFrame
    
    @JuMEG_PipelineFrame
    def apply_suggest_bads(raw_fname=None,raw=None,config=None,fname_out=None):
        # do you stuff here

    __main__
     raw_fname,raw = apply_suggest_bads(raw_fname=raw_fname,raw=raw,config=jpl.config.get("suggest_bads"))
    
    """
    
    def __init__(self,*kargs,**kwargs):
        super().__init__()
        self._function  = None
        self._raw_fname = None
        self._cfg       = None
        self._raw       = None
        self._fname_out = None
        self._verbose   = None
        self._debug     = None
        self._run       = False
        self._raw_ischanged = False
        self._rest_bads  = False
        
        if len(kargs):
           self._function = kargs[0]
        self._update_from_kwargs(**kwargs)
    
    @property
    def verbose(self): return self._verbose
    @verbose.setter
    def verbose(self,v):
        jb.verbose    = v # set jumeg_base obj
        self._verbose = v
        
    @property
    def fname_out(self):
        return self._fname_out
    
    @property
    def raw_fname(self):
        return self._raw_fname
    
    @property
    def raw(self):
        return self._raw
    
    @raw.setter
    def raw(self,v):
        self._raw = v
    
    @property
    def RawIsChanged(self):
        return self._raw_ischanged
    
    @RawIsChanged.setter
    def RawIsChanged(self,v):
        self._raw_ischanged = v
    
    @property
    def run(self):
        return self._run
    
    @property
    def label(self):
        try:
          return self._function.__name__
        except:
          return None
        
    def _update_from_kwargs(self,**kwargs):
        """

         :param kwargs:
         raw_fname=None,raw=None,config=None,function_name=None,verbose=False,debug=False
         :return:
        """
        self._raw_fname = kwargs.get("raw_fname")  # reset
        self._raw       = kwargs.get("raw")
        self._cfg       = kwargs.get("config")
        self.verbose    = kwargs.get("verbose",self._verbose)
        self._debug     = kwargs.get("debug",self._debug)
        self._reset_bads= kwargs.get("reset_bads",False)
        
    def info(self):
        msg = ["  -> raw obj            : {}".format(self._raw),
               "  -> file extention list: {}".format(self._cfg.get("file_extention")),
               "  -> input  raw file    : {}".format(self._raw_fname),
               "  -> output raw file    : {}".format(self._fname_out)]
        if self._raw:
           msg.append("  -> Bads               : {}".format(str(self._raw.info['bads'])) )
        return "\n".join(msg)
    
    def __call__(self,*kargs,**kwargs):
        """
        overwriten method
        
        :param kargs:
        :param kwargs:
        :return:
         fname_out,raw obj
        """
        
        if len(kargs):
           self._function = kargs[0]
     
        if not kwargs:
           return
     
        self._on_enter(**kwargs)
        
        if not self.run or not self.raw:
           return self.fname_out,self.raw
   
        try:
            kwargs["raw_fname"] = self.raw_fname
            kwargs["fname_out"] = self.fname_out
            kwargs["raw"]       = self.raw
            self._fname_out,self._raw,self._raw_ischanged = self._function(**kwargs)
            
        except:
            logger.exception("---> ERROR in {}\n".format(self.label) + self.info())
        
        finally:
            return self._on_exit()
           
    
    def _on_enter(self,**kwargs):
        """
        :return:
         raw_fname, raw obj
        """
        self._update_from_kwargs(**kwargs)
        logger.info("  -> < {} > file name: {}".format(self.label,self._raw_fname))
       
        if self._debug:
           logger.debug(" -> config parameter:\n{}".format(self._cfg))

       # self._raw = None
        self._run = True
        self._fname_out = None

        #--- check file extention in list
        if not jb.check_file_extention(fname=self._raw_fname,file_extention=self._cfg.get("file_extention")):
            logger.info(
                " --> preproc {}:  SKIPPED : <file extention not in extention list>\n".format(self.label) + self.info())
            self._run = False
            return None,None
        
        self._fname_out,self._raw = jb.update_and_save_raw(self._raw,fin=self._raw_fname,fout=None,save=False,
                                                           postfix=self._cfg.get("postfix"),overwrite=False)
       #--- return raw_fname,raw
        if not self._cfg.get("run"):
            logger.info(" --> preproc {}:  SKIPPED : <run> is False\n".format(self.label) + self.info())
            self._run = False
            return self._raw_fname,None
        
        #--- if file exist and do not overwrite
        if os.path.isfile(self._fname_out) and (self._cfg.get("overwrite",False) == False):
            logger.info(
                " --> preproc {}: SKIPPED : do not overwrite existing output file\n".format(self.label) + self.info())
            self._run = False
            return self._fname_out,None
        
        #--- OK load raw, reset bads
        self._raw,self._raw_fname = jb.get_raw_obj(self._raw_fname,raw=self._raw,reset_bads=self._reset_bads)
        logger.info(" --> preproc {}\n".format(self.label) + self.info())
        return self._raw_fname,self._raw
    
    def _on_exit(self,**kwargs):
        #--- save and update filename in raw
        if not self.run: return
        
        save = False
        if self._cfg.get("save"):
           save = self.RawIsChanged
        
        #--- update filename in raw and save if save
        self._fname_out,self._raw = jb.update_and_save_raw(self._raw,fin=self._raw_fname,fout=None,save=save,
                                                           update_raw_filenname=True,postfix=self._cfg.get("postfix"),
                                                           overwrite=True)
        logger.info(" --> done preproc: {} \n".format(self.label) + self.info())
        
        if self._fname_out:
           return self._fname_out,self._raw
        else:
            raise Exception("---> ERROR file name not defined !!!")
    
    
    #--- use in with statement
    #def __enter__(self):
    #    return self
    
    #def __exit__(self,exc_type,exc_value,tb):
    #    if exc_type is not None:
    #        logger.exception("ERROR")
    #        # return False # uncomment to pass exception through
    #    return True


