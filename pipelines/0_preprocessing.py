#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 18.04.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------
"""
JuMEG preprcessing script frame work

Head
- setup defaults

-------------

 preprocessing function


0_preprocessing.py -s $JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/FV -subj 211044 -c fbconfig_file.yaml -log -v -d -r

#stage = "/mnt/meg_store2/exp/JUMEGTest/FV"
#subjects=["211044","211747"]

-------------

MAIN

"""


import os,sys,logging,argparse

import mne
import yaml

#from utils import noise_reduction
#from utils import interpolate_bads_batch

from jumeg.base import jumeg_logger

logger = logging.getLogger("root")

__version__= "2019.04.18.001"

#--------------------------------------------------
#--- ToDo init arparser defaults in a module/script or CLS
#--- init argparser base parameter



#--------------------------------------------------
#--- init part some defaults for argparser
#--- declare here your default parameter and add/update <get_args> function

#---fb
#recordings_dir = "/mnt/meg_store2/exp/JUMEGTest/FV"
#recordings_dir = "$JUMEG_PATH_LOCAL_DATA/exp/JUMEGTest/FV"
#subjects=["211044","211747"]


defaults={
          "stage"          : ".",
          "fif_extention"  : ["meeg-raw.fif","rfDC-empty.fif"],
          "config"         : "config_file.yaml",
          "subjects"       : None,
          "logfile"        : False,
          "overwrite"      : False,
          "verbose"        : False,
          "debug"          : False
         }

def update_and_save_raw(raw,fin=None,fout=None,save=False,overwrite=True,postfix=None,separator="-"):
    """
    renaming filename in raw-obj
    saving mne raw obj to fif format
    
    ToDo:
     update into jumeg_base CLS
     
    Parameters
    ----------
     raw       : raw obj
     fout      : full output filename if set use this
     fin       : input file name <None>
     postfix   : used to generate <output filename> from <input file name> and <postfix>
                 e.g.:
     separator : split to generate output filename with postfix  <->
     save      : save raw to disk
     overwrite : <True>
     
    Returns
    --------
     filename,raw-obj
    """
    from distutils.dir_util import mkpath
   #--- use full filname
    if fout:
       fname =  fname = os.path.expandvars( os.path.expanduser(fout) )
   #--- make output file name
    else:
       fname = fin if fin else raw.filenames[0]
       fname = os.path.expandvars(os.path.expanduser(fname))  # expand envs
       if postfix:
          fpre,fext = fname.rsplit(separator)
          fname = fpre+ "," + postfix + separator + fext
    
    if save:
       try:
           if ( os.path.isfile(fname) and ( not overwrite) ) :
              logger.info(" --> File exist => skip saving data to : " + fname)
           else:
              logger.info(">>>> writing filtered data to disk...\n --> saving: "+ fname)
              mkpath( os.path.dirname(fname) )
              raw.save(fname,overwrite=True)
              logger.info(' --> Bads:' + str( raw.info['bads'] ) +"\n --> Done writing data to disk...")
       except:
           logger.exception("---> error in saving raw object:\n  -> file: {}".format(fname))
           
    return fname,raw


def apply_noise_reducer(raw_fname,raw=None,**cfg):
    '''
    Parameter:
    -----------
    <noise_reducer> parameter used in this function :
     fname_raw  : input raw filename
     raw        : <None>\n
   
    from config file <noise_reducer> part\n
     reflp      : <None>\n
     refhp      : <None>\n
     refnotch   : <None>\n
     save       : <None>\n
     overwrite  : overwrite existing file <True>
     plot_dir   : subdir to save plots
     
    <noise_reducer> parameter extended
       signals=[], noiseref=[], detrending=None,
       tmin=None, tmax=None,
       exclude_artifacts=True, checkresults=True, return_raw=False,
       complementary_signal=False, fnout=None, verbose=False
    
    Return:
    --------
     filename,raw-obj
     
    '''
    from jumeg.jumeg_noise_reducer import noise_reducer, plot_denoising

    logger.debug("  ->config parameter:\n{}".format(cfg) )
    
   #--- noise reduction
    # apply noise reducer thrice to reference channels with different freq parameters
    # !!! overwrite raw-obj !!!
   
    raw_changed=False
    
    #---ToDO start plot denoising orig raw psd
    # avoid reloading raw data
    if cfg.get("plot"):
       #  psds, freqs = psd_welch(raw, picks=picks, fmin=fmin, fmax=fmax,
       #                                 tmin=tmin, tmax=tmax, n_fft=n_fft,
       #                                 n_jobs=n_jobs, proj=proj)
       # open a plot fig in background and plot
       # now ready to overwrite raw obj
       pass
       
    
    #--- 1 low pass filter for freq below 5 hz
    if cfg.get("reflp"):
       raw = noise_reducer(raw_fname, raw=raw, reflp=cfg.get("reflp"),return_raw=True,verbose=cfg.get("verbose"))
       raw_changed = True
    #--- 2 high pass filter
    if cfg.get("refhp"):
       raw = noise_reducer(raw_fname, raw=raw, reflp=cfg.get("refhp"),return_raw=True,verbose=cfg.get("verbose"))
       raw_changed = True
    #--- 3 notch filter to remove power line noise
    if cfg.get("refnotch"):
       raw = noise_reducer(raw_fname, raw=raw,refnotch=cfg.get("refnotch"),fnout=None,return_raw=True,verbose=cfg.get("verbose"))
       raw_changed = True
    #--- save and update filename in raw
    if raw_changed:
       fname_out,raw = update_and_save_raw( raw,fin=raw_fname,fout=None,save=cfg.get("save"),postfix=cfg.get("postfix","nr"),overwrite=cfg.get("overwrite",True) )

   #--- plot results
   #---ToDO start plot denoising orig raw psd
   # avoid reloading raw data
    if cfg.get("plot"):
       #  psds, freqs = psd_welch(raw, picks=picks, fmin=fmin, fmax=fmax,
       #                                 tmin=tmin, tmax=tmax, n_fft=n_fft,
       #                                 n_jobs=n_jobs, proj=proj)
       # save final plotting to file in subdir ./plots
       #plot_name = fname_out.rsplit('-raw.fif')[0] + '-plot'
       #plot_denoising(raw_fname,fname_out],n_jobs = 1,fnout = plot_name,show = False)
       pass
  
    return fname_out,raw
 
#------------------------------------------------------
#--- script part
#------------------------------------------------------
def apply(opt):
    """
    
    :param opt:
    :return:
    
    """
  #--- init logging
    #logger = jumeg_logger.setup_script_logging(opt=opt,level="DEBUG",logfile=False)
    
  #--- load cfg
    with open(opt.config,'r') as f:
         config = yaml.load(f)
   
   #--- get subjects
    subjects = opt.subjects if opt.subjects else config.get("subjects")
   #--- get subjects
    stage    = opt.stage if opt.stage else config.get( "stage",defaults.get("stage") )
   #--- get fif file extentio to start with
    fif_extention = opt.fif_extention if opt.fif_extention else config.get( "fif_extention",defaults.get("fif_extention") )
  
   #--- raw obj
    raw = None
  
    subj_idx = 1
 
   #--- one main for loop do you preprocessing
    for raw_fname,recordings_dir in subject_file_list(subjects,stage=stage,file_extention=fif_extention):
       #--- setup logfile
        if opt.logfile:
           Hlog = jumeg_logger.update_filehandler(logger=logger,path=recordings_dir,prefix=".log")
           logger.info("  -> writing log to: {}".format(Hlog.filename))

        logger.info("---> Start PreProc : {} / {}\n".format(subj_idx,len(subjects))+
                    "  -> raw file name : {}\n".format(raw_fname)+
                    "  -> recordings dir: {}".format(recordings_dir))

    #--- call noise reduction
        cfg = config.get("noise_reducer")
        if cfg.get("run"):
           logger.info(" --> preproc noise_reducer: {} / {} raw file: {}".format(subj_idx,len(subjects),raw_fname))
           if check_file_extention(fname=raw_fname,file_extention=fif_extention):
              raw_fname,raw = apply_noise_reducer(os.path.join(recordings_dir,raw_fname),raw=raw,**cfg)
              # noise_reduction(recordings_dir,raw_fname,**cfg)

       #--- call interploate_bads
        """
        cfg = config.get("interpolate_bads")
        if cfg.get("run"):
           if check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention") ):
              raw_fname,raw = apply_interpolate_bads_batch(raw_fname,raw=raw,**cfg)
            # apply_interpolate_bads_batch(subjects, recordings_dir,**cfg)
       
   
       #--- call filter
        cfg = config.get("filter")
        if cfg.get("run"):
           if check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention") ):
              apply_filter(raw_fname,raw=raw,**cfg)
      
       #--- call resample
        cfg = config.get("resample")
        if cfg.get("run"):
           if check_file_extention(fname=raw_fname,file_extention=cfg.get("file_extention") ):
              apply_resample(raw_fname,raw=raw,**cfg)
        """
        subj_idx += 1

#=========================================================================
# end part helper functions

def check_file_extention(fname=None,file_extention=None):
    if not fname:
       return False
    if file_extention:
        if isinstance(file_extention,(list)):
           if fname.split(",")[-1] in file_extention:
              return True
        elif fname.split(",")[-1] == file_extention:
             return True
    return False


#--- ToDo put in utility/helper script  e.g. jumeg.base.jumeg_base
def subject_file_list(subjects,stage=".",file_extention=None,separator=","):
    """
    something like a contexmanager in a loop
    loop over subjects dir
    avoid copy/paste and boring repetitions
    place more error handling and file checking
    handels system ENVs

    https://stackoverflow.com/questions/29708445/how-do-i-make-a-contextmanager-with-a-loop-inside
    
    :param subjects       : string or list of strings e.g: subject ids
    :param stage          : start dir,stage
    :param file_extention : string or list  <None>
                            if <file_extention> checks filename extention ends with extentio from list
    :param separator      : split file to get file extention <,>
    :return:
     
     Example:
     --------
     for  raw_fname in subject_file_list(subjects,stage_dir= my_stage):
          logger.info("found in subject list: {}".format(raw_fname) )
    """
   #--- expand envs
    start_dir = os.path.expandvars( os.path.expanduser(stage) )
    
    if not isinstance(subjects,(list)):
       subjects=[subjects]
    if not isinstance(file_extention,(list)):
       file_extention=[file_extention]

    for subj in subjects:
        try:
          #--- check if its a dir
            if os.path.isdir( os.path.join(start_dir,subj) ):
               recordings_dir = os.path.join(start_dir,subj)
               for fname in os.listdir( recordings_dir ):
                 #--- ck for file and file extention
                   if file_extention:
                      if fname.split(separator)[-1] in file_extention:
                         yield fname,recordings_dir
                   else:
                      yield fname,recordings_dir
        except:
            logger.exception("---> error subject: {} dirname: {}".format(subj,start_dir))
  

def get_args(argv,parser=None):
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
                  version: {}
                  used python version: {}
                  """.format(__version__,sys.version.replace("\n"," "))
    h_stage     = """
                  stage/base dir: start path for ids from list
                  -> start path to directory structure
                  e.g. /data/megstore1/exp/M100/mne/
                  """
    h_fextention ="fif file extention or list of extentions, looking for files to process with these extention e.g. meeg-raw.fif or [meeg-raw.fif,rfDC-empty.fif]"
    
    h_subjects  = "subject id or list of ids  e.g.: 123 [234,456]"
    h_config    = "script config file, full filename"
    h_verbose   = "bool, str, int, or None"
    h_overwrite = "overwrite existing files"
    
   #--- parser
    if not parser:
       parser = argparse.ArgumentParser(description=description)
    else:
       parser.description = description
   #---
    parser.add_argument("-s",   "--stage",    help=h_stage,default=defaults.get("stage",".") )
    parser.add_argument("-subj","--subjects", help=h_subjects,default=defaults.get("subjects"))
    parser.add_argument("-fif_ext","--fif_extention",help=h_fextention,default=defaults.get("fif_extention") )
   #---
    parser.add_argument("-c","--config",help=h_config,default=defaults.get("config") )
   #--- flags
    parser.add_argument("-o",  "--overwrite",action="store_true",help=h_overwrite, default=defaults.get("overwrite") )
    parser.add_argument("-v",  "--verbose",  action="store_true",help=h_verbose,   default=defaults.get("verbose"))
    parser.add_argument("-d",  "--debug",    action="store_true",help="debug mode",default=defaults.get("debug"))

    parser.add_argument("-r",  "--run",      action="store_true",help="!!! EXECUTE & RUN this program !!!",
                        default=defaults.get("run",True))
    parser.add_argument("-log","--logfile",  action="store_true",help="generate logfile",default=defaults.get("logfile"))
  
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
    
    if opt.run: apply(opt)
    
   
if __name__ == "__main__":
   main(sys.argv)

