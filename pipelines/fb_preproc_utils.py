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

import sys,os,logging,yaml,argparse,glob
#from contextlib import redirect_stdout

import mne

from jumeg.base.jumeg_base         import jumeg_base as jb
from jumeg.base.jumeg_logger       import StreamToLogger # capture stout,stderr
from jumeg.plot.jumeg_plot_preproc import JuMEG_PLOT_PSD
from jumeg.jumeg_noise_reducer     import noise_reducer

from jumeg.jumeg_suggest_bads      import suggest_bads
from jumeg.jumeg_interpolate_bads  import interpolate_bads as jumeg_interpolate_bads

logger = logging.getLogger("root")

__version__= "2019.04.18.001"


#---
def init_globals(opt=None,defaults=None):
    """
     init global parameter
      load config file parameter
      init <subject list>,<stage>,<fif_extention>  from opt or config or defaults
      
    :param opt: arparser option obj
    :param defaults: default dict
    
    :return:
    config parameter: dict
    subject ids     : list
    stage           : start / base dir
    fif_extention   : list of file extentions, files must match to get started
    recursive       : option to search for files in subfolder <True/False>
    """
    #--- init logging
    #logger = jumeg_logger.setup_script_logging(opt=opt,level="DEBUG",logfile=False)
    
    if not defaults:
       defaults = {}
    
    #--- load cfg ToDo in CLS
    fcfg = opt.config if opt.config else defaults.get("config")
    if opt.debug:
        logger.info("  -> loading config file: {} ...".format(fcfg))
    with open(fcfg,'r') as f:
        config = yaml.load(f)
    if opt.debug:
        logger.info("  -> DONE loading config file")
    
    #--- get global parameter from cfg ToDo in CLS
    cfg_global = config.get("global")
    
    #--- get subjects
    subjects = opt.subjects.split(",") if opt.subjects else cfg_global.get("subjects",defaults.get("subjects"))
    
    #--- get subjects
    stage = opt.stage if opt.stage else cfg_global.get("stage",defaults.get("stage"))
    #--- get fif file extentio to start with
    fif_extention = opt.fif_extention if opt.fif_extention else cfg_global.get("fif_extention",defaults.get("fif_extention"))

    recursive = opt.recursive if opt.recursive else cfg_global.get("recursive",defaults.get("recursive"))
    
    return config,subjects,stage,fif_extention,recursive

#---
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


#---
def subject_file_list(subjects,stage=".",file_extention=None,separator=",",recursive=False):
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
    :param recursive      : recursive searching for files in <stage/subject_id> using glob.iglob <False>
    :return:
    fname,recordings_dir,subject id

     Example:
     --------
     for raw_fname,recordings_dir,subject_id in subject_file_list(subjects,stage_dir= my_stage):
         print("---> subject: {}\n  -> dir: {}\n  -> fname: {}".format(subject_id,recordings_dir,raw_fname) )
    """
    #--- expand envs
    start_dir = os.path.expandvars(os.path.expanduser(stage))
    
    if not isinstance(subjects,(list)):
        subjects = [subjects]
    if not isinstance(file_extention,(list)):
        file_extention = [file_extention]
    if recursive:
       fpatt = '**/*'
    else:
       fpatt = '*'

    for subj in subjects:
        try:
         #--- check if its a dir
           if os.path.isdir(os.path.join(start_dir,subj)):
               recordings_dir = os.path.join(start_dir,subj)
              
               with jb.working_directory(recordings_dir):
                    for fext in file_extention:
                        #for fname in glob.glob(fpatt + fext,recursive=recursive):
                        flist = glob.glob(fpatt + fext,recursive=recursive)
                        flist.sort()
                        for fname in flist:
                            dir = os.path.dirname(fname)
                            if dir == "":
                               dir = "."
                            yield os.path.basename(fname),dir,subj
        except:
            logger.exception("---> error subject : {}\n".format(subj)+
                             "  -> recordings dir: {}\n".format(recordings_dir)+
                             "  -> sub dir       : {}\n".format(dir)+
                             "  -> file          : {}\n".format(os.path.basename(fname)) )

#---
def get_args(argv,parser=None,defaults=None):
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
    parser.add_argument("-c","--config",help=h_config,default=defaults.get("config") )
   #--- flags
    parser.add_argument("-o",  "--overwrite",action="store_true",help=h_overwrite, default=defaults.get("overwrite") )
    parser.add_argument("-v",  "--verbose",  action="store_true",help=h_verbose,   default=defaults.get("verbose"))
    parser.add_argument("-d",  "--debug",    action="store_true",help="debug mode",default=defaults.get("debug"))
    parser.add_argument("-rec","--recursive",action="store_true",help="search recursive find files in subfolders",default=defaults.get("recursive",False))

    parser.add_argument("-r",  "--run",      action="store_true",help="!!! EXECUTE & RUN this program !!!",default=defaults.get("run",True))
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
 

#===========================================================
#=== preproc part
#===========================================================

#---------------------------------------------------
#--- apply_noise_reducer
#---------------------------------------------------
def apply_noise_reducer(raw_fname,raw=None,**cfg):
    '''
    apply <magic ee noise reducer> thrice to reference channels with different freq parameters
    
     !!! overwrite raw-obj, works inplace !!!
    
    1) apply nr low pass filter for freq below e.g.: 5Hz  <reflp>
    2) apply nr high pass filter if defined               <reflp>
    3) apply nr notch filter to remove power line noise   <refnotch>
    
    Parameter:
    -----------
    <noise_reducer> parameter used in this function :
     fname_raw  : input raw filename
     raw        : <None>\n
     cfg        : dict, part of config file <None>
        from config file <noise_reducer> part\n
        reflp      : <None>\n
        refhp      : <None>\n
        refnotch   : <None>\n
        save       : <None>\n
        overwrite  : overwrite existing file <True>
        plot_dir   : subdir to save plots
    ToDo add parameter extended
    <noise_reducer> parameter extended
       signals=[], noiseref=[], detrending=None,
       tmin=None, tmax=None,
       exclude_artifacts=True, checkresults=True, return_raw=False,
       complementary_signal=False, fnout=None, verbose=False

    Return:
    --------
     filename,raw-obj
    '''
    
   #--- init plot
    jplt = None
    logger.debug("  ->config parameter:\n{}".format( jb.pp_list2str(cfg) ))
    
   #--- catch stdout,stderr
    sys.stdout = StreamToLogger(logger,logging.INFO)
    sys.stderr = StreamToLogger(logger,logging.ERROR)
    
    
    ToDO noise_reducer set  reject in config file
    
    #--- fb
        reject = dict(grad=4000e-13, # T / m (gradiometers)
                      mag=10e-12,     # T (magnetometers)
                      eeg=40e-6,     # uV (EEG channels)
                      eog=250e-6)    # uV (EOG channels)

    #--- noise reduction
    # apply noise reducer thrice to reference channels with different freq parameters
    # !!! overwrite raw-obj !!!
    
    raw_changed = False

   #--- load raw, reset bads
    jb.verbose = True
    raw,bads = jb.update_bad_channels(raw_fname,raw=raw,save=True)
    
    #raw,raw_fname = jb.get_raw_obj(raw_fname,raw=raw)
    
    #--- start plot denoising orig raw psd
    #    avoid reloading raw data
    
    
    if cfg.get("plot"):
       jplt = JuMEG_PLOT_PSD(n_plots=2,name="denoising",verbose=True)
       jplt.plot(raw,title="orig: " + os.path.basename(raw_fname) )
    
    
    jb.update_bad_channels(raw_fname,raw=raw,save=True,append=True)
    
    
    #--- 1 nr low pass filter for freq below 5 hz
    if cfg.get("reflp"):
       #with redirect_stdout(logger):
       raw = noise_reducer(None,raw=raw,reflp=cfg.get("reflp"),return_raw=True,verbose=cfg.get("verbose"))
       raw_changed = True
    #--- 2 nr high pass filter
    if cfg.get("refhp"):
       #with redirect_stdout(logger):
       raw = noise_reducer(None,raw=raw,reflp=cfg.get("refhp"),return_raw=True,verbose=cfg.get("verbose"))
       raw_changed = True
    #--- 3  nr notch filter to remove power line noise
    if cfg.get("refnotch"):
       #with redirect_stdout(logger):
       raw = noise_reducer(None,raw=raw,refnotch=cfg.get("refnotch"),fnout=None,return_raw=True,
                           verbose=cfg.get("verbose"))
       raw_changed = True

   #--- save and update filename in raw
    if raw_changed:
       fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),
                                               postfix=cfg.get("postfix","nr"),overwrite=cfg.get("overwrite",True))
    
    #--- plot results, avoid reloading raw data
    if cfg.get("plot"):
       jplt.plot(raw,title="denoised: "+os.path.basename(raw_fname))
       if cfg.get("show_plot"):
          jplt.show_plot()
       jplt.save(fname=fname_out,plot_dir=cfg.get("plor_dir","plots"))
    
    return fname_out,raw


#---------------------------------------------------
#--- apply_interpolate_bads
#---------------------------------------------------
def apply_interpolate_bads(raw_fname,raw=None,**cfg):
    """

    :param raw_fname:
    :param raw:
    :param cfg:
    :return:
     filename,raw-obj
    """
    logger.debug("  ->config parameter:\n{}".format(cfg))
    
    #--- interpolate bads
    # !!! overwrite raw-obj !!!
    
    raw_changed = False
    '''
    ib_dict = dict()
    
    subj = op.basename(raw_fname).split('_')[0]
    
    if not op.isfile(bcc_fname):
        raw = mne.io.Raw(op.join(dirname,raw_fname),preload=True)
        # automatically suggest bad channels and plot results for visual inspection
        marked,raw = suggest_bads(raw,show_raw=True)
        
        ib_dict['bads_channels'] = raw.info['bads']
        
        # Interpolate bad channels using jumeg
        raw_bcc = jumeg_interpolate_bads(raw,origin=None,reset_bads=True)
        
        # check if everything looks good
        raw_bcc.plot(block=True)
        raw_bcc.save(bcc_fname,overwrite=True)
        
        ib_dict['output_file'] = bcc_fname
        
        save_state_space_file(ss_dict_fname,process='interpolate_bads',
                              input_fname=raw_fname,process_config_dict=ib_dict)
    
    #--- save and update filename in raw
    if raw_changed:
        fname_out,raw = jb.update_and_save_raw(raw,fin=raw_fname,fout=None,save=cfg.get("save"),
                                               postfix=cfg.get("postfix","nr"),overwrite=cfg.get("overwrite",True))
    '''
   # return fname_out,raw


