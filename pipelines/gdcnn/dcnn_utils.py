#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Authors:
  - Frank Boers: f.boers@fz-juelich.de
  - Jurgen Dammers: j.dammers@fz-juelich.de

"""

import sys, os, argparse

import numpy as np
import mne
from mne.preprocessing import ctps_ as ctps

import pprint
from dcnn_logger import get_logger, init_logfile
logger = get_logger()

__version__= "2020.08.04.001"

#--- FB
def setup_logfile(fname,version=__version__,verbose=False,overwrite=True,level=None):
    Hlog = init_logfile(logger=logger,fname=fname,overwrite=overwrite,level=level)
    msg  = ["DCNN : {}".format(version),
            "  -> logfile  : {}".format(Hlog.filename),
            "  -> overwrite: {}".format(overwrite)]
    logger.info("\n".join(msg))
    return Hlog

def dict2str(d,indent=2):
    '''
    pretty printing
    wrapper for pprint.PrettyPrinter
    Parameters
    ----------
    d : dict
    intent : passed to PrettyPrinter <2>

    Returns
    -------
    strind
    '''
    pp = pprint.PrettyPrinter(indent=2)
    return ''.join(map(str,pp.pformat(d)))


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
                  DCNN Script
                  script version : {}
                  python version : {}

                  Example:

                  dcnn_run.py -cfg config_jumeg_4D.yaml -base /data/exp/DCNN/mne -data 007/DCNN01/20200220_2020/2 -pat *.int-raw.fif -v -ica -ck -log -h

                  """.format(version,sys.version.replace("\n"," "))
    
    h_stage = """
                  stage/base dir: start path for ids from list
                  -> start path to directory structure
                  e.g. /data/megstore1/exp/M100/mne/
                  """
    h_pattern = "fif file extention to search for"
    h_config = "script config file, full filename"
    
    #--- parser
    if not parser:
        parser = argparse.ArgumentParser(description=description,formatter_class=argparse.RawTextHelpFormatter)
    else:
        parser.description = description
    
    if not defaults:
        defaults = { }
    
    #--  parameter settings  if opt  elif config else use defaults
    #parser.add_argument("-f","--fname",help=h_fname)
    #parser.add_argument("-stg","--stage",    help=h_stage)#,default=defaults.get("stage",".") )
    #--
    parser.add_argument("-cfg","--config",help=h_config)
    parser.add_argument("-pat","--pattern",help=h_pattern,default="*-raw.fif")
    #--
    parser.add_argument("-base","--basedir",     help="base dir to search for raw files")
    parser.add_argument("-dmeg","--data_meg",    help=" meg data dir, search for input files ")
    parser.add_argument("-dtrain","--data_train",help="training data dir")
    #--
    parser.add_argument("-logpre","--logprefix",help="logfile prefix",default="dcnn")
    #-- flags
    parser.add_argument("-v","--verbose",action="store_true",help="tell me more")
    # parser.add_argument("-d",  "--debug",  action="store_true",help="debug mode")
    parser.add_argument("-ica","--ica",action="store_true",help="execute dcnn label ica")
    parser.add_argument("-ck","--check",action="store_true",help="execute dcnn label check")
    #--
    parser.add_argument("-jd","--jd",action="store_true",help="use jd test settings")
    parser.add_argument("-fb","--fb",action="store_true",help="use fb test settings")
    #--
    parser.add_argument("-log","--log2file",action="store_true",help="generate logfile")
    parser.add_argument("-logov","--logoverwrite",action="store_true",help="overwrite existing logfile",default=True)
    
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
    return opt,parser

def expandvars(v):
    """
    expand env's from string works on list or string
     => expandvars and expanduser
    :param v: list of strings  or string
    :return: input with expanded env's
    """
    if not v: return None
    if isinstance(v,(list)):
       for i in range(len(v)):
           v[i] = os.path.expandvars(os.path.expanduser( str(v[i]) ))
       return v
    else:
       return os.path.expandvars(os.path.expanduser(str(v)))

    # TODO: the code below is unreachable
    return os.path.expandvars(os.path.expanduser( str(v) ))

def isPath(pin,head="check path exist",exit_on_error=False,logmsg=False,mkdir=False):
    """
    check if file exist

    Parameters
    ----------
    :param <string>     : full path to check
    :param head         : log msg/error title
    :param logmsg       : log msg <False>
    :param mkdir        : makedir <False>
    :param exit_on_error: will exit pgr if not file exist <False>

    :return:
    full path / False or call <exit>

    """
    p = os.path.abspath(expandvars(pin))
    if os.path.isdir(p):
        # avoid unimportant output
        # if logmsg:
        #    logger.info(head+"\n --> dir exist: {}\n  -> abs dir: {}".format(pin,p))
       return p
    elif mkdir:
        os.makedirs(p)
        if logmsg:
           logger.info(head+"\n --> make dirs: {}\n  -> abs dir: {}".format(pin,p))
        return p
   #--- error no such file
    logger.error(head+"\n --> no such directory: {}\n  -> abs dir: {}".format(pin,p))
    if exit_on_error:
       raise SystemError()
    return False


def isFile(f):
    '''
    FB
    check if file exist
    eval env vars inn filename

    Parameters
    ----------
     f : string, full filename

    Return
    -------
     f : string, fullfilename with expanded env's            
    '''
      
    f = expandvars(f)
    if os.path.isfile(f):
       return f
    return None


def file_looper(rootdir='.',pattern='*',recursive=False,version=None,verbose=False,log2file=False,logoverwrite=True,
                level=None):
    """
     # ToDo run for list of files or search in subdirs
     loop over files found with < find_files>
     Looks for all files in the root directory matching the file
     name pattern.
     setup log-file for logging

     Parameters:
     -----------
     rootdir : str
               Path to the directory to be searched.
     pattern : str
               File name pattern to be looked for.
     version : version number
     verbose : False

     log2file: False
     logoverwrite: True
     level: loglevel <None> use effective loglevel

     Returns:
     --------
     None
    """
    
    fnames = find_files(rootdir=rootdir,pattern=pattern,recursive=recursive)
    
    msg_info = [" -> path     : {}".format(rootdir),
                " -> pattern  : {}".format(pattern),
                " -> recursive: {}\n".format(recursive)]
    try:
        fnraw = None
        if not fnames:
            msg = ["ERROR No files found"]
            msg.extend(msg_info)
            raise Exception("\n".join(msg))
        
        msg = ["DCNN files to process: {}\n  -> {}".format(len(fnames),"\n  -> ".join(fnames))]
        
        for fnraw in fnames:
            Hlog = None
            # --- setup logger: log to file handler
            if log2file:
                Hlog = setup_logfile(os.path.splitext(fnraw)[0] + ".log",version=version,verbose=verbose,
                                     overwrite=logoverwrite,level=level)
                msg.append("  -> writing log to   : {}".format(Hlog.filename))
                logger.info("\n".join(msg))
            try:
                #--- do your stuff here
                yield fnraw
            
            except:
                raise Exception("DCNN ERROR in file: {}\n".format(fnraw))
            if Hlog:
                Hlog.close()
    except:
        msg = ["\nDCNN ERROR parameter:",
               " -> file name: {}".format(fnraw),
               "-" * 40]
        msg.extend(msg_info)
        raise Exception("\n".join(msg))


def get_raw_filename(raw,index=0):
        """ get filename from raw obj
        
        Parameters:
        -----------
        raw     : raw-obj to modify
        index   : index in list of filenames from raw.filenames <0>      
                  if index = list return filename list
        Returns:
        ----------
         first filename or None
        
        Example:
        ----------
         from jumeg.jumeg_base import jumeg_base as jb 
         fname = jb.get_raw_filename(raw)
        """
        if raw:
           if hasattr(raw,"filenames"): 
              if index == "list"                : return raw.filenames 
              if abs(index) < len(raw.filenames): return raw.filenames[index]
              return raw.filenames
           return raw.info.get('filename')
        return None 

def read_raw(fname,raw=None,path=None,preload=True,reload_raw=False,reset_bads=False,clean_names=False,
             file_extention={'fif':'.fif','brainvision':'.vhdr','ctf':'.ds','ica':'.ica'},
             system_clock='truncate',verbose=False):
    """
    supported file formats via file extention:
     BrainVision: <.vhdr>
     CTF        : <.ds>
     FIF        : <.fif>
     MNE ICA obj: <.ica>
      
     check for filename or raw obj
     check file format
     if filename -> load fif file
        
        Parameters
        ----------
         fname     : name of raw-file
         raw       : raw obj <None>
                     if raw: return raw and fullfilename of raw
         preload   : True
         reload_raw: reload raw-object via raw.filename <False>
         reset_bads: reset bads <False>
         
         
         CTF parameter:
           clean_names = False,
           system_clock = 'truncate'
        
         verbose: <False>  
        
        Return
        ----------
         raw obj,fname from raw obj
        """
        
    #-- ToDo  make a raw loader CLS
    if verbose:
       msg= ["start reading raw data:\n",
             "  -> raw : {}\n".format(raw),
             "  -> file: {}\n".format(fname),
             "  -> path: {}\n".format(path)]
       if raw:
           msg.append("  -> Bads: {}\n".format(str(raw.info.get('bads'))))
       logger.info("".join(msg) )
  
    if raw:
       fname = None
       if reset_bads:
          if "bads" in raw.info:
             raw.info["bads"] = []
       if reload_raw:
          fn = raw.filenames[0]
       else:
          return raw ,raw.filenames[0]

    if fname:
       fn = expandvars( fname )
       if path:
          path = expandvars(path)
          fn   = os.path.join(path,fn)
    
    if not fn:
       logger.error("ERROR no such file or raw-object:\n  -> raw obj: {}\n  -> fname: {}\n  -> path : {}".
                    format(raw,fname,path))
       return None,None
   
    try:
        if not isFile(fn):
           raise FileNotFoundError("ERROR no file found: {}".format(fn))
        
        if ( fn.endswith(file_extention["brainvision"]) ):
           # --- changed in mne version 019.dev
           # raw = mne.io.read_raw_brainvision(fn,response_trig_shift=self.brainvision_response_shift,preload=preload)
           raw = mne.io.read_raw_brainvision(fn,preload=preload)
           #raw.info['bads'] = []
        elif (fn.endswith(file_extention["ica"])):
            raw = mne.preprocessing.read_ica(fn)
        elif ( fn.endswith(file_extention["ctf"]) ):
            raw = mne.io.read_raw_ctf(fn,system_clock=system_clock,preload=preload,clean_names=clean_names,verbose=verbose)
        else:
           raw = mne.io.Raw(fn,preload=preload)

        if not raw:
           raise FileNotFoundError("ERROR could not load RAW object: {}".format(fn))
    except:
        logger.exception("ERROR: could not get raw obj from file:\n --> FIF name: {}\n  -> file not exist".format(fn))
        return None,None
    
    if reset_bads:
       try:
          if "bads" in raw.info:
             raw.info["bads"] = []
             logger.debug("  -> resetting bads in raw")
       except AttributeError:
            logger.exception("ERROR -> cannot reset bads in raw: {}".format(fn))

    if verbose:
       msg = ["done loading raw obj:",
              "  -> path: {}".format(path),
              "  -> input filename: {}".format(fname),
              "-"*40,
              "  -> used filename : {}".format(fn),
              "  -> raw filename  : {}".format(get_raw_filename(raw)),
              "-"*40,
              "  -> Bads: {}".format(str(raw.info.get('bads')))]

       try:
          msg.append(" --> mne.annotations in RAW:\n  -> {}\n".format(raw.annotations))
       except:
          msg.append(" --> mne.annotations in RAW: None\n")
       logger.info("\n".join(msg))

    return raw,fn 



# ======================================================
#
# find files
# copied from jumeg
#
# ======================================================
def find_files(rootdir='.', pattern='*', recursive=False):
    """
    Looks for all files in the root directory matching the file
    name pattern.

    Parameters:
    -----------
    rootdir : str
        Path to the directory to be searched.
    pattern : str
        File name pattern to be looked for.

    Returns:
    --------
    files : list
        List of file names matching the pattern.
    """
    import os
    import fnmatch
    rootdir = expandvars(rootdir)
    files   = []
    for root, dirnames, filenames in os.walk( rootdir ):
        if not recursive:
            del dirnames[:]
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    files = sorted(files)

    return files


# ======================================================
#
# Function to rescale data
#  copied from jumeg
#
# ======================================================
def rescale(data_arr, minval, maxval):
    """ Function to rescale an array to the desired range. """
    min_data = -1.0 * np.min(data_arr)
    max_data = np.max(data_arr)
    if (max_data + min_data) != 0:
        b = (maxval - minval) / (max_data + min_data)
        data_new = ((data_arr + min_data) * b) + minval
    # if data_arr is a constant function
    else:
        data_new = (max_data - min_data) / 2.0
    return data_new

# ======================================================
#
# helper function to return the union list
#
# ======================================================
def get_unique_list(*args):
    """
    Parameters:
    -----------
    value,list of values,np.array1D
    e.g. a,b,c  or a,[b,c,d],...,[1,2,3]

    Returns:
    -------
     one unique sorted list
    """
    idx = list()
    for ix in args:
        if isinstance(ix, (np.ndarray)):
            idx.extend(ix.tolist())
        elif isinstance(ix, (list)):
            idx.extend(ix)
        else:
            idx.append(ix)

    idx = sorted(set(idx).union())

    if len(idx) > 0:
        if idx[0] == -1:
            idx = idx[1:]

    return idx


# ======================================================
#
# 4D noise reduction
#
# ======================================================
def apply_noise_reduction_4d(raw, refnotch=[], reflp=5., refhp=0.1,hpc=False,vendor="4D"):
    '''
    apply noise ruduction for hcp data via call to
    jumeg_noise_reducer_hcp 
    
    Parameters
    ----------
    raw : TYPE
        DESCRIPTION.
    refnotch : TYPE, optional
        DESCRIPTION. The default is [].
    reflp : TYPE, optional
        DESCRIPTION. The default is 5..
    refhp : TYPE, optional
        DESCRIPTION. The default is 0.1.
    vendor: string, The default is 4D [4D,HCP], work around for HCP

    Returns
    -------
    raw : TYPE
        DESCRIPTION.

    '''
    try: # FB
        if vendor.upper().endswith("HCP"):
           from jumeg.jumeg_noise_reducer_hcp import noise_reducer
        else:
           from jumeg.jumeg_noise_reducer     import noise_reducer
    except:
        if vendor.upper().endswith("HCP"):
           from jumeg.jumeg_noise_reducer_hcp import noise_reducer
        else:
           from jumeg.jumeg_noise_reducer     import noise_reducer 
        
    # apply noise reducer three times to reference channels with different freq parameters
    # 1. low pass filter for freq below 5 Hz
    raw = noise_reducer(None,raw=raw, reflp=reflp, return_raw=True)

    # 2. high pass filter for freq above 0.1 Hz using gradiometer refs
    raw = noise_reducer(None,raw=raw, refhp=refhp, noiseref=['RFG ...'], return_raw=True)

    # 3. remove power line noise
    raw = noise_reducer(None,raw=raw, refnotch=refnotch, return_raw=True)

    return raw


# ======================================================
#
#  determine chop indices and chop times for cropping
#
# ======================================================
def get_chop_times_indices(times, chop_length=180., chop_nsamp=None, strict=False, exit_on_error=False):
    """
    calculate chop times for every X s
    where X=interval.
    
    Author: J.Dammers
    update: F.Boers

    Parameters
    ----------
    times: the time array
    chop_length: float  (in seconds)
    chop_nsamp: int (number of samples per chop)
                if set, chop_length is ignored

    strict: boolean (only when chop_samp=None)
            True: use be strict with the length of the chop
                  If the length of the last chop is less than X
                  the last chop is combined with the penultimate chop.
            False: (default) the full time is equally distributed across chops
                   The last chop will only have a few samples more
    
    exit_on_error: boolean <False>
                   error occures if chop_length < times 
                    -> if True : exit on ERROR  
                    -> if False: try to adjust chop_time
                       e.g. chop_times: is one chop with [ times[0],times[-1] ]                                      

    Returns
    -------
    chop_times : list of float
                 Time range for each chop
    chop_time_indices : list of indices defining the time range for each chop

    """

    n_times = len(times)

    try:
        data_type = times.dtype()
    except:
        data_type = np.float64 

    if chop_nsamp:  # compute chop based on number of samples
        n_chops = int(n_times // chop_nsamp)
        if n_chops == 0:
            n_chops = 1
        n_times_chop = chop_nsamp
    else:  # compute chop based on duration given
        dt = times[1] - times[0]  # time period between two time samples
        n_chops, t_rest = np.divmod(times[-1], chop_length)
        n_chops = int(n_chops)
      
      
        # chop duration in s
        if strict:
          #-- ToDo ck for times[-1] < chop_length
            chop_len = chop_length
        else:
            chop_len = chop_length + t_rest // n_chops  # add rest to chop_length
            
            msg1=[
                  "  -> number of chops      : {}".format(n_chops),
                  "  -> calculated chop legth: {}".format(chop_len),
                  "  -> rest [s]             : {}".format(t_rest),
                  "-"*40,
                  "  -> chop length          : {}".format(chop_length),
                  "  -> numer of timepoints  : {}".format(n_times),
                  "  -> strict               : {}".format(strict),
                  "-"*40,
                  "  -> exit on error        : {}\n".format(exit_on_error)
                 ]
           #---
            try:
               n_times_chop = int(chop_len / dt)
            except:
               if exit_on_error:   
                  msg=["EXIT on ERROR"]
                  msg.extend( msg1 )
                  logger.exception("\n".join(msg))
                  assert (chop_len > 0),"Exit => chop_len: {}\n".format(chop_len)
               else: # data size < chop_length
                  msg=["setting <chop_len> to number of timepoints!!!"]
                  msg.extend(msg1)
                  logger.error( "\n".join(msg) )  
                  
                  n_times_chop = n_times
                  n_chops = 1
                  msg=["data length smaller then chop length !!!",
                       " --> Adjusting:",
                       "  -> number of chops: {}".format(n_chops),
                       "  -> chop time      : {}".format(n_times_chop)
                       ]
                  logger.warning("\n".join(msg))
                  
        # check if chop length is larger than max time (e.g. if strict=True)
        if n_times_chop > n_times:
            n_times_chop = n_times

    # compute indices for each chop
    ix_start = np.arange(n_chops) * n_times_chop  # first indices of each chop
    ix_end = np.append((ix_start - 1)[1:], n_times - 1)  # add last entry with last index

    # chop indices
    chop_indices = np.zeros([n_chops, 2], dtype=np.int)
    chop_indices[:, 0] = ix_start
    chop_indices[:, 1] = ix_end

    # times in s
    chop_times = np.zeros([n_chops, 2], dtype=data_type)
    chop_times[:, 0] = times[ix_start]
    chop_times[:, 1] = times[ix_end]

    return chop_times, chop_indices


    
# ======================================================
#
#  get_ics_cardiac:  determine cardiac related ICs
#  copied from jumeg
#
# ======================================================
def get_ics_cardiac(meg_raw, ica, flow=8, fhigh=25, tmin=-0.4, tmax=0.4,
                    name_ecg='ECG 001', use_CTPS=True, event_id=999,
                    score_func='pearsonr', thresh=0.25):
    '''
    Identify components with cardiac artefacts
    '''

    from mne.preprocessing import find_ecg_events
    idx_ecg = []
    if name_ecg in meg_raw.ch_names:
        # get and filter ICA signals
        ica_raw = ica.get_sources(meg_raw)
        ica_raw.filter(l_freq=flow, h_freq=fhigh, n_jobs=2, method='fft')
        # get ECG events
        events_ecg, _, _  = find_ecg_events(meg_raw, ch_name=name_ecg, event_id=event_id,
                                            l_freq=flow, h_freq=fhigh, verbose=False)

        # CTPS
        if use_CTPS:
            # create epochs
            picks = np.arange(ica.n_components_)
            ica_epochs = mne.Epochs(ica_raw, events=events_ecg, event_id=event_id,
                                    tmin=tmin, tmax=tmax, baseline=None,
                                    proj=False, picks=picks, verbose=False)
            # compute CTPS
            _, pk, _ = ctps.ctps(ica_epochs.get_data())

            pk_max = np.max(pk, axis=1)
            scores_ecg = pk_max
            ic_ecg = np.where(pk_max >= thresh)[0]
        else:
            # use correlation
            idx_ecg = [meg_raw.ch_names.index(name_ecg)]
            ecg_filtered = mne.filter.filter_data(meg_raw[idx_ecg, :][0],
                                                  meg_raw.info['sfreq'], l_freq=flow, h_freq=fhigh)
            scores_ecg = ica.score_sources(meg_raw, target=ecg_filtered, score_func=score_func)
            ic_ecg = np.where(np.abs(scores_ecg) >= thresh)[0]
    else:
        logger.warning(">>>> Warning: Could not find ECG channel %s" % name_ecg)
        events_ecg = []

    if len(ic_ecg) == 0:
        ic_ecg     = np.array([-1])
        scores_ecg = np.zeros(ica.n_components) #scores_ecg = np.array([-1]) ???
        events_ecg = np.array([-1])
    else:
        events_ecg[:,0] -= meg_raw.first_samp    # make sure event samples start from 0

    return [ic_ecg, scores_ecg, events_ecg]


# ======================================================
#
#  get_ics_ocular: determine occular related ICs
#  copied from jumeg
#
# ======================================================
def get_ics_ocular(meg_raw, ica, flow=2, fhigh=20, name_eog='EOG 002',
                   score_func='pearsonr', thresh=0.2, event_id=998):
    '''
    Find Independent Components related to ocular artefacts
    '''

    from mne.preprocessing import find_eog_events

    # ---------------------------
    # vertical EOG
    # ---------------------------
    ic_eog = []
    if name_eog in meg_raw.ch_names:
        idx_eog = [meg_raw.ch_names.index(name_eog)]
        eog_filtered = mne.filter.filter_data(meg_raw[idx_eog, :][0],
                                              meg_raw.info['sfreq'], l_freq=flow, h_freq=fhigh)
        scores_eog = ica.score_sources(meg_raw, target=eog_filtered, score_func=score_func)
        ic_eog = np.where(np.abs(scores_eog) >= thresh)[0]  # count from 0
        # get EOG ver peaks
        events_eog = find_eog_events(meg_raw, ch_name=name_eog, event_id=event_id,
                                     l_freq=flow, h_freq=fhigh, verbose=False)

        # make sure event samples start from 0
        events_eog[:,0] -= meg_raw.first_samp

    else:
        logger.warning(">>>> Warning: Could not find EOG channel %s" % name_eog)
        events_eog = []

    if len(ic_eog) == 0:
        ic_eog     = np.array([-1])
        scores_eog = np.zeros(ica.n_components) #scores_eog = np.array([-1]) ???
        # events_eog = np.array([-1])

    return [ic_eog, scores_eog, events_eog]



# -----------------------------------------------------
#  auto label ECG artifacts
# -----------------------------------------------------
def auto_label_cardiac(raw_chop, ica, name_ecg, flow=8, fhigh=25, tmin=-0.4, tmax=0.4,
                        thresh_ctps=0.20, thresh_corr=None):
    '''
    Identify components with cardiac activity
    '''

    # CTPS
    ic_ctps, scores_ctps, events_ecg = get_ics_cardiac(raw_chop, ica, flow=flow, fhigh=fhigh,
                                                        tmin=tmin, tmax=tmax, name_ecg=name_ecg,
                                                        use_CTPS=True, thresh=thresh_ctps)

    # correlation
    if thresh_corr:
        ic_corr, scores_corr, _ = get_ics_cardiac(raw_chop, ica, flow=flow, fhigh=fhigh,
                                                  tmin=tmin, tmax=tmax, name_ecg=name_ecg,
                                                  use_CTPS=False, thresh=thresh_corr)
    else:
        ic_corr     = np.array([-1])
        scores_corr =  np.zeros(np.shape(scores_ctps)) #scores_corr =  np.array([-1]) ???

    ecg_info = dict(
        events_ecg   = events_ecg,
        ic_ctps      = ic_ctps,
        scores_ctps  = scores_ctps,
        ic_corr      = ic_corr,
        scores_corr  = scores_corr
    )

    return ecg_info


# -----------------------------------------------------
#  determine ocular related ICs
# -----------------------------------------------------
def auto_label_ocular(raw_chop, ica, name_eog_ver, name_eog_hor=None,
                       flow=2.0, fhigh=20, thresh_corr_ver=0.20, thresh_corr_hor=0.20):
    '''
    Find Independent Components related to ocular artefacts
    '''

    # vertical EOG:  correlation
    ic_ver, score_ver, events_ver = get_ics_ocular(raw_chop, ica, flow=flow, fhigh=fhigh,
                                                   thresh=thresh_corr_ver, score_func='pearsonr',
                                                   name_eog=name_eog_ver)

    # horizontal EOG:  correlation
    ic_hor, score_hor, events_hor = get_ics_ocular(raw_chop, ica, flow=flow, fhigh=None,
                                                   thresh=thresh_corr_hor, score_func='pearsonr',
                                                   name_eog=name_eog_hor)
    eog_info = dict(
        ic_ver     = ic_ver,
        scores_ver = score_ver,
        events_ver = events_ver,
        ic_hor     = ic_hor,
        scores_hor = score_hor,
        events_hor = events_hor
    )

    return eog_info


# ======================================================
#
#  update annotations
#
# ======================================================
def update_annotations(raw, description="TEST", onsets=None, duration=None, verbose=False):
        '''
        update annotations in raw

        Parameters
        ----------
        raw         : raw obj
        description : string, description/label for event in anotation <TEST>
        onsets      : np.array of ints,  onsets in samples <None>
        duration    : length in samples

        Returns
        -------
        raw with new annotation
        '''

        try:
            raw_annot = raw.annotations
            orig_time = raw_annot.orig_time
        except:
            raw_annot = None
            orig_time = None

        if not duration:
            duration = np.ones(onsets.shape[0]) / raw.info["sfreq"]

        annot = mne.Annotations(onset=onsets.tolist(),
                                duration=duration.tolist(),
                                description=description,
                                orig_time=orig_time)

        # logger.info("description  : {}\n".format(description)+
        #            "  -> onsets  : {}\n".format(onsets)+
        #            "  -> duration: {}".format(duration)+
        #            " annot:\n {}".format(annot)
        #           )

        msg = ["Update Annotations with description: <{}>".format(description)]

        if raw_annot:
            # -- clear old annotations
            kidx = np.where(raw_annot.description == description)[0]  # get index
            if kidx.any():
                msg.append("  -> delete existing annotation <{}> counts: {}".format(description, kidx.shape[0]))
                raw_annot.delete(kidx)
            raw_annot += annot  # pointer to raw.anotations; add new annot
        else:
            raw.set_annotations(annot)

        if verbose:
            idx = np.where(raw.annotations.description == description)[0]

            msg.extend([
                " --> mne.annotations in RAW:\n  -> {}".format(raw.annotations),
                "-" * 40,
                "  -> <{}> onsets:\n{}".format(description, raw.annotations.onset[idx]),
                "-" * 40])

            logger.info("\n".join(msg))
        return raw


# ======================================================
#
# find closest element in ndarray
#
# ======================================================
def find_nearest(Xarr, value):
    import numpy as np
    X = np.array(Xarr)
    index = np.argmin(np.abs(X - value))
    return X[np.unravel_index(index, X.shape)], index


# --------------------------------------------------------------
#  compare two arrays and keep the largest value in two ndarrays
# --------------------------------------------------------------
def get_largest(arr1, arr2, abs=True):
    import numpy as np
    if abs:
        sc1 = np.abs(arr1)
        sc2 = np.abs(arr2)
    else:
        sc1 = arr1
        sc2 = arr2
    diff = sc1 - sc2

    # copy all elements from first array
    arr_max = sc1.copy()

    # overwrite elements where values in arr2 are larger than arr1
    ix_min = np.where(diff < 0)[0]
    arr_max[ix_min] = sc2[ix_min]

    return arr_max


# ======================================================
#
# transform ICA sources to (MEG) data space
# Note: this routine makes use of the ICA object as defied by MNE-Python
#
# ======================================================
def transform_mne_ica2data(sources, ica, idx_zero=None, idx_keep=None):

    """
    performs back-transformation from ICA to Data space using
    rescaling as used as in MNE-Python
        sources: shape [n_chan, n_samples]
        ica: ICA object from MNE-Python
        idx_zero: list of components to remove (optional)
        idx_keep: list of components to remove (optional)
    return: data re-computed from ICA sources
    """

    import numpy as np
    from scipy.linalg import pinv

    n_features = len(ica.pca_components_)
    n_comp, n_samples = sources.shape
    A = ica.mixing_matrix_.copy()

    # create data with full dimension
    data = np.zeros((n_samples, n_features))

    # if idx_keep is set it will overwrite idx_zero
    if idx_keep is not None:
        idx_all = np.arange(n_comp)
        idx_zero = np.setdiff1d(idx_all, idx_keep)

    # if idx_zero or idx_keep was set idx_zero is always defined
    if idx_zero is not None:
        A[:, idx_zero] = 0.0

    # back transformation to PCA space
    data[:, :n_comp] = np.dot(sources.T, A.T)  # get PCA data

    # back transformation to Data space
    # is compatible to MNE-Python, but not to scikit-learn or JuMEG
    data = (np.dot(data, ica.pca_components_) + ica.pca_mean_).T  # [n_chan, n_samples]

    # restore scaling
    if ica.noise_cov is None:  # revert standardization
        data *= ica.pre_whitener_
    else:
        data = np.dot(pinv(ica.pre_whitener_, cond=1e-14), data)

    return data

# ======================================================
# add_aux_channels
# ======================================================
def add_aux_channels(raws, data_aux, aux_labels, aux_types):
    """
    add aux channels with aux-data to raw objs

    Parameters
    ----------
    raws       : raw obj or list of raws [raw_chop,raw_chop_clean]
    data_aux   : np.array aux data
    aux_labels : channel labels; list , e.g.: ECG, EOGver, EOGhor
    aux_types  : channel types; list,   e.g.: ecg, eog

    Returns
    -------
    list of updated raws
    """
    if not isinstance(raws,(list)):
       raws = [raws]

    for raw in raws:
        aux_info              = mne.create_info(aux_labels, raw.info['sfreq'], aux_types)
        aux_info['meas_date'] = raw.info['meas_date']  # for annotations.orig_time
        aux_ch_raw            = mne.io.RawArray(data_aux, aux_info)
        raw.add_channels([aux_ch_raw], force_update_info=True)
    return raws

# ======================================================
# transform_ica2data
# ======================================================
def transform_ica2data(data_ica, ica):
    """
    transform ica data to raw-obj => recalculate raw-chop & raw-chop-clean
    add aux data to raws

    Parameters
    ----------
    data_ica   : ica data
    ica        : ICA obj

    Returns
    -------
    list of reconstructed raw chops
    [raw, raw_clean]
    """
    # reconstruct MEG data and create raw object
    data_meg = transform_mne_ica2data(data_ica,ica)
    raw      = mne.io.RawArray(data_meg,ica.info)

    # reconstruct MEG data, clean artifacts and create raw clean object
    data_meg = transform_mne_ica2data(data_ica,ica,idx_zero=ica.exclude,idx_keep=None)
    raw_clean = mne.io.RawArray(data_meg,ica.info)

    return raw, raw_clean

# --------------------------------------------------------------
#  get IC label and score info
# --------------------------------------------------------------
def get_ic_info(ics, labels, score_ecg_ctps, score_ecg_corr, score_eog_ver, score_eog_hor):

    import numpy as np

    ics_info = []
    ics_label = []

    # artifact ICs
    for ic in ics:
        sc_ecg1 = np.abs(score_ecg_ctps[ic])
        sc_ecg2 = np.abs(score_ecg_corr[ic])
        #logger.info("eog1: {}".format( score_eog_ver[ic]))

        sc_eog1 = np.abs(score_eog_ver[ic])
        sc_eog2 = np.abs(score_eog_hor[ic])
        info = 'IC#%2d: %s: scores (%.3f, %.3f); (%.3f, %.3f)' % \
               (ic, labels[ic], sc_ecg1, sc_ecg2, sc_eog1, sc_eog2)
        ics_info.append(info)
        ics_label.append(labels[ic])

    return [ics_label, ics_info]


# --------------------------------------------------------------
#  collect ICs which are just below the threshold
# --------------------------------------------------------------
def get_ics_below_threshold(ics_ar, labels, score_ecg_ctps, score_ecg_corr, score_eog_ver, score_eog_hor, n_ics=4):

    '''
    :param ics_ar:  artifact components
    :param labels:  IC labels from all components
    :param score_ecg_ctps:
    :param score_ecg_corr:
    :param score_eog_ver:
    :param score_eog_hor:
    :return:
    '''

    n_comp = len(labels)

    # get non-artifact ICs (i.e., below threshold)
    if n_ics < 1:
        n_ics = 1
    ics_ignored = list(set.difference(set(range(n_comp)), set(ics_ar)))

    # get strongest scores across all scores (below threshold)
    score_ecg = get_largest(score_ecg_ctps[ics_ignored], score_ecg_corr[ics_ignored], abs=True)
    score_eog = get_largest(score_eog_ver[ics_ignored], score_eog_hor[ics_ignored], abs=True)
    score_max = get_largest(score_ecg, score_eog, abs=True)

    # sort to get strongest scores first
    ix_sort = score_max.argsort()[::-1]
    ics_ignored = np.array(ics_ignored)[ix_sort[:n_ics]]

    # get score info
    label_ignored, info_ignored = get_ic_info(ics_ignored, labels,
                                              score_ecg_ctps, score_ecg_corr, score_eog_ver, score_eog_hor)

    return [ics_ignored, label_ignored, info_ignored]


# --------------------------------------------------------------
#  collect info about sources (artifact and ignored ICs)
# --------------------------------------------------------------
def collect_source_info(sources, exclude, pick, n_below_thresh=4, n_kurtosis=2):
    from scipy.stats import kurtosis

    labels = sources.labels[pick]
    data_ica = sources.data_ica[pick]
    score_ecg_ctps = sources.score.ecg_ctps[pick]
    score_ecg_corr = sources.score.ecg_corr[pick]
    score_eog_ver  = sources.score.eog_ver[pick]
    score_eog_hor  = sources.score.eog_hor[pick]
    n_comp = data_ica.shape[0]
    ic_ecg = exclude.ecg[pick]
    ic_eog = exclude.eog[pick]
    n_ecg = len(ic_ecg)
    n_eog = len(ic_eog)

    # init data
    ics_select     = []
    info_select    = []
    ics_label      = []
    sources_select = []

    # ECG components
    if n_ecg > 0:
        ics_label, info_select = get_ic_info(ic_ecg, labels, score_ecg_ctps, score_ecg_corr,
                                                         score_eog_ver, score_eog_hor)
        ics_select = ic_ecg
        sources_select = data_ica[ic_ecg]

    # EOG components
    if n_eog > 0:
        label_eog, info_eog = get_ic_info(ic_eog, labels, score_ecg_ctps, score_ecg_corr,
                                          score_eog_ver, score_eog_hor)
        if n_ecg > 0:
            ics_select     = np.concatenate([ics_select, ic_eog])
            ics_label      = np.concatenate([ics_label, label_eog])
            info_select    = np.concatenate([info_select, info_eog])
            sources_select = np.concatenate([sources_select, data_ica[ic_eog]])
        else:
            ics_select     = ic_eog
            ics_label      = label_eog
            info_select    = info_eog
            sources_select = data_ica[ic_eog]

    # ICs below threshold
    if n_below_thresh > 0:
        ics_ignored, label_ignored, info_ignored = get_ics_below_threshold(ics_select, labels,
                                                                           score_ecg_ctps, score_ecg_corr,
                                                                           score_eog_ver, score_eog_hor,
                                                                           n_ics = n_below_thresh)
        ics_select = np.concatenate([ics_select, ics_ignored])
        ics_label = np.concatenate([ics_label, label_ignored])
        info_select = np.concatenate([info_select, info_ignored])
        if (n_ecg+n_eog) > 0:
            sources_select = np.concatenate([sources_select, data_ica[ics_ignored]])
        else:
            sources_select = data_ica[ics_ignored]

    # ICs with large kurtosis values
    if n_kurtosis > 0:
        ic_other = list(set.difference(set(range(n_comp)), set(ics_select)))
        kurt = kurtosis(data_ica[ic_other], axis=1)
        ix_sort = kurt.argsort()[::-1]
        ic_other = np.array(ic_other)[ix_sort[0:n_kurtosis]]
        kurt = kurt[ix_sort[0:n_kurtosis]]

        label_kurt, info_kurt = get_ic_info(ic_other, labels, score_ecg_ctps, score_ecg_corr,
                                            score_eog_ver, score_eog_hor)
        for icomp in range(n_kurtosis):
            info_kurt[icomp] = info_kurt[icomp] + '; kurtosis %.3f' % kurt[icomp]

        sources_select = np.concatenate([sources_select, data_ica[ic_other]])
        ics_label = np.concatenate([ics_label, label_kurt])
        info_select = np.concatenate([info_select, info_kurt])
        ics_select = np.array(np.concatenate([ics_select, ic_other]), dtype=int)

    return [sources_select, ics_label, info_select, ics_select]


# --------------------------------------------------------------
#  grab image array from figure
# --------------------------------------------------------------
def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
