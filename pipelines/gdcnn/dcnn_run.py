#!/usr/bin/env python
from utils import find_files,get_args,dict2str

#################   FROM FRANK ###################################
import os,sys
from dcnn_base   import DCNN_CONFIG
from dcnn_main   import DCNN
from dcnn_logger import setup_script_logging,init_logfile
__version__= "2020-07-06-001"

####################################################

#==============================================
# USER SETTINGS 4 Magic
# ==============================================
basedir  =  None
data_meg = 'meg_rawdata/Philly'
fnconfig = 'config_CTF_Philly.yaml'
pattern  ='*-raw.fif'
verbose  = True

do_label_ica   = False
do_label_check = True
do_log2file    = False


def run(fnconfig=None,basedir=None,data_meg=None,pattern='-raw.fif',
        verbose=False,do_label_ica=False,do_label_check=False,do_log2file=False):
    
    # ToDo run for list of files or search under dir
    
    # ToDo check if this make sense?
    #  if __DEFAULTS__: # if defaults define as global see on top of the script
    #     cfg = DCNN_CONFIG(defaults=__DEFAULTS__)
    
    
    cfg = DCNN_CONFIG(verbose=verbose)
    cfg.load(fname=fnconfig)
    
    dcnn = DCNN(**cfg.config)  # init object with config details
    dcnn.verbose = True
    
    if basedir: # FB test
       dcnn.path.basedir = basedir
    if data_meg:
       dcnn.path.data_meg = data_meg  # input directory
    
    # ==========================================================
    # run ICA auto labelling
    # ==========================================================
    if do_label_ica:
        # get file list to process
        path_in = os.path.join( dcnn.path.basedir,dcnn.path.data_meg)
        
        fnames = find_files(path_in, pattern=pattern)
        if not fnames:
            logger.exception("ERROR No files found:\n  -> path: {}\n  -> pattern: {}".format(path_in,pattern))
            import sys
            sys.exit()
    
        fnraw = fnames[0]  # use first file for testing
    
       # -- init logfile
        if do_log2file:
           flog = os.path.splitext( fnraw )[0]
           Hlog = init_logfile(logger=logger,fname=flog +".log",overwrite=True,name=flog)
           msg= ["DCNN : {}".format(__version__),
                 "  -> raw file: {}".format(fnraw),
                 "  -> logfile : {}".format(Hlog.filename)]
           logger.info("\n".join(msg))
           if verbose:
              cfg.info()
           
        # -- read raw data and apply noise reduction
        dcnn.meg.update(fname=fnraw)
        
        # -- apply ICA on chops, label ICs save results to disk
        # TODO: store chop-times, ECG,EOG in raw.annotations
        #   - chop:  make use of annotations in get_chop_times_indices()
        fgdcnn = dcnn.label_ica(save=True)
        
        if verbose:
           dcnn.get_info()
        
    # ==========================================================
    # check ICA auto labelling
    # ==========================================================
    if do_label_check:
        path_in = dcnn.path.data_train
        fnames  = find_files(path_in, pattern= '*.npz')
        fname   = fnames[0]  # use first file for testing
    
        dcnn.load_gdcnn(fname)
        
        if verbose:
           dcnn.get_info()
        
           logger.info("dcnn ica chop  dump.\n{}\n{}\n\n".format( dcnn.ica.chop, dict2str(dcnn.ica.chop.dump())  ))
           logger.info("dcnn ica n_chop.\n{}\n\n".format(dcnn.ica.chop.n_chop))


if __name__ == "__main__":
    # --  get parameter / flags from cmd line
    argv = sys.argv
    opt, parser = get_args(argv,version=__version__)
    if len(argv) < 2:
       parser.print_help()
       sys.exit(-1)

    logger = setup_script_logging(name="DCNN",opt=opt,logfile=True,version=__version__,level="DEBUG")
   
    if opt.verbose:
       # show cmd line parameter
       # https://stackoverflow.com/questions/39978186/python-print-all-argparse-arguments-including-defaults/39978305
       msg=["DCNN input parameter:"]
       for k,v in sorted(vars(opt).items()): msg.append("  -> {0:12}: {1}".format(k,v))
       logger.info("\n".join(msg) )
    try:
       if opt.fb: # FB call from shell
          run(fnconfig=opt.config,basedir=opt.basedir,data_meg=opt.data_meg,pattern=opt.pattern,
              verbose=opt.verbose,do_label_ica=opt.ica,do_label_check=opt.check,do_log2file=opt.log2file )
       elif opt.jd: # JD settings
          run(fnconfig=fnconfig,pattern=pattern,
              verbose=verbose,do_label_ica=do_label_ica,do_label_check=do_label_check,do_log2file=do_log2file)
       else:
          run(fnconfig=fnconfig,pattern=pattern,
              verbose=verbose,do_label_ica=do_label_ica,do_label_check=do_label_check,do_log2file=do_log2file)

    except:
       logger.exception("ERROR in DCNNN")
 