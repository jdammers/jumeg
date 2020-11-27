#!/usr/bin/env python
#################  FROM FRANK ###################################
import os,sys
from dcnn_base      import DCNN_CONFIG
from dcnn_main      import DCNN
from dcnn_logger    import setup_script_logging  #FB
from dcnn_utils     import file_looper,get_args,dict2str,expandvars #FB

__version__= "2020-07-06-001"

####################################################

#==============================================
# USER SETTINGS 4 Magic
# ==============================================
basedir  =  None
data_meg = 'meg_rawdata/Philly'
fnconfig = 'config_CTF_Philly.yaml'
pattern  = '-raw.fif'
verbose  = True

do_label_ica         = False
do_label_check       = True
do_performance_check = False
# --
do_log2file    = True


def run(fnconfig=None,basedir=None,data_meg=None,data_train=None,pattern='-raw.fif',
        verbose=False,do_label_ica=False,do_label_check=False,log2file=False):
    
    # ToDo run for list of files or search in subdirs
    
    # -- init config CLS
    cfg = DCNN_CONFIG(verbose=verbose)
    cfg.load(fname=fnconfig)

    #-- init dcnn CLS
    dcnn = DCNN(**cfg.config)  # init object with config details

    #---
    if basedir: # FB test
        dcnn.path.basedir = basedir
    if data_meg:
        dcnn.path.data_meg = data_meg  # input directory
    if data_train:
        dcnn.path.data_train = data_train  # input directory

    dcnn.verbose = True
    dcnn.get_info()

   # ==========================================================
   # run ICA auto labelling
   # ==========================================================
    if do_label_ica:
        #path_in = os.path.join(cfg.config['path']['basedir'],cfg.config['path']['data_meg'])
       path_in = dcnn.path.data_meg

      # -- looper catch error via try/exception setup log2file
       for fnraw in file_looper(rootdir=path_in,pattern=pattern,version=__version__,verbose=verbose,logoverwrite=True,log2file=log2file):
           #logger.info(fnraw)
           # -- read raw data and apply noise reduction
           dcnn.meg.update(fname=fnraw)
            
           # -- apply ICA on chops, label ICs save results to disk
           # ToDo store chop-times, ECG,EOG in raw.annotations
           #   - chop:  make use of annotations in get_chop_times_indices()
           fgdcnn = dcnn.label_ica(save=True)
           
           if verbose:
              dcnn.get_info()
            
    # ==========================================================
    # check ICA auto labelling
    # ==========================================================
    npz_pattern = pattern.split(".",-1)[0] +"-gdcnn.npz"

    if do_label_check:
       path_in = dcnn.path.data_train

       from mne.report import Report
       report = Report(title='Check IC labels')

       for fname in file_looper(rootdir=path_in, pattern=npz_pattern,version=__version__,verbose=verbose,log2file=log2file,logoverwrite=False):
       
           dcnn.load_gdcnn(fname)
        # check IC labels (and apply corrections)
           #dcnn.check_labels(save=True)

           fnreport ="test_report"

        # check IC labels (and apply corrections)
           name = os.path.basename(fname[:-4])
           print ('>>> working on %s' % name)
           figs, captions = dcnn.plot_ica_traces(fname)
           report.add_figs_to_section(figs, captions=captions, section=name, replace=True)
       report.save(fnreport + '.h5', overwrite=True)
       report.save(fnreport + '.html', overwrite=True, open_browser=True)




       #dcnn.check_labels(save=True, path_out=cfg.config['path']['data_train'])



#if verbose: # ToDo set verbose level  True,2,3
           #   logger.debug("dcnn ica chop  dump.\n{}\n{}\n\n".format( dcnn.ica.chop, dict2str(dcnn.ica.chop.dump())  ))
           #   logger.debug("dcnn ica n_chop.\n{}\n\n".format(dcnn.ica.chop.n_chop))
           #   logger.debug("dcnn ica topo data.\n{}\n\n".format(dcnn.ica.topo.data))
           #   logger.debug("dcnn ica topo img.\n{}\n\n".format(dcnn.ica.topo.images))
    
    # ==========================================================
    # ICA performance plot
    # ==========================================================
    if do_performance_check:
       path_in = cfg.config['path']['data_train']
       
       for fname in file_looper(rootdir=path_in, pattern=npz_pattern,version=__version__,verbose=verbose,log2file=log2file,logoverwrite=False):
       
           dcnn.load_gdcnn(fname)
           # ToDo plot performance, save to report
           

if __name__ == "__main__":
    # --  get parameter / flags from cmd line
    argv = sys.argv
    opt, parser = get_args(argv,version=__version__)
    if len(argv) < 2:
       parser.print_help()
       sys.exit(-1)
    
    #flog= "dcnn_"+os.getenv("USER","test")+".log"
    logger = setup_script_logging(name="DCNN",opt=opt,logfile=False,version=__version__,level="DEBUG")
   
    try:
       if opt.jd: # JD settings
          opt.config  = fnconfig
          opt.pattern = pattern
          opt.verbose = verbose
          opt.do_label_ica   = do_label_ica
          opt.do_label_check = do_label_check
          opt.log2file       = do_log2file
       elif opt.fb:  # call from shell
          #opt.basedir    = "$JUMEG_PATH_LOCAL_DATA"+"/gDCNN"
          #opt.data_meg   = "data_examples"
          #opt.data_train = "$JUMEG_PATH_LOCAL_DATA"+"/exp/dcnn/ica_labeled/Juelich"
          #-- 4D
          opt.pattern = "*int-raw.fif" #"*.c,rfDC_bcc,nr-raw.fif"
          opt.config  = "config_4D_Juelich.yaml"
          # ToDo trouble with
          # 205399_MEG94T_121220_1322_2_c,rfDC_EC_bcc-raw-gdcnn.npz
          # eog / ICs found ???

          opt.ica      = do_label_ica
          opt.check    = do_label_check
          opt.log2file = do_log2file
         
       if opt.verbose:
           # show cmd line parameter
           # https://stackoverflow.com/questions/39978186/python-print-all-argparse-arguments-including-defaults/39978305
           msg=["DCNN input parameter:"]
           for k,v in sorted(vars(opt).items()): msg.append("  -> {0:12}: {1}".format(k,v))
           logger.info("\n".join(msg) )

       run(fnconfig=opt.config,basedir=opt.basedir,data_meg=opt.data_meg,data_train=opt.data_train,pattern=opt.pattern,
           verbose=opt.verbose,do_label_ica=opt.ica,do_label_check=opt.check,log2file=opt.log2file)
 
    except:
       logger.exception("ERROR in DCNNN")
 