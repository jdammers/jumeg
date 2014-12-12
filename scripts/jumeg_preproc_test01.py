#!/usr/bin/env python

"""
    meg94t mne preproctest
    FB 04.12.2014
    last updae FB 04.12.2014
"""
import sys, getopt, os, os.path
import numpy as np
import mne   as mne

#--- jumegs functions
import jumeg.jumeg_preproc_data as jppd
import jumeg.jumeg_math         as jumeg_math

#--- jumeg clases
from jumeg.jumeg_base              import jumeg_base
from jumeg.jumeg_epocher           import jumeg_epocher

from jumeg.template.jumeg_template import experiment as jtmp_exp

def usage():
    usage=""" 
     usage: 
     -e --exp --experiment <experiment name> 
     -e name  or --exp=name  or --experiment=name
     experiment name for json template:
     
     -c --condi --condition <c1,c2,c3, ..cn>
     list of conditions to process
     
     -s --stage  
      MNE stage path, additional start path for files from list
      e.g. start path to mne fif file directory structure
      
     -p --pfif
      path to single fif file
     
     -f --fif --ffif
      fif file name
         
     --plist --path_list=
      path to text file with fif files
     --flist
      text file with fif files
      
     ==> Flags:
     -r --run : will execute this program
     -v       : verbose
     -d       : debug test mode
     -h       : show this


    jumeg_preproc_test01.py --exp=TEST01 -s /data/exp/TEST01/mne --plist= /data/exp/TEST01/doc --flist=test01.txt -r -v


    """
    print usage
    sys.exit()

#=========================================================================================
def main(argv):
#-- default argv
    experiment_name = None
    path_list       = os.getcwd()
    fname_list      = None 
    fif_name        = None
    pfif            = os.getcwd()
    path_mne_stage  = None
    verbose, do_run, debug_mode = False, False,False
    condition_list  = None 
    epocher_hdf_fname = None
    
#--- check argv
    try:
        opts, args = getopt.getopt(sys.argv[1:],"e:s:p:f:c,rvhd",
                             ["exp=","experiment=","condi=","condition=","stage=","pfif=","fif=","ffif=",
                              "plist=","path_list=","flist=","help","run","verbose"])
    
    except getopt.GetoptError, err:
           print "Error wrong arguments:\n"
           print str(err) 
           usage()
#--- check argv           
    for o, a in opts:
        if o in ("-e","--exp","--experiment"):
             experiment_name = a
        elif o in ("-c", "--condi", "--condition"):
             condition_list = a.split(',')
        elif o in ("-s", "--stage"):
             path_mne_stage = a
        elif o in ("-p", "--pfif"):
             pfif = a
        elif o in ("-f", "--fif", "--ffif"):
             fif_name = a
        elif o in ("--plist", "--path_list"):
             path_list = a
        elif o in ("--flist"):
             fname_list = a
        elif o in ("-v", "--verbose"):
             verbose=True
        elif o in ("-r", "--run"):
             do_run=True
        elif o in ("-d"):
             debug_mode=True
        elif o in ("-h", "--help"):
             usage()

#--- set debug & test mode
    if debug_mode  :
       experiment_name = 'TEST01'
       path_mne_stage  = '/localdata/frank/data/Chrono/mne'
       #path_list       = "/localdata/frank/data/Chrono/doc"
       #fname_list      = 'chrono_normal_inkomp.txt'
            
       fif_name        = '201195_test.fif'  #None
       pfif            = '201195/Chrono01/110516_1413/1'  #None
       verbose         = True
       do_run          = True     
       #condition_list  = ('LRst','LRrt')
#---   
    if verbose :
           print"\n---> ARGV parameter:"
           print"experiment : " + str(experiment_name)
           print"condition  : " + str(condition_list)
           print"stage      : " + str(path_mne_stage)
           print"pfif       : " + str(pfif)
           print"fif name   : " + str(fif_name)
           print"path list  : " + str(path_list)
           print"fname_list : " + str(fname_list)
           print"verbose    : " + str(verbose)
           print"run        : " + str(do_run)
           print"debug mode : " + str(debug_mode)
           print"\n\n"  
#---
    if not do_run:
       print "===> Done jumeg preprocessing , set run flag for real data processing\n"
       usage()
 
#--- update base  
    jumeg_base.verbose = verbose
 
#--- init experiment template parameter
    jtmp_exp.template_name = experiment_name
    jtmp_exp.verbose       = verbose
    #- read template parameters into dict
    tmp = jtmp_exp.update_template_file()
    #--- make obj from dict 
    TMP      = jtmp_exp.get_as_obj()
    path_exp = TMP.experiment.path.experiment
       
    if path_mne_stage is None:
       path_mne_stage = TMP.experiment.path.mne 
       
    #--- get existing files from list
    fn_raw_list=[]
    if fname_list:
       fn_raw_list = jumeg_base.get_filename_list_from_file(path_list + "/" + fname_list,start_path = path_mne_stage)
    #--- check & add fif file to list
    if fif_name:
       if pfif :
          f = pfif +"/"+ fif_name
       else:
          f = fif_name
       if os.path.isfile(f):
          fn_raw_list.append(f)
       elif os.path.isfile(path_mne_stage + '/' + f):
          fn_raw_list.append(path_mne_stage + '/' + f)
            
    #--- obj short-cut 
    tmp_pp_raw = TMP.experiment.data_preprocessing.raw
    tmp_pp_brs = TMP.experiment.data_preprocessing.brainresponse
          
    #--- loop preproc for each fif file
    for fif_file in (fn_raw_list) :
        raw = None  
        
    #--- epocher search for events save to HDF     
        if tmp_pp_raw.epocher.do_run :
           tmp_pp_raw.epocher.verbose = verbose
           print"\n===> PP Info: start apply epocher => event code search" 
           print"File  :" + fif_file
           if verbose:
              print"Parameter:"
              print tmp_pp_raw.epocher
              print"\n\n"
           (fname,raw,epocher_hdf_fname) = jumeg_epocher.apply_epochs_to_hdf(fif_file,raw=raw,condition_list=condition_list, **tmp_pp_raw.epocher)
        
    #--- noise_covariance
    #--- will search and find empty room file if fif is no empty room file 
        if tmp_pp_raw.noise_covariance.do_run :
           tmp_pp_raw.noise_covariance.verbose = verbose
           print"\n===> PP Info: start apply create noise_covariance" 
           print"File  :" + fif_file
           if verbose:
              print"Parameter:"
              print tmp_pp_raw.noise_covariance
              print"\n\n"
           fname_noise_covariance = jppd.apply_create_noise_covariance(fif_file,raw=raw,**tmp_pp_raw.noise_covariance)
           print"\n\n==> PP Info: done apply create noise_covariance :\n  ---> " + fname_noise_covariance
       
    #--- filter raw data   
        if tmp_pp_raw.filter.do_run :
           tmp_pp_raw.filter.verbose = verbose
           print"\n===> PP Info: start apply filter raw:"
           print"File  : " + fif_file
           if verbose:
              print"Parameter :"
              print tmp_pp_raw.filter
              print"\n\n"
           (fname, raw) = jppd.apply_filter_data(fif_file,raw=raw,**tmp_pp_raw.filter)
           print"\n\n==> PP Info: done apply filter raw\n  ---> " + fname
        else:
             fname = jumeg_base.get_fif_name(fif_file,postfix=tmp_pp_raw.filter.fif_postfix,extention=tmp_pp_raw.filter.fif_extention) 
                          
    #--- average raw filtered data
        if tmp_pp_raw.average.do_run :
           tmp_pp_raw.average.verbose = verbose
           print"\n===> PP Info: start apply averager raw" 
           print"File  :" + fname
           if verbose:
              print"Parameter :"
              print tmp_pp_raw.average
              print"\n\n"
           jppd.apply_averager(fn_raw_list,**tmp_pp_raw.averager)    
           print"\n\n==> PP Info: done apply averager filterd raw data\n"
           
    #--- ocarta
        if tmp_pp_raw.ocarta.do_run :
           tmp_pp_raw.ocarta.verbose = verbose   
           print"\n===> PP Info: start apply ocarta offline"
           print"File  :" + fname
           if verbose :
              print"Parameter :" 
              print tmp_pp_raw.ocarta
              print"\n\n"
           (fname_oca,raw,ecg_events,eog_events) = jppd.apply_ocarta_offline_data(fname,raw=raw,**tmp_pp_raw.ocarta) 
           jumeg_epocher.write_events_to_json(fname_oca,ecg_events=ecg_events,eog_events=eog_events)  
           print"\n\n==> PP Info: done apply ocarta\n  ---> " + fname_oca
        else:
             fname_oca = jumeg_base.get_fif_name(fname,postfix=tmp_pp_raw.ocarta.offline_parameter.fif_postfix,
                                                 extention=tmp_pp_raw.ocarta.offline_parameter.fif_extention) 
        #     (ecg_events, eog_events) = jppd.find_ecg_eog_events(fname_oca)
        
        #jumeg_epocher.write_events_to_json(fname_oca,ecg_events=ecg_events,eog_events=eog_events)  
        
        #jumeg_epocher.apply_epocher(fname_oca,raw=raw, template_name=tmp_pp_brs.epocher.template_name)
        #jumeg_epocher.apply_epocher(fname_oca,raw=raw, **tmp_pp_brs.epocher)
   
    #--- brain response apply ica from mne  
        if tmp_pp_brs.ica.do_run :
           tmp_pp_brs.ica.verbose = verbose
           print "\n===> PP Info: start apply brain-response ica"
           print"File   : " + fname_oca
           if verbose: 
              print"Parameter :" 
              print tmp_pp_brs.ica
              print"\n\n"
           (fname_oca_ica,ica_obj) = jppd.apply_ica_data(fname_oca,raw=raw,**tmp_pp_brs.ica)
           print"\n\n==> PP Info: done apply ica for brain responses\n  ---> " + fname_oca_ica
        else:
             fname_oca_ica = jumeg_base.get_fif_name(fname_oca,postfix=tmp_pp_brs.ica.fif_postfix,extention=tmp_pp_brs.ica.fif_extention) 
       
    #--- brain response ctps 
    #--- run for all condition combine and or exclude CTPs-ICs for different conditions
        if tmp_pp_brs.ctps.do_run :
           tmp_pp_brs.ctps.verbose = verbose
       #--- init freq array     
           #fmax  = tmp_pp_brs.ctps.fmax
           #fmin  = tmp_pp_brs.ctps.fmin
           #fstep = tmp_pp_brs.ctps.fstep
           # freq_ctps = np.array([np.arange(f0, fmax+1  ,fstep/2),np.arange(f0+ fstep, fmax+ fstep+f0 ,fstep/2) ]).T
           
           # freq_ctps = jumeg_math.calc_frequency_windos(tmp_pp_brs.ctps.fmin, tmp_pp_brs.ctps.fmax, tmp_pp_brs.ctps.fstep)
           
           print"\n===> PP Info: start brainresponse ica"
           print"File  : "  + fname_oca_ica
           if verbose :
              print"Parameters:" 
              print tmp_pp_brs.ctps
              print"\n\n"
           fname_oca_ica_brain_event = jppd.apply_ica_select_brain_response_data(fname_oca_ica,raw=raw,**tmp_pp_brs.ctps)
           print"\n\n==> PP Info: done apply ctp for brain responses & cleaned \n  ---> " +fname_oca_ica_brain_event
         
     #--- epocher avg  & time frequency epochs
     #jumeg_epocher.apply_epocher(fname_oca,raw=raw)
     
     
if __name__ == "__main__":
   main(sys.argv)
