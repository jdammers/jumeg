#!/usr/bin/env python

"""
   JuMEG / MNE-python  prepocessing MEEG data
   author: Frank Boers   04.12.2014
---> update FB 22.07.2015
     adding try/exception block to check if preproc-step or template parameter not defined
     
   jumeg_preprocessing_4raw_data.py -fif 109077_Chrono01_110518_1415_1_c,rfDC-raw.fif -pfif /localdata/frank/data/Chrono/mne/109077/Chrono01/110518/1415/1 -exp InKomp -v -r 
"""
import sys,os, os.path

#--- jumegs functions
import jumeg.jumeg_4raw_data_preproc_utils  as jpp_utils
import jumeg.jumeg_4raw_data_preproc        as jppd

#--- jumeg clases
from jumeg.jumeg_base              import jumeg_base
from jumeg.template.jumeg_template import experiment as jtmp_exp



#=========================================================================================
def main(argv):

    epocher_hdf_fname = None
    
 #--- get args from parser   
    opt,parser = jpp_utils.get_args()
  
#--- set debug & test mode
    if opt.debug :
       opt.experiment = 'TEST01'
       opt.stage      = '/localdata/frank/data/Chrono/mne'
       #path_list       = "/localdata/frank/data/Chrono/doc"
       #fname_list      = 'chrono_normal_inkomp.txt'
            
       opt.fifname    = '201195_Chrono01_110516_1413_1_c,rfDC-raw.fif' #'201195_test.fif'  #None
       opt.pathfif    = '201195/Chrono01/110516_1413/1'  #None
       opt.verbose    = True
       opt.run        = True     
       #condition_list  = ('LRst','LRrt')
#---   
    if opt.verbose :
           print"\n---> ARGV parameter:"
           print"experiment  : " + str(opt.experiment)
           print"condition   : " + str(opt.conditions)
           print"stage       : " + str(opt.stage)
           print"path to fif : " + str(opt.pathfif)
           print"fif name    : " + str(opt.fifname)
           print"path to list: " + str(opt.pathlist)
           print"fname list  : " + str(opt.fnamelist)
           print"verbose     : " + str(opt.verbose)
           print"run         : " + str(opt.run)
           print"debug mode  : " + str(opt.debug)
           print"\n\n"  
       
#--- update base  
    jumeg_base.verbose = opt.verbose
 
#--- init experiment template parameter
    jtmp_exp.template_name = opt.experiment
    jtmp_exp.verbose       = opt.verbose

    #- read template parameters into dict
    tmp = jtmp_exp.template_update_file()

    #--- make obj from dict 
    TMP      = jtmp_exp.template_get_as_obj()
    path_exp = TMP.experiment.path.experiment
       
    if opt.stage is None:
       opt.stage = TMP.experiment.path.mne 

#---
    if opt.verbose :
           print"\n---> Experiment Template parameter:"
           print" --> name         : "+ jtmp_exp.template_name
           print" --> template file: "+ jtmp_exp.template_full_file_name
           print"\n"
#---
    if not opt.run:
       print "===> Done jumeg preprocessing , set run flag for real data processing\n"
       print parser.print_help()
       print"\n"
       exit() 


    #--- get existing files from list  e.g.
    # 005/MEG94T/121219_1311/1/005_MEG94T_121219_1311_1_c,rfDC-raw.fif -nc=A1
    # 007/MEG94T/121217_1239/1/007_MEG94T_121217_1239_1_c,rfDC-raw.fif -nc=A1,A2

    fn_raw_list=[]
    fn_raw_bad_channel_dict=[]

    if opt.fnamelist:
       fn_raw_list,fn_raw_bad_channel_dict = jumeg_base.get_filename_list_from_file(opt.pathlist + "/" + opt.fnamelist,start_path = opt.stage)

   #--- check & add fif file to list update bad channel dict
    if opt.fifname:
       if opt.pathfif :
          f = opt.pathfif +"/"+ opt.fifname
       else:
          f = opt.fifname
       if os.path.isfile(f):
          fn_raw_list.append(f)
          if opt.bads:  #--- bad channels
             fn_raw_bad_channel_dict[f]= opt.bads
       elif os.path.isfile(opt.stage + '/' + f):
          fn_raw_list.append(opt.stage + '/' + f)
          if opt.bads:
             fn_raw_bad_channel_dict[f]= opt.bads


   #--- raw obj short-cut
    tmp_pp_raw = TMP.experiment.data_preprocessing.raw

   #--- brainresponse obj short-cut
    tmp_pp_brs = TMP.experiment.data_preprocessing.brainresponse    

   #--- loop preproc for each fif file
    for fif_file in (fn_raw_list) :
        raw          = None  
        fif_file_raw = fif_file

    #--- check / set bad channels
        if ( fif_file in fn_raw_bad_channel_dict ):
           print "\n ===> BAD Channel -> " + fif_file
           print"  --> BADs: "  + str(fn_raw_bad_channel_dict[fif_file])
           if fn_raw_bad_channel_dict[fif_file]:
              raw,bads_dict = jumeg_base.update_bad_channels(fif_file,raw=raw,bads=fn_raw_bad_channel_dict[fif_file],save=True)


    #--- epocher search for events save to HDF
        if hasattr(tmp_pp_raw.epocher,"do_run"):
           if tmp_pp_raw.epocher.do_run :
              tmp_pp_raw.epocher['verbose'] = opt.verbose
              print"\n===> PP Info: start apply epocher => event code search" 
              print"File : " + fif_file
              if opt.verbose:
                 print"Parameter:"
                 print tmp_pp_raw.epocher
                 print"\n\n"
              (fname,raw,epocher_hdf_fname) = jppd.apply_epocher_events_data(fif_file,raw=raw,condition_list=opt.conditions, **tmp_pp_raw.epocher)
        else:
             jpp_utils.msg_on_attribute_error("raw epocher")         
    
       
    #--- noise_covariance
    #--- will search and find empty room file if fif is no empty room file 
        if hasattr(tmp_pp_raw.noise_covariance,"do_run"):
           if tmp_pp_raw.noise_covariance.do_run :
              tmp_pp_raw.noise_covariance['verbose'] = opt.verbose
              print"\n===> PP Info: start apply create noise_covariance" 
              print"File  :" + fif_file
              if opt.verbose:
                 print"Parameter:"
                 print tmp_pp_raw.noise_covariance
                 print"\n\n"
              fname_noise_covariance = jppd.apply_create_noise_covariance_data(fif_file,raw=raw,**tmp_pp_raw.noise_covariance)
              print"\n\n==> PP Info: done apply create noise_covariance :\n  ---> "
              try:
                  print fname_noise_covariance +"\n"
              except:
                  print " !!! not found !!!\n\n"
        else:
           jpp_utils.msg_on_attribute_error("raw noise covariance")
   

    #--- noise_reducer
    #--- will apply magic ee noise reducer7
        if hasattr(tmp_pp_raw.noise_reducer,"do_run"):
           if tmp_pp_raw.noise_reducer.do_run :
              tmp_pp_raw.noise_reducer['verbose'] = opt.verbose
              print"\n===> PP Info: start apply ee noise_reducer"
              print"File  :" + fif_file
              if opt.verbose:
                 print"Parameter:"
                 print tmp_pp_raw.noise_reducer
                 print"\n\n"

              (fif_file, raw) = jppd.apply_noise_reducer_data(fif_file,raw=raw,**tmp_pp_raw.noise_reducer)
              print"\n\n==> PP Info: done apply noise reducer raw\n  ---> " + fif_file
              print"RAW fname: "+raw.info['filename']
           else:
              fif_file = jumeg_base.get_fif_name(fname=fif_file,postfix=tmp_pp_raw.noise_reducer.fif_postfix,extention=tmp_pp_raw.noise_reducer.fif_extention)
              raw      = None
        else:
           jpp_utils.msg_on_attribute_error("raw noise reducer")
        
    #--- filter raw data   
        if hasattr(tmp_pp_raw.filter,"do_run"):
           if tmp_pp_raw.filter.do_run :
              tmp_pp_raw.filter['verbose'] = opt.verbose
              print"\n===> PP Info: start apply filter raw:"
              print"File  : " + fif_file
              if opt.verbose:
                 print"Parameter :"
                 print tmp_pp_raw.filter
                 print"\n\n"
              (fif_file, raw) = jppd.apply_filter_data(fif_file,raw=raw,**tmp_pp_raw.filter)
              print"\n\n==> PP Info: done apply filter raw\n  ---> " + fname
           else:
              fif_file = jumeg_base.get_fif_name(fname=fif_file,postfix=tmp_pp_raw.filter.fif_postfix,extention=tmp_pp_raw.filter.fif_extention) 
              raw   = None
        else:
           jpp_utils.msg_on_attribute_error("raw filter")

    #--- average raw filtered data
    #    if tmp_pp_raw.average.do_run :
    #       tmp_pp_raw.average.verbose = verbose
    #       print"\n===> PP Info: start apply averager raw"
    #       print"File  :" + fname
    #       if verbose:
    #          print"Parameter :"
    #          print tmp_pp_raw.average
    #          print"\n\n"
    #       jppd.apply_averager(fn_raw_list,**tmp_pp_raw.averager)
    #       print"\n\n==> PP Info: done apply averager filterd raw data\n"
           
    #--- ocarta
        if hasattr(tmp_pp_raw.ocarta,"do_run"):
           if tmp_pp_raw.ocarta.do_run :
              tmp_pp_raw.ocarta['verbose'] = opt.verbose
              print"\n===> PP Info: start apply ocarta fit"
              print"File  :" + fif_file
              if opt.verbose :
                 print"Parameter :" 
                 print tmp_pp_raw.ocarta
                 print"\n\n"

              (fif_file,raw,fhdf) = jppd.apply_ocarta_data(fif_file,raw=raw,**tmp_pp_raw.ocarta)
              print"\n\n==> PP Info: done apply ocarta\n  ---> " + fif_file
           else:
              fif_file = jumeg_base.get_fif_name(fname=fif_file,postfix=tmp_pp_raw.ocarta.fif_postfix,extention=tmp_pp_raw.ocarta.fif_extention)
              raw      = None
        else:
           jpp_utils.msg_on_attribute_error("raw ocarta")
 
    #--- events
        if hasattr(tmp_pp_raw.events,"do_run"):
           if tmp_pp_raw.events.do_run :
              tmp_pp_raw.events['verbose'] = opt.verbose
              print"\n===> PP Info: start apply event export"
              print"File  :" + fif_file
              if opt.verbose :
                 print"Parameter :" 
                 print tmp_pp_raw.events
                 print"\n\n"

              (fif_file,raw,event_ids) = jppd.apply_epocher_export_events_data(fif_file,raw=raw,condition_list=opt.conditions,**tmp_pp_raw.events)
              print"\n\n==> PP Info: done apply events\n  ---> " + fif_file
              print event_ids
              print"\n\n"
        else:
           jpp_utils.msg_on_attribute_error("raw events")
 
           
    #--- brain response apply mne ica: fastica
        if hasattr(tmp_pp_brs.ica,"do_run"):
           if tmp_pp_brs.ica.do_run :
              tmp_pp_brs.ica['verbose'] = opt.verbose
              print "\n===> PP Info: start apply brain-response ica"
              print"File   : " + fif_file
              if opt.verbose: 
                 print"Parameter :" 
                 print tmp_pp_brs.ica
                 print"\n\n"

              (fif_file_ica,raw,ICAobj) = jppd.apply_ica_data(fif_file,raw=raw,**tmp_pp_brs.ica)

              print"\n\n==> PP Info: done apply ica for brain responses\n  ---> " + fif_file_ica
           else:
              fif_file_ica = jumeg_base.get_fif_name(fname=fif_file,postfix=tmp_pp_brs.ica.fif_postfix,extention=tmp_pp_brs.ica.fif_extention)
              raw           = None
        else:
           jpp_utils.msg_on_attribute_error("brain response ica")

            # 201195_Chrono01_110516_1413_1_c,rfDC,fihp1n,ocarta-ica.fif

    #--- brain response ctps 
    #--- run for all condition combine and or exclude CTPs-ICs for different conditions
        if hasattr(tmp_pp_brs.ctps,"do_run"):
           if tmp_pp_brs.ctps.do_run :
              tmp_pp_brs.ctps['verbose'] = opt.verbose
              print"\n===> PP Info: start brainresponse ica"
              print"File  : " + fif_file_ica
              if opt.verbose :
                 print"Parameters:" 
                 print tmp_pp_brs.ctps
                 print"\n\n"

              (fif_file,raw,fhdf)=jppd.apply_ctps_brain_responses_data(fif_file,raw=raw,fname_ica=fif_file_ica,ica_raw=None,
                                                                         condition_list=opt.condition,**tmp_pp_brs.ctps)

              print"\n\n==> PP Info: done apply ctp for brain responses\n  ---> " + fif_file
           else:
              fhdf = raw = None
        else:
           jpp_utils.msg_on_attribute_error("brain response ctps")
        
    #--- brain response ctps ica cleaning
    #--- run for global ics, all ctp-ics-condition  create raw,epochs,average
        if hasattr(tmp_pp_brs.clean,"do_run"):
           if tmp_pp_brs.clean.do_run :
              tmp_pp_brs.clean['verbose'] = opt.verbose

              print"\n===> PP Info: start brain-response cleaning"
              print"File  : " + fif_file
              if opt.verbose :
                 print"Parameters:"
                 print tmp_pp_brs.clean
                 print"\n\n"

              fhdf = jppd.apply_ctps_brain_responses_cleaning_data(fif_file,raw=raw,fname_ica=fif_file_ica,ica_raw=None,fhdf=fhdf,
                                                                   condition_list=opt.condition,**tmp_pp_brs.clean)
        else:
           jpp_utils.msg_on_attribute_error("brain response clean")


    print"===> Done JuMEG Pre-Processing Data"
    print" --> FIF file input                : " + fif_file_raw
    print" --> Information stored in HDF file: " + str(fhdf)
    print"\n\n"
    raw = None


if __name__ == "__main__":
   main(sys.argv)
