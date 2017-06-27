#import os
import mne

from jumeg.jumeg_base  import jumeg_base

"""
Created on Tue Jun  2 13:38:32 2015

@author: fboers

---> update 23.12.2016 FB
 --> add opt -feeg
 --> to merge eeg BrainVision with meg in jumeg_processing_batch
---> update 09.01.2017 FB
     apply_merge_meeg: if do_run ==False ->return:  None, fif_name

"""

#################################################################
#
# apply_merge_meeg
#
#################################################################
def apply_merge_meeg(fif_file,raw=None,eeg_fname=None,do_run=False,save=True,verbose=False,**kwargs):
    # startcode=128,
    # meg={'stim_channel':'STI 013'},eeg={'stim_channel':'STI 014','response_shift':1000,'stim_type':'RESPONSE'} ):
    
    """  
    stim_channel               :'STI 013' 
    startcode"                 :  128     
    brainvision_response_shift : 1000
    brainvision_channel_type   : RESPONSE
               
    merge brainvision eeg data into MEG-fif file

    RETURN:
           fname          : fif-file name,
           raw            : raw obj

    """
    from jumeg.jumeg_merge_meeg import JuMEG_MergeMEEG

    JMEEG = JuMEG_MergeMEEG(**kwargs)
    
    JMEEG.meg.filename = fif_file
    JMEEG.meg.raw      = raw
    JMEEG.eeg.filename = eeg_fname

    JMEEG.do_save   = save
    JMEEG.verbose   = verbose

    if do_run:
       return JMEEG.run()
    else:
       return JMEEG.get_fif_name(fname=fif_file,raw=raw,extention=JMEEG.meeg_extention,update_raw_fname=True),None

#################################################################
#
# apply epocher data
#
#################################################################
def apply_epocher_events_data(fname,raw=None,condition_list=None,do_run=False,**kwargs):
    """
    find events and stores epoch information into hdf5 file

    RETURN:
           fname          : fif-file name,
           raw            : raw obj
           fname_epocher  : epocher file name (hdf5 format)

    """
    from jumeg.epocher.jumeg_epocher  import jumeg_epocher
 
    if do_run :
                
       return jumeg_epocher.apply_events_to_hdf(fname,raw=raw,condition_list=condition_list, **kwargs)
   


#######################################################
#                                                     #
# interface for creating the noise-covariance matrix  #
#                                                     #
#######################################################
def apply_create_noise_covariance_data(fname,raw=None,do_filter=True,filter_parameter=None,verbose=False,do_run=True,save=True ):
    '''
    Creates the noise covariance matrix from an empty room file.

    Parameters
    ----------
    fname : string 
        containing the filename of the empty room file (must be a fif-file)
    do_filter: bool
        If true, the empy room file is filtered before calculating
        the covariance matrix. (Beware, filter settings are fixed.)
    filter_parameter: dict
        dict with filter parameter and flags 
    do_run: bool
        execute this
    save: bool
        save output file
    verbose : bool, str, int, or None
        If not None, override default verbose level
        (see mne.verbose).
        default: verbose=None
        
    RETURN
    ---------
    full empty room filname as string
    raw obj of input raw-obj or raw-empty-room obj    
    '''

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    from mne import compute_raw_data_covariance as cp_covariance
    from mne import write_cov
    
    mne.verbose = verbose

    try:
        (fname_empty_room,raw_empty) = jumeg_base.get_empty_room_fif(fname=fname,raw=raw,preload=do_run)
    except:
        return

    if raw_empty :
   #--- picks meg channels
       filter_parameter.picks = jumeg_base.pick_meg_nobads(raw_empty)
  
   #--- filter or get filter name
       filter_parameter.do_run = do_filter

       if do_filter :
          print "Filtering empty room fif with noise variance settings...\n"
          (fname_empty_room,raw_empty) = apply_filter_data(fname_empty_room,raw=raw_empty,**filter_parameter)
    

   #--- update file name for saving noise_cov
    fname_empty_room_cov = fname_empty_room.split('-')[0] + ',empty-cov.fif'
  
   #--- calc nois-covariance matrix
    if do_run :
       noise_cov_mat = cp_covariance(raw_empty,picks=filter_parameter.picks,verbose=verbose)
   #--- write noise-covariance matrix to disk
       if save :
          write_cov( fname_empty_room_cov, noise_cov_mat)
    
    return fname_empty_room_cov


'''
fn_raw_nfr = fn_raw[:fn_raw.rfind('-raw.fif')] + ',nfr-raw.fif'
            noise_reducer(fn_raw, refnotch=50, detrending=False,
                          fnout=fn_raw_nfr, verbose=verbose)

            # apply noise reducer for 60 Hz (and harmonics)
            noise_reducer(fn_raw_nfr, refnotch=60, fnout=fn_raw_nfr, verbose=verbose)

            # apply noise reducer for frequencies below 5 Hz
            noise_reducer(fn_raw_nfr, reflp=5, verbose=verbose)
            fn_raw_n = fn_raw_nfr[:fn_raw_nfr.rfind('-raw.fif')] + ',nr-raw.fif'
            remove(fn_raw_nfr)

            # plot power spectrum
            fn_power_spect = join(plot_dir,  basename(fn_raw_n[:fn_raw_n.rfind('-raw.fif')] + ',denoising'))
            plot_denoising([fn_raw, fn_raw_n], stim_name=stim_name,
                           event_id=event_id, show=False, fnout=fn_power_spect,
                           tmin_stim=tmin_stim, tmax_stim=tmax_stim)
'''

#################################################################
#
# apply noise reducer ( the magic ee once )
# modified for raw obj support imports jumeg_noise_reducer_4raw_data
#################################################################
def apply_noise_reducer_data(fname,raw=None,do_run=True,verbose=False,save=True,plot=False,
                             reflp=None, refhp=None, refnotch=None,fif_postfix="nr",fif_extention="-raw.fif",parameter=None): # **kwargs):
    '''
    Applies the noise reducer to raw obj data or to fif-file.
            the magic ee once
            fb modified for raw obj support
            imports jumeg_noise_reducer_4raw_data
    '''

    import os

#--- import noise_reducer and plot_power_spectrum function
    # from jumeg.jumeg_4raw_data_noise_reducer import noise_reducer_4raw_data, plot_denoising_4raw_data
    from jumeg.jumeg_noise_reducer import noise_reducer, plot_denoising

    fname_out = None
    nr_done   = False
    
    if do_run :  
       raw,fname_raw = jumeg_base.get_raw_obj(fname,raw=raw)
       fname_out = jumeg_base.get_fif_name(raw=raw,postfix=fif_postfix,extention=fif_extention,update_raw_fname=False)
     
#--- apply noise reducer for 50 Hz (and harmonics)
     
       if (reflp or refhp):
          #raw, fname_out = noise_reducer_4raw_data(fname, raw=raw, reflp=reflp, refhp=refhp, verbose=verbose,
          #                                          save=False, **kwargs['parameter'])

          raw,fname_out = noise_reducer(fname,raw=raw,reflp=reflp,refhp=refhp,verbose=verbose,save=False,**parameter)
          #parameter['detrending'] = None
          nr_done = True
       if refnotch:
          for refn in refnotch:
              # raw,fname_out = noise_reducer_4raw_data(None,raw=raw,refnotch=refn,verbose=verbose,save=False,**kwargs['parameter'])
              raw, fname_out = noise_reducer(None, raw=raw, refnotch=refn, verbose=verbose, save=False,**parameter)
              #parameter['detrending'] = None
          nr_done = True
  
     
       # raw.info['filename'] = fname_out
       
       if not nr_done :
          return fname_raw,raw
  
       raw.info['filename'] = fname_out
       
       if save:
          fname_out = jumeg_base.apply_save_mne_data(raw,fname=fname_out)

       if plot:
          print " --> noise reducer plot power spectrum"

          from distutils.dir_util import mkpath

          p,pdf = os.path.split(fname_raw)

          plot_dir = p+ '/plots/'

          mkpath(plot_dir)

          fn_power_spect = plot_dir + pdf[:pdf.rfind('-raw.fif') ]+ ',denoising'

          #plot_denoising_4raw_data([fname_raw, fname_out], show=False, fnout=fn_power_spect)
          plot_denoising([fname_raw, fname_out], show=False, fnout=fn_power_spect)

          print"---> noise reducer plot :"  + fn_power_spect

    print "---> Done noise reducer: "+ fname_out

    return (fname_out,raw)


#################################################################
#
# apply noise reducer ( the magic ee once )
# modified for raw obj support imports jumeg_noise_reducer_4raw_data
#################################################################
#def apply_noise_reducer(fname,raw=None, do_run=True, verbose=False, save=True, plot=False,
#                             reflp=None, refhp=None, refnotch=None, fif_postfix="nr", fif_extention="-raw.fif",**kwargs):

def apply_noise_reducer(fname,raw=None,do_run=True,verbose=False,save=True,plot=False,fif_postfix="nr",fif_extention="-raw.fif",
                        return_raw =True,reflp=None,refhp=None,refnotch=None,parameter={}):
    '''
    Applies the noise reducer to raw obj data or to fif-file.
            the magic ee once
            fb modified for raw obj support
            imports jumeg_noise_reducer_4raw_data
    '''

    import os

    # --- import noise_reducer and plot_power_spectrum function
    # from jumeg.jumeg_4raw_data_noise_reducer import noise_reducer_4raw_data, plot_denoising_4raw_data
    from jumeg.jumeg_noise_reducer import noise_reducer, plot_denoising

    fname_out = None
    nr_done = False

    if do_run:
        raw, fname_raw = jumeg_base.get_raw_obj(fname, raw=raw)
        fname_out = jumeg_base.get_fif_name(raw=raw, postfix=fif_postfix, extention=fif_extention,update_raw_fname=False)

        # --- apply noise reducer for 50 Hz (and harmonics)
        '''
        "signals":[],
        "noiseref":[],
        "detrending": null,
        "tmin": null,
        "tmax": null,
        "exclude_artifacts":true,
        "checkresults":true,
        "complementary_signal":false,
        "return_raw":true
        '''
              
        if ( reflp or refhp ):
           # raw, fname_out = noise_reducer_4raw_data(fname,raw=raw,reflp=reflp,refhp=refhp,verbose=verbose,save=False,**parameter)
           fin = None
           raw = noise_reducer(fin,raw=raw,reflp=reflp,refhp=refhp,return_raw=True,verbose=verbose,**parameter)
           parameter['detrending'] = None
           nr_done = True
            
        if refnotch:
            
           for refn in refnotch:
               # raw,fname_out = noise_reducer_4raw_data(None,raw=raw,refnotch=refn,verbose=verbose,save=False,**kwargs['parameter'])
               raw = noise_reducer(fin, raw=raw, refnotch=refn,return_raw=True,**parameter)
               parameter['detrending'] = None
           nr_done = True

        # raw.info['filename'] = fname_out

        if not nr_done:
            return fname_raw, raw
        
        if save:
            fname_out = jumeg_base.apply_save_mne_data(raw, fname=fname_out)

        if plot:
            print " --> noise reducer plot power spectrum"

            from distutils.dir_util import mkpath

            p, pdf = os.path.split(fname_raw)

            plot_dir = p + '/plots/'

            mkpath(plot_dir)

            fn_power_spect = plot_dir + pdf[:pdf.rfind('-raw.fif')] + ',denoising'

            # plot_denoising_4raw_data([fname_raw, fname_out], show=False, fnout=fn_power_spect)
            plot_denoising([fname_raw, fname_out], show=False, fnout=fn_power_spect)

            print"---> noise reducer plot :" + fn_power_spect

    print "---> Done noise reducer: " + fname_out

    return (fname_out, raw)


#################################################################
#
# apply filter on (raw) data
#
#################################################################
def apply_filter_data(fname,raw=None,filter_method="mne",filter_type='bp',fcut1=1.0,fcut2=45.0,notch=None,notch_max=None,order=4,
                      remove_dcoffset = False,njobs=1,overwrite = False,do_run=True,verbose=False,save=True,picks=None,
                      fif_postfix=None, fif_extention='-raw.fif'):
    ''' 
    Applies the FIR FFT filter [bp,hp,lp,notches] to data array. 
    filter_method : mne => fft mne-filter
                    bw  => fft butterwoth
                    ws  => fft - windowed sinc 
    '''
            
    from jumeg.filter import jumeg_filter
   
 #--- define filter 
    jfilter = jumeg_filter(filter_method=filter_method,filter_type=filter_type,fcut1=fcut1,fcut2=fcut2,njobs=njobs, 
                                remove_dcoffset=True,order=order)
    jfilter.verbose = verbose                     

    
    if do_run :
       raw,fname = jumeg_base.get_raw_obj(fname,raw=raw)
      
       if picks is None :
          picks = jumeg_base.picks.channels_nobads(raw)

    #- apply filter for picks, exclude stim,resp,bads
       jfilter.sampling_frequency = raw.info['sfreq']
    #--- calc notch array 50,100,150 .. max
       if notch :
          jfilter.calc_notches(notch,notch_max)

       jfilter.apply_filter(raw._data,picks=picks )
       jfilter.update_info_filter_settings(raw)

    #--- make output filename
       name_raw = fname.split('-')[0]
       fnfilt   = name_raw + "," + jfilter.filter_name_postfix + fif_extention

       raw.info['filename'] = fnfilt

       if save :
          fnfilt = jumeg_base.apply_save_mne_data(raw,fname=fnfilt)

    else:
     #--- calc notch array 50,100,150 .. max
       if notch :
          jfilter.calc_notches(notch,notch_max)

     #--- make output filename
       name_raw = fname.split('-')[0]
       fnfilt   = name_raw + "," + jfilter.filter_name_postfix + fif_extention

    return (fnfilt, raw)

#################################################################
#
# apply_ocarta_data
#
#################################################################

'''
# ------------------------------------------
        # apply OCARTA to remove ocular and cardiac
        # artifacts
        # ------------------------------------------
        if apply_ocarta:
            # import ocarta module
            from jumeg.decompose import ocarta

            # apply OCARTA
            ocarta_obj = ocarta.JuMEG_ocarta(name_eog='EOG 001', explVar=0.95, percentile_eog=70, thresh_ecg=0.25) #, maxsteps=10, percentile_eog=70, flow=1, fhigh=20)
            fn_ocarta = join(plot_dir, basename(fn_raw[:fn_raw.rfind('-raw.fif')] + ',ocarta_perf'))
            ocarta_obj.fit(fn_raw, fn_perf_img=fn_ocarta, plot_template_OA=True,
                           verbose=verbose, seg_length=120, shift_length=60)

            from jumeg.jumeg_noise_reducer import plot_denoising
            fn_raw_ocarta = fn_raw[:fn_raw.rfind('-raw.fif')] + ',ocarta-raw.fif'
            fn_ocarta = join(plot_dir, basename(fn_raw[:fn_raw.rfind('-raw.fif')] + ',ocarta_stim'))
            plot_denoising([fn_raw, fn_raw_ocarta], stim_name=stim_name,
                           event_id=event_id, show=False, fnout=fn_ocarta,
                           tmin_stim=tmin_stim, tmax_stim=tmax_stim)

        # adjust fn_raw
        fn_raw = fn_raw[:fn_raw.rfind('-raw.fif')] + ',ocarta-raw.fif'
'''
def apply_ocarta_data(fname,raw=None,do_run=True,verbose=False,template_name=None,**kwargs):

    #---- ocarta obj
    # name_ecg='ECG 001', ecg_freq=[10,20],
    # thresh_ecg=0.4, name_eog='EOG 002', eog_freq=[1,10],
    # seg_length=30.0, shift_length=10.0,
    # percentile_eog=80, npc=None, explVar=0.95, lrate=None,
    # maxsteps=50

    #---- ocarta.fit
    # denoising=None,unfiltered=False, notch_filter=True, notch_freq=50,
    # notch_width=None, plot_template_OA=False, verbose=True,
    # name_ecg=None, ecg_freq=None, thresh_ecg=None,
    # name_eog=None, eog_freq=None, seg_length=None, shift_length=None,
    # npc=None, explVar=None, lrate=None, maxsteps=None):

    if do_run :

       from jumeg.decompose import ocarta

       oca = ocarta.JuMEG_ocarta()

       if kwargs['fit_parameter'] :
          kwargs['fit_parameter']['verbose'] = verbose

          (raw,fnout) = oca.fit(fname,meg_raw=raw,**kwargs['fit_parameter'])

       else :
          (raw,fnout) = oca.fit(fname,meg_raw=raw)

       jumeg_base.set_raw_filename(raw,fnout)

      #--- save ocarta results into HDFobj

       if template_name :

          ecg_parameter = {'num_events': oca.idx_R_peak[:,0].size,
                           'ch_name': oca.name_ecg,'thresh' : oca.thresh_ecg,'explvar':oca.explVar,
                           'freq_correlation':None,'performance':oca.performance_ca}

          eog_parameter  = {'num_events': oca.idx_eye_blink[:,0].size,
                            'ch_name':oca.name_eog,'thresh' : None,'explvar':oca.explVar,
                            'freq_correlation':None,'performance':oca.performance_oa}

     #--- save ecg & eog onsets in HDFobj

          from jumeg.epocher.jumeg_epocher import jumeg_epocher

          (fnout,raw,fhdf) = jumeg_epocher.apply_update_ecg_eog(fnout,raw=raw,ecg_events=oca.idx_R_peak[:,0],ecg_parameter=ecg_parameter,
                                                                eog_events=oca.idx_eye_blink[:,0],eog_parameter=eog_parameter,template_name=template_name)

          print "===> save ocarta info & events : " + fhdf

       print "===> Done JuMEG ocarta        : " + fnout + "\n"

    return (fnout,raw,fhdf)


#######################################################
#
#  apply mne-ica (fastica) after ocarta
#
#######################################################
def apply_ica_data(fname,raw=None,do_run=False,verbose=False,save=True,fif_extention=".fif",fif_postfix="-ica",**kwargs):
    """
     apply mne ica

      return
        fnica_out  : fif filename of mne ica-obj
        raw        : fif-raw obj
        ICAobj     : mne-ica-object


             Attributes
        ----------
        current_fit : str
            Flag informing about which data type (raw or epochs) was used for
            the fit.
        ch_names : list-like
            Channel names resulting from initial picking.
            The number of components used for ICA decomposition.
        n_components_` : int
            If fit, the actual number of components used for ICA decomposition.
        n_pca_components : int
            See above.
        max_pca_components : int
            The number of components used for PCA dimensionality reduction.
        verbose : bool, str, int, or None
            See above.
        pca_components_` : ndarray
            If fit, the PCA components
        pca_mean_` : ndarray
            If fit, the mean vector used to center the data before doing the PCA.
        pca_explained_variance_` : ndarray
            If fit, the variance explained by each PCA component
        mixing_matrix_` : ndarray
            If fit, the mixing matrix to restore observed data, else None.
        unmixing_matrix_` : ndarray
            If fit, the matrix to unmix observed data, else None.
        exclude : list
            List of sources indices to exclude, i.e. artifact components identified
            throughout the ICA solution. Indices added to this list, will be
            dispatched to the .pick_sources methods. Source indices passed to
            the .pick_sources method via the 'exclude' argument are added to the
            .exclude attribute. When saving the ICA also the indices are restored.
            Hence, artifact components once identified don't have to be added
            again. To dump this 'artifact memory' say: ica.exclude = []
        info : None | instance of mne.io.meas_info.Info
            The measurement info copied from the object fitted.
        n_samples_` : int
            the number of samples used on fit.

    """
    ICAobj = None

    if do_run :
       raw,fname = jumeg_base.get_raw_obj(fname,raw=raw)
      
       from mne.preprocessing import ICA
       picks = jumeg_base.pick_meg_nobads(raw)

      #--- init MNE ICA obj

       kwargs['global_parameter']['verbose'] = verbose
       ICAobj = ICA( **kwargs['global_parameter'] )

      #--- run  mne ica
       kwargs['fit_parameter']['verbose'] = verbose
       ICAobj.fit(raw, picks=picks,**kwargs['fit_parameter'] )

       fnica_out = fname[:fname.rfind('-raw.fif')] + fif_postfix + fif_extention
      # fnica_out = fname[0:len(fname)-4]+'-ica.fif'

      #--- save ICA object
       if save :
          ICAobj.save(fnica_out)

    print "===> Done JuMEG MNE ICA : " + fnica_out
    print "\n"


    return (fnica_out,raw,ICAobj)


#######################################################
#
#  apply_ctps_ for brain_responses
#
#######################################################
def apply_ctps_brain_responses_data(fname,raw=None,fname_ica=None,ica_raw=None,condition_list=None,**kwargv):

    '''
    raw=None,fname_ica=None,ica_raw=None,condition_list=None,template_name=None,
                                           filter_method="bw",remove_dcoffset=False,jobs=4,
                                           freq_ctps=None,fmin=4,fmax=32,fstep=8,proj=False,exclude_events=None,
                                           ctps_parameter = {'time_pre':None,'time_post':None,'baseline':None},
                                           save_phase_angles=False,
                                           #fif_extention=".fif",fif_postfix="ctps",
                                           do_run=False,verbose=False,save=True):

    ctps= {'time_pre':-0.20,'time_post':0.50,'baseline':[None,0]},

    exclude_events={'eog_events':{ 'tmin':-0.4,'tmax':0.4} }
    '''


    fhdf = None
    """
    if kwargv['do_run']:

       from jumeg.epocher.jumeg_epocher import jumeg_epocher
       jumeg_epocher.verbose = kwargv['verbose']

       if kwargv['do_update']:

          # fname,raw,fhdf = jumeg_epocher.apply_update_brain_responses(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,condition_list=condition_list,**kwargv['parameter'])

          fhdf = jumeg_epocher.ctps_ica_brain_responses_update(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,condition_list=condition_list,template_name=kwargv['template_name'],
                                                               **kwargv['update_parameter'])


       if kwargv['do_select']:
          fhdf=jumeg_epocher.ctps_ica_brain_responses_select(fhdf=fhdf,fname=fname,raw=raw,condition_list=condition_list,template_name=kwargv['template_name'],
                                                             **kwargv['select_parameter'])

       print "===> save ctps ics : " + str(fhdf)





    print "===> Done JuMEG ctps fro brain responses : " + str(fhdf) + "\n"
    """
    return (fname,raw,fhdf)


#######################################################
#
# apply ICA-cleaning
#
#######################################################
def apply_ctps_brain_responses_cleaning_data(fname,raw=None,fname_ica=None,ica_raw=None,fhdf=None,condition_list=None,
                                             njobs=4,fif_extention=".fif",fif_postfix="ctps",template_name=None,
                                             clean_global=None,clean_condition=None,do_run=False,verbose=False):
    """
     # ICA decomposition
     # ica = mne.preprocessing.read_ica(fnica)

     # ica.exclude += list(ic_ecg) + list(ic_eog)
     #    # ica.plot_topomap(ic_artefacts)
     # # apply cleaning
        meg_clean = ica.apply(meg_raw, exclude=ica.exclude,
                              n_pca_components=npca, copy=True)
        meg_clean.save(fnclean, overwrite=True)

          ica.save(fnica)  # save again to store excluded



    """
    
    # print "===> Start JuMEG CTPs ICA cleaning: "

    fout=None
    """
    if do_run:
       from jumeg.epocher.jumeg_epocher import jumeg_epocher

       fhdf = jumeg_epocher.ctps_ica_brain_responses_clean(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,fhdf=fhdf,condition_list=condition_list,
                                                    njobs=njobs,fif_extention=fif_extention,fif_postfix=fif_postfix,template_name=template_name,
                                                    clean_global=clean_global,clean_condition=clean_condition,do_run=do_run,verbose=verbose)

       #TODO:  return global and or ctps-condition  as raw,epochs,average or names

    print "===> Done JuMEG MNE ICA clean: " + fhdf
    
    """
    return fhdf

#######################################################
#
#  apply_epocher_export_events
#
#######################################################
def apply_epocher_export_events_data(fname,raw=None,condition_list=None,**kwargv):
 
    '''
    apply_export_events
    
    fname,raw=None,condition_list=None,do_run=False,verbose=False,save=False
     
    "events": {
          "template_name":"InKomp",
          "event_extention": ".eve",
          "export_condition":{"events":true,"epochs":true,"evoked":true},
          "time":{"time_pre":None,"time_post":None,"baseline":None},
          "baseline":{"type":"avg","channel":"stimulus","baseline":[null,0]},
          "exclude_events":{"eog_events":{"tmin":-0.4,"tmax":0.6} },
          "do_run": true,
          "verbose": false
    '''
    fhdf = None

    if kwargv['do_run']:

       from jumeg.epocher.jumeg_epocher import jumeg_epocher
       jumeg_epocher.verbose = kwargv['verbose']
     #--- get export events from hdf obj, exclude EOG
       fname,raw,fhdf = jumeg_epocher.apply_events_export_events(fname,raw=raw,condition_list=condition_list,**kwargv)

    print "===> Done JuMEG apply epocher export events data: " + str(fhdf) + "\n"

    return (fname,raw,fhdf)
