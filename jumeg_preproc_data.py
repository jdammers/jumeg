import os
import mne

from jumeg.ctps import ctps
from jumeg.decompose import ocarta_offline, _find_eog_events,  qrs_detector

from jumeg.jumeg_base import jumeg_base
from jumeg import jumeg_plot

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
    from mne.io import Raw
    from mne import pick_types
    import os

    mne.verbose = verbose
    
    fname_empty_room = None
    (fname_empty_room,raw_empty) = jumeg_base.get_empty_room_fif(fname=fname,rar=raw,preload=do_run)
 
    #--- picks meg channels
    filter_parameter.picks = jumeg_base.pick_meg_nobads(raw_empty)
  
    #--- filter or get filter name
    filter_parameter.do_run = do_filter
    if do_filter :
       print "Filtering empty room fif with noise variance settings..."
       
    (fname_empty_room,raw_empty) = apply_filter_data(fname_empty_room,raw=raw_empty,**filter_parameter)   
    
    #--- update file name for saving noise_cov
    fname_empty_room_cov = fname_empty_room.split('-')[0] + ',empty-cov.fif'
  
    #--- calc nois-covariance matrix
    if do_run :
       noise_cov_mat = cp_covariance(raw_empty,picks=filter_parameter.picks,verbose=verbose)
       # write noise-covariance matrix to disk
       if save :
          write_cov( fname_empty_room_cov, noise_cov_mat)
    
    return fname_empty_room_cov

#################################################################
#
# apply filter on (raw) data
#
#################################################################
def apply_filter_data(fname,raw=None,filter_method="mne",filter_type='bp',fcut1=1.0,fcut2=45.0,notch=None,notch_max=None,order=4,remove_dcoffset=True,njobs=1,  
                      overwrite=False,do_run=True,verbose=False,save=True,picks=None,fif_postfix=None, fif_extention='-raw.fif'):
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
 #--- calc notch array 50,100,150 .. max
    if notch :
      jfilter.calc_notches(notch,notch_max)
    
 #--- make output filename
    name_raw = fname.split('-')[0]
    fnfilt = name_raw + "," + jfilter.filter_name_postfix + fif_extention
    
    if do_run :
       if raw is None:
          if fname is None:
             print"ERROR no file foumd!!\n" 
             return 
          raw = mne.io.Raw(fname,preload=True)
       
       if picks is None :
          picks = jumeg_base.pick_channels_nobads(raw)
          
    #- apply filter for picks, exclude stim,resp,bads
       jfilter.sampling_frequency = raw.info['sfreq']
       jfilter.apply_filter(raw._data,picks=picks )
       jfilter.update_info_filter_settings(raw)
       
       if save :
          fnfilt = jumeg_base.apply_save_mne_data(raw,fname=fnfilt)   
     
    return (fnfilt, raw)

#################################################################
#
# apply_ocarta_offline_data
#
#################################################################
def apply_ocarta_offline_data(fname,raw=None,do_run=True,**argv):
                              #**offline_parameter
                              #name_ecg='ECG 001',name_eog='EOG 002',
                              #event_chan=None,dir_img='plots',seg_length=30.0,shift_length=10.0,explVar=0.99,init_maxsteps=100,
                              #maxsteps=50,denoising=None,fnout=None,verbose=True,do_run=True,save=True,fif_postfix='aroca',fif_extention='-raw.fif'):
    if do_run :
       (fnout,raw,ecg_events,eog_events) = ocarta_offline(fname,filt_data=raw,**argv['offline_parameter'])
       print "===> Done ocarta offline"
    return (fnout,raw,ecg_events,eog_events)


def find_ecg_eog_events(fname, raw=None, name_ecg='ECG 001', name_eog='EOG 002'):
    
    if raw is None:
       if fname is None:
          print"ERROR no file foumd!!\n" 
          return 
       raw = mne.io.Raw(fname,preload=True)
  
    from jumeg.filter import jumeg_filter
    
    idx_ecg_channel = raw.info['ch_names'].index(name_ecg)
    idx_eog_channel = raw.info['ch_names'].index(name_eog)
    eog_signals     = raw._data[idx_eog_channel]
    ecg_signals     = raw._data[idx_ecg_channel]
    sfreq           = raw.info['sfreq']
    ntsl            = ecg_signals.size
   
    fi_bp_bw = jumeg_filter(filter_method='bw', filter_type='bp',sampling_frequency=sfreq,fcut1=1.0,fcut2=10.0)
    fi_bp_bw.apply_filter( eog_signals )
    idx_eye_blink = _find_eog_events(eog_signals.reshape(1, ntsl), 998, 1, 10, sfreq, 0)[:, 0]

    idx_R_peak = qrs_detector(sfreq, ecg_signals.ravel(),thresh_value="auto")                     # get indices of R-peaks

    return(idx_R_peak,idx_eye_blink)
    
    
#######################################################
#
#  apply ICA for artifact rejection
#
#######################################################
def apply_ica_data(fname,raw=None,n_components=0.99, decim=None, max_pca_components=None, reject={'mag': 5e-12},
                   verbose=True, do_run=True, save=True,fif_postfix="ica",fif_extention=".fif"):

    ''' Applies ICA to a list of (filtered) raw files. '''

    import mne
    from mne.preprocessing import ICA
    import os
    fnout_ica = jumeg_base.get_fif_name(fname,postfix=fif_postfix,extention=fif_extention) 
       
    if do_run :
       if raw is None:
          raw = mne.io.Raw(fname,preload=True)
       picks = jumeg_base.pick_meg_nobads(raw)
       #--- ICA decomposition
       ica = ICA(n_components=n_components, max_pca_components=max_pca_components)
       ica.fit(raw, picks=picks, decim=decim, reject=reject)
       #--- save ICA object
       if save :
          ica.save(fnout_ica)
       
    return (fnout_ica,ica)


