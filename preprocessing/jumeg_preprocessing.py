import os
import numpy as np
import matplotlib.pylab as pl

import mne
from mne.preprocessing import ICA
from jumeg.filter import jumeg_filter
from jumeg.ctps import ctps

class JuMEG_PreProcessing(object):
     def __init__ (self):
         self._jumeg_preprocessing_version   = 0.0314
         self._notch_min = 50.0
         self._notch_max = None
        
#--- version
     def _get_version(self):  
         return self._jumeg_preprocessing_version
       
     version = property(_get_version)

#--- verbose    
     def _set_verbose(self,value):
         self._verbose = value

     def _get_verbose(self):
         return self._verbose
       
     verbose = property(_get_verbose, _set_verbose)

#--- notch
     def _get_notch_min(self):
         return self._notch_min
     
     def _set_notch_min(self, v):
         self._notch_min = v
       
     notch_min = property(_get_notch_min, _set_notch_min)

     def _get_notch_max(self):
         return self._notch_max
     def _set_notch_max(self, v):
         self._notch_max = v
       
     notch_max = property(_get_notch_max, _set_notch_max)

#--- MNE foool fct  -> picks preselected channel groups
#--- mne.pick_types(raw.info, **fiobj.pick_all) 
#    mne.pick_types(info, meg=True, eeg=False, stim=False, eog=False, ecg=False, emg=False, ref_meg='auto',
#                   misc=False, resp=False, chpi=False, exci=False, ias=False, syst=False, include=[], exclude='bads', selection=None)

     def pick_channels(self,raw):
         return mne.pick_types(raw.info,meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude=None)
       
     def pick_channels_nobads(self, raw):
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude='bads')
       
     def pick_all(self, raw):
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude=None)
       
     def pick_all_nobads(self, raw):
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude='bads')
       
     def pick_meg(self,raw):
         return mne.pick_types(raw.info,meg=True)
       
     def pick_meg_nobads(self,raw):
         return mne.pick_types(raw.info, meg=True,exclude='bads')
        
     def pick_ecg_eog(self,raw):
         return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=True,eog=True)
        
     def pick_stim(self,raw):
         return mne.pick_types(raw.info,meg=False,stim=True)
        
     def pick_response(self,raw):
         return mne.pick_types(raw.info,meg=False,resp=True)
        
     def pick_stim_response(self,raw):
         return mne.pick_types(raw.info, meg=False,stim=True,resp=True)

#--- helper function
     def get_files_from_list(self, fin):
         if isinstance(fin, list):
            fout = fin
         else:
            if isinstance(fin, str):
               fout = list([ fin ]) 
            else:
               fout = list( fin )
         return fout


#---
     def apply_filter(self,fname_raw, fcut1=1.0, fcut2=45.0,order=4,filter_type='bp',filter_method="mne",do_run=True,do_notches=True, overwrite=True):
         ''' 
            Applies the FIR FFT filter [bp,hp,lp,notches] to a list of raw files. 
            filter_method : mne => fft mne-filter
                            bw  => fft butterwoth
                            ws  => fft - windowed sinc
          '''
    
         fnout_list = []
   
     #--- define filter obj
         jfilter_obj = jumeg_filter(filter_method=filter_method,filter_type=filter_type,fcut1=fcut1,fcut2=fcut2,
                                    remove_dcoffset=True,sampling_frequency=None,order=order)
         jfilter_obj.verbose=True                     
     #--- calc notch array 50,100,150 .. max
         jfilter_obj.calc_notches( self.notch_min, self.notch_max)
         
     #--- loop across all filenames
         idx = 1        
         for fname in ( self.get_files_from_list( fname_raw ) ):   
             print "\n%3d / %3d Apply Filter : %s" % ( idx, len( fname_raw ),fname )   
             print '>>>raw data: ' + jfilter_obj.filter_info
        
     #--- make output filename
             name_raw = fname.split('-')[0]
             fnfilt = name_raw + "," + jfilter_obj.filter_name_postfix + '-raw.fif'
             fnfilt_psds = name_raw + "," + jfilter_obj.filter_name_postfix + '-raw,' + jfilter_obj.filter_method
       
     #--- start filter
             if do_run :
             #--- load raw data
                if ( os.path.isfile( fnfilt) and (overwrite == False) ) :
                     print "FOUND Data => skip filter"
                else:
                     raw = mne.io.Raw(fname,preload=True)
                     
                     if self.verbose :
                        self.plot_powerspectrum(raw,fname=name_raw )
                     
                     #--- update filter obj for raw data
                     jfilter_obj.sampling_frequency = raw.info['sfreq']
                     #--- apply filter for picks, exclude stim,resp,bads
                     jfilter_obj.apply_filter( raw._data,picks=self.pick_channels_nobads(raw))
                     
                     #raw.plot(start=10.0, duration=60.0)
                     
                     if self.verbose :
                        self.plot_powerspectrum(raw,fname=fnfilt_psds )
                     
                     print ">>>> writing filtered data to disk..."
                     print 'saving: '+ fnfilt
                     raw.save(fnfilt, overwrite=True)
           
             fnout_list.append(fnfilt) 
             idx+=1
             
             jfilter_obj.reset()
             
         return fnout_list
    
 
     def apply_average(self, filenames, stim_channel='STI 014', event_id =None, postfix=None,tmin=-0.2, tmax=0.4, baseline = (None,0), 
                                              save_plot=True, show_plot=False,  trigger_and_mask=255,response_and_mask=2047, output='onset'):

         ''' Performs averaging to a list of raw files. '''  
         
         and_mask = 2**16-1
         # Trigger or Response ?
         if  stim_channel == 'STI 014':      # trigger
            trig_name = 'trigger' 
            and_mask =   trigger_and_mask
         else:
            if  stim_channel == 'STI 013':   # response
               trig_name = 'response'
               and_mask = response_and_mask
            else:
               trig_name = 'trigger'
               
         # check list of filenames  & loop across raw files
         fnout_list = []    # collect output filenames
         for fname in ( self.get_files_from_list(filenames) ):
             name = os.path.split(fname)[1]
             print '>>> average raw data'
             print name
            # load raw data
             raw = mne.io.Raw(fname, preload=True)
             picks = self.pick_meg_nobads(raw)
             pick_stim= self.pick_stim(raw)
            
#--- apply and mask check values float or integer
             trig_idx        = raw.ch_names.index( stim_channel )   #'STI 014' 'STI 013'
             trig_data      = raw._data[trig_idx, :] 
             trig_data[:]  = np.bitwise_and( trig_data.astype(np.int),and_mask)
             stim_events = mne.find_events(raw, stim_channel=stim_channel, consecutive=True, output=output)
             nevents       = len(stim_events)
#  alalalalalal 
 # 205386_MEG94T_12-09-06_14-01_1_c,rfDC,fwsbp1-400ord16n7,o,ica,fwslp45ord16.hdr,avg_meg94t_avginfo.yml
 #    code: '30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30'
 #epochs: '39254,58948,83088,90780,98471,102444,106417,118673,122645,135053,147039,155305,159278,201016,204988,224834,232526,248805,264780,280908,289022,305724,309696,317388,372819,444123,452389,469243,477358,502073,521919,525891,534006,537978,582979,599106,611666,636229,648214,655906'
 
             if nevents > 0:
             # for a specific event ID
                if event_id:
                   ix = np.where(stim_events[:,2] == event_id)[0]
                   stim_events = stim_events[ix,:]
                else:
                   event_id = stim_events[0,2]  
            
                epochs = mne.Epochs(raw, events=stim_events,event_id=event_id, tmin=tmin, tmax=tmax,
                                 picks=picks, preload=True, baseline=baseline)
                avg = epochs.average()

               # save averaged data
                if (postfix):
                    fnout = fname[0:len(fname)-4] + postfix+'.fif'
                else:    
                    fnout = fname[0:len(fname)-4] + ',avg,'+trig_name+'.fif'

                avg.save(fnout)
                print 'saved:'+fnout
                fnout_list.append(fnout)

                if (save_plot):
                    self.plot_average(fnout, show_plot=show_plot)

             else: 
               event_id = None
               print '>>> Warning: Event not found in file: '+fname   
         
         return fnout_list

     
     def plot_average(self, filenames, save_plot=True, show_plot=False):

         ''' Plot Signal average from a list of averaged files. '''

         pl.ioff()  # switch off (interactive) plot visualisation
         factor = 1e15
         for fnavg in ( self.get_files_from_list(filenames) ):
             name = fnavg[0:len(fnavg)-4] 
             basename = os.path.splitext(os.path.basename(name))[0]
             print fnavg
            # mne.read_evokeds provides a list or a single evoked based on the condition.
            # here we assume only one evoked is returned (requires further handling)
             avg = mne.read_evokeds(fnavg)[0]
             ymin, ymax = avg.data.min(), avg.data.max()
             ymin  *= factor*1.1
             ymax  *= factor*1.1
             fig = pl.figure(basename,figsize=(10,8), dpi=100) 
             pl.clf()
             pl.ylim([ymin, ymax])
             pl.xlim([avg.times.min(), avg.times.max()])
             pl.plot(avg.times, avg.data.T*factor, color='black')
             pl.title(basename)

            # save figure
             fnfig = os.path.splitext(fnavg)[0]+'.png'
             pl.savefig(fnfig,dpi=100)

         pl.ion()  # switch on (interactive) plot visualisation


     def apply_ica(self, fname_filtered, n_components=0.99, decim=None, do_run=True):

         ''' Applies ICA to a list of (filtered) raw files. '''
  
        # loop across all filenames
         fnout_list = []
    
         for fname in ( self.get_files_from_list(fname_filtered) ):   
             name      = os.path.split(fname)[1]
             fnica_out = fname.strip('-raw.fif') + '-ica.fif'
        
             print ">>>> perform ICA signal decomposition on :  "+name
           
            # load filtered data
             if do_run :
                raw = mne.io.Raw(fname,preload=True)
                picks = self.pick_meg_nobads(raw)
              # ICA decomposition
                ica = ICA(n_components=n_components, max_pca_components=None)
                ica.fit(raw, picks=picks, decim=decim, reject={'mag': 5e-12})
              # save ICA object 
                ica.save(fnica_out)
        
             fnout_list.append(fnica_out)
   
         return fnout_list
    
     def apply_ica_cleaning(self, fname_ica, n_pca_components=None, 
                           flow_ecg=10.0, fhigh_ecg=20.0, flow_eog=1.0, fhigh_eog=10.0, threshold=0.3,do_run=True):

        ''' Performs artifact rejection based on ICA to a list of (ICA) files. '''

        # loop across all filenames
        fnout_list = []
        
        for fnica in ( self.get_files_from_list(fname_ica) ):   
            name  = os.path.split(fnica)[1]
            #basename = fnica[0:len(fnica)-4]
            basename = fnica.strip('-ica.fif')
            fnfilt = basename+'-raw.fif'
            #fnfilt = basename + '.fif'
            fnclean = basename+',ar-raw.fif'
            fnica_ar = basename+',ica-performance'
            print ">>>> perform artifact rejection on :"
            print '   '+name
            if do_run:
               # load filtered data
                meg_raw = mne.io.Raw(fnfilt,preload=True)
                picks = self.pick_meg_nobads(meg_raw)
               # ICA decomposition
                ica = mne.preprocessing.read_ica(fnica)
               # get ECG and EOG related components
                ic_ecg = self.get_ics_cardiac(meg_raw, ica,flow=flow_ecg, fhigh=fhigh_ecg, thresh=threshold)
                ic_eog = self.get_ics_ocular(meg_raw, ica,flow=flow_eog, fhigh=fhigh_eog, thresh=threshold)
                ica.exclude += list(ic_ecg) + list(ic_eog)
                # ica.plot_topomap(ic_artefacts)
                ica.save(fnica)  # save again to store excluded
    
                # clean and save MEG data 
                if n_pca_components:
                   npca = n_pca_components
                else:
                   npca = picks.size
                print npca
                meg_clean = ica.apply(meg_raw, exclude=ica.exclude,n_pca_components=npca, copy=True)
                meg_clean.save(fnclean, overwrite=True)
    
                # plot ECG, EOG averages before and after ICA 
                print ">>>> create performance image..."
                self.plot_performance_artifact_rejection(meg_raw, ica, fnica_ar,show=False, verbose=False)

            fnout_list.append( fnclean )
        return fnout_list
    
     def get_ics_ocular(self, meg_raw, ica, flow=1, fhigh=10,name_eog_hor = 'EOG 001', name_eog_ver = 'EOG 002',
                       score_func = 'pearsonr', thresh=0.3):
         ''' Find Independent Components related to ocular artefacts '''
         # Note: when using the following:
         #   - the filter settings are different
         #   - here we cannot define the filter range
         # vertical EOG
         # idx_eog_ver = [meg_raw.ch_names.index(name_eog_ver)]
         # eog_scores = ica.score_sources(meg_raw, meg_raw[idx_eog_ver][0])
         # eogv_idx = np.where(np.abs(eog_scores) > thresh)[0]
         # ica.exclude += list(eogv_idx)
         # ica.plot_topomap(eog_idx)
         # horizontal EOG
         # idx_eog_hor = [meg_raw.ch_names.index(name_eog_hor)]
         # eog_scores = ica.score_sources(meg_raw, meg_raw[idx_eog_hor][0])
         # eogh_idx = np.where(np.abs(eog_scores) > thresh)[0]
         # ica.exclude += list(eogh_idx)
         # ica.plot_topomap(eog_idx)
         # print [eogv_idx, eogh_idx]

         # vertical EOG
       
         filter_method = "bw"
         filter_type   = 'bp'  # band pass
         srate         = meg_raw.info['sfreq']
         fi_obj        = jumeg_filter(filter_method=filter_method,filter_type=filter_type,fcut1=flow,fcut2=fhigh,
                                        remove_dcoffset=False,sampling_frequency=srate,order=4)
    
         idx_eog_ver = [meg_raw.ch_names.index(name_eog_ver)]
         eog_ver_filtered = meg_raw[idx_eog_ver, :][0].copy() # ??? fb
        
           #eog_ver_filtered = mne.filter.band_pass_filter(meg_raw[idx_eog_ver, :][0],meg_raw.info['sfreq'], Fp1=flow, Fp2=fhigh)
                                
         fi_obj.apply_filter( eog_ver_filtered )   # fb
        
         eog_ver_scores = ica.score_sources(meg_raw,target=eog_ver_filtered, score_func=score_func)
         ic_eog_ver = np.where(np.abs(eog_ver_scores) >= thresh)[0] +1  # plus 1 for any()
         if not ic_eog_ver.any(): 
                ic_eog_ver = np.array([0])
    
         # horizontal EOG
         
         idx_eog_hor = [meg_raw.ch_names.index(name_eog_hor)]
         eog_hor_filtered = meg_raw[idx_eog_hor, :][0].copy() #fb
         fi_obj.apply_filter( eog_hor_filtered )
        
         #eog_hor_filtered = mne.filter.band_pass_filter(meg_raw[idx_eog_hor, :][0],meg_raw.info['sfreq'], Fp1=flow, Fp2=fhigh)
         eog_hor_scores = ica.score_sources(meg_raw,target=eog_hor_filtered, score_func=score_func)
         ic_eog_hor = np.where(np.abs(eog_hor_scores) >= thresh)[0] +1 # plus 1 for any()
           
         if not ic_eog_hor.any(): 
                ic_eog_hor = np.array([0])
        
         # combine both  
         idx_eog = []
         for i in range(ic_eog_ver.size):
             ix = ic_eog_ver[i] -1
             if (ix >= 0):
                 idx_eog.append(ix)
         for i in range(ic_eog_hor.size):
             ix = ic_eog_hor[i] -1
             if (ix >= 0):
                 idx_eog.append(ix)
                 
         return idx_eog

     #  determine cardiac related ICs
     def get_ics_cardiac(self, meg_raw, ica, flow=10, fhigh=20, tmin=-0.3, tmax=0.3, name_ecg = 'ECG 001', use_CTPS=True,
                         score_func = 'pearsonr', thresh=0.3):
        ''' Identify components with cardiac artefacts '''
    
        filter_method = "bw"
        filter_type   = 'bp'  # band pass
        srate         = meg_raw.info['sfreq']
        fi_bw_bp      = jumeg_filter(filter_method=filter_method,filter_type=filter_type,fcut1=flow,fcut2=fhigh,
                                     remove_dcoffset=False,sampling_frequency=srate,order=4)
     
        event_id_ecg = 999
        # get and filter ICA signals
        ica_raw = ica.get_sources(meg_raw)
        
        #ica_raw.filter(l_freq=flow, h_freq=fhigh, n_jobs=2, method='fft')
        fi_bw_bp.apply_filter( ica_raw._data ) # fb 30.09.14
        
        # get R-peak indices in ECG signal
        idx_R_peak, _, _ = mne.preprocessing.find_ecg_events(meg_raw, \
                            ch_name=name_ecg, event_id=event_id_ecg, \
                            l_freq=flow, h_freq=fhigh,verbose=False)
    
    
        # -----------------------------------
        # default method:  CTPS
        #           else:  correlation
        # -----------------------------------
        if use_CTPS:
            # create epochs
            picks = np.arange(ica.n_components_)
            ica_epochs = mne.Epochs(ica_raw, events=idx_R_peak, \
                                    event_id=event_id_ecg, tmin=tmin, \
                                    tmax=tmax, baseline=None, \
                                    proj=False, picks=picks, verbose=False)
            # compute CTPS
            _, pk, _ = ctps.compute_ctps(ica_epochs.get_data())
    
            pk_max = np.max(pk, axis=1)
            idx_ecg = np.where(pk_max >= thresh)[0]
        else:
            # use correlation
            idx_ecg = [meg_raw.ch_names.index(name_ecg)]
            
            #-- fb 30.09.14
            ecg_filtered = meg_raw[idx_ecg, :][0].copy()
            fi_bw_bp.apply_filter( ecg_filtered )
            #ecg_filtered = mne.filter.band_pass_filter(meg_raw[idx_ecg, :][0], meg_raw.info['sfreq'], Fp1=flow, Fp2=fhigh)
                                    
            ecg_scores = ica.score_sources(meg_raw, \
                                target=ecg_filtered, score_func=score_func)
            idx_ecg = np.where(np.abs(ecg_scores) >= thresh)[0]
    
    
        return idx_ecg


      #  calculate the performance of artifact rejection
     def calc_performance(self, evoked_raw, evoked_clean):
           ''' Gives a measure of the performance of the artifact reduction. 
               Percentage value returned as output. 
           '''
           from jumeg import jumeg_math as jmath

           diff = evoked_raw.data - evoked_clean.data
           rms_diff = jmath.calc_rms(diff, average=1)
           rms_meg = jmath.calc_rms(evoked_raw.data, average=1)
           arp = (rms_diff / rms_meg) * 100.0
           return arp
     
     
     def plot_performance_artifact_rejection(self, meg_raw, ica, fnout_fig, \
                                            show=False, verbose=False):
    
        ''' Creates a performance image of the data before 
        and after the cleaning process.
        '''
    
        import mne
        from jumeg import jumeg_math as jmath
        import matplotlib.pylab as pl
        import numpy as np
    
        name_ecg = 'ECG 001'
        name_eog_hor = 'EOG 001'
        name_eog_ver = 'EOG 002'
        event_id_ecg = 999
        event_id_eog = 998
        tmin_ecg = -0.4
        tmax_ecg =  0.4
        tmin_eog = -0.4
        tmax_eog =  0.4
    
        picks = mne.pick_types(meg_raw.info, meg=True, exclude='bads')
        # Why is the parameter below n_components_ instead of n_pca_components?
        meg_clean = ica.apply(meg_raw, exclude=ica.exclude, n_pca_components=ica.n_components_, copy=True)
    
        # plotting parameter
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        xFigSize = 12
        nrange = 2
    
    
        # ToDo:  How can we avoid popping up the window if show=False ?
        pl.ioff()
        pl.figure('performance image', figsize=(xFigSize, 12))
        pl.clf()
    
    
        # ECG, EOG:  loop over all artifact events
        for i in range(nrange):
            # get event indices
            if i == 0:
                baseline = (None, None)
                event_id = event_id_ecg
                idx_event, _, _ = mne.preprocessing.find_ecg_events(meg_raw,
                                    event_id, ch_name=name_ecg, verbose=verbose)
                idx_ref_chan = meg_raw.ch_names.index(name_ecg)
                tmin = tmin_ecg
                tmax = tmax_ecg
                pl1 = nrange * 100 + 21
                pl2 = nrange * 100 + 22
                text1 = "CA: original data"
                text2 = "CA: cleaned data"
            elif i == 1:
                baseline = (None, None)
                event_id = event_id_eog
                idx_event = mne.preprocessing.find_eog_events(meg_raw,
                                    event_id, ch_name=name_eog_ver, verbose=verbose)
                idx_ref_chan = meg_raw.ch_names.index(name_eog_ver)
                tmin = tmin_eog
                tmax = tmax_eog
                pl1 = nrange * 100 + 23
                pl2 = nrange * 100 + 24
                text1 = "OA: original data"
                text2 = "OA: cleaned data"
    
            # average the signals
            raw_epochs = mne.Epochs(meg_raw, idx_event, event_id, tmin, tmax,
                                picks=picks, baseline=baseline, verbose=verbose)
            cleaned_epochs = mne.Epochs(meg_clean, idx_event, event_id, tmin, tmax,
                                picks=picks, baseline=baseline, verbose=verbose)
            ref_epochs = mne.Epochs(meg_raw, idx_event, event_id, tmin, tmax,
                                picks=[idx_ref_chan], baseline=baseline, verbose=verbose)
    
            raw_epochs_avg = raw_epochs.average()
            cleaned_epochs_avg = cleaned_epochs.average()
            ref_epochs_avg = np.average(ref_epochs.get_data(), axis=0).flatten() * -1.0
            times = raw_epochs_avg.times*1e3
            if np.max(raw_epochs_avg.data) < 1:
                factor = 1e15
            else:
                factor = 1
            ymin = np.min(raw_epochs_avg.data) * factor
            ymax = np.max(raw_epochs_avg.data) * factor
    
            # plotting data before cleaning
            pl.subplot(pl1)
            pl.plot(times, raw_epochs_avg.data.T * factor, 'k')
            pl.title(text1)
            # plotting reference signal
            pl.plot(times, jmath.rescale(ref_epochs_avg, ymin, ymax), 'r')
            pl.xlim(times[0], times[len(times)-1])
            pl.ylim(1.1*ymin, 1.1*ymax)
            # print some info
            textstr1 = 'num_events=%d\nEpochs: tmin, tmax = %0.1f, %0.1f' \
                       %(len(idx_event), tmin, tmax)
            pl.text(times[10], 1.09*ymax, textstr1, fontsize=10, verticalalignment='top', bbox=props)
    
    
            # plotting data after cleaning
            pl.subplot(pl2)
            pl.plot(times, cleaned_epochs_avg.data.T * factor, 'k')
            pl.title(text2)
            # plotting reference signal again
            pl.plot(times, jmath.rescale(ref_epochs_avg, ymin, ymax), 'r')
            pl.xlim(times[0], times[len(times)-1])
            pl.ylim(1.1*ymin, 1.1*ymax)
            # print some info
            #ToDo: would be nice to add info about ica.excluded
            textstr1 = 'Performance: %f\nNum of components used: %d\nn_pca_components: %f' \
                       %( self.calc_performance(raw_epochs_avg, cleaned_epochs_avg), ica.n_components_, ica.n_pca_components)
            pl.text(times[10], 1.09*ymax, textstr1, fontsize=10, verticalalignment='top', bbox=props)
    
        if show:
            pl.show()
    
        # save image
        pl.savefig(fnout_fig + '.tif', format='tif')
        pl.close('performance image')
        pl.ion()
        
    
     def apply_ctps_select_ic(self, fname_ctps, threshold=0.1):
    
        ''' Select ICs based on CTPS analysis. '''
    
        import mne, os, ctps, string
        import numpy as np
        import matplotlib.pylab as pl
    
        # check list of filenames        
        if isinstance(fname_ctps, list):
            fnlist = fname_ctps
        else:
            if isinstance(fname_ctps, str):
                fnlist = list([fname_ctps]) 
            else:
                fnlist = list(fname_ctps)
    
        # loop across all filenames
        pl.ioff()  # switch off (interactive) plot visualisation
        ifile = 0
        for fnctps in fnlist:        
            name  = os.path.splitext(fnctps)[0]
            basename = os.path.splitext(os.path.basename(fnctps))[0]
            print '>>> working on: '+basename
            # load CTPS data
            dctps = np.load(fnctps).item()
            freqs = dctps['freqs']
            nfreq = len(freqs)
            ncomp = dctps['ncomp']
            trig_name = dctps['trig_name']
            times = dctps['times']
            ic_sel = []
            # loop acros all freq. bands
            fig=pl.figure(ifile+1,figsize=(16,9), dpi=100) 
            pl.clf()
            fig.subplots_adjust(left=0.08, right=0.95, bottom=0.05,
                                top=0.93, wspace=0.2, hspace=0.2)            
            fig.suptitle(basename, fontweight='bold')
            nrow = np.ceil(float(nfreq)/2)
            for ifreq in range(nfreq):
                pk = dctps['pk'][ifreq]
                pt = dctps['pt'][ifreq]
                pkmax = pk.max(1)
                ixmax = np.where(pkmax == pkmax.max())[0]
                ix = (np.where(pkmax >= threshold))[0]
                if np.any(ix):
                    if (ifreq > 0):
                        ic_sel = np.append(ic_sel,ix+1)
                    else:
                        ic_sel = ix+1
    
                # do construct names for title, fnout_fig, fnout_ctps
                frange = ' @'+str(freqs[ifreq][0])+'-'+str(freqs[ifreq][1])+'Hz'
                x = np.arange(ncomp)+1
                # do make bar plots for ctps thresh level plots
                ax = fig.add_subplot(nrow,2,ifreq+1)
                pl.bar(x,pkmax,color='steelblue')
                pl.bar(x[ix],pkmax[ix],color='red')
                pl.title(trig_name+frange, fontsize='small')
                pl.xlim([1,ncomp])
                pl.ylim([0,0.5])
                pl.text(2,0.45, 'ICs: '+str(ix+1))
            ic_sel = np.unique(ic_sel)
            nic = np.size(ic_sel)
            info = 'ICs (all): '+str(ic_sel).strip('[]')
            fig.text(0.02,0.01, info,transform=ax.transAxes)
    
            # save CTPS components found
            fntxt = name+'-ic_selection.txt'
            ic_sel = np.reshape(ic_sel,[1,nic])
            np.savetxt(fntxt,ic_sel,fmt='%i',delimiter=', ')
            ifile += 1
    
            # save figure
            fnfig = name+'-ic_selection.png'
            pl.savefig(fnfig,dpi=100)
        pl.ion()  # switch on (interactive) plot visualisation

     def plot_powerspectrum(self, raw, fname='raw'):
         ''' 
       
         '''
         # import matplotlib.pyplot as plt
         #import matplotlib.pylab as pl
         import matplotlib.pyplot as pl
         
         tmin=None
         tmax=None
         fmin, fmax = 0.0, 450.0  # look at frequencies between DC and 300Hz
         n_fft = 4096  # the FFT size (n_fft). Ideally a power of 2
         picks = self.pick_meg(raw)
         #raw.plot_psds(area_mode='range', tmax=10.0)
    
         file_name = fname.split('/')[-1]
         fnfig = fname + '-psds.png'
         pl.figure()
         pl.title('PSDS ' + file_name)
         ax = pl.axes()
         #ax.set_title('PreProc PSDS Check')
         #pl.legend(['MEG' + fname])
        
         #raw.plot_psds(tmin=tmin,tmax=tmax,fmin=fmin,fmax=fmax,n_fft=n_fft,n_jobs=1,proj=False,ax=ax,color=(0, 0, 1),picks=picks)
         fig = raw.plot_psds(fmin=fmin,fmax=fmax,n_fft=n_fft,n_jobs=1,proj=False,ax=ax,color=(0, 0, 1),picks=picks, area_mode='range')
    
        
         fnfig = fname + '-psds.png'
         fig.savefig(fnfig)
         #pl.show()

def  jumeg_preprocessing(**kwargv):
       return JuMEG_PreProcessing()
       
#######################################################
# main
# return the new obj  from JuMEG_PreProcessing class
#######################################################
if __name__ == "__main__":
     jumeg_preprocessing(**kwargv)

