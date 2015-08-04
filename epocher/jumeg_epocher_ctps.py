#import os
import numpy as np
import pandas as pd
import mne

#from jumeg.epocher.jumeg_epocher  import jumeg_epocher
from jumeg.jumeg_base import jumeg_base
from jumeg.epocher.jumeg_epocher_events import JuMEG_Epocher_Events

from jumeg.filter.jumeg_filter  import jumeg_filter
# from jumeg.ctps                 import ctps
import jumeg.jumeg_math         as jumeg_math

from mne.preprocessing.ctps_ import ctps



class JuMEG_Epocher_CTPS(JuMEG_Epocher_Events):
      def __init__ (self):
          super(JuMEG_Epocher_CTPS, self).__init__()


          self.raw=None
          self.ica_raw=None
          self.ica_picks =None


          self.ctps_freq_min= None
          self.ctps_freq_max= None
          self.ctps_freq_step = 1
          self.ctps_freq_bands = None
          self.ctps_freq_bands_list = None


          self.time_window_pre = None
          self.time_window_post = None
          self.time_window_baseline= None

          self.scale_factor = 1000.0
          self.ctps_hdf_parameter = dict()

          self.ctps_pkd_theshold = 110  # 0.15 150/1000
          self.ctps_pkd_min_number_of_timeslices = 4

          self.steady_state_artifact_bands = np.array([ [22,26],[58,62],[48,52],[98,102],[148,152],[198,202],[248,252],[298,302]] )

      def ctps_update_ctps_hdf_parameter_time(self,ctps_parameter=None,ep_param=None):

          """

          :param ctps_parameter:
          :param ep_param:
          :return:
          """
          for k in ('time_pre','time_post','baseline'):
              if ctps_parameter.has_key(k) :
                 if ctps_parameter[k]:
                    self.ctps_hdf_parameter[k] = ctps_parameter[k]
                    continue

              if ep_param.has_key('ctps'):
                 self.ctps_hdf_parameter[k] = ep_param['ctps'][k]
              elif ep_param[k]:
                 self.ctps_hdf_parameter[k] = ep_param[k]

              if not self.ctps_hdf_parameter.has_key(k) :
                 self.ctps_hdf_parameter[k] = None

          if self.verbose:
             print " --> ctps  hdf parameter: "
             print self.ctps_hdf_parameter

          return self.ctps_hdf_parameter



      def ctps_init_freq_bands(self,freq_ctps=None,fmin=None,fmax=None,fstep=None):
          """

          :param freq_ctps:
          :param fmin:
          :param fmax:
          :param fstep:
          :return:
          """

          if freq_ctps.any():
             self.ctps_freq_bands = freq_ctps
             self.ctps_freq_min   = self.ctps_freq_bands[0,0]
             self.ctps_freq_max   = self.ctps_freq_bands[-1,-1]
             self.ctps_freq_step  = self.ctps_freq_bands[0,1] - self.ctps_freq_bands[0,0]

          else:
             self.ctps_freq_min = fmin
             self.ctps_freq_max = fmax
             self.ctps_freq_step= fstep

             self.ctps_freq_bands = jumeg_math.calc_sliding_window(fmin,fmax,fstep)

             self.ctps_hdf_parameter['nfreq']= self.ctps_freq_bands.shape[0]

          self.ctps_freq_bands_list = '\n'.join('-'.join(str(cell) for cell in row) for row in self.ctps_freq_bands).split('\n')

      def ctps_init_brain_response_data(self,fname,raw=None,fname_ica=None,ica_raw=None,template_name=None):
          """

          :param fname:
          :param raw:
          :param fname_ica:
          :param ica_raw:
          :param template_name:
          :return:
          """

          print " ---> Start CTPS  init select brain responses"

         #--- ck template
          if template_name:
             self.template_name = template_name
          else:
             assert "ERROR no <template_name> specified !!\n\n"

          self.raw,fname = jumeg_base.get_raw_obj(fname,raw=raw)
          
          self.ica_raw,fname_ica = jumeg_base.get_ica_raw_obj(fname_ica,ica_raw=ica_raw)
       
          self.ica_picks = np.arange( self.ica_raw.n_components_ )

         #--- open HDFobj
          self.hdf_obj_open(fname=fname,raw=self.raw)

         #---
          self.ctps_hdf_parameter['fnica'] = self.ica_raw.info['filename'],
          self.ctps_hdf_parameter['ncomp'] = len(self.ica_picks),
          self.ctps_hdf_parameter['sfreq'] = self.ica_raw.info['sfreq'],
          self.ctps_hdf_parameter['scale_factor'] = self.scale_factor

      def ctps_init_brain_response_clean_data(self,fname,raw=None,fname_ica=None,ica_raw=None,fhdf=None,template_name=None):
          """

          :param fname:
          :param raw:
          :param fname_ica:
          :param ica_raw:
          :param fhdf:
          :param template_name:
          :return:
          """
          print " ---> Start CTPS  init clean brain responses"

         #--- ck template
          if template_name:
             self.template_name = template_name
         # else:
         #    assert "ERROR no <template_name> specified !!\n\n"
          self.raw,fname = jumeg_base.get_raw_obj(fname,raw=raw)
          
         #--- load ica raw obj & init
          if ica_raw is None:
             if fname_ica is None:
                assert "ERROR no file foumd!!\n\n"
             self.ica_raw = mne.preprocessing.read_ica(fname_ica)
          else:
             self.ica_raw = ica_raw

          self.ica_picks = np.arange( self.ica_raw.n_components_ )

         #--- open HDFobj
          self.hdf_obj_open(fname=fname,raw=self.raw,fhdf=fhdf)


      def ctps_update_artifact_time_window(self,aev=None,tmin=None,tmax=None):
          """

          :param aev:
          :param tmin:
          :param tmax:
          :return:
          """
          import numpy as np
          import mne

          artifact_events = dict()

          for kbad in ( aev.keys() ):
              node_name = '/ocarta/' + kbad

              if self.HDFobj.get(node_name) is None:
                 continue

              artifact_events[kbad]= {'tmin':None,'tmax':None,'tsl':np.array([])}

              if tmin:
                 tsl0= self.raw.time_as_index(tmin)
                 artifact_events[kbad]['tmin'] = tmin
              else:
                 tsl0= self.raw.time_as_index( aev[kbad]['tmin'] )
                 artifact_events[kbad]['tmin'] = aev[kbad]['tmin']

              if tmax:
                 tsl1= self.raw.time_as_index(tmax)
                 artifact_events[kbad]['tmax'] = tmax
              else:
                 tsl1= self.raw.time_as_index(aev[kbad]['tmax'] )
                 artifact_events[kbad]['tmax'] = aev[kbad]['tmax']

              df_bad = self.HDFobj.get(node_name)
              artifact_events[kbad]['tsl'] = np.array([ df_bad['onset'] +tsl0, df_bad['onset'] +tsl1 ] )
              # aev[0,ixd] ->tsl0   aev[1,idx] -> tsl1
              #artifact_events[kbad]['tsl'] = np.zeros( shape =( df_bad['onset'].size,2) )
              #artifact_events[kbad]['tsl'][:,0] = df_bad['onset'] +tsl0
              #artifact_events[kbad]['tsl'][:,1] = df_bad['onset'] +tsl1

          return artifact_events


      def ctps_update_hdf_condition_list(self,condition_list,node='/epocher/'):
          """

          :param condition_list:
          :param node:
          :return:
          """

          clist = []

          if condition_list :
             for k in condition_list :
                 if k in self.hdf_obj_list_keys_from_node(node):
                    clist.append(node +'/'+ k )
             return clist
          else :
             return self.hdf_obj_list_keys_from_node(node)



      def ctps_update_condition_parameter(self,condi,exclude_events=None):

         stim = dict()

         key= '/epocher/' + condi

        #--- get pandas data frame
         df = self.hdf_obj_get_dataframe(key)

        #--- get HDF StorerObj
        # Hstorer = self.HDFobj.get_storer(key)

        #--- get stored attributes -> epocher_parameter -> ...
         ep_param   = self.hdf_obj_get_attributes(key=key,attr='epocher_parameter')
         info_param = self.hdf_obj_get_attributes(key=key,attr='info_parameter')

        #--- get channel e.g.  'STI 014'  stimulus.events.stim_channel
         stim['channel'] = ep_param[ ep_param['marker_channel'] ]['events']['stim_channel']

        #--- get output type [onset/offset]
         stim['output'] = ep_param[ ep_param['marker_channel'] ]['events']['output']

        #---  <onset/offset> for stimulus onset/offset <rt_onset/rt_offset> for response onset/offset
         stim['marker_type'] = ep_param['marker_type']

        #--- ck for artefacts => set bads to -1
         if exclude_events :
            for kbad in ( exclude_events.keys() ):
                for idx in range( exclude_events[kbad]['tsl'].shape[-1] ) :    #df.index :
                    df['bads'][ ( exclude_events[kbad]['tsl'][0,idx] < df[stim['output']] ) & ( df[stim['output']] < exclude_events[kbad]['tsl'][1,idx] ) ] = self.idx_bad


        #--- store df with updated bads & restore user-attribute
         storer_attrs = {'epocher_parameter': ep_param,'info_parameter':info_param}
         self.hdf_obj_update_dataframe(df,key=key,reset=False,**storer_attrs)


        #--- response type idx to process
         if ep_param['response_matching'] :
            if ep_param['response_matching_type'] is None:
               rt_type_idx = self.rt_type_as_index('HIT')
            else:
               rt_type_idx = self.rt_type_as_index(ep_param['response_matching_type'])
           #--- find   response type idx
            events = df[ stim['marker_type'] ][ (df['rt_type'] == rt_type_idx) & (df['bads'] != self.idx_bad) ]
         else :
            events = df[ stim['marker_type'] ][ df['bads'] != self.idx_bad ]

        # self.verbose = True
         #if self.verbose :
         print" ---> Update Artifact & Bads Info:"
         print"      events: %d" % events.shape
         bads = df[ stim['marker_type'] ][ (df['bads']== self.idx_bad)  ]
         print"      bads  : " + str(bads.shape)
         print bads

         self.ctps_hdf_parameter['stim']      = {k: stim[k] for k in ('channel','output','marker_type')}
         self.ctps_hdf_parameter['condition'] = condi,
         self.ctps_hdf_parameter['nevent']    = events.size
         self.ctps_hdf_parameter['event_id']  = self.idx_hit

        #--- create mne event array
         import numpy as np
         stim['events'] = np.zeros((events.size, 3), dtype=np.float64)
         stim['events'][:,0]= events
         stim['events'][:,2]= self.idx_hit


         return stim,ep_param,info_param

      # jumeg_epocer.ctps_ica_steady_state_artifacts_update(fname,fname_ica=fname_ica,template_name='MEG94T',verbose=True)

      def ctps_ica_steady_state_artifacts_update(self,fname,raw=None,fname_ica=None,ica_raw=None,template_name=None,condition_list=None,
                                 filter_method="bw",remove_dcoffset=False,jobs=4,
                                 freq_ctps=None,proj=False,njobs=None,
                                 ctps_parameter = {'time_pre':-1.0,'time_post':1.0,'baseline':None},
                                 save_phase_angles=False,verbose=False):


          self.ctps_init_brain_response_data(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,template_name=template_name)

          if not freq_ctps:
            self.ctps_init_freq_bands(freq_ctps=self.steady_state_artifact_bands)
          else:
            self.ctps_init_freq_bands(freq_ctps=freq_ctps)

         #--- init/define bw-bp-filter-obj for ctps  to filter ica_raw data within freq bands
          jfi_bw = jumeg_filter(filter_method=filter_method,filter_type='bp',fcut1=None,fcut2=None,remove_dcoffset=remove_dcoffset,njobs=njobs)
          jfi_bw.sampling_frequency = self.ica_raw.info['sfreq']

         #--- init ctps_hdf_parameter
          ctps_parameter['dt'] = 4 # ~every 4 sec
          ctps_parameter['t0'] = 1
          ctps_parameter['t1'] = int ( self.raw.index_as_time( self.raw.n_times)[0] )

          self.ctps_update_ctps_hdf_parameter_time(ctps_parameter=ctps_parameter)

         #--- make evs
          ev_id = 2048

          tpoints = np.linspace(ctps_parameter['t0'],ctps_parameter['t1'],int(ctps_parameter['t1']/ctps_parameter['dt']),endpoint=False)

          # dtp= tp[1:]-tp[0:-1]

          ev = np.zeros(( tpoints.size, 3), dtype=np.float64)
          ev[:,0]= self.raw.time_as_index(tpoints)
          ev[:,2]= ev_id

          # dtsls=tsls[1:]-tsls[0:-1]

         #---  make HDF node for steady-state artifacsts
          storer_attrs = {'ctps_parameter': self.ctps_hdf_parameter}
          stst_key='/artifacts/steady-state'
          self.hdf_obj_update_dataframe(pd.DataFrame( self.ctps_freq_bands ).astype(np.int16),key=stst_key,**storer_attrs )

         #--- get fresh IC's data & filter inplace
          print " ---> get ica sources ...\n"
          ica_orig = self.ica_raw.get_sources(self.raw)

         #---for filter bands
          for idx_freq in range( self.ctps_freq_bands.shape[0] ):
              print " ---> START CTPS  Steady-State Artifact Detection Filter Band ==> %d  / %d\n" % (idx_freq+1, self.ctps_freq_bands.shape[0]+1 )
              print self.ctps_freq_bands[idx_freq]

          #--- get fresh IC's data & filter inplace
              print " ---> copy ica sources ...\n"
              ica = ica_orig.copy() # self.ica_raw.get_sources(self.raw)

              print " ---> apply filter ...\n"
              jfi_bw.fcut1 = self.ctps_freq_bands[idx_freq][0]
              jfi_bw.fcut2 = self.ctps_freq_bands[idx_freq][1]
              jfi_bw.verbose = self.verbose
              jfi_bw.apply_filter(ica._data)

              pk_dynamics_key = stst_key +'/pk_dynamics/'+ self.ctps_freq_bands_list[idx_freq]

              ica_epochs = mne.Epochs(ica,events=ev,event_id=ev_id,
                                      tmin=self.ctps_hdf_parameter['time_pre'],
                                      tmax=self.ctps_hdf_parameter['time_post'],
                                      baseline=self.ctps_hdf_parameter['baseline'],
                                      verbose=self.verbose,proj=proj)

              print " ---> Steady-State Artifact -> apply compute_ ctps ...\n"

             #--- compute CTPS
             #-------
             #--- ks_dynamics : ndarray, shape (n_sources, n_times)
             #     The kuiper statistics.
             #--- pk_dynamics : ndarray, shape (n_sources, n_times)
             #     The normalized kuiper index for ICA sources and
             #     time slices.
             #--- phase_angles : ndarray, (n_epochs, n_sources, n_times) | None
             #     The phase values for epochs, sources and time slices. If ``assume_raw``
             #    is False, None is returned.

              if save_phase_angles :
                 phase_angles_key = stst_key +'/phase_angle/'+ self.ctps_freq_bands_list[idx_freq]
                 _,pk_dynamics_f64,phase_angles_f64 = ctps( ica_epochs.get_data() )
                 self.HDFobj[ phase_angles_key ]= pd.Panel( (phase_angles_f64 * self.ctps_hdf_parameter['scale_factor'])).astype( np.int16 )

              else :
                 _,pk_dynamics_f64,_ = ctps( ica_epochs.get_data() )
                 self.HDFobj[pk_dynamics_key] = pd.DataFrame( (pk_dynamics_f64 * self.ctps_hdf_parameter['scale_factor']) ).astype( np.int16 )

              print " ---> done Steady-State Artifact -> "+ pk_dynamics_key
              print "Max : %f" % ( pk_dynamics_f64.max() )
              print "\n"

              self.HDFobj.flush()


          fhdr = self.HDFobj.filename
          self.HDFobj.close()

          return fhdr

#===============================================================
      def ctps_ica_brain_responses_update(self,fname,raw=None,fname_ica=None,ica_raw=None,template_name=None,condition_list=None,
                                 filter_method="bw",remove_dcoffset=False,njobs=None,
                                 freq_ctps=np.array([]),fmin=4,fmax=32,fstep=8,proj=False,exclude_events=None,
                                 ctps_parameter = {'time_pre':None,'time_post':None,'baseline':None},
                                 save_phase_angles=False,fif_extention=".fif",fif_postfix="ctps"):
          """

          :param fname:
          :param raw:
          :param fname_ica:
          :param ica_raw:
          :param template_name:
          :param condition_list:
          :param filter_method:
          :param remove_dcoffset:
          :param njobs:
          :param freq_ctps:
          :param fmin:
          :param fmax:
          :param fstep:
          :param proj:
          :param exclude_events:
          :param ctps_parameter:
          :param save_phase_angles:
          :param fif_extention:
          :param fif_postfix:
          :return:
          """

          self.ctps_init_brain_response_data(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,template_name=template_name)

          self.ctps_init_freq_bands(freq_ctps=freq_ctps,fmin=fmin,fmax=fmax,fstep=fstep)

          artifact_events = self.ctps_update_artifact_time_window(aev=exclude_events)

       #--- init/define bw-bp-filter-obj for ctps  to filter ica_raw data within freq bands
          jfi_bw = jumeg_filter(filter_method=filter_method,filter_type='bp',fcut1=None,fcut2=None,remove_dcoffset=remove_dcoffset,njobs=njobs)
          jfi_bw.sampling_frequency = self.ica_raw.info['sfreq']


          epocher_condition_list = self.ctps_update_hdf_condition_list(condition_list)

          for condi in epocher_condition_list:
              self.hdf_obj_reset_key('/ctps/'+ condi)

       #--- get fresh IC's data & filter inplace
          print " ---> get ica sources ...\n"
          ica_orig = self.ica_raw.get_sources(self.raw)

       #---for filter bands
          for idx_freq in range( self.ctps_freq_bands.shape[0] ):
              print " ---> START CTPS  Filter Band ==> %d  / %d\n" % (idx_freq+1, self.ctps_freq_bands.shape[0]+1 )
              print self.ctps_freq_bands[idx_freq]

          #--- get fresh IC's data & filter inplace
              print " ---> copy ica sources ...\n"
              ica = ica_orig.copy() # self.ica_raw.get_sources(self.raw)

              print " ---> apply filter ...\n"
              jfi_bw.fcut1 = self.ctps_freq_bands[idx_freq][0]
              jfi_bw.fcut2 = self.ctps_freq_bands[idx_freq][1]
              jfi_bw.verbose = self.verbose
              jfi_bw.apply_filter(ica._data)


          #--- for epocher condition
              for condi in epocher_condition_list:

                  print " ---> START condition : " + condi + " CTPS  Filter Band ==> %d  / %d \n" % (idx_freq+1, self.ctps_freq_bands.shape[0] )
                  ctps_key = '/ctps/' + condi

                  #stim,ep_param,info_param = self.ctps_update_condition_parameter(condi,artifact_events)

                  #self.ctps_update_ctps_hdf_parameter_time(ctps_parameter=ctps_parameter,ep_param=ep_param)

                  if not( ctps_key in self.HDFobj.keys() ):

                     print"---> NEW HDF key: " + ctps_key

                     stim,ep_param,info_param = self.ctps_update_condition_parameter(condi,artifact_events)
                     self.ctps_update_ctps_hdf_parameter_time(ctps_parameter=ctps_parameter,ep_param=ep_param)

                     self.HDFobj[ctps_key] = pd.DataFrame( self.ctps_freq_bands ).astype(np.int16)

                     Hstorer   = self.HDFobj.get_storer(ctps_key)
                     Hstorer.attrs['ctps_hdf_parameter'] = self.ctps_hdf_parameter
                     self.HDFobj[ctps_key+'/events'] = pd.Series( stim['events'][:,0] ).astype(np.int32)

                      # d=np.zeros( [ len(self.ctps_freq_bands_list ),248,1000 ] ).astype(np.int16)


                     self.HDFobj.flush()

                     print"--->done update storer: " + ctps_key

                  else:
                     ev = self.HDFobj.get(ctps_key+'/events')
                     Hstorer = self.HDFobj.get_storer(ctps_key)
                     self.ctps_hdf_parameter = Hstorer.attrs.ctps_hdf_parameter

                     stim['events'] = np.zeros(( ev.size, 3), dtype=np.float64)
                     stim['events'][:,0]= ev
                     stim['events'][:,2]= self.idx_hit

                  #pk_dynamics_key = ctps_key +'/pk_dynamics'
                  pk_dynamics_key = ctps_key +'/pk_dynamics/'+ self.ctps_freq_bands_list[idx_freq]

                 #--- make epochs
                  # print self.ctps_hdf_parameter
                  ica_epochs = mne.Epochs(ica,events=stim['events'],picks=self.ica_picks,
                                          event_id=self.ctps_hdf_parameter['event_id'],
                                          tmin=self.ctps_hdf_parameter['time_pre'],
                                          tmax=self.ctps_hdf_parameter['time_post'],
                                          baseline=self.ctps_hdf_parameter['baseline'],verbose=self.verbose,proj=proj)
                 #--- compute CTPS
                 #-------
                 #--- ks_dynamics : ndarray, shape (n_sources, n_times)
                 #     The kuiper statistics.
                 #--- pk_dynamics : ndarray, shape (n_sources, n_times)
                 #     The normalized kuiper index for ICA sources and
                 #     time slices.
                 #--- phase_angles : ndarray, (n_epochs, n_sources, n_times) | None
                 #     The phase values for epochs, sources and time slices. If ``assume_raw``
                 #    is False, None is returned.

                  print " ---> apply compute_ctps ...\n"

                  if save_phase_angles :
                     phase_angles_key = ctps_key +'/phase_angle/'+ self.ctps_freq_bands_list[idx_freq]
                     _,pk_dynamics_f64,phase_angles_f64 = ctps( ica_epochs.get_data() )
                     #_,pk_dynamics_f64,phase_angles_f64 = ctps.compute_ctps( ica_epochs.get_data() )

                     self.HDFobj[ phase_angles_key ]= pd.Panel( (phase_angles_f64 * self.ctps_hdf_parameter['scale_factor'])).astype( np.int16 )

                  else :
                     _,pk_dynamics_f64,_ = ctps( ica_epochs.get_data() )
                   #_,pk_dynamics_f64,_ = ctps.compute_ctps( ica_epochs.get_data() )

                  self.HDFobj[pk_dynamics_key] = pd.DataFrame( (pk_dynamics_f64 * self.ctps_hdf_parameter['scale_factor']) ).astype( np.int16 )

                  self.HDFobj.flush()

                 # print self.HDFobj[pk_dynamics_key].transpose().max()

          fhdr = self.HDFobj.filename
          self.HDFobj.close()

          return fhdr



      def ctps_ica_brain_responses_select(self,fhdf=None,fname=None,raw=None,condition_list=None,template_name=None,
                                          ctps_pkd_theshold=None,ctps_pkd_min_number_of_timeslices=None):
          """

          :param fhdf:
          :param fname:
          :param raw:
          :param condition_list:
          :param template_name:
          :param ctps_pkd_theshold:
          :param ctps_pkd_min_number_of_timeslices:
          :return:
          """

         #--- get HDF obj
          if fhdf:
             self.HDFobj = pd.HDFStore(fhdf)
          else:
             if raw:
                self.raw = raw
             self.template_name = template_name

             self.HDFobj = self.hdf_obj_open(fname=fname,raw=self.raw)

          ctps_condition_list = self.ctps_update_hdf_condition_list(condition_list,node='/ctps/')


          if ctps_pkd_theshold :
             self.ctps_pkd_theshold = ctps_pkd_theshold

          if ctps_pkd_min_number_of_timeslices:
             self.ctps_pkd_min_number_of_timeslices = ctps_pkd_min_number_of_timeslices


          #print ctps_condition_list

          condi_ics_dict = {}

          for condi in ctps_condition_list:
              ctps_key= '/ctps/' + condi

              ics_key= ctps_key+'/ics_selected'

              HST = self.HDFobj.get_storer(ctps_key)

              ics = np.zeros(HST.attrs['ctps_hdf_parameter']['ncomp'], dtype=np.bool)

              for node in self.HDFobj.get_node(ctps_key+'/pk_dynamics'):
                  df_pkd = self.HDFobj.select(node._v_pathname).copy()

                  df_pkd[ df_pkd < self.ctps_pkd_theshold] = 0
                  df_pkd[ df_pkd > 0 ] = 1
                  d1 = np.array( np.where(df_pkd > 0))
                  if not d1[1].shape[0]:
                     continue

                 #-- ck ics debouncing  -> for more than X tsl in phase
                  for ic in np.unique( d1[0,:] ):
                      if ics[ic]: continue

                     # print"Found: %d  " %(ic)

                      tsl_seq = d1[1,np.where(d1[0] == ic)].copy().flatten()
                      tsl_list= np.array_split(tsl_seq,np.where(np.diff(tsl_seq)!=1)[0]+1)
                      #print"TSL list: "
                      #print tsl_list
                      for tsls in tsl_list:
                         # print tsls
                          if tsls.size >= self.ctps_pkd_min_number_of_timeslices:
                             ics[ic]=True
                             break

              self.HDFobj[ics_key] = pd.Series( ics ).astype( np.bool )

              condi_ics_dict[condi] = np.array(np.where(ics))

              self.HDFobj.flush()

                  # self.HDFobj[ics_key][ics_idx] = True

          ics_global = np.array([])

          for c in condi_ics_dict.keys():
              print "---> "+ c   +":  %d" % (condi_ics_dict[c].size)
              print condi_ics_dict[c]
              print
              ics_global = np.unique( np.append( ics_global,condi_ics_dict[c] ) )



 #--- store df with updated bads & restore user-attribute
          storer_attrs = {'ics_parameter': {'nelem':ics_global.size,'ctps_pkd_theshold':self.ctps_pkd_theshold,'ctps_pkd_min_number_of_timeslices':self.ctps_pkd_min_number_of_timeslices}}
          self.hdf_obj_update_dataframe(pd.Series( ics_global ).astype( np.int16 ),key='/ics_global',reset=True,**storer_attrs )


          print"ICs GLOBAL : %d"  % (ics_global.size)
          print ics_global


          fhdf = self.HDFobj.filename
          self.HDFobj.close()

          return fhdf


      def ctps_ica_brain_responses_clean(self,fname,raw=None,fname_ica=None,ica_raw=None,fhdf=None,template_name=None,
                                         condition_list=None, njobs=4,fif_extention=".fif",fif_postfix="ctps",
                                         clean_global={'save_raw':False,'save_epochs':False,'save_evoked':False},
                                         clean_condition={'save_raw':False,'save_epochs':False,'save_evoked':False},
                                         do_run=False,verbose=False,epocher_proj=None):
          """

          clean ica brainresponses ics_global
          clean ctps/condition/ics_selected  -> where(ics ==True)

          :param fname:
          :param raw:
          :param fname_ica:
          :param ica_raw:
          :param fhdf:
          :param condition_list:
          :param njobs:
          :param fif_extention:
          :param fif_postfix:
          :param clean_global:
          :param clean_condition:
          :param save_epochs:
          :param save_average:
          :param save_raw:
          :param do_run:
          :param verbose:
          :return: fhdf
          """
          # add ctps ica artifact steady state e.g. 23,8 Hz,  60z
          # TODO: split in fkt cleaning global & condition

          from jumeg.jumeg_base import jumeg_base

          self.ctps_init_brain_response_clean_data(fname,raw=raw,fname_ica=fname_ica,ica_raw=ica_raw,fhdf=fhdf,template_name=template_name)

          ctps_condition_list = self.ctps_update_hdf_condition_list(condition_list,node='/ctps/')

         #--- clean global
          if any( clean_global.values() ) :

            #--- get global Ics from  pd.Series
             ics       = self.HDFobj['ics_global']
             ica_picks = ics.values

             print "CTPs Clean Global RAW"
             raw_ctps_clean = self.ica_raw.apply(self.raw,include=ica_picks,n_pca_components=None,copy=True)

             fout = jumeg_base.get_fif_name(fname,postfix=fif_postfix+'-raw',extention=fif_extention)

             if clean_global['save_raw'] :
                raw_ctps_clean.save(fout,overwrite=True)
                print"---> save raw ctps global clean\n ---> " + fout +"\n"

             if ( clean_global['save_epochs'] or clean_global['save_evoked'] ):

                for condi in ctps_condition_list:
                    print "---> Init parameter global : " + condi
                    print " --> save epochs : %r" %(clean_global['save_epochs'])
                    print " --> save evoked : %r" %(clean_global['save_evoked'])
                    print " --> ICs global count : %d" %(ica_picks.size)

                    if not ica_picks.size: continue

                 #--- read epocher condi parameter & info
                    epocher_key   = '/epocher/' + condi

                    ep_param      = self.hdf_obj_get_attributes(key=epocher_key,attr='epocher_parameter')

                    #stimulus_info = self.hdf_obj_get_attributes(key=epocher_key,attr='info_parameter')

                    if self.verbose:
                       print " --> Epocher Parameter: "
                       print ep_param
                       #print " --> Stimulus Info :"
                       print "\n\n"

                 #--- make event array from < ctps condition events>
                    ctps_key= '/ctps/' + condi
                    ev_tsl  = self.hdf_obj_get_dataframe(key= ctps_key+'/events')
                    ev      = np.zeros(( ev_tsl.size, 3), dtype=np.int32)
                    try:
                        ev[:,2] = ep_param['stimulus']['event_id']
                    except:
                        ev[:,2] = ep_param['stimulus']['event_id'][0]
                    else:
                        ev[:,2] = 1

                    ev[:,0] = ev_tsl

                   #--- get global epochs
                    for k in ('time_pre','time_post','reject'):
                        if ep_param.has_key(k): continue
                        ep_param[k] = None

                    dout = mne.Epochs(raw_ctps_clean,events=ev,event_id=dict( condi = int(ev[0,2]) ),
                                      tmin=ep_param['time_pre'],tmax=ep_param['time_post'],baseline=ep_param['baseline'],
                                      reject=ep_param['reject'],
                                      verbose=self.verbose,proj=epocher_proj)


                  #--- global condi epochs
                    if clean_global['save_epochs']:
                       fout = jumeg_base.get_fif_name(fname,postfix=fif_postfix+'-'+condi+'-epo',extention=fif_extention)
                       dout.save(fout)
                       print"---> save epochs ctps global condition clean: " + condi +"\n"+ fout +"\n"

                  #---  global evoked ( average condi epochs )
                    if clean_global['save_evoked']:
                       print "Global Evoked (Averager) :" + condi
                       fout = jumeg_base.get_fif_name(fname,postfix=fif_postfix+'-'+condi+'-ave',extention=fif_extention)
                       dout = dout.average()
                       dout.save(fout)
                       print"---> save evoked ctps global condition clean: " + condi +"\n"+ fout + "\n"

                    print"---> Done ctps global condition clean: " + condi + "\n"
         #------
          if any( clean_condition.values() ):
             print "CTPs Clean Condition RAW"

             for condi in ctps_condition_list:
                 print "---> Init parameter condition : " + condi
                 print " --> save epochs : %r" %(clean_condition['save_epochs'])
                 print " --> save evoked : %r" %(clean_condition['save_evoked'])

                 ctps_key = '/ctps/' + condi

                 ics       = self.HDFobj[ctps_key+'/ics_selected']
                 ica_picks = np.array( np.where( ics ),dtype=np.int16).flatten()
                 print" ICs counts: %d" %( ica_picks.size )
                 print "ICS:"
                 print ica_picks

                 if not ica_picks.size: continue

                 raw_ctps_clean = self.ica_raw.apply(self.raw,include=ica_picks,n_pca_components=None,copy=True)

                 fout = jumeg_base.get_fif_name(fname,postfix=fif_postfix+'_co-'+ condi+'-raw',extention=fif_extention)

                 if clean_condition['save_raw']:
                    raw_ctps_clean.save(fout) # mne
                    print"---> save raw ctps condition clean: " + condi +"\n"+ fout +"\n"

                 if ( clean_condition['save_epochs'] or clean_condition['save_evoked'] ):

                  #--- read epocher condi parameter & info
                    epocher_key = '/epocher/' + condi
                    ep_param    = self.hdf_obj_get_attributes(key=epocher_key,attr='epocher_parameter')

                    if self.verbose:
                       print " --> CTPs Condition Epocher Parameter: " + condi
                       print ep_param
                       #print " --> Stimulus Info :"
                       print "\n\n"

                 #--- make event array from < ctps condition events>
                    ev_tsl  = self.hdf_obj_get_dataframe(key= ctps_key+'/events')
                    ev      = np.zeros(( ev_tsl.size, 3), dtype=np.int32)
                    try:
                        ev[:,2] = ep_param['stimulus']['event_id']
                    except:
                        ev[:,2] = ep_param['stimulus']['event_id'][0]
                    else:
                        ev[:,2] = 1

                    ev[:,0] = ev_tsl

                   #--- get global epochs
                    for k in ('time_pre','time_post','reject'):
                        if ep_param.has_key(k): continue
                        ep_param[k] = None

                    dout = mne.Epochs(raw_ctps_clean,events=ev,event_id=dict( condi = int( ev[0,2] ) ),
                                      tmin=ep_param['time_pre'],tmax=ep_param['time_post'],baseline=ep_param['baseline'],
                                      reject=ep_param['reject'],
                                      verbose=self.verbose,proj=epocher_proj)

                  #--- global condi epochs
                    if clean_condition['save_epochs']:
                       fout = jumeg_base.get_fif_name(fname,postfix=fif_postfix+'_co-'+condi+'-epo',extention=fif_extention)
                       dout.save(fout)
                       print"---> save epochs ctps condition clean: " + condi +"\n"+ fout +"\n"

                  #---  global evoked ( average condi epochs )
                    if clean_condition['save_evoked']:
                       print "Global Evoked (Averager) :" + condi
                       fout = jumeg_base.get_fif_name(fname,postfix=fif_postfix+'_co-'+condi+'-ave',extention=fif_extention)
                       dout = dout.average()
                       dout.save(fout)
                       print"---> save evoked ctps condition clean: " + condi +"\n"+ fout +"\n"

                    print"---> Done ctps condition clean: " + condi + "\n"



          fhdf = self.HDFobj.filename
          self.HDFobj.close()

          return fhdf


         #------


         #--- plot x time
         # x=np.arange(pkd.shape[1],dtype=np.float32)
         # HST.attrs['ctps_hdf_parameter']['time_pre']
         # x/=HST.attrs['ctps_hdf_parameter']['sfreq']
         # x+=HST.attrs['ctps_hdf_parameter']['time_pre']
         # pylab.imshow(pkd,aspect='auto')
         # pylab.imshow(pkd.T,aspect='auto',extent=[x[0],x[-1],1,pkd.shape[0]] )



#import matplotlib.pyplot as plt

#times = epochs.times * 1e3
#plt.figure()
#plt.title('single trial surrogates')
#plt.imshow(surrogates[conditions.argsort()], origin='lower', aspect='auto',
#           extent=[times[0], times[-1], 1, len(surrogates)],
#           cmap='RdBu_r')
#plt.xlabel('Time (ms)')
#plt.ylabel('Trials (reordered by condition)')

#plt.figure()
#plt.title('Average EMS signal')

#mappings = [(k, v) for k, v in event_ids.items() if v in conditions]
#for key, value in mappings:
#    ems_ave = surrogates[conditions == value]
#    ems_ave *= 1e13
#    plt.plot(times, ems_ave.mean(0), label=key)
#plt.xlabel('Time (ms)')
#plt.ylabel('fT/cm')
#plt.legend(loc='best')



# ave=mne.read_evokeds(fif)
#  ave[0].plot()
# ave.data
#ave[0].data.dtype=np.float32
#ave[0].data..astype('int16').tofile(filename)

jumeg_epocher_ctps = JuMEG_Epocher_CTPS()
