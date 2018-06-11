'''Class JuMEG_Epocher_Events

Class to extract event/epoch information and save to hdf5

Author:
         Frank Boers     <f.boers@fz-juelich.de>
----------------------------------------------------------------
extract mne-events per condition, save to HDF5 file

#--- example via obj:
from jumeg.epocher.jumeg_epocher import jumeg_epocher
epocher= {"template_name": "LDAEP",
          "fif_extention": ".fif",
          "verbose":True,
          "save": True}

fname=test.fif
raw=None

fname,raw,fhdf = jumeg_epocher.apply_events_to_hdf(fname, raw=raw,**epocher)

---> update 10.01.2017 FB
     check event-code/conditions for none existing

'''


import os
import warnings
import numpy as np
import pandas as pd
# import matplotlib.pylab as pl
import mne

from jumeg.jumeg_base import jumeg_base

from jumeg.epocher.jumeg_epocher_hdf import JuMEG_Epocher_HDF

class JuMEG_Epocher_Events(JuMEG_Epocher_HDF):
    def __init__ (self):

        super(JuMEG_Epocher_Events, self).__init__()

        self.__rt_type_list             = ['MISSED', 'TOEARLY', 'WRONG', 'HIT']
        self.__data_frame_stimulus_cols = ['id','onset','offset']
        self.__data_frame_response_cols = ['rt_type','rt','rt_id','rt_onset','rt_offset','rt_index','rt_counts','bads','selected','weighted_selected']

        self.__stat_postfix = '-epocher-stats.csv'

        self.__idx_bad = -1

#---
    def __get_idx_bad(self):
        return self.__idx_bad
    idx_bad = property(__get_idx_bad)

#---
    def __get_data_frame_stimulus_cols(self):
        return self.__data_frame_stimulus_cols
    def __set_data_frame_stimulus_cols(self,v):
        self.__data_frame_stimulus_cols = v
    data_frame_stimulus_cols = property(__get_data_frame_stimulus_cols,__set_data_frame_stimulus_cols)

#---
    def __get_data_frame_response_cols(self):
        return self.__data_frame_response_cols
    def __set_data_frame_response_cols(self,v):
        self.__data_frame_response_cols = v
    data_frame_response_cols = property(__get_data_frame_response_cols,__set_data_frame_response_cols)


#--- rt_type list: 'MISSED', 'TOEARLY', 'WRONG', 'HIT'
    def __get_rt_type_list(self):
        return self.__rt_type_list
    rt_type_list = property(__get_rt_type_list)

#--- rt type index: 'MISSED', 'TOEARLY', 'WRONG', 'HIT'
    def rt_type_as_index(self,s):
        return self.__rt_type_list.index( s.upper() )

    def __get_idx_missed(self):
        return self.__rt_type_list.index( 'MISSED')
    idx_missed = property(__get_idx_missed)

    def __get_idx_toearly(self):
        return self.__rt_type_list.index( 'TOEARLY')
    idx_toearly = property(__get_idx_toearly)

    def __get_idx_wrong(self):
        return self.__rt_type_list.index( 'WRONG')
    idx_wrong = property(__get_idx_wrong)

    def __get_idx_hit(self):
        return self.__rt_type_list.index( 'HIT')
    idx_hit = property(__get_idx_hit)


#--- events stat file (output as csv)
    def __set_stat_postfix(self, v):
         self.__stat_postfix = v
    def __get_stat_postfix(self):
         return self.__stat_postfix
    stat_postfix = property(__get_stat_postfix,__set_stat_postfix)

#---
    def events_find_events(self,raw,**param):
        """
        some how calls <mne.find_events()>
        input:
             raw obj,
             e.g. parameters:
              {'event_id': 40, 'and_mask': 255,
               'events': {'consecutive': True, 'output':'step','stim_channel': 'STI 014',
               'min_duration':0.002,'shortest_event': 2,'mask': 0}
              }

         return:
                pandas data-frame with epoch event structure for stimulus or response channel
                    id       : event id
                    offset   : np array with TSL event code offset
                    onset    : np array with TSL event code onset
                    counts   : number of events
                    bads     : np.array with index of bad events
                    #=> RESPONSE MATCHING
                    #rt_type     : NAN => np array with values  MISSED=0 ,TOEARLY=1,WRONG=2,HIT=3
                    #rt      : NAN => np array with reaction time [TSL]
                    #rt_onset: NAN => np array with response onset [TSL]
                    #rt_id       : NAN => np array with response key / id

                dict() with event structure for stimulus or response channel
                sfreq    : sampling frequency  => raw.info['sfreq']
                duration : {mean,min,max}  in TSL
                system_delay_is_applied : True/False
                 --> if true <system_delay_ms> converted to TSLs and added to the TSLs in onset,offset
                (TSL => timeslices , your samples)
        """
        if raw is None:
           print "ERROR in  <get_event_structure: No raw obj \n"
           return None,None
       #---
       # import pandas as pd   done
        df = pd.DataFrame(columns = self.data_frame_stimulus_cols)

        ev_id_idx = np.array([])
        ev_onset  = np.array([])
        ev_offset = np.array([])
       #---
        events           = param['events'].copy()
        events['output'] = 'step'
        ev               = mne.find_events(raw, **events) #-- return int64

       #--- apply and mask e.g. 255 get the first 8 bits in Trigger channel
        if param['and_mask']:
           ev[:, 1:] = np.bitwise_and(ev[:, 1:], param['and_mask'])
           ev[:, 2:] = np.bitwise_and(ev[:, 2:], param['and_mask'])

        ev_onset  = np.squeeze( ev[np.where( ev[:,2] ),:])  # > 0
        ev_offset = np.squeeze( ev[np.where( ev[:,1] ),:])

        if param['event_id']:
           ev_id = jumeg_base.str_range_to_numpy(param['event_id'],exclude_zero=True)
           
           #ev_onset  = np.squeeze( ev[np.where( np.in1d( ev[:,2],ev_id ) ), :])
           #ev_offset = np.squeeze( ev[np.where( np.in1d( ev[:,1],ev_id ) ), :])

          #--- check if code in events
           if ( ev_id in np.unique(ev[:, 2]) ):
              ev_id_idx = np.squeeze( np.where( np.in1d( ev_onset[:,2],ev_id )))
              if ( ev_id_idx.size > 0 ):
                   ev_onset = ev_onset[ ev_id_idx,:]
                   ev_offset= ev_offset[ev_id_idx,:]
              else:
                  print'Warning => No such event code(s) found (ev_id_idx) -> event: ' + str( param['event_id'] )
                  return None,None
           else:
               print'Warning => No such event code(s) found (ev_id) -> event: ' + str(param['event_id'])
               return None,None

       #---- use all event ids
        if ( ev_onset.size == 0 ):
            print'Warning => No such event code(s) found -> event: ' + str(param['event_id'])
            return None,None

       #--- apply system delay if is defined e.g. auditory take`s 20ms to subjects ears
        if param['system_delay_ms']:
           system_delay_tsl = raw.time_as_index( param['system_delay_ms']/1000 ) # calc in sec
           ev_onset[:, 0] += system_delay_tsl
           ev_offset[:, 0]+= system_delay_tsl
           system_delay_is_applied = True
        else:
           system_delay_is_applied = False
       
        
       #-- avoid invalid index/dimension error if last offset is none
        df['id']     = ev_onset[:,2]
        df['onset']  = ev_onset[:,0]
        df['offset'] = np.zeros( ev_onset[:,0].size,dtype=np.long )
        div = np.zeros( ev_offset[:,0].size )
        try:
            if ( ev_onset[:,0].size >= ev_offset[:,0].size ):
               div = ev_offset[:,0] - ev_onset[:ev_offset[:,0].size,0]
               df['offset'][:ev_offset[:,0].size] = ev_offset[:,0]
        
            else:
               idx_max = ev_offset[:,0].size
               div = ev_offset[:,0] - ev_onset[:idx_max,0]    
               df['offset'][:] = ev_offset[:idx_max,0]
        except:
            assert "ERROR dims onset offset will not fit\n"
            
            print ev_onset[:,0].size
        print ev_offset[:,0].size
        
        
        return df,dict( {
                         'sfreq'        : raw.info['sfreq'],
                         'duration'     :{'mean':np.rint(div.mean()),'min':div.min(),'max':div.max()},
                         'system_delay_is_applied' : system_delay_is_applied
                         } )

#---
    def events_response_matching(self,raw,stim_df=None,resp_df=None, **param ):
        """
        matching correct responses with respect to <stimulus channel> <output type> (onset,offset)
        input:
                stim_df = <stimulus channel data frame>
                res_df  = <response channel data frame>
                param :  parameter dict
                   e.g.:{ 'response':{'counts':1,'window':[0,0.5],'event_id':2,'include_early_ids':[1,4],'events':{'output':'onset'} },
                         'stimulus':{'events':{'output'.'onset'}} }
        return:
                <stimulus data frame>  pandas DataFrame obj
                with added cols
                rt_type       : response type MISSED,EARLY,WRONG,HIT
                rt        : response time as tsl (RT)
                rt_id         : event id button press code
                rt_onset      : response onset  [tsl]
                rt_offset     : response offset [tsl]
                rt_index      : index in response onset / offset
                rt_counts     : number of responses in <response time window>
                bads          : flag for bad epochs; e.g. later use in combination with ecg/eog events
        """
      #--- ck errors
        err_msg = None

        if raw is None:
           err_msg = "ERROR in <apply_response_matching> : no raw obj \n"
        if (stim_df is None):
           err_msg +="\nERROR no Stimulus-Data-Frame obj. provided\n"
        if (resp_df is None):
           err_msg +="\nERROR no Response-Data-Frame obj. provided\n"
        if (param is None):
           err_msg +="\nERROR no Parameter obj. provided\n"
       #--- ck RT window range
        if ( param['response']['window'][0] >= param['response']['window'][1] ):
            err_msg += "ERROR in parameter response windows\n"

        if err_msg:
           print "ERROR in <apply_response_matching>\n" + err_msg +"\n\n"
           return None

       #--- extend stimulus data frame
        for x in self.data_frame_response_cols :
            stim_df[x]= 0 #np.NaN

       #--- convert rt window [ms] into tsl
        (r_window_tsl_start, r_window_tsl_end ) = raw.time_as_index( param['response']['window'] );

       #--- get respose code -> event_id [int or string] as np array
        r_event_id = jumeg_base.str_range_to_numpy( param['response']['event_id'] )

       #--- ck if any toearly-id is defined, returns None if not
        r_event_id_toearly = jumeg_base.str_range_to_numpy( param['response']['include_early_ids'] )

       #--- get output type: onset or offset
        stim_output_type = param['stimulus']['events']['output']
        resp_output_type = param['response']['events']['output']

       #--- loop for all stim events
        ridx = 0
       #--- get rt important part of respose df
        RESP_TSLs = resp_df[resp_output_type]

        for idx in stim_df.index :
            st_tsl_onset   = stim_df[stim_output_type][idx]
            st_window_tsl0 = stim_df[stim_output_type][idx] + r_window_tsl_start
            st_window_tsl1 = stim_df[stim_output_type][idx] + r_window_tsl_end
            # st_id          = stim_df['id'][idx]

          #--- check for to TOEARLY responses
            toearly_tsl0 = st_tsl_onset
            toearly_tsl1 = st_window_tsl0
          #--- look for part of response dataframe ...
            res_df_early = resp_df[(toearly_tsl0 <= RESP_TSLs ) & ( RESP_TSLs < toearly_tsl1 )]
            if res_df_early.index.size > 0 :
               if not res_df_early.isin( r_event_id_toearly ).all :
                      ridx                   = res_df_early.index[0]
                      stim_df.rt_counts[idx] = res_df_in.index.size
                      stim_df.rt_onset[idx]  = resp_df.onset[ridx]
                      stim_df.rt_offset[idx] = resp_df.offset[ridx]
                      stim_df.rt_id[idx]     = resp_df.id[ridx]
                      stim_df.rt_index[idx]  = ridx
                      stim_df.rt_type[idx]   = self.idx_toearly
                      continue

          #--- find index of responses from window-start till end of res_event_type array [e.g. onset / offset]
            # res_df_in = resp_df[ ( st_window_tsl0 <= RESP_TSLs ) & ( RESP_TSLs <= st_window_tsl1) ]
            resp_df_in_idx = resp_df[ ( st_window_tsl0 <= RESP_TSLs ) & ( RESP_TSLs <= st_window_tsl1) ].index

          #--- MISSED response
            if not np.any( resp_df_in_idx ): 
               continue

          #--- WRONG or HIT;
           # if res_df_in.index.size > 0 :
           #    ridx = res_df_in.index[0]
            if resp_df_in_idx.size > 0 :
               ridx = resp_df_in_idx[0]
               stim_df.rt_counts[idx] = resp_df_in_idx.size
               stim_df.rt_onset[idx]  = resp_df.onset[ridx]
               stim_df.rt_offset[idx] = resp_df.offset[ridx]
               stim_df.rt_type[idx]   = self.idx_wrong
               stim_df.rt_id[idx]     = resp_df.id[ridx]
               #stim_df.rt_id[idx]     = resp_df.id[resp_df_in_idx].max()
               
               #stim_df.rt_id[idx]     = np.bitwise_and(resp_df.id[resp_df_in_idx], 72).max()
               stim_df.rt_index[idx]  = ridx
               
          #--- HIT; ck number of responses; ck pressed buttons; wrong if more than count
               if ( stim_df.rt_counts[idx] <= param['response']['counts'] ):
                  #if np.all( res_df_in.id.isin( r_event_id ) ) :
                  if np.all( resp_df.id[ resp_df_in_idx].isin( r_event_id ) ) :
                     stim_df.rt_type[idx] = self.idx_hit
                
              # if (stim_df.rt_id[idx] <1) and(stim_df.rt_type[idx]  == self.idx_wrong):
              #      print"!!ERROR"
              #      print resp_df_in_idx
              #      print resp_df.onset[resp_df_in_idx]
              #      print resp_df.id[resp_df_in_idx]
              #      assert "erroe rt"
   #--- MISSED response
            else: 
               continue

      #---  calc reaction time (rt in tsl)
        if stim_output_type == 'offset' :
           sto = stim_df.offset
        else:
           sto = stim_df.onset
        if resp_output_type == 'offset' :
           rto = stim_df.rt_offset
        else:
           rto = stim_df.rt_onset

        stim_df.rt = rto - sto
        stim_df.rt[ (stim_df.rt_type == self.idx_missed) ] = 0
       #---
        if self.verbose:
           for kidx in range( len( self.rt_type_list ) ) :
               print "\n\n---> Stimulus DataFrame Type: " + self.rt_type_list[kidx]
               ddf = stim_df[ stim_df.rt_type == kidx ]
               if ddf.index.size > 0 :
                  print ddf.describe()
               else :
                  print "---> EMPTY"
               print"---------------------------"
           print "\n\n"

        #import sys
        #char = sys.stdin.read(1)
        return stim_df

 #---
    def events_store_to_hdf(self,raw,condition_list=None,overwrite_hdf=False):
        """
        find & store epocher data to hdf5:
        -> readding parameter from epocher template file
        -> find events from raw-obj using mne.find_events
        -> apply response matching if is true
        -> save results in pandas dataframes & HDF fromat
        input:
             raw : raw obj
             condition_list: list of conditions to process
                             select special conditions from epocher template
                             default: condition_list=None , will process all
             overwrite_hdf : flag for overwriting output HDF file
                             default: overwrite_hdf=True

         return:
                 HDF filename
        """
        import pandas as pd

       #---  init obj
       # overwrite_hdf=False
        self.hdf_obj_init(raw=raw,overwrite=overwrite_hdf)

       #--- condi loop
        for condi, param, in self.template_data.iteritems():
         
          #--- check for real condition
            if condi == 'default': continue
          
          #--- check for condition in list
            if condition_list :
               if condi not in condition_list: continue
                   
            print '===> start condition: '+ condi
           #--- update & merge condi parameter with defaults
            parameter = self.template_data['default'].copy()
            parameter = self.template_update_and_merge_dict(parameter,param)
           #--- stimulus init dict's & dataframes
            stimulus_info          = dict()
            stimulus_data_frame    = None

            if self.verbose:
               print'===>EPOCHER  Template: %s Condition:%s' %(self.template_name,condi)
               print'find events and epochs,  generate epocher output HDF5'
               print"\n---> Parameter :"
               print parameter
               print"\n"
           #--- select stimulus channel e.g. "stimulus" -> STIM 014 or "response" ->STIM 013
            if parameter['stimulus_channel'] in ['stimulus','response'] :
               print"STIMULUS CHANNEL -> find events: "+ condi +" ---> "+ parameter['stimulus_channel']
               if self.verbose:
                  print "---> Stimulus Channel Parameter:"
                  print parameter[ parameter['stimulus_channel'] ]
                  print"\n\n"
              #--- get stimulus channel epochs from events as pandas data-frame
               stimulus_data_frame,stimulus_info = self.events_find_events(raw,**parameter[ parameter['stimulus_channel'] ])

               if self.verbose:
                  print "---> Stimulus Epocher Events Data Frame [stimulus channel]: "+ condi
                  print stimulus_data_frame
                  print"\n\n"

               if stimulus_data_frame is None: continue

              #--- RESPONSE Matching task
              #--- match between stimulus and response
              #--- get all response events for condtion e.g. button press 4
               if parameter['response_matching'] :
                  print"RESPONSE MATCHING -> matching stimulus & response channel: " + condi
                  print"stimulus channel : " + parameter['stimulus_channel']
                  print"response channel : " + parameter['response_channel']

                 #--- look for all responses => 'event_id' = None
                  res_param = parameter[ parameter['response_channel'] ].copy()
                  res_param['event_id'] = None

                 #--- get epochs from events as pandas data-frame
                  response_data_frame,response_info = self.events_find_events(raw,**res_param)

                  if self.verbose:
                     print "---> Response Epocher Events Data Frame [respnse channel] : " + parameter['response_channel']
                     print response_data_frame
                     print"\n\n"

                 #--- update stimulus epochs with response matching
                  stimulus_data_frame = self.events_response_matching(raw,stimulus_data_frame,response_data_frame,**parameter )

                 #--- store dataframe to HDF format

               else:
                  stimulus_data_frame['bads']= np.zeros_like( stimulus_data_frame['onset'],dtype=np.int8 )
              
              #--- for later mark selected epoch as 1
               stimulus_data_frame['selected']         = np.zeros_like( stimulus_data_frame['onset'],dtype=np.int8 )
               stimulus_data_frame['weighted_selected']= np.zeros_like( stimulus_data_frame['onset'],dtype=np.int8 ) 
               
               key = '/epocher/'+condi
               storer_attrs = {'epocher_parameter': parameter,'info_parameter':stimulus_info}
               self.hdf_obj_update_dataframe(stimulus_data_frame.astype(np.int32),key=key,**storer_attrs )

       #--- write stat info into hdf and as csv/txt
        df_stats = self.events_condition_stats(save=True)
        
        key = '/conditions/statistic/'
        storer_attrs = {'epocher_parameter': parameter,'info_parameter':stimulus_info}
        self.hdf_obj_update_dataframe(df_stats.astype(np.float32),key=key,**storer_attrs )
        
        fhdf= self.HDFobj.filename
        self.HDFobj.close()

        print" ---> DONE save epocher data into HDF5 :"
        print"  --> " + fhdf +"\n\n"
        return fhdf

#--
    def events_condition_stats(self,save=False):
        """
        return:
        <pandas data frame>
        """
        import pandas as pd

       #--- ck error

        if not self.hdf_obj_is_open():
           assert "ERROR no HDF obj open\n"
           return

       #---

        #cols = ['EvID','Hit','Wrong', 'TOEarly', 'Missed', 'RTmean','RTmedian','RTstd', 'RTmin', 'RTmax']
        cols = ['STID','RTID','Hit','Wrong', 'TOEarly', 'Missed', 'RTmean','RTmedian','RTstd', 'RTmin', 'RTmax']
        #Index([u'id', u'onset', u'offset', u'rt_type', u'rt', u'rt_id', u'rt_onset', u'rt_offset', u'rt_index', u'rt_counts', u'bads'], dtype='object')

        index_keys= []
        for w in self.HDFobj.keys():
            if w.startswith('/epocher'):
               index_keys.append( w.replace('/epocher', '').replace('/', '') )

        df_stat = pd.DataFrame(index=index_keys,columns = cols)
        
       # s = Series(randn(5), index=['a', 'b', 'c', 'd', 'e'])
       #d = {'one' : Series([1., 2., 3.], index=['a', 'b', 'c']),
#   ....:      'two' : Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
        idx = 0

        for condi in index_keys:
            k='/epocher/'+condi

            #print k
            #print condi
            df = self.HDFobj[k]
            # df_stat['EvID'][idx] = condi
            
           #--- get sampling frquency from  attrs epocher_info
            Hstorer = self.HDFobj.get_storer(k)
            try:
                sfreq = Hstorer.attrs.info_parameter['sfreq']
            except:
                sfreq = 1.0

            try:
                rtm = Hstorer.attrs.epocher_parameter['response_matching']
            except:
              #  df_stat['EvID'][idx] = np.array_str( np.unique(df.id) )
                df_stat['Hit'][idx]  = df.id.size
                rtm = False
            
            df_stat['STID'][idx] = np.unique(df.id)[0]
            df_stat['RTID'][idx] = 0.0
            
            if rtm :              
                
           #--- missed
               missed_idx = df[ df['rt_type'] == self.idx_missed ].index
               if missed_idx.size :
                  df_stat['Missed'][idx] = missed_idx.size
           #--- early
               toearly_idx = df[ df['rt_type'] == self.idx_toearly ].index
               if toearly_idx.size :
                  df_stat['TOEarly'][idx] = toearly_idx.size
           #--- wrong
               wrong_idx = df[ df['rt_type'] == self.idx_wrong ].index
               if wrong_idx.size:
                  df_stat['Wrong'][idx] = wrong_idx.size

           #--- hit
               hit_idx = df[ df['rt_type'] == self.idx_hit ].index
               if hit_idx.size:
                  df_stat['Hit'][idx] = hit_idx.size
                  df_stat['RTID'][idx] = df['rt_id'][ hit_idx[0] ]

            #--- RTmean

               df_stat['RTmean'][idx]   = df.rt[hit_idx].mean() / sfreq
               df_stat['RTmedian'][idx] = df.rt[hit_idx].median() / sfreq
               df_stat['RTstd'][idx]    = df.rt[hit_idx].std() / sfreq
               df_stat['RTmin'][idx]    = df.rt[hit_idx].min() / sfreq
               df_stat['RTmax'][idx]    = df.rt[hit_idx].max() / sfreq

            else:
               df_stat['STID'][idx] =  np.unique(df.id)[0]
               df_stat['Hit'][idx]  = df.id.size
                        
            idx += 1

        #--- save stats data frame to csv
        fcsv = None
        
        if save:
           fcsv = self.HDFobj.filename.replace(self.hdf_postfix,self.stat_postfix)
           #--- float formating not working due to pandas float_format bug 12.12.14
           df_stat.to_csv(fcsv,na_rep='0',float_format='%.3f')

        if self.verbose :
           print"\n  --> Condition Statistic Data Frame\n"
           print df_stat
           print"\n\n"
           if save :
              print " --> Stat DataFrame saved as: "+ fcsv

        return df_stat
            
 
    def events_export_events(self,raw=None,fhdf=None,condition_list=None,fif_postfix="evt",
                             event_extention=".eve",picks=None,reject=None,proj=False,
                             save_condition={"events":True,"epochs":True,"evoked":True},
                             time={"time_pre":None,"time_post":None,"baseline":None},
                             baseline_correction={"type":None,"channel":None,"output":None,"baseline":None},
                             exclude_events = None,weights=None ):
          
        '''
        raw=None,fhdf=None,condition_list=None,fif_postfix="evt",
        event_extention=".eve",picks=None,reject=None,proj=False,
        save_condition={"events":True,"epochs":True,"evoked":True},
        time={"time_pre":None,"time_post":None,"baseline":None},
        baseline_correction={"type":None,"channel":None,"output":None,"baseline":None},
        exclude_events = None,weights=None   
        '''
        
        if raw:
           self.raw = raw
      
      #--- get HDF obj
        if fhdf:
           self.HDFobj = pd.HDFStore(fhdf)
        else:
           if self.raw:
              self.HDFobj = self.hdf_obj_open(raw=self.raw)

        if not self.hdf_obj_is_open():
           assert "ERROR could not open HDF file:\n --> raw: "+self.raw.filename+"\n --> hdf: "+self.HDFObj.filename+"\n"
         
        epocher_condition_list = self.hdf_get_key_list(node='/epocher',key_list=condition_list)
         
        event_id   = dict()
        
        time_param = dict()        
        for k in time:
            if time[k]:
               time_param[k]= time[k] 
        
        bc_param = dict()        
        for k in baseline_correction:
            if baseline_correction[k]:
               bc_param[k]= baseline_correction[k] 
       
       #--- init exclude_events e.g.  eog onset
        exclude_events = self.events_update_artifact_time_window(aev=exclude_events)
        
        for condi in epocher_condition_list:
            evt  = self.events_hdf_to_mne_events(condi,exclude_events=exclude_events,time=time_param,baseline=bc_param,weights=None)
            if evt['events'].size:
               event_id[condi] = {'id':evt['event_id'],'trials': evt['events'].shape[0],'trials_weighted':0}
             #---
               ep,bc = self.events_apply_epochs_and_baseline(self.raw,evt=evt,reject=reject,proj=proj,picks=picks)
               self.events_save_events(evt=evt,condition=condi,postfix=fif_postfix,picks=picks,reject=reject,proj=proj,save_condition=save_condition)

         #--- ck weighted events
         # "weights":{"mode":"equal_counts","selection":"median","skipp_first":null},
        if hasattr(weights,'mode') :      
           print "\n ---> Applying Weighted Export Events"            
                     
           if weights['mode'] == 'equal':
              weights['min_counts'] = event_id[ event_id.keys()[0] ]['trials'] 
          
              for condi in event_id.keys() :
                  if event_id[ condi ]['trials'] <  weights['min_counts']: 
                     weights['min_counts'] = event_id[ condi ]['trials']
           
              for condi in event_id.keys():
                  evt = self.events_hdf_to_mne_events(condi,exclude_events=exclude_events,time=time_param,baseline=bc_param,weights=weights)
                #---
                  if evt['events'].size:
                     ep,bc = self.events_apply_epochs_and_baseline(self.raw,evt=evt,reject=reject,proj=proj,picks=picks)
                     event_id[condi]['trials_weighted'] = evt['events'].shape[0]
                     self.events_save_events(evt=evt,condition=condi,postfix=fif_postfix+'W',picks=picks,reject=reject,proj=proj,save_condition=save_condition)
            
        
        fhdf = self.HDFobj.filename
        self.HDFobj.close()        
        return event_id    
          
          
    def events_save_events(self,evt=None,condition=None,postfix="evt",
                                picks=None,reject=None,proj=False,
                                save_condition={"events":True,"epochs":True,"evoked":True}):
        
        from jumeg.preprocbatch.jumeg_preprocbatch_plot import jumeg_preprocbatch_plot as jplt
        jplt.verbose = self.verbose
      
        ep,bc = self.events_apply_epochs_and_baseline(self.raw,evt=evt,reject=reject,proj=proj,picks=picks)      
      
        postfix += '_' + condition
        if bc:
           postfix += '_bc'
               
      #--- save events to txt file    
        if save_condition["events"]:
           fname = jumeg_base.get_fif_name(raw=self.raw,postfix=postfix,extention=".eve",update_raw_fname=False)
           mne.event.write_events( fname,evt['events'] )
           print" ---> done jumeg epocher save events as => EVENTS :" +fname
          
      #--- save epoch data
        if save_condition["epochs"]:
           fname = jumeg_base.get_fif_name(raw=self.raw,postfix=postfix,extention="-epo.fif",update_raw_fname=False)
           ep.save( fname )
           print" ---> done jumeg epocher save events as => EPOCHS :" +fname
          
      #--- save averaged data
           if save_condition["evoked"]:
              fname = jumeg_base.get_fif_name(raw=self.raw,postfix=postfix,extention="-ave.fif",update_raw_fname=False)
              mne.write_evokeds( fname,ep.average() )              
              print" ---> done jumeg epocher save events as => EVOKED (averaged) :" +fname 
              fname = jumeg_base.get_fif_name(raw=self.raw,postfix=postfix,extention="-ave",update_raw_fname=False)  
            #--- plot evoked
              fname = jplt.plot_evoked(ep,fname=fname,condition=condition,show_plot=False,save_plot=True,plot_dir='plots')
              print" ---> done jumeg epocher plot evoked (averaged) :" +fname 
    
    
    def events_hdf_to_mne_events(self,condi,exclude_events=None,time=None,baseline=None,weights=None):
        '''
         export HDF events to mne events structure
         input:
               condition name  
               exclude_events = None
               time           = None
               weights:{"mode":"equal_counts","selection":"median","skipp_first":null,min_counts=None},
    
         return:
               events for mne
        '''       
      #-----
        events_idx = np.array([],dtype=np.int64)

        print " ---> START EPOCHER extract condition : " + condi 
        ep_key = '/epocher/' + condi
            
        if not( ep_key in self.HDFobj.keys() ): return
        
       #--- get pandas data frame
        df = self.hdf_obj_get_dataframe(ep_key)
       #--- get stored attributes -> epocher_parameter -> ...
        ep_param   = self.hdf_obj_get_attributes(key=ep_key,attr='epocher_parameter')
        info_param = self.hdf_obj_get_attributes(key=ep_key,attr='info_parameter')

        evt = dict()
       #--- get channel e.g.  'STI 014'  stimulus.events.stim_channel
        evt['channel'] = ep_param[ ep_param['marker_channel'] ]['events']['stim_channel']
       #--- get output type [onset/offset]
        evt['output']  = ep_param[ ep_param['marker_channel'] ]['events']['output']
       #---  <onset/offset> for stimulus onset/offset <rt_onset/rt_offset> for response onset/offset
        evt['marker_type'] = ep_param['marker_type']
       #--- time
        evt['time']= self.events_get_parameter(hdf_parameter=ep_param,param=time)
       
       #--- baseline
        evt['bc'] = self.events_get_parameter(hdf_parameter=ep_param['baseline'],
                                              param=baseline,key_list=('output','channel','baseline','type'))
                             
       #--- ck for artefacts => set bads to -1
        if exclude_events :
           for kbad in ( exclude_events.keys() ):
               ep_bads_cnt0 = df['bads'][ df['bads'] == self.idx_bad ].size

               for idx in range( exclude_events[kbad]['tsl'].shape[-1] ) :    #df.index :
                   df['bads'][ ( exclude_events[kbad]['tsl'][0,idx] < df[evt['output']] ) & ( df[evt['output']] < exclude_events[kbad]['tsl'][1,idx] ) ] = self.idx_bad
               
               ep_bads_cnt1 = df['bads'][df['bads'] == self.idx_bad].size 
               
               if self.verbose:         
                  print"\n---> Exclude artefacts " + condi + " : " + kbad
                  print"---> Tmin: %0.3f Tmax %0.3f" % (exclude_events[kbad]['tmin'],exclude_events[kbad]['tmax'])
                  print"---> bad epochs     : %d" %(ep_bads_cnt0)
                  print"---> artefact epochs: %d" %(ep_bads_cnt1 - ep_bads_cnt0)
                  print"---> excluded epochs: %d" %(ep_bads_cnt1)
                  #if  (ep_bads_cnt1 - ep_bads_cnt0) > 0:
                  #    assert "FOUND"
                      
        df['selected']          = 0  
        df['weighted_selected'] = 0  
        
       #--- response type idx to process
        if ep_param['response_matching'] :
           if ep_param['response_matching_type'] is None:
              rt_type_idx = self.rt_type_as_index('HIT')
           else:
              rt_type_idx = self.rt_type_as_index(ep_param['response_matching_type'])
       #--- find   response type idx
           events_idx = df[ evt['marker_type'] ][ (df['rt_type'] == rt_type_idx) & (df['bads'] != self.idx_bad) ].index

           if events_idx.size:
              #df.loc['selected',events_idx] = 1
               df['selected'][events_idx] = 1
              # data.loc[data['name'] == 'fred', 'A'] = 0

              #--- apply weights to reduce/equalize number of events for all condition
               if hasattr(weights,'min_counts'):
                  if weights['min_counts']:
                     assert weights['min_counts'] <= events_idx.size,"!!!ERROR minimum required trials greater than number of trials !!!"
                     w_value = 0.0
                     if weights['min_counts'] < events_idx.size:

                        if weights['method']=='median':
                             w_value = df['rt'][events_idx].median()
                        elif weights['method']=='mean':
                             w_value = df['rt'][events_idx].mean()
                        #else: # random
                       #     w_idx = np.random.shuffle( np.array( events_idx ) )

                      #--- find minimum from median as index from events_idx => index of index !!!
                        w_idx = np.argsort( np.abs( np.array( df['rt'][events_idx] - w_value )))
                        w_events_idx = np.array( events_idx[ w_idx[ 0:weights['min_counts'] ] ] )
                        w_events_idx.sort()

                        df['weighted_selected'][w_events_idx] = 1
                        #df.loc['weighted_selected',w_events_idx] = 1

                        if self.verbose:
                           print"RESPONSE MATCHING => Weighted event index => method:" + weights['method']+" => value: %0.3f" % (w_value)
                           print w_events_idx
                           print"RT :"
                           print df['rt'][w_events_idx]

                     elif weights['min_counts'] == events_idx.size:
                          df['weighted_selected'][events_idx] = 1
                          #df.loc['weighted_selected',events_idx] = 1

                          #--- update new weighted event idx
                     events_idx = df[ df['weighted_selected'] > 0 ].index

                 
        else :
   #--- no response matching
           events_idx = df[ evt['marker_type'] ][ df['bads'] != self.idx_bad ].index
         #-- ck for eaqual number of trials over conditons#
           if events_idx.size:
               if hasattr(weights,'min_counts'):
                  if weights['min_counts']:
                     assert weights['min_counts'] <= events_idx.size,"!!!ERROR minimum required trials greater than number of trials !!!"

                     if weights['min_counts'] < events_idx.size:
                        w_idx = np.array( events_idx.values )
                        np.random.shuffley(w_idx)
                        w_events_idx = w_idx[0:weights['min_counts'] ]
                        df['weighted_selected'][w_events_idx] = 1
                        events_idx = df[ df['weighted_selected'] > 0 ].index

                     elif weights['min_counts'] == events_idx.size:
                          df['weighted_selected'][events_idx] = 1

        if events_idx.size:

          #--- make event array
            evt['events']      = np.zeros(( events_idx.size, 3), dtype=np.long)
            evt['events'][:,0] = df[ evt['marker_type'] ][events_idx]

            if  ep_param['marker_channel'] == 'response':
                evt['events'][:,2] = ep_param['response']['event_id']
            else:
                evt['events'][:,2] = df['id'][0]

           #--- ck if events exist
            evt['event_id'] = int( evt['events'][0,2] )

           #--- baseline events
            #evt['bc']['output']      = ep_param[ evt['bc']['channel'] ]['events']['output']
            evt['bc']['events']      = np.zeros((events_idx.size,3),dtype=np.long)
            evt['bc']['events'][:,0] = df[ evt['bc']['output'] ][events_idx]

            if ep_param[ evt['bc']['channel'] ] == 'response':
               evt['bc']['events'][:,2] = ep_param['response']['event_id']
            else:
                evt['bc']['events'][:,2] = df['id'][0]
            evt['bc']['event_id'] = int( evt['bc']['events'][0,2] )
        else:
       #--- no events
            evt['events']       = np.array([])
            evt['event_id']     = np.array([])
            evt['bc']['events'] = np.array([])


    #---update HDF: store df with updated bads & selected & restore user-attribute
        storer_attrs = {'epocher_parameter': ep_param,'info_parameter':info_param}
        self.hdf_obj_update_dataframe(df,key=ep_key,reset=False,**storer_attrs)
      
      #--- 
        if self.verbose:
           print" ---> Export Events from HDF to MNE-Events for condition: " + condi
           print"      events: %d" % events_idx.size
           bads = df[ evt['marker_type'] ][ (df['bads']== self.idx_bad)  ]
           print"      bads  : " + str(bads.shape)
           print bads
           print"\nEvent Info:"
           print evt
           print"\n\n"
           
        return evt


    def events_get_parameter(self,hdf_parameter=None,param=None,key_list=('time_pre','time_post') ):

          """

          :param hdf_parameter:
          :param ep_param
          :param key_list=('time_pre','time_post','baseline')
          
          :return:
            param_out
          """
          param_out = dict()
          
          for k in key_list:
              if param.has_key(k):
                 param_out[k] = param[k]
              elif hdf_parameter.has_key(k) :
                   if hdf_parameter[k]:
                      param_out[k] = hdf_parameter[k]
           
          if self.verbose:
             print " --> Parameter: "
             print param_out

          return param_out
          
    def events_update_artifact_time_window(self,aev=None,tmin=None,tmax=None):
          """

          :param aev:
          :param tmin:
          :param tmax:
          :return:
          """
          import numpy as np
     
          artifact_events = dict()

          for kbad in ( aev.keys() ):
              node_name = '/ocarta/' + kbad
            #--- ck if node exist 
              try:             
                  self.HDFobj.get(node_name)
              except:
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
              artifact_events[kbad]['tsl'] = np.array([ df_bad['onset']+tsl0, df_bad['onset']+tsl1 ] )
              # aev[0,ixd] ->tsl0   aev[1,idx] -> tsl1
              #artifact_events[kbad]['tsl'] = np.zeros( shape =( df_bad['onset'].size,2) )
              #artifact_events[kbad]['tsl'][:,0] = df_bad['onset'] +tsl0
              #artifact_events[kbad]['tsl'][:,1] = df_bad['onset'] +tsl1

          return artifact_events
          
             
    def events_apply_epochs_and_baseline(self,raw,evt=None,reject=None,proj=None,picks=None):
        '''
        generate epochs from raw and apply baseline correction
        input:
              raw obj
              evt=event dict
              check for bad epochs due to short baseline onset/offset intervall and drop them
              
        output:
              baseline corrected epoch data
              !!! exclude trigger channels: stim and resp !!!
              bc correction: true /false
        '''
           
        ep_bads    = None
        ep_bc_mean = None
        bc         = False
        
     #--- get epochs no bc correction 
        ep = mne.Epochs(self.raw,evt['events'],evt['event_id'],evt['time']['time_pre'],evt['time']['time_post'],
                        baseline=None,picks=picks,reject=reject,proj=proj,preload=True,verbose=False) # self.verbose)
            
        if ('bc' in evt.keys() ): 
          if evt['bc']['baseline']:   
             if len( evt['bc']['events'] ): 
                if evt['bc']['baseline'][0]:        
                   bc_time_pre = evt['bc']['baseline'][0] 
                else:
                   bc_time_pre = evt['time']['time_pre']  
                if evt['bc']['baseline'][1]:        
                   bc_time_post = evt['bc']['baseline'][1] 
                else:
                   bc_time_post = evt['time']['time_post']  
                
                picks_bc = jumeg_base.picks.exclude_trigger(ep)
                
               #--- create baseline epochs -> from stimulus baseline
                ep_bc = mne.Epochs(self.raw,evt['bc']['events'],evt['bc']['event_id'],
                                   bc_time_pre,bc_time_post,baseline=None,
                                   picks=picks_bc,reject=reject,proj=proj,preload=True,verbose=self.verbose)
             
                
               #--- check for equal epoch size epoch_baseline vs epoch for np bradcasting
                ep_goods = np.intersect1d(ep_bc.selection,ep.selection)
               
               #--- bad epochs & drop them
                ep_bads  = np.array( np.where(np.in1d(ep_bc.selection,ep_goods,invert=True)) )
                if ep_bads:
                   ep_bc.drop(ep_bads.flatten())
              #--- calc mean from bc epochs 
                ep_bc_mean = np.mean(ep_bc._data, axis = -1)
                ep._data[:,picks_bc,:] -= ep_bc_mean[:,:,np.newaxis]
                bc = True
 #--- 
        if self.verbose:
           print" ---> Epocher apply epoch and baseline -> mne epochs:" 
           print ep
           print"      id: %d  <pre time>: %0.3f <post time>: %0.3f" % (evt['event_id'],evt['time']['time_pre'],evt['time']['time_post'])
           print" --> baseline correction : %r" %(bc)
         
           if bc:
                print"     done -> baseline correction"
                print"     bc id: %d  <pre time>: %0.3f <post time>: %0.3f" % (evt['bc']['event_id'],bc_time_pre,bc_time_post)
       
                print"\n --> Epoch selection: "
                print ep.selection
                print" --> Baseline epoch selection: "
                print ep_bc.selection
                print"\n --> good epochs selected:"
                print ep_goods
                print"\n --> bad epochs & drop them:"             
                print ep_bads
           print"\n"
       
        return ep,bc
          
jumeg_epocher_events = JuMEG_Epocher_Events()

