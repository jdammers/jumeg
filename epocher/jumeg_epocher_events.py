import os
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
        self.__data_frame_response_cols = ['rt_type','rt','rt_id','rt_onset','rt_offset','rt_index','rt_counts','bads']

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
               'events': {'consecutive': True, 'output':'steps','stim_channel': 'STI 014',
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
           return None
       #---
        import pandas as pd
        df = pd.DataFrame(columns = self.data_frame_stimulus_cols)

       #---
        events           = param['events'].copy()
        events['output'] = 'step'
        ev               = mne.find_events(raw, **events)

       #--- apply and mask e.g. 255 get the first 8 bits in Trigger channel
        if param['and_mask']:
           ev[:, 1:] = np.bitwise_and(ev[:, 1:], param['and_mask'])
           ev[:, 2:] = np.bitwise_and(ev[:, 2:], param['and_mask'])

       #--- search and return only events with event_id in event_id-array
       #--- shape =>(trials,3)  => [trial idx ,tsl-onset/ tsl-offset,event id]
       #--- split this in onset offset
        if param['event_id']:
           ev_id     = jumeg_base.str_range_to_numpy(param['event_id'])
           ev_onset  = np.squeeze( ev[np.where( np.in1d( ev[:,2],ev_id ) ), :])
           ev_offset = np.squeeze( ev[np.where( np.in1d( ev[:,1],ev_id ) ), :])

       #--- get all ids e.g. response for later response matching
        else:
           ev_onset  = ev[0::2,]
           ev_offset = ev[1::2,]

        if ( ev_onset.size == 0 ):
           return df

       #--- apply system delay if is defined e.g. auditory take`s 20ms to subjects ears
        if param['system_delay_ms']:
           system_delay_tsl = raw.time_as_index( param['system_delay_ms']/1000 ) # calc in sec
           ev_onset[:, 0] += system_delay_tsl
           ev_offset[:, 0]+= system_delay_tsl
           system_delay_is_applied = True
        else:
           system_delay_is_applied = False

        div          = ev_offset[:,0] - ev_onset[:,0]
        df['id']     = ev_onset[:,2]
        df['onset']  = ev_onset[:,0]
        df['offset'] = ev_offset[:,0]
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
            st_id          = stim_df['id'][idx]

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
            res_df_in = resp_df[ ( st_window_tsl0 <= RESP_TSLs ) & ( RESP_TSLs <= st_window_tsl1) ]

          #--- MISSED response
            if res_df_in.empty : continue

          #--- WRONG or HIT;
            if res_df_in.index.size > 0 :
               ridx = res_df_in.index[0]
               stim_df.rt_counts[idx] = res_df_in.index.size
               stim_df.rt_onset[idx]  = resp_df.onset[ridx]
               stim_df.rt_offset[idx] = resp_df.offset[ridx]
               stim_df.rt_type[idx]   = self.idx_wrong
               stim_df.rt_id[idx]     = resp_df.id[ridx]
               stim_df.rt_index[idx]  = ridx
          #--- HIT; ck number of responses; ck pressed buttons; wrong if more than count
               if (  stim_df.rt_counts[idx] <= param['response']['counts'] ):
                   if res_df_in.id.isin( r_event_id ).all :
                      stim_df.rt_type[idx] = self.idx_hit


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
          #--- check for condition in list
            if condition_list :
               if condi not in condition_list: continue

          #--- check for real condition
            if condi == 'default': continue

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
                  print "---> Stimulus Epocher Events Data Frame [stimulus channel]:"
                  print stimulus_data_frame
                  print"\n\n"

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

               key = '/epocher/'+condi
               storer_attrs = {'epocher_parameter': parameter,'info_parameter':stimulus_info}
               self.hdf_obj_update_dataframe(stimulus_data_frame.astype(np.int32),key=key,**storer_attrs )

       #--- write stat info (.csv)
        self.events_condition_stats(save=True)

        fhdf= self.HDFobj.filename
        self.HDFobj.close()

        print" ---> DONE save epocher data into HDF5 :"
        print"  --> " + fhdf +"\n\n"
        return fhdf

#--
    def events_condition_stats(self,save=False):
        """
        return:
        < pandas data frame>
        """
        import pandas as pd

       #--- ck error

        if not self.hdf_obj_is_open():
           return


       #---

        cols = ['EvID','Hit','Wrong', 'TOEarly', 'Missed', 'RTmean','RTmedian','RTstd', 'RTmin', 'RTmax']
        #Index([u'id', u'onset', u'offset', u'rt_type', u'rt', u'rt_id', u'rt_onset', u'rt_offset', u'rt_index', u'rt_counts', u'bads'], dtype='object')

        index_keys= []
        for w in self.HDFobj.keys():
            if w.startswith('/epocher'):
               index_keys.append( w.replace('/epocher', '').replace('/', '') )

        df_stat = pd.DataFrame(index=index_keys,columns = cols)

        idx = 0

        for condi in index_keys:
            k='/epocher/'+condi

            #print k
            #print condi
            df = self.HDFobj[k]

           #--- get sampling frquency from  attrs epocher_info
            Hstorer = self.HDFobj.get_storer(k)
            try:
                sfreq = Hstorer.attrs.info_parameter['sfreq']
            except:
                sfreq = 1.0

            try:
                rtm = Hstorer.attrs.epocher_parameter['response_matching']
            except:
                df_stat['EvID'][idx] = np.array_str( np.unique(df.id) )
                df_stat['Hit'][idx]  = df.id.size
                rtm = False

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
            #--- RTmean

               df_stat['RTmean'][idx]   = df.rt[hit_idx].mean() / sfreq
               df_stat['RTmedian'][idx] = df.rt[hit_idx].median() / sfreq
               df_stat['RTstd'][idx]    = df.rt[hit_idx].std() / sfreq
               df_stat['RTmin'][idx]    = df.rt[hit_idx].min() / sfreq
               df_stat['RTmax'][idx]    = df.rt[hit_idx].max() / sfreq

            else:
               df_stat['EvID'][idx] =  np.unique(df.id)[0]
               # df_stat['EvID'][idx] = np.array_str( np.unique(df.id) )
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


jumeg_epocher_events = JuMEG_Epocher_Events()