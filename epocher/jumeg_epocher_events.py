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

---> update 13.04.2018 FB
     check event-code/conditions for none existing
     update for eye tracking events
     new template file key words

'''
import numpy as np
import pandas as pd

import mne
from jumeg.jumeg_base import jumeg_base
from jumeg.epocher.jumeg_epocher_hdf import JuMEG_Epocher_HDF

__version__="2018.04.13.001"

class JuMEG_Epocher_Events(JuMEG_Epocher_HDF):
    def __init__ (self):

        super(JuMEG_Epocher_Events, self).__init__()
        self.event_data_info = dict()
        self.event_data_parameter={"events":{
                                             "stim_channel"   : "STI 014",
                                             "output"         : "onset",
                                             "consecutive"    : True,
                                             "min_duration"   : 0.0001,
                                             "shortest_event" : 1,
                                             "mask"           : 0
                                            },
                                    "event_id" : None,        
                                    "and_mask" : None,
                                    "system_delay_ms" : 0.0
                                   }
        
        self.__rt_type_list             = ['MISSED', 'TOEARLY', 'WRONG', 'HIT']
        self.__data_frame_stimulus_cols = ['id','onset','offset']
        self.__data_frame_response_cols = ['rt_type','rt','rt_id','rt_onset','rt_offset','rt_index','rt_counts','bads','selected','weighted_selected']

        self.__stat_postfix = '-epocher-stats.csv'

        self.__idx_bad = -1
        self.parameter = None    
#---
    @property
    def stimulus_channel(self):      return self.parameter['stimulus_channel']  
    @property
    def stimulus_stim_channel(self): return self.parameter[ self.stimulus_channel ]["events"]["stim_channel"]   
#---    
    def stimulus_parameter(self,p=None):
        if p:
           return self.parameter[ self.stimulus_channel ][p] 
        return self.parameter[ self.stimulus_channel ]
#---
    @property
    def response_channel(self):      return self.parameter['response_channel']
    @property
    def response_stim_channel(self): return self.parameter[self.response_channel]["events"]["stim_channel"] 
#---
    def response_parameter(self,p=None):
        if p:
           return self.parameter[ self.response_channel ][p] 
        return self.parameter[ self.response_channel ]
#---    
    @property
    def event_data_stim_channel(self): return self.event_data_parameter["events"]["stim_channel"]
    @event_data_stim_channel.setter
    def event_data_stim_channel(self,v): self.event_data_parameter["events"]["stim_channel"]=v   
#---
    @property
    def idx_bad(self): return self.__idx_bad
#---
    @property
    def data_frame_stimulus_cols(self): return self.__data_frame_stimulus_cols
    @data_frame_stimulus_cols.setter
    def data_frame_stimulus_cols(self,v): self.__data_frame_stimulus_cols = v   
#---
    @property
    def data_frame_response_cols  (self): return self.__data_frame_response_cols
    @data_frame_response_cols.setter
    def data_frame_response_cols(self,v): self.__data_frame_response_cols = v
    
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
        """some how calls <mne.find_events()>
        
        Parameters
        ---------
        raw   : raw obj
        param : parameter like <**kwargs>
               {'event_id': 40, 'and_mask': 255,
               'events': {'consecutive': True, 'output':'step','stim_channel': 'STI 014',
               'min_duration':0.002,'shortest_event': 2,'mask': 0}
                }

        Returns
        --------
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
        
        
        print param
       #---
        events           = param['events'].copy()
        events['output'] = 'step'
        self.pp( events )
        ev = mne.find_events(raw, **events) #-- return int64

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
           #print" --> event_id"
           #print ev_id
           
           evt_ids=np.where(np.in1d(ev[:,2],ev_id))
           
          #--- check if code in events
           #if ( ev_id in np.unique(ev[:, 2]) ):
           if len( np.squeeze(evt_ids) ):   
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
            
        #    print ev_onset[:,0].size
        #print ev_offset[:,0].size
        
        
        return df,dict( {
                         'sfreq'        : raw.info['sfreq'],
                         'duration'     :{'mean':np.rint(div.mean()),'min':div.min(),'max':div.max()},
                         'system_delay_is_applied' : system_delay_is_applied
                         } )

#---
    def events_response_matching(self,raw,stim_df=None,resp_df=None,stim_param=None,resp_param=None): #, **param ):
        """
        matching correct responses with respect to <stimulus channel> <output type> (onset,offset)
        input:
                stim_df = <stimulus channel data frame>
                res_df  = <response channel data frame>
                param :  self.parameter dict
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
        err_msg =[]
        if raw is None:
           err_msg.append("ERROR in <apply_response_matching> : no raw obj")
        if (stim_df is None):
           err_msg.append("ERROR no Stimulus-Data-Frame obj. provided")
        if (resp_df is None):
           err_msg.append("ERROR no Response-Data-Frame obj. provided")
        if (self.parameter is None):
           err_msg.append("ERROR no self.parameter obj. provided")
      
       #--- print parameter  
        if self.verbose:
           self.pp(stim_param,head="STI param")
           self.pp(resp_param,head="RESP param")
       
       #--- ck RT window range
        if ( resp_param['window'][0] >= resp_param['window'][1] ):
       # if ( self.response_parameter['response']['window'][0] >= param['response']['window'][1] ):
            err_msg.append("ERROR in self.parameter response windows")

        if err_msg:
           print" --> ERROR in <apply_response_matching>"
           for msg in err_msg: 
               print" --> " + msg
           return None

       #--- extend stimulus data frame
        for x in self.data_frame_response_cols :
            stim_df[x]= 0 

       #--- convert rt window [ms] into tsl
        (r_window_tsl_start, r_window_tsl_end ) = raw.time_as_index( resp_param['window'] );

       #--- get respose code -> event_id [int or string] as np array
        r_event_id = jumeg_base.str_range_to_numpy( resp_param['event_id'] )

       #--- ck if any toearly-id is defined, returns None if not
        if resp_param["include_early_ids"] != 'all':
           r_event_id_toearly = jumeg_base.str_range_to_numpy( resp_param['include_early_ids'] )
        else:
           r_event_id_toearly=None 
            
       #--- get output type: onset or offset
        stim_output_type = stim_param['events']['output']
        resp_output_type = resp_param['events']['output']

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
            if resp_param["include_early_ids"] == 'all':
               res_df_early = None# np.array([])
            else:
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
               if resp_param['counts']: 
                  if ( stim_df.rt_counts[idx] <= resp_param['counts'] ):
                     if np.all( resp_df.id[ resp_df_in_idx ].isin( r_event_id ) ) :
                        stim_df.rt_type[idx] = self.idx_hit
              
              #--- take all if counts==None e.g.: eyetracking saccades after stimulus onset
              #--- ok problem sccarde and fixation follows each other 
               else: # take all if counts==None
                 #--- get true idxs  
                  resp_df_in_idx_hits = np.where( resp_df.id[ resp_df_in_idx ].isin( r_event_id ) )[0]
                  if resp_df_in_idx_hits.shape[0]:
                     ridx_hits = resp_df_in_idx[resp_df_in_idx_hits]
                    #--- get closes onset as tsl 
                     if ridx_hits.shape[0]>1:
                        ridx = np.argmin( abs(resp_df.onset[ridx_hits] - stim_df.rt_onset[idx]) )
                     else :   
                        ridx = resp_df_in_idx[resp_df_in_idx_hits][0]
                     
                    #print resp_df.take( ridx_hits )
                     
                     stim_df.rt_type[idx]   = self.idx_hit
                     stim_df.rt_onset[idx]  = resp_df.onset[ridx]
                     stim_df.rt_offset[idx] = resp_df.offset[ridx]
                     stim_df.rt_id[idx]     = resp_df.id[ridx]
                     
                     
                 # print "HIT idx"
                 # print hit_idx
                 # print"\n"
                  #if np.all( resp_df.id[ resp_df_in_idx].isin( r_event_id ) ) :
                  #      stim_df.rt_type[idx] = self.idx_hit 
                  #if np.any( resp_df.id[ resp_df_in_idx].isin( r_event_id ) ) :
                  #   stim_df.rt_type[idx] = self.idx_hit 
                   
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
        """find & store epocher data to hdf5:
        -> readding self.parameter from epocher template file
        -> find events from raw-obj using mne.find_events
        -> apply response matching if is true
        -> save results in pandas dataframes & HDF fromat
        
        Parameter
        ---------
        raw : raw obj
        condition_list: list of conditions to process
                        select special conditions from epocher template
                        default: <None> , will process all
        overwrite_hdf : flag for overwriting output HDF file <True>

        Results
        -------
        HDF filename
        """
      
       #---  init obj
       # overwrite_hdf=False
        self.hdf_obj_init(raw=raw,overwrite=overwrite_hdf)
        
        self.parameter         = None
        self.event_data_frames = dict()
        self.event_data_info   = dict()
        
       #--- condi loop
        for condi, param, in self.template_data.iteritems():
         
          #--- check for real condition
            if condi == 'default': continue
          
          #--- check for condition in list
            if condition_list :
               if condi not in condition_list: continue
           
            self.line()
            print"===> Jumeg Epocher Events ====\n"
            print" --> Start Events Store into HDF"      
            print" --> condition: "+ condi
            self.line()
            
           #--- update & merge condi self.parameter with defaults
            self.parameter = self.template_data['default'].copy()
            self.parameter = self.template_update_and_merge_dict(self.parameter,param)
           #--- stimulus init dict's & dataframes
            stimulus_info          = dict()
            stimulus_data_frame    = None

            if self.verbose:
               print' --> EPOCHER  Template: %s  Condition: %s' %(self.template_name,condi)
               print'  -> find events and epochs,  generate epocher output HDF5'
               print"-"*50
               print"  -> parameter :"
               print self.parameter
               print"-"*50+"\n"
               
           #--- select stimulus channel e.g. "stimulus" -> STIM 014 or "response" ->STIM 013
            #if self.parameter['stimulus_channel']: # in ['stimulus','response'] :
            if self.stimulus_parameter(): # in ['stimulus','response'] :
             #print"  -> STIMULUS CHANNEL -> find events: "+ condi +" ---> "+ self.parameter['stimulus_channel']
               print"  -> STIMULUS CHANNEL -> find events => condition: "+ condi +" ---> stimulus channel: "+ self.stimulus_channel
               if self.verbose:
                  print "  -> Stimulus Channel parameter:"
                  #print self.parameter[ self.parameter['stimulus_channel'] ]
                  print"-"*50
                  print self.stimulus_parameter()
                  print"-"*50
                  print"\n"
              #--- get stimulus channel epochs from events as pandas data-frame
               #stimulus_data_frame,stimulus_info = self.events_find_events(raw,**self.parameter[ self.parameter['stimulus_channel'] ])
               stimulus_data_frame,stimulus_info = self.events_find_events(raw,**self.stimulus_parameter())
               
               # pevents = self.stimulus_parameter(p=events)
               
               self.event_data_stim_channel  = self.stimulus_stim_channel 
               self.event_data_frames[self.stimulus_channel],self.event_data_info[self.stimulus_channel] = self.events_find_events(raw,**self.event_data_parameter)
               

               if self.verbose:
                  print"  -> Stimulus Epocher Events Data Frame [stimulus channel]: "+ condi
                  print stimulus_data_frame
                  print"\n"

               if stimulus_data_frame is None: continue

              #--- RESPONSE Matching task
              #--- match between stimulus and response
              #--- get all response events for condtion e.g. button press 4
               if self.parameter['response_matching'] :
                  print"  -> RESPONSE MATCHING -> matching stimulus & response channel: " + condi
                  #print"  -> stimulus channel : " + self.parameter['stimulus_channel']
                  print"  -> stimulus channel : " + self.stimulus_channel
                  print"  -> response channel : " + self.response_channel
                 #--- look for all responses => 'event_id' = None
                  # res_param = self.parameter[ self.parameter['response_channel'] ].copy()
                  res_param = self.response_parameter().copy()
                  res_param['event_id'] = None

                 #--- get epochs from events as pandas data-frame
                  response_data_frame,response_info = self.events_find_events(raw,**res_param)
               
                  self.event_data_stim_channel = self.response_stim_channel
                  self.event_data_frames[self.response_channel],self.event_data_info[self.response_channel] = self.events_find_events(raw,**self.event_data_parameter)
                  
               
                  if self.verbose:
                     print"---> Response Epocher Events Data Frame [response channel] : " + self.response_channel
                     print self.response_parameter()
                     #print"---> Response Epocher Events Data Frame [respnse channel] : " + self.parameter['response_channel']
                     print response_data_frame
                     print"\n\n"

                 #--- update stimulus epochs with response matching
                  #stimulus_data_frame = self.events_response_matching(raw,stimulus_data_frame,response_data_frame,**parameter )
                  # stimulus_data_frame = self.events_response_matching(raw,stimulus_data_frame,response_data_frame,**self.parameter )
                  stimulus_data_frame = self.events_response_matching(raw,stimulus_data_frame,response_data_frame,
                                                                      stim_param = self.stimulus_parameter(),
                                                                      resp_param = self.response_parameter() )
                 #--- store dataframe to HDF format

               else:
                  stimulus_data_frame['bads']= np.zeros_like( stimulus_data_frame['onset'],dtype=np.int8 )
              
              #--- for later mark selected epoch as 1
               #stimulus_data_frame['selected']         = np.zeros_like( stimulus_data_frame['onset'],dtype=np.int8 )
               #stimulus_data_frame['weighted_selected']= np.zeros_like( stimulus_data_frame['onset'],dtype=np.int8 ) 
              
               stimulus_data_frame['selected']         = np.ones_like( stimulus_data_frame['onset'],dtype=np.int8 )
               stimulus_data_frame['weighted_selected']= np.ones_like( stimulus_data_frame['onset'],dtype=np.int8 ) 
               
               key = self.hdf_node_name_epocher +'/'+condi
               storer_attrs = {'epocher_parameter': self.parameter,'info_parameter':stimulus_info}
               self.hdf_obj_update_dataframe(stimulus_data_frame.astype(np.int32),key=key,**storer_attrs )

       #--- write stat info into hdf and as csv/txt
        df_stats = self.events_condition_stats(save=True)
        if self.parameter:
            
           key = self.hdf_node_name_conditions +"/statistic/"
           storer_attrs = {'epocher_parameter': self.parameter,'info_parameter':stimulus_info}
           self.hdf_obj_update_dataframe(df_stats.astype(np.float32),key=key,**storer_attrs )
        
        
        for ch in self.event_data_frames:
            key = self.hdf_node_name_channels +"/"+ ch
            storer_attrs = {'info_parameter': self.event_data_info[ch]}
            self.hdf_obj_update_dataframe(self.event_data_frames[ch].astype(np.int64),key=key,**storer_attrs )
        
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
            if w.startswith(self.hdf_node_name_epocher):
               index_keys.append( w.replace(self.hdf_node_name_epocher,'').replace('/', '') )

        df_stat = pd.DataFrame(index=index_keys,columns = cols)
        
       # s = Series(randn(5), index=['a', 'b', 'c', 'd', 'e'])
       #d = {'one' : Series([1., 2., 3.], index=['a', 'b', 'c']),
#   ....:      'two' : Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
        idx = 0

        for condi in index_keys:
            k=self.hdf_node_name_epocher + '/' + condi

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
           # pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)
           pd.options.display.float_format = '{:8,.3f}'.format
           print df_stat
           print"\n\n"
           if save :
              print " --> Stat DataFrame saved as: "+ fcsv

        return df_stat
            
jumeg_epocher_events = JuMEG_Epocher_Events()

"""
template example
{
"default":{
           "experiment"       : "FreeView",
           "postfix"          : "test",
           "time_pre"         : -0.20,
           "time_post"        :  0.65,
           "baseline"         : {"type":"avg","channel":"stimulus","output":"onset","baseline": [null,0]},
           "stimulus_channel" : "StimImageOnset",
           "response_channel" : "ETevents",
           "marker_channel"   : "StimImageOnset",
           "marker_type"      : "onset",
           "response_matching": false,
           "response_matching_type": "hit",
           "ctps"             : {"time_pre":-0.20,"time_post":0.50,"baseline":[null,0]},
           "reject"           : {"mag": 5e-9},
          
           "ETevents":{
                       "events":{
                                  "stim_channel"   : "ET_events",
                                  "output"         : "onset",
                                  "consecutive"    : true,
                                  "min_duration"   : 0.0001,
                                  "shortest_event" : 1,
                                  "mask"           : 0
                                 },
                        "and_mask"          : 255,
                        "system_delay_ms"   : 0.0,
                        "event_id"          : null,
                        "window"            : [0.0,7.0],
                        "counts"            : null,
                        "system_delay_ms"   : 0.0,
                        "include_early_ids" : "all"
                        
                       },
                     
           "StimImageOnset":{
                       "events":{
                                  "stim_channel"   : "STI 014",
                                  "output"         : "onset",
                                  "consecutive"    : true,
                                  "min_duration"   : 0.0001,
                                  "shortest_event" : 1,
                                  "mask"           : 0
                                 },
                        
                        "event_id"        : 84,        
                        "and_mask"        : 255,
                        "system_delay_ms" : 0.0,
                        "window"            : [-7.0,0.0],
                        "counts"            : null,
                        "system_delay_ms"   : 0.0,
                        "include_early_ids" : "all"
                        },                                                

            "IOD":{
                        "events":{
                                  "stim_channel"   : "STI 013",
                                  "output"         : "onset",
                                  "consecutive"    : true,
                                  "min_duration"   : 0.002,
                                  "shortest_event" : 2,
                                  "mask"           : 0
                                 },
                        "and_mask"             : 128,
                        "window"               : [-0.10,0.20],
                        "counts"               : 1,
                        "system_delay_ms"      : 0.0,
                        "include_early_ids"    : null,
                        "event_id"             : 128
                       }                                    
              },
 
"FVImoBc":{
         "postfix"   :"FVImo", 
         "info"      :"freeviewing, image onset, iod onset, baseline correction",
         "time_pre"  : -0.20,
         "time_post" :  7.00,
         "baseline"  : null,
         "stimulus_channel" : "StimImageOnset",
         "response_channel" : "IOD",
         "marker_channel"   : "StimImageOnset",
         "response_matching": true,
         "marker_type"      : "onset",
         "StimImageOnset"   : {"event_id":94},
         "IOD"              : {"event_id":128}
         }, 

"FVsaccardeBc":{
         "postfix"   :"FVSac", 
         "info"      :"freeviewing, saccard onset, baseline correction",
         "time_pre"  : -0.20,
         "time_post" :  0.65,
         "baseline"  : null,
         "stimulus_channel" : "ETevents",
         "response_channel" : "StimImageOnset",
         "marker_channel"   : "ETevents",
         "response_matching": true,
         "marker_type"      : "onset",
         "StimImageOnset"   : {"event_id":94},
         "ETevents"         : {"event_id":250}
         }, 

"FVfixationBc":{
         "postfix"   :"FVfix", 
         "info"      :"freeviewing, fixation onset, baseline correction",
         "time_pre"  : -0.20,
         "time_post" :  0.65,
         "baseline"  : {"type":"avg","channel":"stimulus_channel","output":"onset","baseline": [null,0]},
         "stimulus_channel" : "StimImageOnset",
         "response_channel" : "ETevents",
         "marker_channel"   : "StimImageOnset",
         "response_matching": true,
         "marker_type"      : "onset",
         "StimImageOnset"   : {"event_id":94},
         "ETevents"         : {"event_id":251}
         }, 
"""
