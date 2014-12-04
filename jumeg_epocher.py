import os
import numpy as np
import pandas as pd 
# import matplotlib.pylab as pl
import mne

from jumeg.jumeg_base import jumeg_base
from jumeg.template.jumeg_template import JuMEG_Template_Epocher

class JuMEG_Epocher(JuMEG_Template_Epocher):
    def __init__ (self,template_name="DEFAULT",do_run=False,do_average=True,verbose=False,save=True):
        super(JuMEG_Epocher, self).__init__()
        
        self._do_run        = do_run
        self._do_average    = do_average
        self._verbose       = verbose
        self._save          = save
        self._epocher_postfix      = '-epocher.hdf5'
        self._epocher_stat_postfix = '-epocher-stats.csv'
      
        self._epocher_file_name = None
      #--- constant 
        self._IDX_HIT     = 3
        self._IDX_WRONG   = 2
        self._IDX_TOEARLY = 1
        self._IDX_MISSED  = 0
        self.rt_type_list  = ['MISSED', 'TOEARLY', 'WRONG', 'HIT']
        
        self._epoch_data_frame_stimulus_cols = ['id','onset','offset']
        self._epoch_data_frame_response_cols = ['rt_type','rt_tsl','rt_id','rt_tsl_onset','rt_tsl_offset','rt_index','rt_counts','bads']     

#-- inherited from JuMEG_Template_Epocher
        self.template_name = template_name
        self.verbose       = verbose

#=== getter & setter  
#--- flags    
    def _set_do_run(self, v):
         self._do_run = v
    def _get_do_run(self):
         return self._do_run
    do_run = property(_get_do_run,_set_do_run)
#---
    def _set_do_average(self, v):
         self._do_average = v 
    def _get_do_average(self):
         return self._do_average
    do_average = property(_get_do_average,_set_do_average)     
#---
    def _set_save(self, v):
         self._save = v 
    def _get_save(self):
         return self._save
    do_save = property(_get_save,_set_save)

#--- epocher file (output in hdf5)        
    def _set_epocher_postfix(self, v):
         self._epocher_postfix = v
    def _get_epocher_postfix(self):
         return self._epocher_postfix
    epocher_postfix = property(_get_epocher_postfix,_set_epocher_postfix)
#--- epocher stat file (output in hdf5)        
    def _set_epocher_stat_postfix(self, v):
         self._epocher_stat_postfix = v
    def _get_epocher_stat_postfix(self):
         return self._epocher_stat_postfix
    epocher_stat_postfix = property(_get_epocher_stat_postfix,_set_epocher_stat_postfix)

#--- epocher filename for HDF obj
    def _get_epocher_file_name(self):
         return self._epocher_file_name
    epocher_file_name = property(_get_epocher_file_name)
    
    def epocher_file_name_init(self,fname=None,raw=None):
        if fname is None:
           if raw : 
              fname = raw.info['filename']
           else:
              fname = "TEST"
        self._epocher_file_name = fname.split('c,rfDC')[0].strip('.fif') +'_' +self.template_name + self.epocher_postfix 
        self._epocher_file_name = self._epocher_file_name.replace('__', '_') 
        return self._epocher_file_name

#--- init epocher output pandas HDF obj
    def epocher_HDFStoreObj_init(self,fname=None,raw=None,overwrite=True):
        """
        create epocher pandas HDF5-Obj and file 
        input: 
            fname : fif-filename 
            raw       = raw  => mne raw obj
            overwrite = True => will overwrite existing output file
        return: pandas.HDFStore obj   
        """
        import pandas as pd
        fHDF = self.epocher_file_name_init(fname=fname,raw=raw)
        if os.path.exists(fHDF):
           if overwrite:
              os.remove(fHDF)
        return pd.HDFStore( fHDF )
  
    def epocher_HDFStoreObj_open(self,fname=None,raw=None):
         """
         open epocher pandas HDF5-Obj 
         input: 
            fname : fif-filename 
            raw   = raw => mne raw obj
         return: pandas.HDFStore obj
         """
         return self.epocher_HDFStoreObj_init(self,fname=fname,raw=raw,overwrite=False)
#---
    def epocher_find_epochs(self,raw,**param):
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
                    #rt_tsl      : NAN => np array with reaction time [TSL]  
                    #rt_tsl_onset: NAN => np array with response onset [TSL] 
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
        df = pd.DataFrame(columns = self._epoch_data_frame_stimulus_cols)

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
    def epocher_response_matching(self,raw,stim_df=None,resp_df=None, **param ):
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
                rt_tsl        : response time as tsl (RT)
                rt_id         : event id button press code
                rt_tsl_onset  : response onset  [tsl]
                rt_tsl_offset : response offset [tsl]
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
        for x in self._epoch_data_frame_response_cols :
            stim_df[x]= np.NaN 
            
       #--- convert rt window [ms] into tsl
        (r_window_tsl_start, r_window_tsl_end ) = raw.time_as_index( param['response']['window'] );
                
       #--- get respose code -> event_id [int or string] as np array
        r_event_id = jumeg_base.str_range_to_numpy( param['response']['event_id'] )
       
       #--- ck if any toearly-id is defined, returns None if not
        r_event_id_toearly = jumeg_base.str_range_to_numpy( param['response']['include_early_ids'] )
        
       #--- get output type: onset or offset 
        stim_output_type = param['stimulus']['events']['output']
        resp_output_type = param['response']['events']['output']
        STIM_TSLs = stim_df[stim_output_type]
        RESP_TSLs = resp_df[resp_output_type]
          
       #--- loop for all stim events    
        ridx = 0  
       #--- get rt important part of respose df
        RESP_TSLs = resp_df[resp_output_type]  #[ridx:]
        
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
                      ridx                       = res_df_early.index[0]
                      stim_df.rt_counts[idx]     = res_df_in.index.size
                      stim_df.rt_tsl_onset[idx]  = resp_df.onset[ridx]
                      stim_df.rt_tsl_offset[idx] = resp_df.offset[ridx]
                      stim_df.rt_id[idx]         = resp_df.id[ridx]
                      stim_df.rt_index[idx]      = ridx
                      stim_df.rt_type[idx]       = self._IDX_TOEARLY
                      continue
               
          #--- find index of responses from window-start till end of res_event_type array [e.g. onset / offset]
            res_df_in = resp_df[ ( st_window_tsl0 <= RESP_TSLs ) & ( RESP_TSLs <= st_window_tsl1) ] 
          
          #--- MISSED response  
            if res_df_in.empty : continue    
            
          #--- WRONG or HIT;     
            if res_df_in.index.size > 0 :
               ridx = res_df_in.index[0]
               stim_df.rt_counts[idx]     = res_df_in.index.size
               stim_df.rt_tsl_onset[idx]  = resp_df.onset[ridx]
               stim_df.rt_tsl_offset[idx] = resp_df.offset[ridx]
               stim_df.rt_type[idx]       = self._IDX_WRONG
               stim_df.rt_id[idx]         = resp_df.id[ridx]
               stim_df.rt_index[idx]      = ridx
          #--- HIT; ck number of responses; ck pressed buttons; wrong if more than count
               if (  stim_df.rt_counts[idx] <= param['response']['counts'] ):
                   if res_df_in.id.isin( r_event_id ).all :
                      stim_df.rt_type[idx] = self._IDX_HIT
            
         
      #---  calc reaction time (rt in tsl)
        if stim_output_type == 'offset' :
           sto = stim_df.offset
        else:
           sto = stim_df.onset
        if resp_output_type == 'offset' :
           rto = stim_df.rt_tsl_offset
        else:
           rto = stim_df.rt_tsl_onset
     
        stim_df.rt_tsl = rto - sto
        
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
    def epocher_store_epochs_to_hdf5(self,raw,condition_list=None,overwrite_epocher =True):
        '''
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
             overwrite_epocher : flag for overwriting output HDF file
                                 default: overwrite_epocher=True
                                 
         return:
                 HDF filename         
        '''
        import pandas as pd
         
       #---  init obj
        HDFobj = self.epocher_HDFStoreObj_init(raw=raw, overwrite=overwrite_epocher)

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
            parameter = self.update_and_merge_dict(parameter,param)
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
               stimulus_data_frame,stimulus_info = self.epocher_find_epochs(raw,**parameter[ parameter['stimulus_channel'] ])
              
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
                
                #--- response init dict's & dataframes
                  response_info          = dict() 
                  response_data_frames   = None  
     
                 #--- look for all responses => 'event_id' = None
                  res_param = parameter[ parameter['response_channel'] ].copy()
                  res_param['event_id'] = None
                 #--- get epochs from events as pandas data-frame   
                  response_data_frame,response_info = self.epocher_find_epochs(raw,**res_param)
               
                  if self.verbose:
                     print "---> Response Epocher Events Data Frame [respnse channel] : " + parameter['response_channel']
                     print response_data_frame
                     print"\n\n"
                 #--- update stimulus epochs with response matching 
                  stimulus_data_frame = self.epocher_response_matching(raw,stimulus_data_frame,response_data_frame,**parameter )
                 
                 #--- store dataframe to HDF format 
                  key         = '/'+condi+'/epocher'
                  HDFobj[key] = stimulus_data_frame
                  storer      = HDFobj.get_storer(key)
                 #--- save condition parameter in HDF attributes
                  storer.attrs.epocher_parameter = parameter
                 #--- save condition info in HDF attributes
                  storer.attrs.epocher_info      = stimulus_info
                  HDFobj.flush()
              
                  if self.verbose :
                     print "\n---> save epocher condition to HDF5: " + condi +" ==> " +key  
                     print "---> PARAMETER => stored as attrs.epoch_parameter"
                     print parameter
                     print "\n---> INFO => stored as attrs.epoch_info"
                     print stimulus_info
                     print"\n --> HDF obj"
                     print HDFobj
                     print "\n---> done : " + condi +"\n\n" 
                     
        self.epocher_condition_stats(HDFobj, save=True)
                     
        fhdf= HDFobj.filename
        HDFobj.close()
        
        print" ---> DONE save epocher data into HDF5 :"
        print"  --> " + fhdf +"\n\n"
        return fhdf
       
    def epocher_condition_stats(self, HDFobj, save=False):
        """
        return:
        < pandas data frame>
        """
        import pandas as pd 
       #--- ck error 
        if HDFobj is None:
           return
        if not HDFobj.is_open :
          return
       #---  
      
        cols = ['EvID','Hit','Wrong', 'TOEarly', 'Missed', 'RTmean', 'RTstd', 'RTmin', 'RTmax']
        #Index([u'id', u'onset', u'offset', u'rt_type', u'rt_tsl', u'rt_id', u'rt_tsl_onset', u'rt_tsl_offset', u'rt_index', u'rt_counts', u'bads'], dtype='object')
        index_keys = [w.replace('/epocher', '').replace('/', '')  for w in HDFobj.keys()]  
       
        df_stat = pd.DataFrame(index=index_keys,columns = cols)
        idx = 0
        
        for k in HDFobj.keys():
            print k
            df = HDFobj[k]
           #--- get sampling frquency from  attrs epocher_info
            Hstorer = HDFobj.get_storer(k)
            try:
                sfreq = Hstorer.attrs.epocher_info['sfreq']
            except:
                sfreq = 1.0
                
            #df_stat['EvID'][idx] = np.array_str( np.unique(df.id) )
            df_stat['EvID'][idx] = np.array_str( np.unique(df.id) )
        
           #--- missed 
            missed_idx = df[ df['rt_type'] == self._IDX_MISSED ].index
            if missed_idx.size :
               df_stat['Missed'][idx] = missed_idx.size
           #--- early
            toearly_idx = df[ df['rt_type'] == self._IDX_TOEARLY ].index
            if toearly_idx.size :
               df_stat['TOEarly'][idx] = toearly_idx.size
           #--- wrong
            wrong_idx = df[ df['rt_type'] == self._IDX_WRONG ].index
            if wrong_idx.size:
               df_stat['Wrong'][idx] = wrong_idx.size
            
           #--- hit
            hit_idx = df[ df['rt_type'] == self._IDX_HIT ].index
            if hit_idx.size:
               df_stat['Hit'][idx] = hit_idx.size
            #--- RTmean   
               df_stat['RTmean'][idx] = df.rt_tsl[hit_idx].mean() / sfreq
               df_stat['RTstd'][idx]  = df.rt_tsl[hit_idx].std() / sfreq
               df_stat['RTmin'][idx]  = df.rt_tsl[hit_idx].min() / sfreq
               df_stat['RTmax'][idx]  = df.rt_tsl[hit_idx].max() / sfreq
            
            idx += 1
            
        #--- save stats data frame to csv
        fcsv = None
        if save:
           fcsv = HDFobj.filename.replace(self.epocher_postfix,self.epocher_stat_postfix)
           df_stat.to_csv(fcsv)
           
        if self.verbose :
           print"\n  --> Condition Statistic Data Frame\n"
           print df_stat
           print"\n\n"
           if save :
              print " --> Stat DataFrame saved as: "+ fcsv
              
        return df_stat

    def apply_epochs_to_hdf(self, fname,raw=None,condition_list=None,picks=None,**kwargv):
        '''
        test
        '''
        if kwargv['template_name']:
           self.template_name = kwargv['template_name']
       
        if kwargv['do_run']:
           self.do_run = kwargv['do_run']
           
        if kwargv['verbose']:
           self.verbose = kwargv['verbose']
     
        self.update_template_file()
       
        fhdf = None
        
        if self.do_run :
           if raw is None:
              raw   = mne.io.Raw(fname,preload=True)
              
           fhdf = self.epocher_store_epochs_to_hdf5(raw,condition_list=condition_list)
            #if picks is None :
           #   picks = jumeg_base.pick_channels_nobads(raw)
          
           #self.do_apply_epocher(raw, picks=picks)
       
        print "===> DONE  apply epoches to HDF\n"
       
        return (fname,raw,fhdf)
             
#fhdf ='/localdata/frank/data/Chrono/mne/201195/Chrono01/110516_1413/1/201195_testTEST01_epocher.hdf5'
# HDF = pd.HDFStore(fhdf)
# hs  = HDF.get_storer('/LRrt/df')
# hs.attrs.epocher_parameter['averager_channel']

#    
        
jumeg_epocher = JuMEG_Epocher()
