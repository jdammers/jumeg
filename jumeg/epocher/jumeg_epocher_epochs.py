#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:00:03 2018

@author: fboers

ToDo:
baseline correction:
check in confidence intervall for non artifac intervall e.g. non-eyeblink 

"""
import os,sys,logging
from copy import deepcopy
import numpy as np
import mne

from jumeg.base.jumeg_base               import jumeg_base
from jumeg.epocher.jumeg_epocher_events  import JuMEG_Epocher_Events,JuMEG_Epocher_Events_Channel,JuMEG_Epocher_Channel_Baseline
from jumeg.epocher.jumeg_epocher_plot    import jumeg_epocher_plot as jplt

#--- setup logger
logger = logging.getLogger('jumeg')
__version__="2019.05.14.001"

class JuMEG_Epocher_Marker(JuMEG_Epocher_Events_Channel):
    '''
    class for marker and response channel
            
    ''' 
    def __init__(self,label=None,parameter=None):
        super(JuMEG_Epocher_Marker,self).__init__(label=label,parameter=parameter)
        self.baseline = JuMEG_Epocher_Channel_Baseline(parameter=parameter)
   #--- 
    @property
    def postfix(self):   return self._param['postfix']  
    @postfix.setter
    def postfix(self,v): self._param['postfix']=v  
   #--- 
    @property
    def file_postfix(self):   
        return self.postfix +'-'+ self.type_result[0].lower()
   #--- 
    @property
    def type_id(self): return self.prefix +'_id'  
   
   #---type_result: "hit","wrong","missed"
    @property
    def type_result(self): return self.get_channel_parameter(key="type_result").upper()  
    @type_result.setter
    def type_result(self,v): self.set_channel_parameter(key="type_result",val=v)
 
 
class JuMEG_Epocher_OutputMode(object):
    """
    
    """
    __slots__ =["events","epochs","evoked","stage","use_condition_in_path","path"]
    
    def __init__(self,**kwargs):
        super().__init__()
        self._init(**kwargs)
        
    def get_parameter(self,key=None):
       """
       get  host parameter
       :param key:)
       :return: parameter dict or value of  parameter[key]
       """
       if key: return self.__getattribute__(key)
       return {slot: self.__getattribute__(slot) for slot in self.__slots__}
  
    def _init(self,**kwargs):
      #--- init slots
       for k in self.__slots__:
           self.__setattr__(k,None)
       self._update_from_kwargs(**kwargs)

    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)

    def _update_from_kwargs(self,**kwargs):
        if not kwargs: return
        for k in kwargs:
            try:
               self.__setattr__(k,kwargs.get(k))
             # self.__setattr__(k,kwargs.get(k)) #,self.__getattribute__(k)))
            except:
               pass#
            
   #--- update output_path
    def make_output_path(self,condition=None,**kwargs):
        """
        makes dirtree along stage and condition
        :param condition:
        :param kwargs:
        :return:
        stage
        
        """
        
        self._update_from_kwargs(**kwargs)
        self.path = self.stage
        
        if self.path:
           self.path = jumeg_base.expandvars(self.path)
           if self.use_condition_in_path:
              if condition:
                 self.path = os.path.join(self.path,condition)
           try:
              os.makedirs(self.path,exist_ok=True)
           except:
              msg=[" --> can not create epocher epochs output path: {}".format(self.path),"  -> ouput mode parameter: "]
              for k in self.__slots__:
                  msg.append("  -> {} : {}".format(k,self.__getattribute__(k)))
              logger.exception("\n".join(msg))
              return None
            
        return self.path
     
      

class JuMEG_Epocher_Epochs(JuMEG_Epocher_Events):
    """ CLS JuMEG_Epocher_Epochs
    input:
         raw           : raw obj
         fhdf          : HDF fileneame
         condition_list: list of conditionto process [None: process all]
         fif_postfix   : output postfix [evt]
         event_extention: [.eve]
         save_mode     : dict with flags for export events,epochs and evoked and parameter
                        events: True,epochs: True,evoked: True
                        parameter:{"stage":None,"use_condition_in_path":True}
                            if stage use stage as output path
                            if "use_condition_in_path" add condition to stage as output path
       
        time            : dict
               time_pre : [None] in sec
               time_post: [None] in sec
                      
        baseline       : dict 
               type    : avg,median [None]
               channel : stimulus,response [None]
               output  : [None]
               baseline: [None] baseline time interval in sec e.g [None,0]
                -->  baseline: {"type":"avg","channel":"stimulus","output":"onset","baseline": [null,0]},
               
        exclude_events : dict [None]
              ecg_events: dict [None] [tmin,tmax] 
              eog_events: dict [None] [tmin,tmax]              
                       
        weights        : dict select epochs for output avg,evt   
              mode     : [equal] overall conditions same number of epochs  
              method   : [median] choose reaction time 
              skipp_first: [None] skip first N epochs
               -->  weights:{"mode":"equal","method":"median","skipp_first":3}
        
         output_mode    : dict of
                         events:True
                         epochs:True
                         evoked:True
                         stage :None
                         use_condition_in_path:True
         mne parameter:
          picks          : channels to process [None: all]
          reject         : mne rejection [None]
          proj           : False
         
    """
    def __init__ (self,raw=None,fhdf=None,fname=None,condition_list=None,
                  fif_postfix="evt",event_extention=".eve",type_result=None,
                  picks=None,reject=None,proj=False,template_name=None,
                  time={"time_pre":None,"time_post":None},
                  baseline={"type":None,"type_input":None,"baseline":None},
                  #baseline={"type":None,"channel":None,"output":None,"baseline":None},
                  weights=None,     
                  output_mode={"events":True,"epochs":True,"evoked":True,"stage":None,"use_condition_in_path":True},
                  exclude_events = {"eog_events":None,"ecg_events":None},
                  verbose=False):

        super(JuMEG_Epocher_Epochs, self).__init__()
       
      #--- obj for dict from parameter[condi][marker] or[response]  
        self.marker         = None
        self.evt_excluded   = None        
        self.event_id       = None
        
        self.template_name  = template_name
        self.raw            = raw
        self.fhdf           = fhdf
        self.fname          = fname
        self.comdition_list = condition_list
        self.fif_postfix    = fif_postfix 
        self.event_extention= event_extention 
        self.baseline       = baseline
        self.time           = time
        self.weights        = weights
        self.exclude_events = exclude_events
        self.picks          = picks
        self.reject         = reject
        self.proj           = proj                     
        self.verbose        = verbose
        self.type_result    = type_result
        self.OutputMode     = JuMEG_Epocher_OutputMode(**output_mode)
#--- 
    def apply_hdf_to_epochs(self,**kwargs):
        """ apply(**kwargs) refer to CLS docstring """
        self._epochs_update(**kwargs)
        self._epochs_run()
        return self.raw,self.fname 
   
#---
    def _epochs_update(self,**kwargs):
        self.condition_list = []
      
        self._epochs_update_data(**kwargs)
     
        self.evt_excluded = None
        
       #--- init exclude_events e.g. {eog onset:[tmin,tmax] }
       #--- ToDo: store artifact events in hdf
        if self.exclude_events:
           self.evt_excluded = self._epochs_update_artifact_time_window(**self.exclude_events)            
    
#--- 
    def _epochs_get_events(self,condi,ck_weights_skip_first=False,ck_weights_equalize=False):
        """
        get events: 
            read hdf data
            update marker time
            check weights for skip first number of events 
            check weights for equalize number of events over conditions
        
        Parameters:
        -----------     
        condition
        ck_weights_skip_first : falg to check to skipp first number of events <False>
        ck_weights_equalize   : flag to check for equalize number of events overall conditios <False>
        
        Results:
        --------
        pandas dataframe
        dict with event and epoch structure
        dict of epoch parameter from HDF
        dict of info  parameter from HDF
        """
        
      #--- read hdf get dataframe
        #print(condi)
        df,ep_param,info_param = self._epochs_read_hdf_data(condi) 
        
        #print("  --> EP param")
        #print(ep_param)
       
        self.response = JuMEG_Epocher_Events_Channel(label="response",parameter=ep_param)
        
      #--- check in case of changing epoch time 
        self._epochs_update_marker_time()
      #--- matching e.g. hits 
        df,events_idx = self._epochs_hdf_data_get_event_index(df)
        event_cnts = events_idx.shape[0]
      #--- 
        if ck_weights_skip_first:
           df,events_idx = self._epochs_check_weights_to_skip_first(df,events_idx)
      #--- update event index weight 
        if ck_weights_equalize:
           df,events_idx = self._epochs_check_weights_to_equalize(df,events_idx)  
      #--- init dict with mne-events & baseline-event
        evt = self._epochs_init_events(df,events_idx)
         
      #---update HDF: store df with updated bads & selected & restore user-attribute    
        self._epochs_save_hdf_data(df,ep_param,info_param,condi)
        
      #--- 
        if self.verbose:
           msg=["---> Export Events from HDF to MNE-Events for condition: " + condi,
                "  -> number of original events   : {}".format(event_cnts),
                "  -> number of events            : {}".format( evt['events'].shape[0]),
                "  -> check weights to skip first : {}".format(ck_weights_skip_first),
                "  -> check weights to equalize   : {}".format(ck_weights_equalize)]
           
           bads = df[ self.marker.type_output ][ (df['bads']== self.idx_bad) ]
           
           msg.append("  -> bad events : " + str(bads.shape))
           if bads.shape[0]:
              msg.append("  -> bads: {}".format(bads.to_string()))
           logger.info( "\n".join(msg) )
           
        return(df,evt,ep_param,info_param)
       
#--- 
    def _epochs_run(self):
        """ apply(**kwargs) refer to CLS docstring """
        self.event_id = dict()
     
        if not self.hdf_obj_is_open():
           logger.exception(" --> no <HDF obj> defined please call <apply_epocher> first\n"+
                        "  -> HDF filename: {}\n".format(self.fhdf)+
                        "  -> FIF filename: {}\n".format(self.fname))
           return
       #----
        for condi in self.condition_list:
            logger.info( "---> JuMEG EpocherEpochs Apply condition: "+condi)
          #---
            df,evt,ep_param,info_param = self._epochs_get_events(condi,ck_weights_skip_first=True,ck_weights_equalize=False)
       
            #print("EP param")
            #print(ep_param)
            #print("TEST")
            #print(self.param)
            
          #---     
            if not evt['events'].size: continue
         
          #--- avg epochs events 
            evt = self._epochs_get_epochs_and_apply_baseline(self.raw,evt=evt)
      
          #--- set event_id  and number of used epochs  ep.selection   
            self.event_id[condi] = {'id': np.unique(self.marker.type_id),'trials': evt['events'].shape[0],'trials_weighted':0}
          
          #--- save all epochs no weights
            self._epochs_save_events(evt=evt,condition=condi,postfix=self.marker.postfix,postfix_extention=self.fif_postfix,weighted=False,
                                     picks=self.picks,reject=self.reject,proj=self.proj)
         
          #--- end for condi
     
  
      #--- ck weighted events
      #--- "weights":{"mode":"equal","selection":"median","skipp_first":null},
        if self.weights:
           if self.weights.get('mode') == 'equal':
              self.Log.info("\n ---> Applying Weighted Export Events")
              
              self.weights['min_counts'] = self.event_id[ self.event_id.keys()[0] ]['trials'] 
              for condi in self.event_id.keys() :
                  if self.event_id[ condi ]['trials'] < self.weights['min_counts']:
                     self.weights['min_counts'] = self.event_id[ condi ]['trials']
           
              for condi in self.event_id.keys():
                 #---
                  df,evt,ep_param,info_param = self._epochs_get_events(condi,ck_weights_skip_first=False,ck_weights_equalize=True)
    
                 #--- weighted avg epochs events
                  if not evt['events'].size: continue
                   
                 #--- avg epochs events 
                  evt = self._epochs_get_epochs_and_apply_baseline(self.raw,evt=evt)
          
                  self.event_id[condi]['trials_weighted'] = evt['events'].shape[0]
                 #---
                  self._epochs_save_events(evt=evt,condition=condi,postfix=ep_param['postfix']+'-W',postfix_extention=self.fif_postfix,
                                           weighted=True,picks=self.picks,reject=self.reject,proj=self.proj)
                     
      
        self.HDFobj.close()        
          
  
#---  
    def _epochs_read_hdf_data(self,condi): #,exclude_events=None,time=None,baseline=None,weights=None):
        """export HDF events to <mne events> structure for mne.Epochs
        read HDFobj attributes <epocher parameter> and <info parameter>
        init event dict
        apply weights to events and pandas dataframe
        update dataframe and save to HDFobj
        
        Parameters
        ----------
          condi          : condition name <None>  
          exclude_events : dict <None> e.g.:{ecg onset:[tmin,tmax],eog onset:[tmin,tmax]}
          time           : dict <None>
                           time_pre : in sec
                           time_post: in sec
          weights        : dict <None>
                           e.g:
                           {"mode":"equal_counts","selection":"median","skipp_first":None,min_counts=None},
    
        Results
        ----------
        <events>          : dict: events,event_id,bc:{events,event_id,baseline}
        <epoch parameter< : dict: response_matching,response_matching_type,marker_type
        <info  parameter> : dict to HDFobj key info
        <pandas dataframe> 
        """       
     
        self.marker   = None

        logger.info("---> EPOCHER HDF to MNE events -> extract condition : " + condi)
      
       #--- check and get hdf key make node
        ep_key = self.hdf_get_hdf_key(condi)
        if not ep_key :
           logger.exception("  -> No HDF-key found for condition: {}\n".format(condi)+
                            "  -> HDF file: {}".format(self.hdf_filename))
           sys.exit()

       #--- get pandas data frame from HDF
        df = self.hdf_obj_get_dataframe(ep_key)
       #--- get stored attributes -> epocher_parameter -> ...
        ep_param   = self.hdf_obj_get_attributes(key=ep_key,attr=self.hdf_obj_attribute_epocher)
        info_param = self.hdf_obj_get_attributes(key=ep_key,attr=self.hdf_obj_attribute_info)   
        
        if self.debug:
           logger.debug("  -> HDF attr: epoch parameter <ep_param>  : {}\n".format(self.pp_list2str(ep_param) )+
                        "  -> HDF attr: info  parameter <info_param>: {}\n".format(self.pp_list2str(info_param)))
        
        if "marker" in ep_param:  
           self.marker = JuMEG_Epocher_Marker(label="marker",parameter=ep_param)
           if self.type_result:
              self.marker.type_result = self.type_result 
        else: return
           
        return df,ep_param,info_param 
    
#---
    def _epochs_hdf_data_get_event_index(self,df):
        """ get correct event index form dataframe
            depending on marker.type_result (hit,wrong,missed) and bads-column
        
        Parameters
        ----------
         pandas dataframe
                
        Results
        -------
        dataframe,event indx array
        """
        event_idx = None
        mrk_value = self.rt_type_as_index( self.marker.type_result )
        mrk_result= self.marker.type_result
       
        if self.response.matching:
           mrk_type = self.response.prefix
        else:
           mrk_type = self.marker.prefix
        
        if not mrk_type.endswith("_type"):
           mrk_type += "_type"

       #--- old: find all xyz_type => img_type iod_type sac_type
        # idx= np.where( df.filter(regex='_type$') == mkr_type    )[0]
        
        mrk_idx = np.where( df[mrk_type] == mrk_value)[0]
        idx =  np.where( df.loc[mrk_idx,'bads'] != self.idx_bad )[0]
        event_idx = deepcopy(mrk_idx[idx])
   
      #--- reset df selection
        df['selected']          = 0
        df['weighted_selected'] = 0
          
        if event_idx.size:
           df.loc[event_idx,'selected']          = 1
           df.loc[event_idx,'weighted_selected'] = 1
       
        if self.verbose:
           logger.info("---> input DataFrame:\n {}\n".format(df)+
                       "\n  -> Matching event index counts: {}\n  -> event idx: {}\n".format(event_idx.size,event_idx)+
                       "  -> matched events in DataFrame:\n {}\n".format( df.loc[event_idx])+
                       "\n --> epocher_hdf_data_get_event_index:\n"+
                       "  -> response matching : {}\n".format(self.response.matching)+
                       "  -> marker type       : {}\n".format(mrk_type)+
                       "  -> marker type result: {}\n".format(self.marker.type_result)+
                       "  -> marker type value : {}\n".format(mrk_value)+
                       "  -> number of events  : {}".format(event_idx.size) )
                      
        return df,event_idx

#---
    def _epochs_check_weights_to_skip_first(self,df,events_idx,weights=None):
        """ check in weights for skip first
        
        Parameters:
        -----------
        pandas dataframe
        np array of event index to use in dataframe
        weights : dict weight parameter <None>
                  if None use global,default weight parameter
                  else overwrite global weight parameter
                  e.g: {"mode":"equal_counts","selection":"median","skipp_first":None,min_counts=None},
        Results
        -------
        pandas dataframe, weighted event index
         
        """
        if weights:
           self.weights = weights
            
      #--- ck for skip first number of trails  
        if self.weights:
           if self.weights.get("skip_first"):
              cnt = self.weights.get("skip_first")
              if self.is_number(cnt):
                 df.loc[ events_idx[0:cnt], 'weighted_selected'] = 0
                 if ( 0 < cnt <= events_idx.size) :
                    events_idx = np.delete( events_idx,np.arange(cnt) )  
        return df,events_idx
    
#---            
    def _epochs_check_weights_to_equalize(self,df,events_idx,weights=None):      
        """ update weights to reduce/equalize number of trials/events for all condition
  
        Parameters
        ----------
        pandas dataframe
        np array of event index to use in dataframe
        weights : dict weight parameter <None>
                  if None use global,default weight parameter
                  else overwrite global weight parameter
                  e.g: {"mode":"equal_counts","selection":"median","skipp_first":None,min_counts=None},
        Results
        -------
        pandas dataframe, weighted event index
        """
       #--- 
        if weights:
           self.weights = weights
           
        if not self.weights:
           return df,events_idx
            
        w_events_idx = None
        w_value      = None
        method       = ''     
    
    #---   
        if self.weights.get('min_counts'):
           df['weighted_selected'] = 0 
           method = self.weights['method']
           
           if self.weights['min_counts']:
              assert self.weights['min_counts'] <= events_idx.size,"!!!ERROR minimum required trials greater than number of trials !!!"
              w_value = 0.0
              
              if self.weights['min_counts'] < events_idx.size:
                 if self.marker.type_input != self.marker.type_output:  # matching
                    w_val= df[ self.marker.type_input ][events_idx] - df[ self.marker.type_output ][events_idx] 
                    if self.weights['method']=='median':
                       w_value = np.median(w_val) 
                    elif self.weights['method']=='mean':
                       w_value = np.mean(w_val)
                   #--- find minimum from median as index from events_idx => index of index !!!self.fif_postfix
                    w_idx = np.argsort( np.abs( np.array( w_val - w_value )))
                    w_events_idx = events_idx[ w_idx[ 0:self.weights['min_counts'] ] ]
                 else: # input==output no matching use random
                    np.random.shuffle(events_idx) # inplace
                    method = 'random'
                    w_events_idx = np.array( events_idx[ 0:self.weights['min_counts'] ] )
                
                 w_events_idx.sort()
                 df.loc[ w_events_idx,'weighted_selected' ] = 1
                 
              elif weights['min_counts'] == events_idx.size:
                   df.loc[ events_idx,'weighted_selected' ] = 1
       
      #--- update new weighted event idx   
      # events_idx = df[ df['weighted_selected'] > 0 ].index
      #--- update new weighted event idx   
        events_idx = np.where( df['weighted_selected'] > 0 )[0] #.index
        
        if self.verbose:
           msg=["---> Weighted Marker => event index => method:" + method]
           if self.is_number(w_value):
              msg.append("   -> value           : %0.3f" % (w_value))

           msg.append("   -> number of events: {}".format( events_idx.shape[0] ))
           msg.append("   -> min counts      : {}".format( self.weights['min_counts'] ))
           msg.append("   -> type output     : " + self.marker.type_output)
           msg.append( "-"*40 )
           logger.info(msg)
        
        return df,events_idx
    
#---    
    def _epochs_update_marker_time(self,time=None):
        """ update marker time
        copy dict [time_pre,time_post] to marker time dict if defined
        Parameters
        ----------
        dict of time
        
        """
        if time:
           self.time=time
        
       #--- set marker time pre,post   
        if self.is_number(self.time["time_pre"]):
           self.marker.time_pre = self.time["time_pre"] 
        if self.is_number(self.time["time_post"]):
           self.marker.time_post = self.time["time_post"] 
              
#---        
    def _epochs_init_events(self,df,events_idx): 
        """ init mne event parameter 
        Parameters
        ----------
         pandas dataframe
         numpy array with event index  
        
        Results
        -------
         event dictionary 
         evt['events']  : <np.array([])>  from mne.find_events
         evt['event_id']: <None> list of event ids
         evt['baseline_corrected']: True/False
         baseline:
         evt['bc']['events']   = np.array([])
         evt['bc']['event_id'] = None
       
        """
        evt = dict()
        evt['events']  = np.array([])
        evt['event_id']= None #np.array([])
        evt['baseline_corrected'] = False
        
        if events_idx.size:      
           evt['events']      = np.zeros(( events_idx.size, 3),dtype=np.int64)
          #--- init onset 
           evt['events'][:,0] += df[ self.marker.type_output ][events_idx]
          #--- init event ids 
           evt['events'][:,2] += df[ self.marker.type_id ][events_idx]
          # evt['event_id']    = np.asarray( np.unique( evt['events'][:,2] ),dtype=np.int  ) #.min()
           evt['event_id']    = np.unique( evt['events'][:,2] ).tolist()
       #---  init baseline events  
           evt['bc']             = dict() 
           evt['bc']['events']   = np.array([])
           evt['bc']['event_id'] = None
           
           if self.marker.baseline.type_input:  
              evt['bc']['events']       = np.zeros( (events_idx.size,3),dtype=np.int64 )   
              evt['bc']['events'][:,0] += df[ self.marker.baseline.type_input][events_idx]
              evt['bc']['events'][:,2] += df[ self.marker.type_id ][events_idx]
              evt['bc']['event_id']     = np.unique( evt['bc']['events'][:,2] ).tolist()
      #-----  
        if self.verbose:
           msg=["---> epocher init mne events & baseline-events",
                "  -> number of events         : " + str( evt["events"].shape )]
           if evt.get("bc"):
               msg.append("  -> number baseline of events: " + str( evt["bc"]["events"].shape ))
           else:
               msg.append("  -> baseline not defined")
           logger.info( "\n".join(msg))
        return evt  
#---   
    def _epochs_save_hdf_data(self,df,ep_param,info_param,condi) :
        """save dataframe to hdf
           store df with updated bads & selected & restore user-attribute   
         
        Parameters
        ----------
         data frame
         dict epoch parameter
         dict info parameter
         condition name
        
        Results
        -------
         None
        """    
       #--- get hdf key make node  
        ep_key = self.hdf_get_hdf_key(condi)
        if not ep_key :
           logger.info(" --> ERROR in <epocher_save_hdf_data> !!!"+
                       " --> no HDF-key found for condition: "+ condi)
           return
       
        storer_attrs = {'epocher_parameter': ep_param,'info_parameter':info_param} 
        self.hdf_obj_update_dataframe(df,key=ep_key,reset=False,**storer_attrs)    
    
#---
    def _epochs_update_exclude_events(self,column=None,exclude_events=None,df=None,condi=None): 
        """update/mark exclude events in dataframe with <idx_bad>
        
        Parameters
        ----------
        column        : column name to check tsls within time window to exclude
        exclude_events: dict : {ecg onset:[tmin,tmax],eog onset:[tmin,tmax]}
        df            : pandas dataframe from HDFobj
        condi         : epocher condition         
        
        
        reset df fields <selected> <weighted_selected'>
        
        Returns
        ----------
        pandas dataframe
         
        """
        
        for kbad in ( exclude_events.keys() ):
            ep_bads_cnt0 = df['bads'][ df['bads'] == self.idx_bad ].size

            for idx in range( exclude_events[kbad]['tsl'].shape[-1] ) :    #df.index :
                df['bads'][ ( exclude_events[kbad]['tsl'][0,idx] < df[ column ] ) & ( df[ column ] < exclude_events[kbad]['tsl'][1,idx] ) ] = self.idx_bad
               
            ep_bads_cnt1 = df['bads'][df['bads'] == self.idx_bad].size 
               
            if self.verbose:
               logger.info("---> Exclude artefacts {} : {}\n".format(condi,kbad)+
                           "  -> Tmin: {0.3f} Tmax {0.3f}\n".format(exclude_events[kbad]['tmin'],exclude_events[kbad]['tmax'])+
                           "  -> bad epochs     : {}\n".format(ep_bads_cnt0)+
                           "  -> artefact epochs: {}\n".format(ep_bads_cnt1 - ep_bads_cnt0)+
                           "  -> excluded epochs: {}\n".format(ep_bads_cnt1) )
                     
        df['selected']          = 1  
        df['weighted_selected'] = 1  
        
        return df
    
#---          
    def _epochs_update_artifact_time_window(self,**kwargs):
        """update artifact time window for ecg- and eog-artifacts in HDFobj
        
        Parameters
        ----------
        :dict: "ecg_events":[-0.4,0.6],"eog_events":[-0.4,0.6]
               "eog_events":[-0.4,0.6],"eog_events":[-0.4,0.6]
        Returns
        ----------
        artifact events dict
        artifact_events[ artifact type ]
          ={tmin: onset time in sec,tmax: offset time in sec, tsl: range in timeslices as np.array[ onset,offset] }
        """
        import numpy as np
     
        artifact_events = dict()
        for kbad,time_window in kwargs.items():
            node_name = self.hdf_node_name_artifact + '/' + kbad
          #--- ck if node exist 
            try:             
               self.HDFobj.get(node_name)
            except:
               continue

            artifact_events[kbad]= {'tmin':time_window[0],'tmax':time_window[1],'tsl':np.array([])}

            tsl0= self.raw.time_as_index(time_window[0])
            artifact_events[kbad]['tmin'] = time_window[0]

            tsl1= self.raw.time_as_index(time_window[1])
            artifact_events[kbad]['tmax'] = time_window[1]

            df_bad = self.HDFobj.get(node_name)
            artifact_events[kbad]['tsl'] = np.array([ df_bad['onset']+tsl0, df_bad['onset']+tsl1 ] )
       
        return artifact_events
          
#---           
    def _epochs_get_epochs_and_apply_baseline(self,raw,evt=None,picks=None):
        """generate epochs from raw and apply baseline correction if baseline is not None
        exclude epochs due to short baseline onset/offset intervall or
        epochs which will not fit in time window[timepre <> time_post]
        
        Parameters
        ----------
        raw obj
        evt: event dict
                                    
        evt['bc']['baseline']    : time range in sec [None,0.0]
                                   if evt['bc']['baseline'] is None or [] no baseline correction applied        
        evt['bc']['events']      : feed to mne.Epochs as <events>
        evt['bc']['event_id']    : feed to mne.Epochs as <event_id>
                   
        check for bad epochs due to short baseline onset/offset intervall and drop them
              
        Returns
        ----------
        updated event dict
         evt["epochs"]             : mne epoch obj
         evt["baseline_corrected"] : True if baseline correction 
         
        FYI: digital channels like <stimulus> and <response> are excluded from baseline correction
             e.g. <STI 013> <STI 014>
        """
           
        ep_bc_corrected           = None
        evt['epochs']             = None
        evt['baseline_corrected'] = False
        
        if raw:
           self.raw=raw
           
       #--- update and load raw     obj
        self.raw,self.fname = jumeg_base.get_raw_obj(self.fname,raw=self.raw)
       
        try:
            ep = mne.Epochs(self.raw,evt['events'],event_id=evt['event_id'],tmin=self.marker.time_pre,tmax=self.marker.time_post,
                            baseline=None,picks=picks,reject=self.reject,proj=self.proj,preload=True,verbose=False)
        except:
            logger.exception("---> Error\nevents:\n{}\nevent id: {}\n".format(evt['events'],evt['event_id']))
            
        if self.verbose: # for later show difference min max with and without bc
           meg_picks = jumeg_base.picks.meg_nobads(self.raw)
           meg_min   = ep._data[:,meg_picks,:].min()
           meg_max   = ep._data[:,meg_picks,:].max()
          
       #--- calc baseline correction  
        if self.marker.baseline.method:
           if evt['bc']['events'].any() and ep.selection.any(): 
            
             #--- ck for no unique baseline events and apply bc correction, standard task 
              ep_bc_corrected = self._calc_baseline_correction_for_events(ep,evt['bc']['events'])
              
             #--- ck for unique events and apply bc correction with unique baseline events, e.g. one baseline intervall used for multi stimuli 
              if not ep_bc_corrected:
                 ep_bc_corrected = self._calc_baseline_correction_for_unique_events(ep,evt['bc']['events'])
            
              if ep_bc_corrected:
                 evt['epochs'] = ep_bc_corrected
                 evt['baseline_corrected'] = True
        else:
           evt['epochs'] = ep
            
        if self.verbose:
           if evt['epochs']:
              ep_min =  "{:0.15f}".format( evt['epochs']._data[:,meg_picks,:].min())
              ep_max =  "{:0.15f}".format( evt['epochs']._data[:,meg_picks,:].max())
           else:
              ep_min=ep_max="None"
           msg=["---> Epocher apply epoch and baseline -> mne epochs:",
                " --> fname     : {}".format(self.fname),
                " --> ids       : {} ".format(evt['event_id']),
                " --> <pre time>: {:0.3f} <post time>: {:0.3f}".format(self.marker.time_pre,self.marker.time_post),
                " --> baseline correction : {}".format(evt['baseline_corrected']),
                "   " +"-" * 40,
                " --> Epochs selected: {}\n   -> index: {}".format( ep.selection.shape , np.array2string(ep.selection)),
                "   " +"-" * 40,
                "  -> MEG min   : {:0.15f}".format( meg_min ),
                "  -> MEG min BC: "+ep_min,
                "  -> MEG max   : {:0.15f}".format( meg_max ),
                "  -> MEG max BC: "+ep_max ]
           if evt['baseline_corrected']:
              msg+=["   " +"-" * 40,
                    " --> baseline correction: DONE",
                    "  -> bc id: {}".format(evt['bc']['event_id']),
                    "  -> < pre time >: {:0.3f} < post time >: {:0.3f}".format(self.marker.time_pre,self.marker.time_post),
                    "  -> < tsl0 >: {} <post time>: {}".format(self.marker.baseline.onset,self.marker.baseline.offset)]
           logger.info("\n".join(msg))
        return evt
    
#---
    def _calc_baseline_correction_for_events(self,ep,bc_events,method=None):
        """
        calc baseline type correction e.g. calc mean or median baseline epochs value
        for all epoch-events who are in baseline events
        
        bad epochs are droped!!!
        
        Parameters
        ----------
        <mne.epochs.Epochs> obj: output from mne.Epochs, e.g. epochs, bads are droped
        bc_events: np.array, mne.Events to get baseline epochs
        method  : string [median,mean], type of baseline value calculation <None>
        
        Returns
        -------
        mean or median baseline value for each channel, each epochs 
        epoch.sahpe=> 10,7,500 [epochs,channels,tsls] => return  ep_bc => 10,7 => [epochs,channels]
        
        default bbaseline correction type is <marker.baseline.method> or mean
        
        """
        logger.info(" --> Epocher baseline correction for events")
        
        ep_bc_cor_values = np.array([])
        
        if not(bc_events.any()) or (not ep.selection.any() ): return
        picks_bc = jumeg_base.picks.exclude_trigger( ep ) # no trigger, response
        
        if not picks_bc.any(): return # check e.g. if only trigger channels
       #--- ck if number of baseline events will match events
        if len(np.unique(bc_events[:,0])) != len(bc_events): return
       
       #--- 
        tmin,tmax = self._ck_baseline_range()
    
       #--- ToDo ck for unique epoch-events and use unique's only 
        ep_bc = mne.Epochs(self.raw,bc_events,tmin=tmin,tmax=tmax,baseline=None,picks=picks_bc,preload=True,verbose=self.verbose)
        ep_bc.drop_bad(verbose=self.verbose)  #- exclude bad epochs e.g: to short
     
       #--- ck if epochs needs to drop in ep and ep_bc
       #--- get index back of good baseline epochs       
       #--- ck and exclude epoch-events not in bc-events => get common selection ep.selectiom with ep_bc.selection
        drop_ep_idx_not_in = np.where(np.isin(ep.selection,ep_bc.selection,invert=True))[0]
        if drop_ep_idx_not_in.any() :
           ep.drop( drop_ep_idx_not_in ) 
           ep.drop_bad()
           
       #--- ck and exclude bc-events not in epoch events => get common selection ep.selectiom with ep_bc.selection
        drop_ep_idx_not_in = np.where(np.isin(ep_bc.selection,ep.selection,invert=True))[0]
        if drop_ep_idx_not_in.any() :
           ep_bc.drop( drop_ep_idx_not_in ) 
           ep_bc.drop_bad()
   
       #--- calc baseline values
        ep_bc_cor_values = self._calc_baseline_values(ep_bc,method=method)
        if not ep_bc_cor_values.any(): return
      
       #---Test for equal drops
        if self.verbose:
           logger.debug("---> Epochs :\n"
                        "  --> epochs : {}\n".format(ep)+
                        "   -> epoch selection:\n{}\n".format(ep.selection)+
                        "   -> epoch events   :\n{}\n".format(ep.events)+
                        "   -> epoch event_id : {}\n".format(ep.event_id)+
                        "    "+"-"*40+"\n"+
                        "  --> Baseline epochs:\n"+
                        "   -> bc epochs : {}\n".format(ep_bc) +
                        "   -> bc epoch selection:\n{}\n".format(ep_bc.selection) +
                        "   -> bc epoch events   :\n{}\n".format(ep_bc.events) +
                        "   -> bc epoch event_id : {}".format(ep_bc.event_id) )

           
           #for epoch and baseline - epochs: "+
           #ep.save("epoch_test-epo.fif")
           #ep_bc.save("epoch_test_bc-epo.fif")

           try:
               ck = (ep.selection - ep_bc.selection).max()
               logger.info("---> Epochs check equal number of epoch and baseline-epoch events => errors: {}".format( ck) )
           except:  # if ck != 0:
                 logger.exception( "ERROR and EXIT no equal number of epoch events for epoch and baseline-epochs !!!")
                 sys.exit()
           
           logger.info("  -> epoch and basseline-epochs: number of epochs is equal  epochs: {} epochs bc: {}".format(ep.selection.shape[0],ep_bc.selection.shape[0] ))
       
        ep._data[:,picks_bc,:] -= ep_bc_cor_values[:,:,np.newaxis]
       
        logger.info("  -> Done Epocher baseline correction for events\n  -> baseline intervall :[ %3.3f , %3.3f] " %(tmin,tmax))
        return ep 
    
#---
    def _calc_baseline_correction_for_unique_events(self,ep,bc_events,method=None):
        """calc baseline type correction e.g. calc mean or median baseline epochs value 
        work around if several epoch events are using same baseline event 
        for unique baseline events mne.Epocher (mne version 15.2) will complaine about unique events
        
        bad epochs are droped!!!
        
        Parameters
        ----------
        <mne.epochs.Epochs> obj: output from mne.Epochs, e.g. epochs, bads are droped
        bc_events: np.array, like mne.Events for baseline epochs
        method  : string [median,mean], type of baseline value calculation <None>
        
        Returns
        -------
        mean or median baseline value for each channel, each epochs 
        epoch.sahpe=> 10,7,500 [epochs,channels,tsls] => return  ep_bc => 10,7 => [epochs,channels]
        
        default bbaseline correction type is <marker.baseline.method> or mean
        
        """

        logger.info("  -> Epocher baseline correction for unique events")
        
        ep_bc_cor_values = np.array([])
        
        if ( not bc_events[:,0].any() ) or ( not ep.selection.any() ): return
        picks_bc = jumeg_base.picks.exclude_trigger( ep ) # no trigger, response
            
        if not picks_bc.any(): return # check e.g. if only trigger channels
     
       #--- ck take only unique events   
       #- unique        : https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
       #- unique_indices: indices of the first occurrences of the unique values in the original array
       #- unique_inverse: indices to reconstruct the original array from the unique array.
        evts_bc_uni,evts_bc_uni_idx,evts_bc_uni_inv = np.unique(bc_events[:,0],return_inverse=True,return_index=True)
      
       #- no unique events but it should be at least one  and all events are different 
        if (evts_bc_uni.size == bc_events.shape[-1]): return 
       #--- 
        tmin,tmax = self._ck_baseline_range()
       #---  
        ep_bc_uni = mne.Epochs(self.raw,bc_events[evts_bc_uni_idx],tmin=tmin,tmax=tmax,baseline=None,
                               picks=picks_bc,preload=True,verbose=self.verbose)                  
        ep_bc_uni.drop_bad() #- exclude bad epochs e.g: to short            
        
        ep_bc_cor_values = self._calc_baseline_values(ep_bc_uni,method=method)
       
        if not ep_bc_cor_values.any(): return
               
       #--- ck if epochs needs to drop in ep and ep_bc
       #--- get index back of good baseline epochs      
        ep_bc_idx_goods = np.where(np.isin(evts_bc_uni_inv,ep_bc_uni.selection))[0]
       #--- ck and exclude epochs not in bc-epochs => get common selection ep.selectiom with ep_bc.selection
        drop_ep_idx_not_in_bc = np.where(np.isin(ep.selection,ep_bc_idx_goods,invert=True))[0]
     
        if drop_ep_idx_not_in_bc.any() :
           ep.drop( drop_ep_idx_not_in_bc ) 
           ep.drop_bad()
    
        idx_bc = 0
        for idx in ep_bc_uni.selection:
       #--- find epochs which are in epochs & baseline-epochs no droped epochs!!! 
            ep_in_bc_idx = np.where( evts_bc_uni_inv[ ep.selection ]== idx )[0]
          
            if not ep_in_bc_idx.any() : continue
          
          #--- use epochs which are in baseline epochs    
            bc_values = ep_bc_cor_values[idx_bc,:]
            ep._data[ np.ix_(ep_in_bc_idx,picks_bc) ]-= bc_values[:,np.newaxis]
            idx_bc+=1
    
        if self.verbose:                      
           logger.info(" --> Baseline correction for unique basline epochs: {}".format(ep))

        logger.info("  -> Done Epocher baseline correction for unique events\n  -> baseline intervall :[ %3.3f , %3.3f] " %(tmin,tmax))
        return ep
    
#---
    def _ck_baseline_range(self):
        tmin = self.marker.baseline.onset
        tmax = self.marker.baseline.offset
        
        if not self.is_number(tmin):
           tmin = self.marker.time_pre
        if not self.is_number(tmax):
           tmax = self.marker.time_post                                
        return tmin,tmax   
#---
    def _calc_baseline_values(self,epochs,method=None):
        """calc baseline type correction e.g. calc mean or median baseline epochs value 
        
        Parameters
        ----------
        epochs: output from mne.Epochs, e.g. epochs from baseline time intervall, bads are droped
        
        method: string [median,mean], type of baseline value calculation <None>
        
        Returns
        -------
        mean or median baseline value for each channel, each epochs 
        epoch.sahpe=> 10,7,500 [epochs,channels,tsls] => return  ep_bc => 10,7 => [epochs,channels]
        
        default bbaseline correction type is <marker.baseline.method> or mean
        
        """
        if not method:
           method = self.marker.baseline.method
        if method == "median":
          return np.median(epochs.get_data(),axis = -1)
        else: # mean
          return np.mean(epochs.get_data(),axis = -1)
          
#---   
    def _epochs_save_events(self,raw=None,evt=None,condition=None,postfix=None,postfix_extention="evt",weighted=False,picks=None,reject=None,
                            proj=False,output_mode=None ):
        
       """saving event-,epoch-,evoked data using mnefunctions with file nameing conventions
       saving results as *.png in /XYZ/plots folder
         
       Parameters  
       ----------- 
       raw      : raw obj <None> use global
       evt      : event dict <None>
                 condition: name of condition <None>
                 
       postfix  : filename postfix = <condition postfix> from HDF and template file <None>
                  postfix_extention: final postfix extention e.g.: CONDI_evt  <"evt">  
                 
       MNE parameter for mne.epochs 
       picks    : channel index array []
       reject   : None
       proj     : False,
                 
       output_mode : dict <None>; parametes set by kwargs of <apply_hdf_to_epochs>
                  {"events":True,"epochs":True,"evoked":True}>
                   events: save mne events <...,CONDI_evt.eve>
                   epochs: save mne epochs <...,CONDI_evt-epo.fif> 
                   evoked: save mne evoked, averaged data <...,CONDI_evt_bc-ave.fif>  
                   if baseline corrected:  <...,CONDI_evt_bc.xyz>
                   parameter:
                     stage: <None>, if stage save output to stage
                     use_condition_in_path: <True> if True: save output file into stage/condition directory
                 
       weighted  : sort of equal number of trail over all conditions  <False>
                   e.g: response matching task or EOG artifact rejection
                   if weighted            <...,CONDI_evtW.xyz>
                   if weighted and baseline corrected 
                                          <...,CONDI_evtW_bc.xyz>
              
       Returns
       -----------
       None                                                    
                                                    
       """
       if raw:
          self.raw = raw
           
       if postfix:
          postfix += "_" + postfix_extention
       else:
          postfix = postfix_extention 
           
       if evt['baseline_corrected']:
          postfix += '_bc'
   
      #--- update OutputMode CLS e.g. output stage ,save flags like events ..
       if not output_mode:
          output_mode={}
          
       self.OutputMode.make_output_path(condition=condition,**output_mode)
       
      #--- save events to txt file    
       if self.OutputMode.events:
          fname = jumeg_base.get_fif_name(raw=self.raw,postfix=postfix,extention=".eve",update_raw_fname=False,path=self.OutputMode.path)
          mne.event.write_events( fname,evt['events'] )
          logger.info(" ---> done jumeg epocher save events as => EVENTS :{}\n".format(fname)+
                      "   -> number of events : %d" %(evt['events'].shape[0]))
      #--- save epoch data
       if self.OutputMode.epochs:
          fname = jumeg_base.get_fif_name(raw=self.raw,postfix=postfix,extention="-epo.fif",update_raw_fname=False,path=self.OutputMode.path)
          evt['epochs'].save( fname )
          logger.info("---> done jumeg epocher save events as => EPOCHS : " +fname)
          
      #--- save averaged data
       if self.OutputMode.evoked:
          fname = jumeg_base.get_fif_name(raw=self.raw,postfix=postfix,extention="-ave.fif",update_raw_fname=False,path=self.OutputMode.path)
          mne.write_evokeds( fname,evt['epochs'].average( picks = jumeg_base.picks.all(self.raw) ) ) # in case avg only trigger or response
          logger.info("---> done jumeg epocher save events as => EVOKED (averaged) : " +fname)
          fname = jumeg_base.get_fif_name(raw=self.raw,postfix=postfix,extention="-ave",update_raw_fname=False,path=self.OutputMode.path)
         #--- plot evoked
          fname = jplt.plot_evoked(evt,fname=fname,condition=condition,show_plot=False,save_plot=True,plot_dir='plots')
          logger.info("---> done jumeg epocher plot evoked (averaged) :" +fname)
             
#---
    def __clear(self):
        """ clear all CLs parameter"""
        self.raw            = None
        self.template_name  = None    
        self.fhdf           = None
        self.fname          = None
        self.comdition_list = None
        self.fif_postfix    = None 
        self.event_extention= None 
        self.baseline       = None
        self.time           = None
        self.weights        = None
        self.save_conditions= None
        self.exclude_events = None
        self.picks          = None
        self.reject         = None
        self.proj           = None  
            
#---
    def _epochs_update_kwargs(self,**kwargs):
        """ internal use only use update() to update obj parameter"""
        if not kwargs: return
        if kwargs.get("output_mode"):
           self.OutputMode.update(**kwargs["output_mode"])
        
        for k,v in kwargs.items():
            if k == 'template_name':
                 self.template_name = v
            elif k=='fname':
                 self.fname=v
            elif k=='raw':
                 self.raw=v
            elif k=='type_result':
                 self.type_result=v
            elif k=='fhdf':
                 self.fhdf=v
            elif k=='condition_list':
                 self.condition_list=v
            elif k=='exclude_events':
                 self.exclude_events = v
            elif k=='baseline':
                 self.baseline = v
            elif k=='time':
                 self.time = v
            elif k=='weights':
                 self.weights = v
            elif k=='verbose':
                 self.verbose = v          
#--- 
    def _epochs_update_data(self,**kwargs):
        """
        :param template_name:
        :param fname:
        :param raw:
        :param fhdf:
        :param condition_list:
        :param exclude_events:
        :param baseline:
        :param time:
        :param weights:
       
        :return: None
        
        update parameter in cls from **kwargs
        open HDFobj
        check condition list for int
        """
        self._epochs_update_kwargs(**kwargs)
        
        self.HDFobj = self.hdf_obj_open(fhdf=self.fhdf,fname=self.fname,raw=self.raw,hdf_path=kwargs.get("hdf_path"))
        
        if not self.hdf_obj_is_open():
           assert "ERROR could not open HDF file:\n --> raw: "+self.raw.filename+"\n --> hdf: "+self.HDFObj.filename+"\n"
        
        if not self.condition_list:
           self.condition_list = self.hdf_get_key_list(node=self.hdf_epocher_node_name,key_list=self.condition_list)
    
        if self.verbose:
           logger.info("\n".join([
                         "---> Epocher Epochs condition list form HDF:",
                         "  -> HDF isOpen: {}".format(self.hdf_obj_is_open()),
                         "   "+"-"*40,
                         "  -> raw file: {}".format(self.fname),
                         "  -> hdf file: {}".format(self.hdf_filename),
                         "   "+"-"*40,
                         "  -> "+ ' '.join( self.condition_list ) +"\n","-"*40
                         ]))

          
jumeg_epocher_epochs = JuMEG_Epocher_Epochs()


