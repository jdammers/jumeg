#JuMEG Epocher
```
https://github.com/jdammers/jumeg)
Created on Mon Jun 18 09:07:46 2018
@author: fboers f.boers@fz-juelich.de
```
##Features
1. template dirven approach to extract event structure
2. provide channel marker information for later processes
3. export epochs and averaged data in mne-fif format

###**event features**
1. read condition parameter from epocher-template ( [json](http://jsonapi.org/) format )
2. find mne-events for conditions defined in epocher-template, 
3. store event-data in pandas dataframe for each condition
4. add additional marker information to data frame  
	- image onset detection (IOD)
    - response matching 
    - eye-tracking events
5. save event structure into HDF5 format 

###**epocher features**    
1. apply baseline correction calculated from a common or unique time intervall
2. response matching for arbitrary tasks based on columns in dataframe 
3. support for arbitrary digital marker channels
4. combine events (stimulus, response, eye-tracking, image onset detection)
5. for each condition generate mne-events, epochs, averages, statistic from HDF

####**Example**
~~~python
from jumeg.epocher.jumeg_epocher import jumeg_epocher
        
#--- input fif file
fraw  = '007_FREEVIEW01_180115_1414_1_c,rfDC-raw.fif'
raw   = None

#--- set template path & file name
template_name = 'FreeView'
template_path = "./"

#--- info flag
verbose = True

#--- conditions / events to process
condition_list= ["FVImgBc","FVImgIodBc"]

#--- finding events   
print "---> EPOCHER Events"
print "  -> FIF File: "+ fname
print "  -> Template: "+ template_name+"\n"

evt_param = { 
              "condition_list":condition_list,
              "template_path": template_path, 
              "template_name": template_name,
              "verbose"      : verbose
            }
              
raw,fname,epocher_hdf_fname = jumeg_epocher.apply_events(fname,raw=raw,**evt_param)

#--- Epocher   
print "---> EPOCHER Epochs"
print "  -> File            : "+ fname
print "  -> Epocher Template: "+ template_name+"\n"   

ep_param={
          "condition_list": condition_list,
          "template_path" : template_path, 
          "template_name" : template_name,
          "verbose"       : verbose,
          "parameter":{
                       "event_extention": ".eve",
                       "save_condition":{"events":True,"epochs":True,"evoked":True}
                      }}  

raw,fname,epocher_hdf_fname = jumeg_epocher.apply_epochs(fname=fname,raw=raw,**ep_param)
~~~



# Epocher Template

>The epocher-template is written in [json](http://jsonapi.org/) format. For syntax checking you can use [_jsonlint-py_]( https://gist.github.com/Constellation/264387).

#####**Template File Naming Convention**

>The template naming is like: **_experiment name_**_jumeg_epocher_template.json

####template path
```text
- set variable by user
- if not defined use environment variable <JUMEG_TEMPLATE_PATH_EPOCHER>
- else if not defined use < default template path >
```
        
#### template name
```text
-  the name of the experiment or scan as prefix, defined by user
```

####template_file_name
```text
* template_name 	: filename prefix like experiment name (e.g. M100, M100A)
* postfix 			: jumeg_epocher_template
* extention 		: .json 
```
    
**Example**
```python
from jumeg.epocher.jumeg_epocher import jumeg_epocher as jep
jep.template_name = "M100"

print"---> check if  env jumeg_epocher_template_path is defined: " + jep.env_template_path
print"  -> template path          : " +jep.template_path
print"  -> template name          : " +jep.template_name
print"  -> template filename      : " +jep.template_filename 
print"  -> template full filename : " +jep.template_full_filename
print"  -> template default path  : " +jep.template_path_default

```

---------------------------
#### **Template Parameter**
---------------------------
>*jumeg epocher template* example   (json) format:  { default{ ... }, condition1{ ... }, condition2{ ... } }

**Example**
~~~json
{
"type": "jumeg epocher template",
"id"    : "1",
"attributes": {
                 "title"      : "FreeViewing",
                 "version"    : "2018-06-19-001",
                 "experiment" : "FreeView",
                },
                
"default":{
           "version"          :"2018-06-19-001",
           "experiment"       : "FreeView",
           "postfix"          : "test",
           "time_pre"         : -0.20,
           "time_post"        :  0.65,
        
           "baseline" :{"method":"mean","type_input":"iod_onset","baseline": [null,0]},
           
           "marker"   :{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"sac_onset","prefix":"iod","type_result":"hit"},
           "response" :{"matching":true,"channel":"ETevents","type_input":"sac_onset","type_offset":"sac_offset","prefix":"sac"},         
           
           "iod"      :{"marker"  :{"channel":"StimImageOnset","type_input":"img_onset","prefix":"img"},
                        "response":{"matching":true,"channel":"IOD","type_input":"iod_onset","type_offset":"iod_onset","prefix":"iod"}},
       
           "reject"   : {"mag": 5e-9},
          
           "ETevents":{
                       "events":{
                                  "stim_channel"   : "ET_events",
                                  "output"         : "onset",
                                  "consecutive"    : true,
                                  "min_duration"   : 0.0005,
                                  "shortest_event" : 1,
                                  "mask"           : 0
                                 },
                        "and_mask"          : 255,
                        "event_id"          : null,
                        "window"            : [0.02,5.0],
                        "counts"            : "all",
                        "system_delay_ms"   : 0.0,
                        "early_ids_to_ignore" : "all"
                        
                       },
                     
           "StimImageOnset":{
                       "events":{
                                  "stim_channel"   : "STI 014",
                                  "output"         : "onset",
                                  "consecutive"    : true,
                                  "min_duration"   : 0.0005,
                                  "shortest_event" : 1,
                                  "mask"           : 0
                                 },
                        
                        "event_id"           : 84,        
                        "and_mask"           : 255,
                        "system_delay_ms"    : 0.0,
                        "early_ids_to_ignore" : null
                        },                                                

            "IOD":{
                        "events":{
                                  "stim_channel"   : "STI 013",
                                  "output"         : "onset",
                                  "consecutive"    : true,
                                  "min_duration"   : 0.0005,
                                  "shortest_event" : 1,
                                  "mask"           : 0
                                 },
                        
                        "window"               : [0.0,0.2],
                        "counts"               : "first",
                        "system_delay_ms"      : 0.0,
                        "early_ids_to_ignore"  : null,
                        "event_id"             : 128,
                        "and_mask"             : 255
                       }                            
              },
}
~~~
#### Template Condition Parameter
#####**resource object part**: 

```text
type,id,attributes ... for information and identification
```

#####**default condition**:

```text
every parameter defined in the < default > condition is valid in all condition.
Inside a condition parameters can be changed and the changes are only valid within these condition
```

#####**postfix**:
```text
labeling condtion in output file name
e.g.	postfix : "ImoIOD" 

example:
		input  file name: 211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr-raw.fif
        output file name: 211776_FREEVIEW01_180115_1414_1_c,rfDC,meeg_bcc,tr,ImoIOD_evt-ave.fif
```
#####**time_pre**:
```text
epoch time onset [s] derived from mne.Epochs 
```
#####**time_post**:
```text
epoch time offset [s] derived from mne.Epochs 
```

#####**reject**:
```text
recjection parameter derived from mne.Epochs 
e.g. "reject"   : {"mag": 5e-12},
```
#####**baseline**:
```text
baseline : <null> no baseline correction
baseline : {"method":"mean","type_input":"iod_onset","baseline": [null,0]}
	* method	: baseline value calculation method [mean,median]
	* type_input: data frame column with timesliches as window start points [iod_onset]
	* baseline	: derived from mne.Epocher e.g. [null,0] [-0.2,0]	
```
#####**iod**:
```text
response matching task to include image onset detection

Task:
	A photodiode signal (TTL) were recorded in a digital channel to mark the image onset on the screen.
	These channel (e.g. STI 013) is defined within the <default>-object parameter settings
    e.g. in < jumeg epocher template file >  { "defaults":{ ..., "iod":{ ...}, ...} } 
    This can be used for arbitrary marker in order to find onset or offset events

Parameter:
"iod":{ "marker":{...},"response":{ ... } },

1. marker:{"channel":"StimImageOnset","type_input":"img_onset","prefix":"img"}
	* channel	 : channel-object keyword defined within <default>-object
    	  		   e.g.: StimImageOnset, ETevents
                   { "defaults":{ ..., "ETevents":{ ...}, ...} }
	* type_input : dataframe column with timeslices to use as onset for iod-matching [img_onset]
	* prefix	 : prefix for dataframe columns  [img]
    
2. response:{"matching":true,"channel":"IOD","type_input":"iod_onset","type_offset":"iod_onset","prefix":"iod"}}
	* matching	 : apply IOD matching [true or false]
	* channel	 : channel-object keyword defined under <defaults> [IOD]
    
	* type_input : dataframe column with timeslices to use as response for iod-matching [iod_onset]
	* type_offset: dataframe column with timeslices to use in response-matching
    	 		      for identification of <to-early-events> in iod-matching [iod_onset]
	* prefix	 : prefix for dataframe columns  [iod]
```

#####**channel object definition:**
```text
keyword to label a channel-object (e.g. ImageOnsetDetection, IOD, ETevents)

defaults{ ... "ETevents":{ "events":{ ... }, ... }, ...},

	* events   : event structure derived from <mne.find_events>
   		  	     e.g. {"stim_channel":"ET_events","output":"onset","consecutive":true,"min_duration":0.0005,
               		"shortest_event":1,"mask":0},
	* and_mask : apply and mask for <stim_channel> values [null,255]
	* event_id : event code to search for with mne.find_events
	* window   : time window [s] to look for responses if <response matching> is true
	* counts   : maximum number of responses allowed to be inside the response window [all,first,1]
			  	 - number: maximum number of responses (e.g 1 if more the epoch will be labeld as wrong)
				 - first	: should be at least one to be correct, others ignored
				 - all	: accept all responses defined in <response event_id> 
                   		(e.g. eye-tracking task fixation and saccards) 
                            
	* system_delay_ms     : system delay to add to timeslice columns [ms]
	* early_ids_to_ignore : list of event_id`s to exclude as to-early-events [1,2,3]
```

####**_marker_**:  
```text
keyword to label a marker-object defined within <default>-object

{ "defaults":{ ... "marker":{ ... }, ...} }

"marker":{"channel":"StimImageOnset","type_input":"iod_onset","type_output":"sac_onset","prefix":"iod","type_result":"hit"},

	* channel		: channel-object keyword defined within <default>-object e.g.: StimImageOnset, ETevents
      	            { "defaults":{ ..., "ETevents":{ ...}, ...} }
                    
	* type_input	: dataframe column with timeslices to use as marker [iod_onset]
	* type_output	: dataframe column with timeslices 
    * prefix		: prefix for dataframe columns  [iod]
	* type_result	: event type to process ["hit","wrong","missed"]
 
```
####**_response_**:  
```text
keyword to label a response-object defined within <default>-object

{ "defaults":{ ... "response":{ ... }, ...} }

"response" :{"matching":true,"channel":"ETevents","type_input":"sac_onset","type_offset":"sac_offset","prefix":"sac"},         
	
    * matching      : apply response-matching [true or false]
    * channel		: channel-object keyword defined within <default>-object e.g.: StimImageOnset, ETevents, IOD
      	            { "defaults":{ ..., "ETevents":{ ...}, ...} }
                    
	* type_input	: dataframe column with timeslices to use as marker [iod_onset]
	* type_output	: dataframe column with timeslices 
    * prefix		: prefix for dataframe columns [sac]
 
```

------------------------------
### jumeg_epocher.apply_events
	 evt_param = { "condition_list":condition_list,
	                 "template_path": template_path, 
	                 "template_name": template_name,
	                 "verbose"      : verbose
	               }
              
	  raw,fname,epocher_hdf_fname = jumeg_epocher.apply_events(fname,raw=raw,**evt_param)


### jumeg_epocher.apply_epochs
	  ep_param={
	          "condition_list": condition_list,
	          "template_path" : template_path, 
	          "template_name" : template_name,
	          "verbose"       : verbose,
	          "parameter":{
	                       "event_extention": ".eve",
	                       "save_condition":{"events":True,"epochs":True,"evoked":True},
                           "weights":{"mode":"equal","method":"median","skipp_first":null}
	                      }} 
	 raw,fname,epocher_hdf_fname = jumeg_epocher.apply_epochs(fname=fname,raw=raw,**ep_param)
                       

