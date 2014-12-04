import glob, os, re, sys
import json
# from pprint import pprint

'''
----------------------------------------------------------------------
--- JuMEG Template Class            ----------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 last update: 05.11.2014
 version    : 0.0114

---------------------------------------------------------------------- 
 set env variables to the path of your templates files
 JUMEG_PATH_TEMPLATE
 JUMEG_PATH_TEMPLATE_EXPERIMENTS
 JUMEG_PATH_TEMPLATE_Averager
---------------------------------------------------------------------- 
 r/w jumeg experiment template files in json format
 r/w jumeg averager   template files in json format
----------------------------------------------------------------------
 file name:
 <experiment>_jumeg_experiment_template.json
 <experiment>_jumeg_averager_template.json
---------------------------------------------------------------------- 
 default_jumeg_experiment_template.json
 default_jumeg_averager_template.json
----------------------------------------------------------------------

from jumeg.template.jumeg_template import experiment,averager

'''

__version__ = 0.0001


'''
 Helper Class +++ thnx to 
 http://stackoverflow.com/questions/1305532/convert-python-dict-to-object
 dict2obj_new expample
''' 
class dict2obj(dict):
    def __init__(self, dict_):
        super(dict2obj, self).__init__(dict_)
        for key in self:
            #if isinstance(key, unicode):
            #   print key 
            #   key = key.encode('utf-8')
            #   print key
            item = self[key]
            if isinstance(item, list):
               for idx, it in enumerate(item):
                    if isinstance(it, dict):
                        item[idx] = dict2obj(it)
            elif isinstance(item, dict):
                self[key] = dict2obj(item)

    def __getattr__(self, key):
        # Enhanced to handle key not found.
        if self.has_key(key):
            return self[key]
        else:
            return None
            
    
       
#import os
#import inspect
#print 'inspect.getfile(os) is:', inspect.getfile(os)

class JuMEG_Template(object):
    def __init__ (self):
        self._JUMEG_PATH_TEMPLATE = os.path.abspath( os.path.dirname(__file__) ) + '/../examples/templates'
        self._template_path    = os.getenv('JUMEG_PATH_TEMPLATE',self._JUMEG_PATH_TEMPLATE)
        self._template_name    = "test"
        self._template_list    = []
        self._template_postfix = "template"
        self._template_suffix  = '.json'
        self._template_dic     = {}
        self._template_data    = dict()
        self._verbose          = False
        self._subject_id       = None
        
        self.update_template_name_list()
 
#--- verbose    
    def _set_verbose(self,value):
         self._verbose = value

    def _get_verbose(self):
         return self._verbose
    verbose = property(_get_verbose, _set_verbose)
    
#--- subject id    
    def _set_subject_id(self,value):
         self._subject_id = value

    def _get_subject_id(self):
         return self._subject_id
    verbose = property(_get_subject_id, _set_subject_id)

#--- template name    
    def _get_template_data(self):
         return  self._template_data
    
    def _set_template_data(self, value):
         self._template_data = value
    template_data = property(_get_template_data,_set_template_data)
    
#--- template name    
    def _get_template_name(self):
         return  self._template_name
    
    def _set_template_name(self, value):
         self._template_name = value
    template_name = property(_get_template_name,_set_template_name)
#---  
    def _get_template_path(self):
         return  self._template_path
    
    def _set_template_path(self, value):
         self._template_path = value
    template_path = property(_get_template_path,_set_template_path)
#---  
    def _get_template_postfix(self):
         return  self._template_postfix
    
    def _set_template_postfix(self, value):
         self._template_postfix = value
    template_postfix = property(_get_template_postfix,_set_template_postfix)

#---  
    def _get_template_suffix(self):
         return  self._template_suffix
    
    def _set_template_suffix(self, value):
         self._template_suffix = value
    template_suffix = property(_get_template_suffix,_set_template_suffix)
#--- 
    def get_template_name_from_list(*args):
        if type( args[1] ) == int :
           return args[0]._template_name_list[ args[1] ]  # self = args[0]
        else :
           return args[0]._template_name_list  
    
    def _get_template_name_list(self):
         return self._template_name_list  
   
    def _set_template_name_list(self,value):
         self._template_name_list = value
    template_name_list = property(_get_template_name_list,_set_template_name_list)
    
    def update_template_name_list(self):
         """ read experiment template dir & update experiment names """
         self.template_name_list = []
         flist = glob.glob( self._template_path + '/*' + self._template_postfix + self._template_suffix)
         pat = re.compile( (self.template_path + '|/|'+ '_' + self.template_postfix + self._template_suffix) )
         self.template_name_list = pat.sub('', str.join(',',flist) ).split(',')
#---

    def _get_template_file_name(self):
         return  self.template_name +"_"+ self._template_postfix +"_"+ self._tempalte_suffix
    template_file_name = property(_get_template_file_name)
    
    
    def _get_full_template_file_name(self):
        return self._template_path + '/' + self.template_name +'_'+ self._template_postfix + self._template_suffix
    full_template_file_name = property(_get_full_template_file_name)

    def update_template_file(self):
          self.template_data = dict()
          # print self.full_template_file_name 
          f = open( self.full_template_file_name )
          self.template_data = json.load(f)
          f.close()
          #self.exp_obj = dict2obj( self._template_dic_yaml )
          return self.template_data
          
    def get_as_obj(self):
          # return dict2obj( self._template_dic_yaml )
          return dict2obj( self.template_data )
    
    def update_and_merge_dict(self, d, u, depth=-1):
        """
        copy from:
        http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        
        Recursively merge or update dict-like objects. 
        >>> update({'k1': {'k2': 2}}, {'k1': {'k2': {'k3': 3}}, 'k4': 4})
        {'k1': {'k2': {'k3': 3}}, 'k4': 4}
        return dict
        """
        import collections
        for k, v in u.iteritems():
            if isinstance(v, collections.Mapping) and not depth == 0:
               r = self.update_and_merge_dict(d.get(k, {}), v, depth=max(depth - 1, -1))
               d[k] = r
            elif isinstance(d, collections.Mapping):
               d[k] = u[k]
            else:
               d = {k: u[k]}
        
        return d
       
    def update_and_merge_obj(self, d, u, depth=-1):
        """
        copy from:
        http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        
        Recursively merge or update dict-like objects. 
        >>> update({'k1': {'k2': 2}}, {'k1': {'k2': {'k3': 3}}, 'k4': 4})
        {'k1': {'k2': {'k3': 3}}, 'k4': 4}
        return new object updated and merged
        """
        return dict2obj(self._dict_update_and_merge(d, u, depth=depth) )
        
      #class jumeg_template:
      #def __init__(self, **entries): 
      # self.__dict__.update(entries)

    def read_json(self,fjson):
        d = dict()
        if ( os.path.isfile( fjson ) ):
            FID = open( fjson )
            try:
                d = json.load(FID)
            except:
                d = dict()
                print"\n\n!!! ERROR NO JSON File Format:\n  ---> " + fjson
                print"\n\n"
            FID.close()
        return d
        
    def write_json(self,fjson, d):
        with open(fjson, 'wb') as FOUT:
             json.dump(d,FOUT, sort_keys=True)   
             # json.dump(d,FOUT, sort_keys=True,indent=4, separators=(',', ': ') )   
             FOUT.close()
        # return d
   
class JuMEG_Template_Experiments(JuMEG_Template):
    def __init__ (self):
        super(JuMEG_Template_Experiments, self).__init__()
        self._JUMEG_PATH_TEMPLATE_EXPERIMENTS = self._JUMEG_PATH_TEMPLATE+'/jumeg_experiments'
        self._template_path    = os.getenv('JUMEG_PATH_TEMPLATE_EXPERIMENTS',self._JUMEG_PATH_TEMPLATE_EXPERIMENTS)
        self._template_name    = 'default'
        self._template_list    = []
        self._template_postfix = 'jumeg_experiment_template'
      

class JuMEG_Template_Epocher(JuMEG_Template):
    def __init__ (self):
        super(JuMEG_Template_Epocher, self).__init__()
        self._JUMEG_PATH_TEMPLATE_EPOCHER = self._JUMEG_PATH_TEMPLATE +'/jumeg_epocher'
        self._template_path    = os.getenv('JUMEG_PATH_TEMPLATE_EPOCHER',self._JUMEG_PATH_TEMPLATE_EPOCHER)
        self._template_name    = 'default'
        self._template_list    = []
        self._template_postfix = 'jumeg_epocher_template'
      

#jumeg_template_experiment = JuMEG_Template_Experiments()
#jumeg_template_averager   = JuMEG_Template_Averager()

#jumeg_template_experiment = JuMEG_Template_Experiments()
#jumeg_template_epocher    = JuMEG_Template_Epocher()

experiment = JuMEG_Template_Experiments()
epocher    = JuMEG_Template_Epocher()
