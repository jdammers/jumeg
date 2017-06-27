import glob, os, re, sys
import json
from jumeg.jumeg_base import JuMEG_Base_Basic


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

__version__ = 0.003141


"""
 Helper function
 http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python
 work around json unicode-utf8 and python-2.x string conversion
"""
def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv

"""
 Helper function
 http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python
 work around json unicode-utf8 and python-2.x string conversion
"""
def _decode_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
           key = key.encode('utf-8')
        if isinstance(value, unicode):
           value = value.encode('utf-8')
        elif isinstance(value, list):
             value = _decode_list(value)
        elif isinstance(value, dict):
             value = _decode_dict(value)
        rv[key] = value
    return rv


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
                    #elif isinstance(it, unicode):
                    #     it = it.encode('utf-8')

            elif isinstance(item, dict):
                self[key] = dict2obj(item)

    def __getattr__(self, key):
        # Enhanced to handle key not found.
        if self.has_key(key):
            return self[key]
        else:
            return None

class JuMEG_Template(JuMEG_Base_Basic):
    def __init__ (self,template_name='DEFAULT'):
        super(JuMEG_Template, self).__init__()

        #self.__template_path     = os.getenv('JUMEG_PATH_TEMPLATE',self.__JUMEG_PATH_TEMPLATE)
        self.__template_name     = template_name
        self.__template_list     = []
        self.__template_name_list= []
        self.__template_postfix  = "template"
        self.__template_suffix   = '.json'
        self.__template_dic      = {}
        self.__template_data     = dict()
        self.__verbose           = False
        self.__template_isUpdate = False


        self.__template_path = self.template_path_default
        self.template_update_name_list()

#---
    def __get_template_path_default(self):
        return os.path.abspath( os.path.dirname(__file__) ) + '/../examples/templates'
    template_path_default = property(__get_template_path_default)

#--- template name
    def __get_template_name(self):
        return self.__template_name

    def __set_template_name(self, value):
         self.__template_name = value
    template_name = property(__get_template_name,__set_template_name)

#--- template path
    def __get_template_path(self):
        return self.__template_path

    def __set_template_path(self,v):
        self.__template_path = v
    template_path = property(__get_template_path,__set_template_path)

#--- template data
    def __get_template_data(self):
        return  self.__template_data

    def __set_template_data(self, value):
        self.__template_data = value
    template_data = property(__get_template_data,__set_template_data)

#--- template_postfix
    def __get_template_postfix(self):
        return self.__template_postfix
    def __set_template_postfix(self,v):
        self.__template_postfix = v
    template_postfix = property(__get_template_postfix,__set_template_postfix)

#--- template_suffix
    def __get_template_suffix(self):
        return  self.__template_suffix
    
    def __set_template_suffix(self, value):
        self.__template_suffix = value
    template_suffix = property(__get_template_suffix,__set_template_suffix)

#---template_isUpdate
    def __get_template_isUpdate(self):
        return self.__template_isUpdate
    template_isUpdate = property(__get_template_isUpdate)

#---
    def template_get_name_from_list(*args):
        if type( args[1] ) == int :
           return args[0].__template_name_list[ args[1] ]  # self = args[0]
        else :
           return args[0].__template_name_list
    
    def __get_template_name_list(self):
         return self.__template_name_list
   
    def __set_template_name_list(self,value):
         self.__template_name_list = value
    template_name_list = property(__get_template_name_list,__set_template_name_list)
    
    def template_update_name_list(self):
         """ read experiment template dir & update experiment names """
         self.template_name_list = []
         flist = glob.glob( self.template_path + '/*' + self.template_postfix + self.template_suffix)
         pat = re.compile( (self.template_path + '|/|'+ '_' + self.template_postfix + self.template_suffix) )
         self.template_name_list = pat.sub('', str.join(',',flist) ).split(',')
#---

    def __get_template_file_name(self):
         return  self.template_name +"_"+ self.template_postfix +"_"+ self.tempalte_suffix
    template_file_name = property(__get_template_file_name)
    
    
    def __get_full_template_file_name(self):
        return self.template_path + '/' + self.template_name +'_'+ self.template_postfix + self.template_suffix
    template_full_file_name = property(__get_full_template_file_name)


    def template_update_file(self):
          self.template_data = dict()
          self.__template_isUpdate = False

          f = open( self.template_full_file_name )
          try:
              self.template_data = json.load(f)
              self.template_data = _decode_dict(self.template_data)
              f.close()
              self.__template_isUpdate = True
          except ValueError as e:
              print "\n---> ERROR loading Template File: " +self.template_full_file_name 
              print(' --> invalid json: %s' % e)
         
          assert self.template_data,"---> ERROR in template file format [json]\n"  
          return 
          
    def template_get_as_obj(self):
          # return dict2obj( self._template_dic_yaml )
          return dict2obj( self.template_data )
    
    def template_update_and_merge_dict(self, d, u, depth=-1):
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
               r = self.template_update_and_merge_dict(d.get(k, {}), v, depth=max(depth - 1, -1))
               d[k] = r
            elif isinstance(d, collections.Mapping):
               d[k] = u[k]
            else:
               d = {k: u[k]}
        
        return d


      # obj = json.loads(s, object_hook=_decode_dict)

    def template_update_and_merge_obj(self, d, u, depth=-1):
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

    def template_read_json(self,fjson):
        d = dict()
        if ( os.path.isfile( fjson ) ):
            FID = open( fjson )
            try:
                d = json.load(FID)
                d = _decode_dict(d)
            except:
                d = dict()
                print"\n\n!!! ERROR NO JSON File Format:\n  ---> " + fjson
                print"\n\n"
            FID.close()
        return d
        
    def template_write_json(self,fjson, d):
        with open(fjson, 'wb') as FOUT:
             json.dump(d,FOUT, sort_keys=True)   
             # json.dump(d,FOUT, sort_keys=True,indent=4, separators=(',', ': ') )   
             FOUT.close()
        # return d
   
class JuMEG_Template_Experiments(JuMEG_Template):
    def __init__ (self):
        super(JuMEG_Template_Experiments, self).__init__()
        self.template_path    = os.getenv('JUMEG_PATH_TEMPLATE_EXPERIMENTS',self.template_path_default + '/jumeg_experiments')
        self.template_name    = 'default'
        self.template_postfix = 'jumeg_experiment_template'



'''
class JuMEG_Template_Epocher(JuMEG_Template):
    def __init__ (self):
        super(JuMEG_Template_Epocher, self).__init__()
        self.__JUMEG_PATH_TEMPLATE_EPOCHER = self._JUMEG_PATH_TEMPLATE + '/jumeg_epocher'
        self.template_path    = os.getenv('JUMEG_PATH_TEMPLATE_EPOCHER',self._JUMEG_PATH_TEMPLATE_EPOCHER)
        self.template_name    = 'default'
        self.template_list    = []
        self.template_postfix = 'jumeg_epocher_template'
      
template_epocher  = JuMEG_Template_Epocher()


#jumeg_template_experiment = JuMEG_Template_Experiments()
#jumeg_template_averager   = JuMEG_Template_Averager()

#jumeg_template_experiment = JuMEG_Template_Experiments()
#jumeg_template_epocher    = JuMEG_Template_Epocher()

'''

experiment = JuMEG_Template_Experiments()


