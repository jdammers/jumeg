import glob, os, re, sys, yaml
import json
from pprint import pprint

'''
----------------------------------------------------------------------
--- JuMEG Template Experiment Class ----------------------------------
---------------------------------------------------------------------- 
 autor      : Frank Boers 
 email      : f.boers@fz-juelich.de
 last update: 17.09.2014
 version    : 0.0113
---------------------------------------------------------------------- 
 r/w jumeg experiment template files in json format
----------------------------------------------------------------------
 <experiment>_jumeg_experiment_template.json
---------------------------------------------------------------------- 
 default_jumeg_experiment_template.json
----------------------------------------------------------------------

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
        self._JUMEG_PATH_TEMPLATE = os.path.abspath( os.path.dirname(__file__) )+'/../examples/templates/jumeg_experiments'
        self._template_path    = os.getenv('JUMEG_PATH_TEMPLATE_EXPERIMENTS',self._JUMEG_PATH_TEMPLATE)
        self._template_name    = 'default'
        self._template_list    = []
        self._template_postfix = 'jumeg_experiment_template'
        self._template_suffix  = '.json'
        self._template_dic     = {}
        
        self.update_template_name_list()
        
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
         #print "NAME %s: " % self._template_path
         #print "NAME %s: " % self._template_name
         #print "NAME %s: " % self._template_postfix
         #print "NAME %s: " % self._template_suffix
         #print s
         #return s
         return self._template_path + '/' + self.template_name +'_'+ self._template_postfix + self._template_suffix
         
         
    full_template_file_name = property(_get_full_template_file_name)

    def update_template_file(self):
          self.exp = []
          # print self.full_template_file_name 
          f = open( self.full_template_file_name )
          self.exp = json.load(f)
          f.close()
          #self.exp_obj = dict2obj( self._template_dic_yaml )
          return self.exp
    def get_as_obj(self):
          # return dict2obj( self._template_dic_yaml )
          return dict2obj( self.exp )
          
      #class jumeg_template:
      #def __init__(self, **entries): 
      # self.__dict__.update(entries)

jumeg_template_experiment = JuMEG_Template()
jumeg_template_events     = JuMEG_Template()

#if __name__ == '__main__':
#   TMP = JuMEG_Template_Experiment()
#   TMP.update_template_name_list()
   
   #TMP.template_name = "MEG94T"
   #TMP.update_template_file()

   #print"dic"
   #print TMP.exp['experiment']['name']
   
   #print "OO"
   #print TMP.exp.experiment.name
   #print TMP.exp.experiment.data_processing.preproc.raw.keys()
   #print TMP.exp.experiment.data_processing.preproc.raw.filter.fcut1
   
   #s = jumeg_template(**template)
   
   #print s.__dict__
   
   #print s['experiment']['name']
   
   #json.dump(s)
   
   #t=dict2obj_new(template)
   
   #print t.experiment.path
   #print t.experiment.path.mri
   
   
   
  # for person in ab.person:
  #print "Person ID:", person.id
#pprint.PrettyPrinter(template)

#self.template_name_list = re.search( ( self.template_path + '/'+ '(.+?)' + '_' + self.template_postfix + self._template_suffix) , str.join(',', flist) )

