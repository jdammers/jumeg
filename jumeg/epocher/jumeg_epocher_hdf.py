"""
Created on Tue Jun  2 13:38:32 2015

@author: fboers

---> update 02.04.2018 FB
  -> add opt -feeg
---> update 04.04.2018 FB
  -> properties rt idx
---> update 13.04.2018 FB
  -> epocher rebuild new properties
"""

import os
import pandas as pd

from jumeg.jumeg_base import jumeg_base
from jumeg.template.jumeg_template import JuMEG_Template

__version__="2018.04.13.001"

class JuMEG_Epocher_Template(JuMEG_Template):
    def __init__ (self):
        super(JuMEG_Epocher_Template, self).__init__()
        self.template_path    = os.getenv('JUMEG_TEMPLATE_PATH_EPOCHER',self.template_path_default + '/jumeg_epocher')
        self.template_postfix = 'jumeg_epocher_template'
       
    @property
    def env_template_path(self)  : return os.getenv('JUMEG_TEMPLATE_PATH_EPOCHER','./')
    @env_template_path.setter
    def env_template_path(self,v): os.environ['JUMEG_TEMPLATE_PATH_EPOCHER']=v


class JuMEG_Epocher_HDF(JuMEG_Epocher_Template):
    """ HDF I/O"""
    def __init__ (self):
        super(JuMEG_Epocher_HDF, self).__init__()

        self._hdf_obj           = None
        self._hdf_filename      = None
        self._hdf_node_names    = {"epo":"epocher","eve":"events","oca":"ocarta","cha":"channel_events","con":"conditions","art":"artifacts" }
        
        self._hdf_postfix            = '-epocher.hdf5'
        self._hdf_stat_postfix       = '-epocher-stats.csv'
        self._hdf_obj_attributes     = {'epocher':'epocher_parameter','info':'info_parameter'}     
   
        self._raw_postfix            = 'c,rfDC'
        self._raw_extention          = '.fif'
       
  #---
    @property
    def hdf_obj_attribute_epocher(self):return self._hdf_obj_attributes['epocher']
  #---
    @property
    def hdf_obj_attribute_info(self):return self._hdf_obj_attributes['info']
  
  #--- node names "epocher":"epocher","events":"events","ocarta":"ocarta","channels":"channels","artifacts":"artifacts"
    def hdf_node_name(self,node=None,name=None):
        if name:
           self._hdf_node_names[node] = name 
        return self._hdf_node_names[node]  
    
    @property
    def hdf_node_name_epocher(self):   return  "/"+self._hdf_node_names["epo"]
  #---
    @property
    def hdf_node_name_events(self):    return  "/"+self._hdf_node_names["eve"]
  #---
    @property
    def hdf_node_name_ocarta(self):    return  "/"+self._hdf_node_names["oca"]
  #---
    @property
    def hdf_node_name_channel_events(self):  return  "/"+self._hdf_node_names["cha"]
 #---
    @property
    def hdf_node_name_conditions(self):return  "/"+self._hdf_node_names["con"]
  #---
    @property
    def hdf_node_name_artifact(self):  return  "/"+self._hdf_node_names["art"]
  #---
   
    @property
    def hdf_stat_postfix(self): return self._hdf_stat_postfix
    @hdf_stat_postfix.setter
    def hdf_stat_postfix(self,v): self._hdf_stat_postfix=v  
  #---
    @property
    def raw_postfix(self): return self._raw_postfix
    @raw_postfix.setter
    def raw_postfix(self,v): self._raw_postfix=v  
  #---
    @property
    def raw_extention(self): return self._raw_extention
    @raw_extention.setter
    def raw_extention(self,v): self._raw_extention=v  
  #--- HDFobj
    @property
    def HDFobj(self): return self._hdf_obj
    @HDFobj.setter
    def HDFobj(self,v): self._hdf_obj=v 
  #--- hdf epocher file (output in hdf5)
    @property
    def hdf_postfix(self): return self._hdf_postfix
    @hdf_postfix.setter
    def hdf_postfix(self,v): self._hdf_postfix=v   
  #--- hdf epocher filename for HDF obj
    @property
    def hdf_filename(self): return self._hdf_filename    
    
    
  #---
    def hdf_filename_init(self,fname=None,raw=None):
        """ init hdf filename
        Parameters
        ----------
        fname: <None>
        raw  : <None>
        
        Returns
        ----------
        hdf file name 
        """
        fname = jumeg_base.get_fif_name(fname=fname,raw=raw)
        self._hdf_filename = fname.split( self.raw_postfix )[0].strip( self.raw_extention ) +'_' + self.template_name + self.hdf_postfix
        self._hdf_filename = self._hdf_filename.replace('__', '_')

        return self._hdf_filename

#--- init epocher output pandas HDF obj
    def hdf_obj_init(self,fhdf=None,fname=None,raw=None,overwrite=True):
        """create pandas HDF5-Obj and file 
        
        Parameters
        ----------
        fhdf  : hdf5 filename <None>
        fname : fif-filename  <None>
        raw   : mne raw obj   <None>
        overwrite: will overwrite existing output file <True>
        
        Returns
        ---------
        pandas.HDFobj   
        """
        if not fhdf :
           fhdf = self.hdf_filename_init(fname=fname,raw=raw)

        if ( os.path.exists(fhdf) and overwrite ):
           os.remove(fhdf)
        # return pd.HDFStore( fhdf,format='table' ) not usefull with float16 !! ????
        self.HDFobj= pd.HDFStore( fhdf )

        if self.verbose:
           print"Open HDF file: "+ self.HDFobj.filename
       
        return self.HDFobj

    def hdf_obj_open(self,fhdf=None,fname=None,raw=None):
         """open  HDF file; pandas HDF5-Obj 
         
         Parameters
         ----------
            fhdf  : hdf5 filename or,
            fname : fif-filename  or
            raw   = raw => mne raw obj
         
         Returns
         ---------
         pandas.HDFStore obj
         """
         return self.hdf_obj_init(fhdf=fhdf,fname=fname,raw=raw,overwrite=False)

    def hdf_obj_reset_key(self,k):
        if k in self.HDFobj.keys():
           self.HDFobj.remove(k)
        return k

    def hdf_obj_is_open(self):
        if self.HDFobj.is_open:
           return True
        else:
           print "\n\n!!! ERROR HDFobj is not open !!!\n"
           return None

    def hdf_obj_list_keys_from_node(self,node):
        """ get key list from HDF node
        
        Parameters
        ----------
        node: HDFobj node
              e.g:  for node in HDFobj.keys(): 
                    HDFobj["/epcher/M100"]
        Returns
        ---------
        key list from node
        
        Example:
        ---------    
        self.hdf_obj_list_keys_from_node("/epocher")
            [condition1,condition2 ... conditionN ]
        """
        return self.HDFobj.get_node(node)._v_groups.keys()

    def hdf_obj_get_dataframe(self,key):
        """ get pandas dataframe from HDFobj
        
        Parameters
        ----------
        key: full dataframe key </node + /key ... + /keyN>
        
        Returns
        ----------
        pandas dataframe 
        
        Example
        ----------            
         df = self.hdf_obj_get_dataframe("/epocher/M100")
         
        """
        return self.HDFobj.get(key)

    def hdf_obj_set_dataframe(self,data=None,key=None):
        """set dataframe in HDFobj for key
        
        Parameters
        ----------
        data: pandas dataframe
        key : full dataframe key </node + /key ... + /keyN>
                
        Returns
        ----------
        None 
        
        Example
        ----------            
         self.hdf_obj_set_dataframe(data=<M100-dataframe>,key="/epocher/M100")
       
        """
        self.HDFobj[key]=data

    def hdf_obj_get_attributes(self,HStorer=None,key=None,attr=None):
        """
        Parameters
        ----------
        HStorer: Hdf Storer Obj, to get information from attribute dict  <None>
                if None : use self.HDFobj
        key    : full dataframe key </node + /key ... + /keyN>   <None>
        attr   : name of attribute dictionary 
       
        Returns
        ----------    
        attribute parameter as dictionary
        
        Example
        ----------       
         my_attribut_dict  = self.hdf_obj_get_attributes(key="/epocher/M100",attr="epocher_parameter")
         
         epocher_parameter = self.hdf_obj_get_attributes(key=ep_key,attr=self.hdf_obj_attribute_epocher)
        """
        if HStorer:
           try:
               if HStorer.is_exists:
                  return HStorer.get_storer(key).attrs[attr]
           except:
               print "\nERROR in hdf_obj_get_attributes => can not store key attributes no such Storer-Obj"
               print "HDF : " + self.HDFobj.filename
               print "key : " + key
               print "Attr: " + attr
               print "\n"
               return

        elif self.hdf_obj_is_open():
             return self.HDFobj.get_storer(key).attrs[attr]

    def hdf_obj_store_attributes(self,HStorer=None,key=None,overwrite=True,**storer_attrs):
        """update init HDF Storer attributes
        used to save additional information as attributes in HDFobj[key], dictionary structure
        if HStorer is None: get HDF Storer obj from HDFobj[key]
        
        Parameters
        ----------
        HStorer  : HDF Storer obj <None>
        key      : HDF key
        overwrite: overwrite HDF[key] data structure <True>
        storer_attrs: attributes key,value or dict  <**kwargs>
        
        Returns
        ----------
        HDFStorer obj
        
        Example
        ----------    
        storer_attrs = {'epocher_parameter': self.parameter,'info_parameter':stimulus_info}
        HStorerObj   = self.hdf_obj_store_attributes(key="/epocher/M100",**storer_attrs)
        """

        if not HStorer:

           if self.hdf_obj_is_open():
              HStorer = self.HDFobj.get_storer(key)
           else:
              return None

       #--- overwrite HDF storer attributes
        if overwrite:
           for atr in storer_attrs :
               HStorer.attrs[atr] = storer_attrs[atr]
        else:
       #---  merge attributes
           for atr in storer_attrs :
               if HStorer.attrs[atr] :
                  HStorer.attrs[atr]= self.template_update_and_merge_dict(HStorer.attrs[atr],storer_attrs[atr])
               else:
                  HStorer.attrs[atr] = storer_attrs[atr]

        self.HDFobj.flush()

        if self.verbose :
           print "\n---> HDFobj store attributes to HDF5 : " + key
           print self.HDFobj.filename
           for atr in storer_attrs :
               print"---> PARAMETER "+  atr +":"
               print HStorer.attrs[atr]
               print"\n"

        return HStorer

    def hdf_obj_update_dataframe(self,df,key=None,reset=True,**storer_attrs):
        """store & update data[frame] and user attributes to HDFobj
           call <hdf_obj_store_attributes> to update user attributes in HDFobj
           
        Parameters
        ----------
        df   : pandas DataFrame or Panel
        key  : node name <None
        attrs: user atributes <**storer_attr>
        reset: will reset/clear dataframe first <True>
        
        Returns
        ----------
        HDFStorer obj
        
        Example
        ----------
        storer_attrs = {'epocher_parameter': self.parameter,'info_parameter':stimulus_info}
        self.hdf_obj_update_dataframe(stimulus_data_frame.astype(np.int32),key=key,**storer_attrs )
        
        """

        if not self.hdf_obj_is_open():
           return None

       #--- fist clean HDF df & parameter
        if reset:
           self.hdf_obj_reset_key(key)

       #--- copy dataframe to HDF
        self.HDFobj[key] = df

       #--- update attributes e.g. save dicts like parameter,info ...
        return self.hdf_obj_store_attributes(key=key,**storer_attrs)

    
    def hdf_get_key_list(self,node=None,key_list=None):
          """get list of keys from HDFobj at <node>
          
          Parameters
          ----------
          node    :  HDFobj node  <epocher node name>
          key_list: list of keys in HDFobj[node] <None>
        
          Returns
          ----------
          list of keys
         
          Example
          ----------
          
          """     
          
         # import re
         # s='/epocher////LLst//'
         # re.sub('//*','/',s)

          clist = []             
          if not node:
             node = self.hdf_node_name_epocher
             
          node = node.strip('/')
          node_list = self.hdf_obj_list_keys_from_node(node)          
          
          if self.verbose:
             print"\n---> HDF get keys from node: " + node          
             print"---> node list: "
             print node_list
             print"\n"
             
          if key_list :
             for k in key_list :
                 if k in self.hdf_obj_list_keys_from_node(node):
                    clist.append(node +'/'+ k )
             return clist
          else :
             return self.hdf_obj_list_keys_from_node(node)
           
#---
    def hdf_get_hdf_key(self,condi):
        """get key from HDFobj to extract dataframe
        
        Parameters
        ----------
        condi : epocher condition 
        
        Returns
        ----------
        HDFobj key  e.g.: /epocher/< condition name >
        """
        
        print " ---> START EPOCHER extract condition : " + condi
        if condi.startswith('epocher'):
           ep_key = '/'+condi
        else:
           ep_key = '/epocher/'+condi
        
        if ( ep_key in self.HDFobj.keys() ): return ep_key
        return 
                 

jumeg_epocher_hdf= JuMEG_Epocher_HDF()