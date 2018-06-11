import os
import pandas as pd

from jumeg.template.jumeg_template import JuMEG_Template

class JuMEG_Epocher_Template(JuMEG_Template):
    def __init__ (self):
        super(JuMEG_Epocher_Template, self).__init__()

        self.template_path    = os.getenv('JUMEG_PATH_TEMPLATE_EPOCHER',self.template_path_default + '/jumeg_epocher')
        #self.template_name    = template_name
        self.template_postfix = 'jumeg_epocher_template'


class JuMEG_Epocher_HDF(JuMEG_Epocher_Template):
    def __init__ (self):
        super(JuMEG_Epocher_HDF, self).__init__()

        self.__hdf_obj       = None
        self.__hdf_file_name = None
        self.__hdf_postfix   = '-epocher.hdf5'

#=== getter & setter

   #---  cp from jume_base compensate mne-changes  0.14dev
    def set_raw_filename(self, raw, v):
        if raw.info.has_key('filename'):
            raw.info['filename'] = v
        else:
            raw._filename = [v]

    def get_raw_filename(self, raw):
        if raw.info.has_key('filename'):
            return raw.info['filename']
        else:
            return raw.filenames[0]

        #--- HDFobj
    def __set_hdf_obj(self, v):
         self.__hdf_obj = v
    def __get_hdf_obj(self):
         return self.__hdf_obj
    HDFobj = property(__get_hdf_obj,__set_hdf_obj)


#--- hdf epocher file (output in hdf5)
    def __set_hdf_postfix(self, v):
         self.__hdf_postfix = v
    def __get_hdf_postfix(self):
         return self.__hdf_postfix
    hdf_postfix = property(__get_hdf_postfix,__set_hdf_postfix)


#--- hdf epocher filename for HDF obj
    def __get_hdf_file_name(self):
         return self.__hdf_file_name
    hdf_file_name = property(__get_hdf_file_name)

#---
    def hdf_file_name_init(self,fname=None,raw=None):
        if fname is None:
           if raw :
              fname = self.get_raw_filename(raw)
           else:
              fname = "TEST"
        self.__hdf_file_name = fname.split('c,rfDC')[0].strip('.fif') +'_' + self.template_name + self.hdf_postfix
        self.__hdf_file_name = self.__hdf_file_name.replace('__', '_')

        return self.__hdf_file_name

#--- init epocher output pandas HDF obj
    def hdf_obj_init(self,fhdf=None,fname=None,raw=None,overwrite=True):
        """
        create epocher pandas HDF5-Obj and file 
        input:
            fhdf  : hdf5 filename or,
            fname : fif-filename  or
            raw       = raw  => mne raw obj
            overwrite = True => will overwrite existing output file
        return: pandas.HDFStore obj   
        """
        if not fhdf :
           fhdf = self.hdf_file_name_init(fname=fname,raw=raw)

        if ( os.path.exists(fhdf) and overwrite ):
           os.remove(fhdf)
        # return pd.HDFStore( fhdf,format='table' ) not usefull with float16 !! ????
        self.HDFobj= pd.HDFStore( fhdf )

        if self.verbose:
           print"Open HDF file: "+ self.HDFobj.filename
        
        return self.HDFobj

    def hdf_obj_open(self,fhdf=None,fname=None,raw=None):
         """
         open epocher pandas HDF5-Obj 
         input:
            fhdf  : hdf5 filename or,
            fname : fif-filename  or
            raw   = raw => mne raw obj
         return: pandas.HDFStore obj
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

        return self.HDFobj.get_node(node)._v_groups.keys()


    def hdf_obj_get_dataframe(self,key):
        """

        :param key:
        :return:
        """
        return self.HDFobj.get(key)


    def hdf_obj_set_dataframe(self,data=None,key=None):
        """

        :param data:
        :param key:
        :return:
        """
        self.HDFobj[key]=data


    def hdf_obj_get_attributes(self,HStorer=None,key=None,attr=None):
        """

        :param HStorer:
        :param key:
        :param attr:
        :return:
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
        """

        :param HStorer:
        :param key:
        :param overwrite:
        :param storer_attrs:
        :return:
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
        """
         store & update data[frame] and user attributes to HDFobj
        :param HDFobj
        :param df   : pandas DataFrame or Panel
        :param key  : node name [key=None]
        :param attrs: user atributes [**storer_attr]
        :return     : HDFobj
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

    
    def hdf_get_key_list(self,node='epocher',key_list=None):
          """

          :param node:
          :param key_list=None
          :return:
          """     
          
         # import re
         # s='/epocher////LLst//'
         # re.sub('//*','/',s)

          clist = []             
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
            
             

jumeg_epocher_hdf= JuMEG_Epocher_HDF()