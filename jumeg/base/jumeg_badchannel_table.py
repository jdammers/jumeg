#!/usr/bin/env python3
# -+-coding: utf-8 -+-

"""
"""

#--------------------------------------------
# Authors: Frank Boers <f.boers@fz-juelich.de> 
#
#-------------------------------------------- 
# Date: 15.05.19
#-------------------------------------------- 
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import os,logging
import pandas as pd

from jumeg.base.jumeg_base import jumeg_base

from jumeg.base import jumeg_logger
logger = logging.getLogger('jumeg')

__version__="2019.05.16.001"


class JuMEG_BadChannelTable(object):
    """
    CLS to store noisy/bad channel information for all files within an experiment
    HDF5format

    id,scan,filename,file path, bads
    
    Example:
    --------
     from jumeg.base.jumeg_badchannel_table import JuMEG_BadChannelTable
     BCT = JuMEG_BadChannelTable(verbose=True)
     BCT.open(path=path,fname=fname,overwrite=False)
    
     id    = "123456"
     scan  = "FREEVIEW01"
     fname = "$JUMEG_TEST_DATA/mne/211747/FREEVIEW01/180109_0955/1/211747_FREEVIEW01_180109_0955_1_c,rfDC,meeg-raw.fif"
     bads  = "MEG 007,MEG 142,MEG 156,RFM 011"
     BCT.set_bads(id=id,scan=scan,fname=fname,bads=bads)
     BCT.show()
    
     fname = "$JUMEG_TEST_DATA/mne/211747/FREEVIEW01/180109_0955/1/211747_FREEVIEW01_180109_0955_2_c,rfDC,meeg-raw.fif"
     bads  = "MEG 107,MEG 242,MEG 256,RFM 012"
     BCT.set_bads(id=id,scan=scan,fname=fname,bads=bads)
     BCT.show()
    
     BCT.set_bads(id=id,scan=scan,fname=fname,bads="NIX")
     BCT.show()
    
     print("bads: {}".format( BCT.get_badslist(fname=fname) ))

    """
    
    def __init__(self,**kwargs):
        self._HDF      = None
        self._fname    = None
        self.verbose   = False
        self.postfix   =  'bads'
        self.extention = ".hdf5"
        
        self._update_from_kwargs(**kwargs)
   
    def _update_from_kwargs(self,**kwargs):
        #self._fname    = kwargs.get("filename",self._fname)
        self.postfix   = kwargs.get("postfix",self.postfix)
        self.extention = kwargs.get("extention",self.extention)
        self.verbose   = kwargs.get("verbose",self.verbose)
    
    def update(self,**kwargs):
        self._update_from_kwargs(**kwargs)
      
     
   #--- HDFobj
    @property
    def HDF(self):
        return self._HDF
    
    @property
    def is_open(self):
        try:
            if self.HDF.is_open:
               return True
        except:
            logger.error(" HDFobj is not open !!!\n")
        return False

    #--- HDFobj filename
    @property
    def filename(self):
        return self._fname
    #---
    def update_filename(self,path=None,fname=None,postfix=None,extention=None):
        """
        make filename from fname  and path
        remove file extention add postfix and extention
        mkdir if not exist
        
        :param path:
        :param fname:
        :param postfix:
        :param extention:
        :return:

        hdf fullfilename
        """
        
        if fname:
           self._fname = os.path.expandvars(os.path.expanduser(fname))
           self._fname = os.path.splitext( fname )[0] + "_" + self.postfix + self.extention
        
        if path:
           try:
               path = os.path.expandvars(os.path.expanduser(path))
               os.makedirs(path,exist_ok=True)
               self._fname = os.path.join(path,self._fname)
           except:
                logger.exception(" can not create HDF directory path: {}".format(path) +
                                 " HDF file: {}".format(self._fname))
                return None
                
        if self.verbose:
           logger.info("  -> setting HDF file name to: {}".format(self.filename))
        
        return self._fname
        
    #--- init output pandas HDF obj
    def init(self,fhdf=None,path=None,fname=None,overwrite=False):
        """create pandas HDF5-Obj and file

        Parameters
        ----------
        fhdf     : hdf5 filename <None>
        fname    : fif-filename  <None>
        hdf_path : path to hdf file <None> if None use fif-file path
        overwrite: will overwrite existing output file <False>

        Returns
        ---------
        pandas.HDFobj
        """
        
        if not fhdf:
           fhdf = self.update_filename(fname=fname,path=path)
        
        if (os.path.exists(fhdf) and overwrite):
            if self.verbose:
               logger.info("---> HDF file overwriting : {}".format(fhdf))
            self.close()
            os.remove(fhdf)
            
        
        # return pd.HDFStore( fhdf,format='table' ) not usefull with float16 !! ????
        try:
            self._HDF = None
            self._HDF = pd.HDFStore(fhdf)
            if self.verbose:
               logger.info("---> HDF file created: {}".format(fhdf))
        except:
            logger.exception(" error open or creating HDFobj; e.g.: release lock?" +
                             "  -> HDF filename: {}".format(fhdf))
        if self.verbose:
           logger.info("---> Open HDF file: {}".format(self.HDF.filename))
        
        return self._HDF
        
    def open(self,fhdf=None,path=None,fname=None,overwrite=False):
        """
        open  HDF file; pandas HDF5-Obj
            if exists and not overwrite
            if fhdf use fhdf as fullfilename
            else construct filename and mkdir from path and fname

        Parameters
        ----------
           fhdf     : hdf5 filename or,
           fname    : fif-filename  or
           overwrite: <False>

        Returns
        ---------
        pandas.HDFStore obj
        """
        return self.init(fhdf=fhdf,path=path,fname=fname,overwrite=overwrite)
    
    def close(self):
        if self.is_open:
           if self.verbose:
              logging.debug("  -> HDF closing: {}".format(self.HDF.filename))
           self._HDF.close()
       
        
    def reset_key(self,k):
        if kf in self.HDF.keys():
           self.HDF.remove(k)
        return k
      
    def list_keys_from_node(self,node):
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
        return self.HDF.get_node(node)._v_groups.keys()
    
    def get_dataframe(self,key):
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
        return self.HDF.get(self.key2hdfkey(key))
    
    def set_dataframe(self,data=None,key=None):
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
        self.HDF[key] = data
    
    def obj_get_attributes(self,HStorer=None,key=None,attr=None):
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
                logger.exception("ERROR in hdf_obj_get_attributes => can not store key attributes no such Storer-Obj" +
                                 "HDF : {}\n".format(self.HDF.filename) +
                                 "key : {}\n".format(key) +
                                 "Attr: {}\n".format(attr))
                return
        
        elif self.is_open:
            return self.HDF.get_storer(key).attrs[attr]
    
       
    def update_dataframe(self,df,key=None,reset=True,**storer_attrs):
        """store & update data[frame] and user attributes to HDFobj
           call <hdf_obj_store_attributes> to update user attributes in HDFobj

        Parameters
        ----------
        df   : pandas DataFrame or Panel
        key  : node name <None
        attrs: user atributes <**storer_attr>
        reset: will reset/clear dataframe first <True>

        """
        #-- avoid error in python3 pandas HDF  e.g: STI 13 => STI-13
        
        if not self.is_open:
           return None
        
        #--- fist clean HDF df & parameter
        if reset:
           self.reset_key(key)
        #--- copy dataframe to HDF
        self.HDF[key] = df
        
        #--- update attributes e.g. save dicts like parameter,info ...
       # return self.store_attributes(key=key,**storer_attrs)

    def bads2list(self,bads,sep=","):
        """
        
        :param bads:
        :param sep:
        :return:
        """
        if not bads:
           return []
        elif isinstance(bads,str):
             return bads.split(sep)
        return bads

    def get_key(self,id=None,scan=None,fname=None):
        """
        
        :param id:
        :param scan:
        :param fname:
        :return:
         key  e.g:  id + "/" + scan + "/" + run_name
        
        Example:
        --------
         input : fname = 123456_FREEVIEW01_200108_0955_1_c,rfDC,meeg-raw.fif
         output: 123456/FREEVIEW01/123456_FREEVIEW01_200108_0955_1
        """
        pdf_name = "_".join(os.path.basename(fname).split("_")[0:-1])
        id       = id if id else pdf_name.split("_")[0]
        scan     = scan if scan else pdf_name.split("_")[1]
        
        return id + "/" + scan + "/" + pdf_name

    def get_bads(self,id=None,scan=None,fname=None,key=None):
        """
        
        :param id:
        :param scan:
        :param fname:
        :return:
        """
        if not self.is_open: return
        if not key:
           if not fname: return
           key = self.get_key(id=id,scan=scan,fname=fname)
        
        try:
           return self.HDF[key]
        except:
           logger.exception("\n".join(["---> ERROR can not get bads from HDF:"
                                       "  -> filename: {}".format(fname),
                                       "  -> bads    : {}".format(bads),
                                       "  -> HDF key : {}".format(key),
                                       "  -> HDf file: {}".format(self.HDF.filename)]))
     
    def get_badslist(self,**kwargs):
        """
        
        :param id:
        :param scan:
        :param fname:
        :return:
        """
        s = self.get_bads(**kwargs)
        if isinstance(s,pd.Series):
          return s.tolist()
        
    def set_bads(self,id=None,scan=None,fname=None,bads=None):
        """
        set list of bads in HDF as pd.Series
        build HDF key : id + "/" + scan + "/" + pdf_name
        :param id:
        :param scan:
        :param fname: (full)filename
        :param bads: string or list of strings
        :return:
        
        Example:
        --------
        
        id    = "123456"
        scan  = "FREEVIEW01"
        fname = "$JUMEG_TEST_DATA/mne/123456/FREEVIEW01/180109_0955/1/123456_FREEVIEW01_180109_0955_1_c,rfDC,meeg-raw.fif
        bads  = "MEG 007,MEG 142,MEG 156,RFM 011"
        
        BCT.set_bads(fname=fname,bads=bads)
        BCT.set_bads(id=id,scan=scan,fname=fname,bads=bads)
        
        """
        if not self.is_open: return
        if not fname       : return
        key = self.get_key(id=id,scan=scan,fname=fname)
        
        try:
           self.HDF[key] = pd.Series( self.bads2list(bads) )  # ["MEG 007","MEG 142","MEG 156","RFM 011"]
        except:
           logger.exception("\n".join(["---> ERROR can not set bads in HDF:"
                                       "  -> filename: {}".format(fname),
                                       "  -> bads    : {}".format(bads),
                                       "  -> HDF key : {}".format(key),
                                       "  -> HDf file: {}".format(self.HDF.filename)]))
        self.HDF.flush()
        
    def show(self):
        if self.is_open:
           msg=[]
           for key in self.HDF.keys():
               msg.append("  -> {} : {}".format(key,self.get_badslist(key=key)))
           logger.info( "\n".join(msg))



def update_bads_in_hdf(fhdf=None,bads=None,path=None,fname=None,overwrite=False,verbose=False):
    """
    :param fhdf
    :param path:
    :param fname:
    :param overwrite:
    :param verbose:
    :return:
    """
    BCT = JuMEG_BadChannelTable(verbose=True)
    BCT.open(fhdf=fhdf,path=path,fname=fname,overwrite=overwrite)
    BCT.set_bads(fname=fname,bads=bads)
    BCT.close()
    
    
#--- test
def _test():
    path  = "$JUMEG_TEST_DATA/mne"
    fname = "FREEVIEW01"
    
    BCT = JuMEG_BadChannelTable(verbose=True)
    
    BCT.open(path=path,fname=fname,overwrite=True)
    
    id    = "123456"
    scan = "FREEVIEW01"
    fname = "$JUMEG_TEST_DATA/mne/211747/FREEVIEW01/180109_0955/1/211747_FREEVIEW01_180109_0955_1_c,rfDC,meeg-raw.fif"
    bads = "MEG 007,MEG 142,MEG 156,RFM 011"
    BCT.set_bads(id=id,scan=scan,fname=fname,bads=bads)
    
    BCT.show()
    
    fname = "$JUMEG_TEST_DATA/mne/211747/FREEVIEW01/180109_0955/1/211747_FREEVIEW01_180109_0955_2_c,rfDC,meeg-raw.fif"
    BCT.set_bads(id=id,scan=scan,fname=fname,bads=bads)
    BCT.show()
    
    BCT.set_bads(id=id,scan=scan,fname=fname,bads="NIX")
    BCT.show()
    print("bads: {}".format( BCT.get_badslist(fname=fname) ))
    
    BCT.close()


if __name__ == "__main__":
  #--- init/update logger
   logger=jumeg_logger.setup_script_logging(logger=logger,level=logging.DEBUG)
   _test()
   