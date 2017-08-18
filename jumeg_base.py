# -*- coding: utf-8 -*-

'''
JuMEG Base Class to provide wrapper & helper functions

Authors: 
         Prank Boers     <f.boers@fz-juelich.de>
         Praveen Sripad  <praveen.sripad@rwth-aachen.de>
License: BSD 3 clause

---> update 23.06.2016 FB

---> update 20.12.2016 FB
 --> add eeg pick-cls
 --> eeg BrainVision IO support

---> update 23.12.2016 FB
 --> add opt feeg in get_filename_list_from_file
 --> to merge eeg BrainVision with meg in jumeg_processing_batch

---> update 04.01.2017 FB
 --> add neww CLS JuMEG_Base_FIF_IO
 --> to merge eeg BrainVision with meg in jumeg_processing_batch

'''

import os
import mne

'''
class AccessorType(type):
    """
    meta class example
    http://eli.thegreenplace.net/2011/08/14/python-metaclasses-by-example
    """
    def __init__(self, name, bases, d):
        type.__init__(self, name, bases, d)
        accessors = {}
        prefixs = ["__get_", "__set_", "__del_"]
        for k in d.keys():
            v = getattr(self, k)
            for i in range(3):
                if k.startswith(prefixs[i]):
                    accessors.setdefault(k[4:], [None, None, None])[i] = v
        for name, (getter, setter, deler) in accessors.items():
            # create default behaviours for the property - if we leave
            # the getter as None we won't be able to getattr, etc..

            # [...] some code that implements the above comment

            setattr(self, name, property(getter, setter, deler, ""))

'''

class JuMEG_Base_Basic(object):
    def __init__ (self):
        super(JuMEG_Base_Basic, self).__init__()

        self.__version__     = 0.00014
        self.__verbose       = False
        self.__template_name = None
        self.__do_run        = False
        self.__do_save       = False
        self.__do_plot       = False

#--- version
    def __get_version(self):
        return self.__version
    def __set_version(self,v):
        self.__version=v
    version = property(__get_version,__set_version)

#=== FLAGS ==========
#--- verbose
    def __set_verbose(self,value):
        self.__verbose = value
    def __get_verbose(self):
        return self.__verbose
    verbose = property(__get_verbose, __set_verbose)

#--- run
    def __set_do_run(self, v):
        self.__do_run = v
    def __get_do_run(self):
        return self.__do_run
    do_run = property(__get_do_run,__set_do_run)

#--- save
    def __set_do_save(self, v):
        self.__do_save = v
    def __get_do_save(self):
        return self.__do_save
    do_save = property(__get_do_save,__set_do_save)

#--- plot
    def __set_do_plot(self, v):
        self.__do_plot = v
    def __get_do_plot(self):
        return self.__do_plot
    do_plot = property(__get_do_plot,__set_do_plot)


class JuMEG_Base_PickChannels(object):
     """ MNE Wrapper Class for mne.pick_types
         return list of channel index from mne.raw obj e.g. for special groups
        
         Wrapper call to    
         --> mne.pick_types(raw.info, **fiobj.pick_all) 
         --> mne.pick_types(info, meg=True, eeg=False, stim=False, eog=False, ecg=False, emg=False, ref_meg='auto',
                           misc=False, resp=False, chpi=False, exci=False, ias=False, syst=False,
                           include=[], exclude='bads', selection=None)

         https://github.com/mne-tools/mne-python/blob/master/mne/io/pick.py  lines 20ff
         type : 'grad' | 'mag' | 'eeg' | 'stim' | 'eog' | 'emg' | 'ecg' |'ref_meg' | 'resp' | 'exci' | 'ias' | 'syst' | 'misc'|'seeg' | 'chpi'  
     
         Example:
         picks = JuMEG_Base_PickChannels()

         picks.meg( raw )
         return meg channel index array => 4D Magnes3600 => [0 .. 247]
         
         picks.meg_nobads( raw )
         return meg channel index array without bad channels
         
     """        
     def __init__ (self):
         import mne
         self.__version__  = 20160623
        
#--- MNE foool fct  -> picks preselected channel groups

     def channels(self,raw):
         ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude=None'''
         return mne.pick_types(raw.info,meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude=[])       
     def channels_nobads(self, raw):
         ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude='bads' '''
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude='bads')
       
     def all(self, raw):
         ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude=None '''
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude=[])       
    
     def all_nobads(self, raw):
         ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude='bads' '''
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude='bads')
       
     def meg(self,raw):
         ''' call with meg=True'''
         return mne.pick_types(raw.info,meg=True)      
     def meg_nobads(self,raw):
         ''' call with meg=True,exclude='bads' '''
         return mne.pick_types(raw.info, meg=True,exclude='bads')
    
     def ref(self,raw):
         ''' call with ref=True'''
         return mne.pick_types(raw.info,ref_meg=True,meg=False,eeg=False,stim=False,eog=False)
     def ref_nobads(self,raw):
         ''' call with ref=True,exclude='bads' '''
         return mne.pick_types(raw.info,ref_meg=True,meg=False,eeg=False,stim=False,eog=False,exclude='bads')
        
     def meg_and_ref(self,raw):
         ''' call with meg=True,ref_meg=True'''
         return mne.pick_types(raw.info, meg=True,ref_meg=True, eeg=False, stim=False,eog=False)
     def meg_and_ref_nobads(self,raw):
         ''' call with meg=mag,ref_meg=True,exclude='bads' '''
         return mne.pick_types(raw.info,meg=True,ref_meg=True,eeg=False,stim=False,eog=False,exclude='bads')
  
     def meg_ecg_eog_stim(self,raw):
         ''' call with meg=True,ref_meg=False,ecg=True,eog=Truestim=True,'''
         return mne.pick_types(raw.info, meg=True,ref_meg=False,eeg=False,stim=True,eog=True,ecg=True)
     def meg_ecg_eog_stim_nobads(self,raw):
         ''' call with meg=True,ref_meg=False,ecg=True,eog=True,stim=True,exclude=bads'''
         return mne.pick_types(raw.info, meg=True,ref_meg=False,eeg=False,stim=True,eog=True,ecg=True,exclude='bads')
       
     def ecg(self,raw):
         ''' meg=False,ref_meg=False,ecg=True,eog=False '''
         return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=True,eog=False)
     def eog(self,raw):
         ''' meg=False,ref_meg=False,ecg=False,eog=True '''
         return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=False,eog=True)
   
     def ecg_eog(self,raw):
         ''' meg=False,ref_meg=False,ecg=True,eog=True '''
         return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=True,eog=True)

     def eeg(self,raw):
         ''' meg=False,ref_meg=False,ecg=False,eog=False '''
         return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=False,eog=False,eeg=True)
     def eeg_nobads(self, raw):
         ''' meg=False,ref_meg=False,ecg=False,eog=False '''
         return mne.pick_types(raw.info, meg=False, ref_meg=False, ecg=False, eog=False, eeg=True, exclude='bads')

     def eeg_ecg_eog(self, raw):
         ''' meg=False,ref_meg=False,ecg=True,eog=True,eeg=True '''
         return mne.pick_types(raw.info, meg=False, ref_meg=False, ecg=True, eog=True, eeg=True)
     def eeg_ecg_eog_nobads(self, raw):
         ''' meg=False,ref_meg=False,ecg=True,eog=True,eeg=True '''
         return mne.pick_types(raw.info, meg=False, ref_meg=False, ecg=True, eog=True, eeg=True, exclude='bads')

     def stim(self,raw):
         ''' call with meg=False,stim=True '''
         return mne.pick_types(raw.info,meg=False,stim=True)
        
     def response(self,raw):
         ''' call with meg=False,resp=True'''
         return mne.pick_types(raw.info,meg=False,resp=True)
        
     def stim_response(self,raw):
         ''' call with meg=False,stim=True,resp=True'''
         return mne.pick_types(raw.info, meg=False,stim=True,resp=True)

     def exclude_trigger(self, raw):
         ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude=None '''
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=False,resp=False)       

     def bads(self,raw):
         """ return raw.info[bads] """
         return raw.info['bads']

class JuMEG_Base_StringHelper(object):
    """ Helper Class to work with strings """
    
    def __init__ (self):
        self.__version__  = 20160623
         
    def isString(self, s):
        """
         http://ideone.com/uB4Kdc
        """
        if (isinstance(s, str)):
           return True
        try:
           if (isinstance(s, basestring)):
              return True
        except NameError:
           return False
        
        return False    

    def str_range_to_list(self, seq_str):
        """
         inpiut: string   txt= '1,2,3-6,8,111'
         output: list     [1,2,3,4,5,6,8,111]
         copy from:
         http://stackoverflow.com/questions/6405208/how-to-convert-numeric-string-ranges-to-a-list-in-python
        """
        xranges = [(lambda l: xrange(l[0], l[-1]+1))(map(int, r.split('-'))) for r in seq_str.split(',')]
         # flatten list of xranges
        return [y for x in xranges for y in x]
      
    def str_range_to_numpy(self, seq_str,exclude_zero=False): 
        """
         converts array to numpy array 
         input : [1,2,3]
         out numpy array with unique numbers
        """
        import numpy as np

        if seq_str is None:
           return np.unique( np.asarray( [ ] ) )
        if self.isString(seq_str):
           anr = self.str_range_to_list( seq_str )
        else:
           anr = np.unique( np.asarray( [seq_str] ) )
        if exclude_zero:
           return anr[ np.where(anr) ] 
        return anr

class JuMEG_Base_FIF_IO(JuMEG_Base_Basic,JuMEG_Base_StringHelper):
    def __init__ (self):
        super(JuMEG_Base_FIF_IO, self).__init__()
        
    def set_raw_filename(self,raw,v):
        if raw.info.has_key('filename'):
            raw.info['filename'] = v
        else:
            raw._filenames = []
            raw._filenames.append(v)

    def get_raw_filename(self,raw):
        if raw:
           if raw.info.has_key('filename'):
              return raw.info['filename']
           else:
              return raw.filenames[0]
        return None    
      
    def __get_from_fifname(self,v=None,f=None):
        try:
           return os.path.basename(f).split('_')[v]
        except:
           return os.path.basename(f)

    def get_id(self,v=0,f=None):
        return self.__get_from_fifname(v=v,f=f)

    def get_scan(self,v=1,f=None):
        return self.__get_from_fifname(v=v,f=f)

    def get_session(self,v=2,f=None):
        return self.__get_from_fifname(v=v,f=f)

    def get_run(self,v=3,f=None):
        return self.__get_from_fifname(v=v,f=f)

    def get_postfix(self,f=None):
        return os.path.basename(self.raw.info['filename']).split('_')[-1].split('.')[0]

    def get_extention(self,f=None):
        if f:
            fname = f
        else:
            fname = self.raw.info['filename']
        return os.path.basename(fname).split('_')[-1].split('.')[-1]

    def get_postfix_extention(self,f=None):
        if f:
            fname = f
        else:
            fname = self.raw.info['filename']
        return os.path.basename(fname).split('_')[-1]

   

class JuMEG_Base_IO(JuMEG_Base_FIF_IO):
    def __init__ (self):
        super(JuMEG_Base_IO, self).__init__()
        
        self.picks = JuMEG_Base_PickChannels()

        self.__version__  = 20160623
        self.verbose      = False
      #--- ToDo --- start implementig BV support may new CLS
        self.brainvision_response_shift = 1000
        self.brainvision_extention      = '.vhdr'
        
    def get_fif_name(self, fname=None, raw=None, postfix=None, extention="-raw.fif", update_raw_fname=False):
        """
        Returns fif filename
        based on input file name and applied operation

        Parameters
        ----------
        fname      : base file name
        raw              = <raw obj>     : if defined get filename from raw obj
        update_raw_fname = <False/True>  : if true and raw is obj will update raw obj filename in place
        postfix          = <my postfix>  : string to add for applied operation [None]
        extention        = <my extention>: string to add as extention  [raw.fif]

        """
        if raw:
           fname = self.get_raw_filename(raw)
        try:
            p, pdf = os.path.split(fname)
            fname = p + "/" + pdf[:pdf.rfind('-')]
            if postfix:
               fname += "," + postfix
               fname = fname.replace(',-', '-')
            if extention:
               fname += extention

            if update_raw_fname:
               self.set_raw_filename(raw,fname)

        except:
            return False
        return fname    
        
    def update_bad_channels(self,fname,raw=None,bads=None,preload=True,append=False,save=False,interpolate=False):
        """

         :param fname:
         :param raw:
         :param bads:
         :param preload:
         :param append:
         :param save:
         :return: raw, bad channel list

        """
         #TODO: if  new bads ==  old bads in raw then  exit

        if save:
           preload = True
        raw,fname = self.get_raw_obj(fname,raw=raw,preload=preload)

        if not append:
           raw.info['bads']=[]

        if not isinstance(bads,list):
           bads = bads.split(',')

        if not bads:
           if not append:
              raw.info['bads']=[]
        else:
           for b in bads:
               bad_ch = None
               if (b in raw.ch_names):
                  bad_ch = b
               else:
                  if b.startswith('A'):
                     bad_ch = 'MEG '+ b.replace(" ","").strip('A').zfill(3)
                  elif b.startswith('MEG'):
                     bad_ch = 'MEG '+ b.replace(" ","").strip('MEG').zfill(3)
               if bad_ch:
                  if bad_ch not in raw.info['bads']:
                     raw.info['bads'].append(bad_ch)

        if self.verbose:
           print "\n --> Update bad-channels : " + self.get_raw_filename(raw)
           print raw.info['bads']
           print"\n"
        
        #--- save raw without interpolating 
        if save:
           raw.save( self.get_raw_filename(raw),overwrite=True)

        if ( interpolate and raw.info['bads'] ) :
           print " --> Update BAD channels => interpolating: " + raw.info['filename']
           print " --> BADs : " 
           print raw.info['bads'] 
           print "\n\n"
           raw.interpolate_bads()
             
        return raw,raw.info['bads']

#--- helper function
    def get_ica_raw_obj(self,fname_ica,ica_raw=None):
        ''' 
           check for <ica filename> or <ica raw obj>
           if filename -> load ica fif file
           input : filename
                   raw_ica = icaraw obj 
           return: raw ica obj,filename from ica raw obj 
        '''
        if ica_raw is None:
           if fname_ica is None:
              assert "---> ERROR no file foumd!!\n\n"
              if self.verbose:
                 print "<<<< Reading ica raw data ..."
        
           ica_raw = mne.preprocessing.read_ica(fname_ica)
         
           if ica_raw is None:
              assert "---> ERROR in jumeg.jumeg_base.get_ica_raw_obj => could not get ica raw obj:\n ---> FIF name: " + fname_ica   
   
        return ica_raw,self.get_raw_filename(ica_raw)
            
    def get_raw_obj(self,fname_raw,raw=None,path=None,preload=True):
        ''' 
           check for filename or raw obj
           chek for meg or brainvision eeg data *.vhdr
           if filename -> load fif file
           input : filename
                   raw : raw obj 
           return: raw obj,fname from raw obj 
        '''

        fnout = None

        if raw is None:
           assert(fname_raw),"---> ERROR no file foumd!!\n"
           if self.verbose:
              print "<<<< Reading raw data ..."
           fn = fname_raw
           if path:
              fn = path+"/"+fname_raw
           if ( fn.endswith(self.brainvision_extention) ):
               raw = mne.io.read_raw_brainvision(fn,response_trig_shift=self.brainvision_response_shift,preload=preload)
               raw.info['bads'] = []
               #--- ToDo may decide for eeg-name .eeg or.vhdr
           else:
               raw = mne.io.Raw(fn,preload=preload)
               fnout=self.get_raw_filename(raw)

        assert(raw), "---> ERROR in jumeg.jumeg_base.get_raw_obj => could not get raw obj:\n ---> FIF name: " + fname_raw
   
        return raw,fnout


    def get_files_from_list(self, fin):
        ''' 
            input : filename or list of filenames
             return: files as iterables lists 
        '''
        if isinstance(fin, list):
           fout = fin
        else:
           if isinstance(fin, str):
              fout = list([ fin ]) 
           else:
              fout = list( fin )
        return fout

    def get_filename_list_from_file(self, fin, start_path = None):
        ''' 
            input : text file to open
                    start_path = <start_dir> [None]
                    txt file format e.g:
                    fif-file-name  --bads=MEG1,MEG123

                    0815/M100/121130_1306/1/0815_M100_121130_1306_1_c,rfDC-raw.fif --bads=A248
                    0815/M100/120920_1253/1/0815_M100_120920_1253_1_c,rfDC-raw.fif
                    0815/M100/130618_1347/1/0815_M100_130618_1347_1_c,rfDC-raw.fif --bads=A132,MEG199


            return: list of existing files with full path and dict with bad-channels (as string e.g. A132,MEG199,MEG246)
        '''
        found_list = []
        # bads_dict  = dict()
        opt_dict = dict()

        try:
            fh = open( fin )
        except:
            assert "---> ERROR no such file list: " + fin
             #return found_list
         
        try:
            for line in fh :
                line  = line.rstrip()
                fname = None
                if line :
                   if ( line[0] == '#') : continue
                   opt = line.split()
                   print opt
                   if start_path :
                      if os.path.isfile( start_path + "/" + opt[0] ):
                         fname = start_path + "/" + opt[0]
                      else:
                         fname = opt[0]
                   if os.path.isfile( fname ):
                      found_list.append(fname)
                      opt_dict[fname]= {'bads': None, 'feeg': None}
                      for opi in opt[1:]:
                          if ('--bads' in opi):
                              _,opt_dict[fname]['bads'] = opi.split('--bads=')
                          if ('--feeg' in opi):
                              _,opt_dict[fname]['feeg'] = opi.split('--feeg=')

        finally:           
           fh.close()
       
        if self.verbose :
           print " --> INFO << get_filename_list_from_file >> Files found: %d" % ( len(found_list) )
           print found_list
           print "\n --> BADs: "
           print opt_dict
           print"\n"

        return found_list,opt_dict

    def apply_save_mne_data(self,raw,fname="test.fif",overwrite=True):
        '''
            Apply saving mne raw obj as fif 
            input : raw=raw obj, fname=file name, overwrite=True
            return: filename
        '''
        from distutils.dir_util import mkpath
         
        if ( os.path.isfile( fname) and (overwrite == False) ) :
           print " --> File exist => skip saving data to : " + fname
        else:
          print ">>>> writing filtered data to disk..."
          print ' --> saving: '+ fname
          mkpath( os.path.dirname(fname) )
          raw.save(fname,overwrite=True)
          print ' --> Bads:' + str( raw.info['bads'] )
          print " --> Done writing data to disk..."
        
        return fname

    def get_empty_room_fif(self,fname=None,raw=None, preload=True):
        '''
            find empty room file for input file name or RAWobj 
            assuming <empty room file> is the last recorded file for this id scan at this specific day
            e.g.: /data/mne/007/M100/131211_1300/1/1007_M100_131211_1300_1_c,rfDC-raw.fif
                  search for id:007 scan:M100 date:13:12:11 and extention: <empty.fif>
            input : raw=raw obj, fname=file name,
                    preload=True will load and return empty-room-raw obj instead of raw
            return: full empty room filename, empty-room-raw obj or raw depends on preload option
        '''
        import glob

        fname_empty_room = None
       
        if raw is not None:
           fname = self.get_raw_filename(raw)
         #--- first trivial check if raw obj is the empty room obj   
           if fname.endswith('epmpty.fif'):
              return(fname,raw)
               
         #--- ck if fname is the empty-room fie  
        if fname.endswith('epmpty.fif'):
           fname_empty_room = fname  
         #--- ok more difficult lets start searching ..
        else : 
            # get path and pdf (in memory of 4D filenames) from filename
           p,pdf = os.path.split(fname)
            # get session dat from file
           session_date = pdf.split('_')[2]

            # get path to scan from p and pdf
           path_scan    = p.split( session_date )[0]
            #--- TODO: may check for the latest or earliest empty-room file
           try:
               fname_empty_room = glob.glob( path_scan + session_date +'*/*/*-empty.fif' )[0]
           except:
               print"---> ERROR can not find empty room file: " + path_scan + session_date
               return

        if fname_empty_room and preload:
           if self.verbose:
              print "\n --> Empty Room FIF file found: %s \n"  % (fname_empty_room)

           return( fname_empty_room, mne.io.Raw(fname_empty_room, preload=True) )
        

#---
jumeg_base       = JuMEG_Base_IO()
jumeg_base_basic = JuMEG_Base_Basic()


"""


class JuMEG_Base_FIFIO(JuMEG_Base_IO):
   '''
       support for good old 4D file structure on harddisk
       => id scan session run postfix extention
        0007_ODDBall_190101_1200_1_c,rfDC-raw.fif

        set parameter:
        start_path = '.'
        experiment = 'nn'
        type       = 'mne' ['mne' or 'eeg'] folder

        id        = '007'
        scan      = 'TEST'
        session   = '130314_1131'
        run       = '1'
        postfix   = 'c,rfDC-raw'
        extention = '.fif'

    '''
    def __init__ (self):
        super(JuMEG_Base_FIFIO, self).__init__()
        self.__version__  = 20160623
        self.verbose      = False

        self.start_path = '.'
        self.experiment = 'nn'
        self.type       = 'mne'

        self.id        = None
        self.scan      = None
        self.session   = None
        self.run       = '1'
        self.postfix   = 'c,rfDC-raw'
        self.extention = '.fif'

    def __ck_pdfs(self):
        l = []
        if self.id:
           l.append(self.id)
        if self.scan:
           l.append(self.scan)
        if self.session:
           l.append(self.session)
        if self.run:
           l.append(self.run)
        return l

    def __get_mne_path(self):
        return self.start_path + os.sep + self.experiment + os.sep + self.type + os.sep
    type_path = property(__get_mne_path)

    def __get_pfif(self):
        return os.sep.join( self.__ck_pdfs() )
        # return self.id + os.sep + self.scan + os.sep + self.session + os.sep + self.run + os.sep
    pfif = property(__get_pfif)

    def __get_full_path(self):
        return self.type_path + os.sep + self.pfif
    path = property(__get_full_path)

    def __get_name(self):
        l = self.__ck_pdfs()
        if self.postfix:
           l.append(self.postfix)
        if self.extention:
           return '_'.join(l) + self.extention
        return '_'.join(l)
        # return self.id + '_' + self.scan +'_'+ self.session +'_'+ self.run +'_'+ self.postfix + self.extention
    name = property(__get_name)

    def __get_full_name(self):
        return self.type_path + os.sep + self.pfif  +os.sep + self.name
    full_name = property(__get_full_name)


class JuMEG_Base_BrainVisionIO(JuMEG_Base_FIFIO):
    '''

    '''

    def __init__(self):
        super(JuMEG_Base_BrainVisionIO, self).__init__()
        self.__version__ = 20160623
        self.verbose = False

        self.start_path = '.'
        self.experiment = 'nn'
        self.type = 'eeg'

        self.id = None
        self.scan = None
        self.session = None
        self.run = '1'
        self.postfix = None
        self.extention = '.vhdr'

    def __get_mne_path(self):
        return self.start_path + os.sep + self.experiment + os.sep + self.type + os.sep

    type_path = property(__get_mne_path)

    def __get_full_path(self):
        return self.type_path + os.sep + self.scan + os.sep

    path = property(__get_full_path)

    def __get_full_name(self):
        return self.path + self.name

    full_name = property(__get_full_name)


"""