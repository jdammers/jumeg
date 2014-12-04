'''
JuMEG Base Class to provide wrapper & helper functions

Authors: 
         Prank Boers     <f.boers@fz-juelich.de>
         Praveen Sripad  <praveen.sripad@rwth-aachen.de>
License: BSD 3 clause

last update 22.10.2014 FB

'''


import os
# from distutils.dir_util import mkpath
import numpy as np
import matplotlib.pylab as pl
import mne

class JuMEG_Base(object):
     def __init__ (self):
        self._jumeg_base_version   = 0.0001
        self._verbose              = False
        #self._extention_empty_room ="empty.fif" 
        
#--- version
     def _get_version(self):  
         return self._jumeg_filter_base_version
       
     version = property(_get_version)

#--- verbose    
     def _set_verbose(self,value):
         self._verbose = value

     def _get_verbose(self):
         return self._verbose
       
     verbose = property(_get_verbose, _set_verbose)

#--- get_files_from_list
     def get_files_from_list(fin):
        ''' Return string of file or files as iterables lists '''
        if isinstance(fin, list):
            fout = fin
        else:
            if isinstance(fin, str):
                fout = list([fin])
            else:
                fout = list(fin)
        return fout

#--- MNE foool fct  -> picks preselected channel groups
#--- mne.pick_types(raw.info, **fiobj.pick_all) 
#    mne.pick_types(info, meg=True, eeg=False, stim=False, eog=False, ecg=False, emg=False, ref_meg='auto',
#                   misc=False, resp=False, chpi=False, exci=False, ias=False, syst=False, include=[], exclude='bads', selection=None)

     def pick_channels(self,raw):
         ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude=None'''
         return mne.pick_types(raw.info,meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude=None)
       
     def pick_channels_nobads(self, raw):
         ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude='bads' '''
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True,stim=False,resp=False,exclude='bads')
       
     def pick_all(self, raw):
         ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude=None '''
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude=None)
       
     def pick_all_nobads(self, raw):
         ''' call with meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude='bads' '''
         return mne.pick_types(raw.info, meg=True,ref_meg=True,eeg=True,ecg=True,eog=True,emg=True,misc=True, stim=True,resp=True,exclude='bads')
       
     def pick_meg(self,raw):
         ''' call with meg=True'''
         return mne.pick_types(raw.info,meg=True)
       
     def pick_meg_nobads(self,raw):
         ''' call with meg=True,exclude='bads' '''
         return mne.pick_types(raw.info, meg=True,exclude='bads')
        
     def pick_ecg_eog(self,raw):
         ''' meg=False,ref_meg=False,ecg=True,eog=True '''
         return mne.pick_types(raw.info,meg=False,ref_meg=False,ecg=True,eog=True)
        
     def pick_stim(self,raw):
         ''' call with meg=False,stim=True '''
         return mne.pick_types(raw.info,meg=False,stim=True)
        
     def pick_response(self,raw):
         ''' call with meg=False,resp=True'''
         return mne.pick_types(raw.info,meg=False,resp=True)
        
     def pick_stim_response(self,raw):
         ''' call with meg=False,stim=True,resp=True'''
         return mne.pick_types(raw.info, meg=False,stim=True,resp=True)

#--- helper function
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
             return: list of existing files with full path
         '''
         found_list = []    
         try:
             fh = open( fin )
         except:
             return found_list 
         
         try:
             for f in fh :
                 f = f.rstrip()
                 if ( f[0] != '#') :
                      if start_path :
                         if os.path.isfile( start_path + "/" + f ):
                            found_list.append( start_path + "/" + f )
                      else :
                          if os.path.isfile( f ):
                             found_list.append(  f )
         finally:           
             fh.close()
       
         if self.verbose :
            print "--> INFO << get_filename_list_from_file >> Files found: %d" % ( len(found_list) )
            print found_list

         return found_list

     def get_trig_name(name_stim):
         ''' check stim_channel name and return trigger or response'''  
         if name_stim == 'STI 014':      # trigger
            return 'trigger'
         if name_stim == 'STI 013':   # response
            return 'response'
         
         return 'trigger'

     def apply_save_mne_data(self, raw,fname="test.fif",overwrite=True):
         '''
             Apply saving mne raw obj as fif 
             input : raw=raw obj, fname=file name, overwrite=True
             return: filename
         '''
         from distutils.dir_util import mkpath
         
         if ( os.path.isfile( fname) and (overwrite == False) ) :
            print "File exist => skip saving data to : " + fname
         else:
           print ">>>> writing filtered data to disk..."
           print 'saving: '+ fname
           mkpath( os.path.dirname(fname) )
           raw.save(fname,overwrite=True)    
           print ">>>> Done writing filtered data to disk..."
        
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
         
         fname_empty_room = None
         
         if raw is not None:
            fname = raw.info.get('filename')
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
            # get path to scan from p
            path_scan    = p.split( pdf[2] )[0] 
            # get session dat from file 
            session_date = fn.split('_')[2]
                          
           #--- TODO: may check for the latest or earliest empty-room file
            #session_time = fn.split('_')[3]
               
            fname_empty_room = glob.glob( path_scan + session_date +'*/*/*.empty.fif' )[0]
               
         if preload:
            return( fname_empty_room, mne.io.Raw(name_empty_room, preload=True) )   
        
         return fname_empty_room
    
     def get_fif_name(self,fname,postfix=None,extention="-raw.fif"):
        """ 
        Returns fif filename
        based on input file name and applied operation

        Parameters
        ----------
        fname      : base file name
        postfix  = <my postfix>  : string to add for applied operation [None]
        extention= <my extention>: string to add as extention  [raw.fif]
       
        """
        # fname = ( fname.split('-')[0] ).strip('.fif')
        p,pdf = os.path.split(fname) 
        fname = p +"/" + pdf[:pdf.rfind('-')]
        if postfix:
           fname += "," + postfix

        return fname + extention 
    
     def isString(self, s):
         """
         http://ideone.com/uB4Kdc
         """
         isString = False;
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
      
     def str_range_to_numpy(self, seq_str): 
         """
         converts array to numpy array 
         input : [1,2,3]
         out numpy array with unique numbers
         """
         if seq_str is None:
            return np.unique( np.asarray( [ ] ) )
         if self.isString(seq_str):
            return self.str_range_to_list( seq_str )
         else:
            return np.unique( np.asarray( [seq_str] ) )
         
#--- 
jumeg_base = JuMEG_Base()
