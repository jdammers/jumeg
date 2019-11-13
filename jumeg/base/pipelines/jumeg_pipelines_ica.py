#!/usr/bin/env python3
# -+-coding: utf-8 -+-

import contextlib,os,os.path as op
import logging,yaml
import mne
import numpy as np
from distutils.dir_util import mkpath


#from utils import set_directory

from jumeg.decompose.ica_replace_mean_std import ICA, read_ica, apply_ica_replace_mean_std
from jumeg.jumeg_preprocessing import get_ics_cardiac, get_ics_ocular
from jumeg.jumeg_plot import plot_performance_artifact_rejection  # , plot_artefact_overview

from jumeg.base.jumeg_base import jumeg_base as jb
from jumeg.base.jumeg_base_config import JuMEG_CONFIG_YAML_BASE

from jumeg.base            import jumeg_logger

logger = logging.getLogger("jumeg")

__version__= "2019.10.08.001"



def apply_ica_and_plot_performance(raw, ica, name_ecg, name_eog, raw_fname, clean_fname, picks=None,
                                   reject=None, replace_pre_whitener=True, save=False):
    """
    Applies ICA to the raw object and plots the performance of rejecting ECG and EOG artifacts.

    Parameters
    ----------
    raw : mne.io.Raw()
        Raw object ICA is applied to
    ica : ICA object
        ICA object being applied d to the raw object
    name_ecg : str
        Name of the ECG channel in the raw data
    name_eog : str
        Name of the (vertical) EOG channel in the raw data
    raw_fname : str | None
        Path for saving the raw object
    clean_fname : str | None
        Path for saving the ICA cleaned raw object
    picks : array-like of int | None
        Channels to be included for the calculation of pca_mean_ and _pre_whitener.
        This selection SHOULD BE THE SAME AS the one used in ica.fit().
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude. This parameter SHOULD BE
        THE SAME AS the one used in ica.fit().
        Valid keys are 'grad', 'mag', 'eeg', 'seeg', 'ecog', 'eog', 'ecg',
        'hbo', 'hbr'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
                          )

        It only applies if `inst` is of type Raw.
    replace_pre_whitener : bool
        If True, pre_whitener is replaced when applying ICA to
        unfiltered data otherwise the original pre_whitener is used.
    save : bool
        Save the raw object and cleaned raw object

    Returns
    -------
    raw_clean : mne.io.Raw()
        Raw object after ICA cleaning
    """

    # apply_ica_replace_mean_std processes in place -> need copy to plot performance
    raw_copy = raw.copy()
    ica = ica.copy()

    raw_clean = apply_ica_replace_mean_std(raw, ica, picks=picks, reject=reject,
                                           exclude=ica.exclude, n_pca_components=None,
                                           replace_pre_whitener=replace_pre_whitener)
    if save:
        if raw_fname is not None:
            raw_copy.save(raw_fname, overwrite=True)
        raw_clean.save(clean_fname, overwrite=True)

    overview_fname = clean_fname.rsplit('-raw.fif')[0] + ',overview-plot'
    plot_performance_artifact_rejection(raw_copy, ica, overview_fname,
                                        meg_clean=raw_clean,
                                        show=False, verbose=False,
                                        name_ecg=name_ecg,
                                        name_eog=name_eog)
    print('Saved ', overview_fname)

    raw_copy.close()

    return raw_clean


def fit_ica(raw, picks, reject, ecg_ch, eog_hor, eog_ver,
            flow_ecg, fhigh_ecg, flow_eog, fhigh_eog, ecg_thresh,
            eog_thresh, use_jumeg=True, random_state=42):
    """
    Fit an ICA object to the raw file. Identify cardiac and ocular components
    and mark them for removal.

    Parameters:
    -----------
    inst : instance of Raw, Epochs or Evoked
        Raw measurements to be decomposed.
    picks : array-like of int
        Channels to be included. This selection remains throughout the
        initialized ICA solution. If None only good data channels are used.
    reject : dict | None
        Rejection parameters based on peak-to-peak amplitude.
        Valid keys are 'grad', 'mag', 'eeg', 'seeg', 'ecog', 'eog', 'ecg',
        'hbo', 'hbr'.
        If reject is None then no rejection is done. Example::

            reject = dict(grad=4000e-13, # T / m (gradiometers)
                          mag=4e-12, # T (magnetometers)
                          eeg=40e-6, # V (EEG channels)
                          eog=250e-6 # V (EOG channels)
                          )

        It only applies if `inst` is of type Raw.
    ecg_ch : array-like | ch_name | None
        ECG channel to which the sources shall be compared. It has to be
        of the same shape as the sources. If some string is supplied, a
        routine will try to find a matching channel. If None, a score
        function expecting only one input-array argument must be used,
        for instance, scipy.stats.skew (default).
    eog_hor : array-like | ch_name | None
        Horizontal EOG channel to which the sources shall be compared.
        It has to be of the same shape as the sources. If some string
        is supplied, a routine will try to find a matching channel. If
        None, a score function expecting only one input-array argument
        must be used, for instance, scipy.stats.skew (default).
    eog_ver : array-like | ch_name | None
        Vertical EOG channel to which the sources shall be compared.
        It has to be of the same shape as the sources. If some string
        is supplied, a routine will try to find a matching channel. If
        None, a score function expecting only one input-array argument
        must be used, for instance, scipy.stats.skew (default).
    flow_ecg : float
        Low pass frequency for ECG component identification.
    fhigh_ecg : float
        High pass frequency for ECG component identification.
    flow_eog : float
        Low pass frequency for EOG component identification.
    fhigh_eog : float
        High pass frequency for EOG component identification.
    ecg_thresh : float
        Threshold for ECG component idenfication.
    eog_thresh : float
        Threshold for EOG component idenfication.
    use_jumeg : bool
        Use the JuMEG scoring method for the identification of
        artifact components.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the FastICA estimation.
        As the estimation is non-deterministic it can be useful to
        fix the seed to have reproducible results. Defaults to None.

    Returns:
    --------
    ica : mne.preprocessing.ICA
        ICA object for raw file with ECG and EOG components marked for removal.

    """
    # increased iteration to make it converge
    # fix the number of components to 40, depending on your application you
    # might want to raise the number
    # 'extended-infomax', 'fastica', 'picard'
    ica = ICA(method='fastica', n_components=40, random_state=random_state,
              max_pca_components=None, max_iter=5000, verbose=False)
    ica.fit(raw, picks=picks, decim=None, reject=reject, verbose=True)

    #######################################################################
    # identify bad components
    #######################################################################

    # get ECG and EOG related components using MNE
    print('Computing scores and identifying components..')

    if use_jumeg:

        # get ECG/EOG related components using JuMEG
        ic_ecg = get_ics_cardiac(raw, ica, flow=flow_ecg, fhigh=fhigh_ecg,
                                 thresh=ecg_thresh, tmin=-0.5, tmax=0.5, name_ecg=ecg_ch,
                                 use_CTPS=True)[0]
        ic_eog = get_ics_ocular(raw, ica, flow=flow_eog, fhigh=fhigh_eog,
                                thresh=eog_thresh, name_eog_hor=eog_hor, name_eog_ver=eog_ver,
                                score_func='pearsonr')
        ic_ecg = list(set(ic_ecg))
        ic_eog = list(set(ic_eog))
        ic_ecg.sort()
        ic_eog.sort()

        # if necessary include components identified by correlation as well
        bads_list = list(set(list(ic_ecg) + list(ic_eog)))
        bads_list.sort()
        ica.exclude = bads_list

        print('Identified ECG components are: ', ic_ecg)
        print('Identified EOG components are: ', ic_eog)

    else:

        ecg_scores = ica.score_sources(raw, target=ecg_ch, score_func='pearsonr',
                                       l_freq=flow_ecg, h_freq=fhigh_ecg, verbose=False)
        # horizontal channel
        eog1_scores = ica.score_sources(raw, target=eog_hor, score_func='pearsonr',
                                        l_freq=flow_eog, h_freq=fhigh_eog, verbose=False)
        # vertical channel
        eog2_scores = ica.score_sources(raw, target=eog_ver, score_func='pearsonr',
                                        l_freq=flow_eog, h_freq=fhigh_eog, verbose=False)

        # print the top ecg, eog correlation scores
        ecg_inds = np.where(np.abs(ecg_scores) > ecg_thresh)[0]
        eog1_inds = np.where(np.abs(eog1_scores) > eog_thresh)[0]
        eog2_inds = np.where(np.abs(eog2_scores) > eog_thresh)[0]

        highly_corr = list(set(np.concatenate((ecg_inds, eog1_inds, eog2_inds))))
        highly_corr.sort()

        highly_corr_ecg = list(set(ecg_inds))
        highly_corr_eog1 = list(set(eog1_inds))
        highly_corr_eog2 = list(set(eog2_inds))

        highly_corr_ecg.sort()
        highly_corr_eog1.sort()
        highly_corr_eog2.sort()

        print('Highly correlated artifact components are:')
        print('    ECG:  ', highly_corr_ecg)
        print('    EOG 1:', highly_corr_eog1)
        print('    EOG 2:', highly_corr_eog2)

        # if necessary include components identified by correlation as well
        ica.exclude = highly_corr

    print("Plot ica sources to remove jumpy component for channels 4, 6, 8, 22")

    return ica


class JuMEG_ICA_CONFIG(JuMEG_CONFIG_YAML_BASE):
    """
    CLS for ICA config file obj

    Example:
    --------
    self.CFG = JuMEG_ICA_CONFIG(**kwargs)
    self.CFG.update(**kwargs)
    """
    def __init__(self,**kwargs):
        super().__init__()

class JuMEG_ICA_FILTER(object):
    __slots__ = ["postfix","raw","flow","fhigh","picks","save"]
    def __init__(self,**kwargs):
        super().__init__()
        for k in self.__slots__:
            self.__setattr__(k,None)
        self._update_from_kwargs(**kwargs)

    @property
    def fname(self): return jb.get_raw_filename(self.raw,index=0)
    
    def _update_from_kwargs(self,**kwargs):
        for k in self.__slots__:
            self.__setattr__(k,kwargs.get(k,self.__getattribute__(k)))
        
    def _update_postfix(self,**kwargs):
        """return filter extention """
        self._update_from_kwargs(**kwargs)
        fi_fix = None
    
        if self.flow and self.fhigh:
            fi_fix = "fibp"
            fi_fix += "%0.2f-%0.1f" % (self.flow,self.fhigh)
        elif self.flow:
            fi_fix = "fihp"
            fi_fix += "%0.2f" % self.flow
        elif self.fhigh:
            fi_fix = "filp"
            fi_fix += "%0.2f" % (self.fhigh)
        
        self.postfix = fi_fix
        return fi_fix

    def apply(self,**kwargs):
        """
        :param kwargs:
         flow,fhigh,raw,picks
        :return:
         fname
        
        """
        self._update_from_kwargs(**kwargs)
        
        logger.info("---> Filter start: {}".format(self.fname))
        self.raw.filter(l_freq=self.flow,h_freq=self.fhigh,picks=self.picks)
        self._update_postfix()
        fname,ext = self.fname.rsplit('-',1) #raw.fif'
        fname += ","+ self.postfix +"-"+ ext
        if self.save:
           fname= jb.apply_save_mne_data(self.raw,fname=fname,overwrite=True)
           jb.set_raw_filename(self.raw,fname)
        logger.info("---> Filter done: {}".format(self.fname))
        return fname
    

class JuMEG_PIPELINES_ICA(object):
    def __init__(self,**kwargs):
        super().__init__()
        self._CFG  = JuMEG_ICA_CONFIG(**kwargs)
        self._FiPre= JuMEG_ICA_FILTER()
        self._clear()
        
    @property
    def path(self): return self._raw_path
    @path.setter
    def path(self,v):
        if v:
           self._raw_path = jb.isPath(v)

    @property
    def raw(self): return self._raw

    @property
    def raw_fname(self): return self._raw_fname

    @raw_fname.setter
    def raw_fname(self,v):
        self._raw_fname = jb.isFile(v,path=self.path)

    @property
    def CFG(self): return self._CFG
    @property
    def cfg(self): return self._CFG._data

    def _clear(self):
        self._path      = None
        self._raw       = None
        self._raw_path  = None
        self._raw_fname = None
        self._raw_isfiltered = False
        
        self._ica_obj    = None
        self._picks     = None
        self._chop_times= None
        self._filter_prefix = ""
        self._filter_fname  = ""
        
    def _update_from_kwargs(self,**kwargs):
        self._raw      = kwargs.get("raw",self._raw)
        self.path      = kwargs.get("path",self._path)
        self.raw_fname = kwargs.get("raw_fname",self._raw_fname)

    def trunc_nd(self,n,d):
        """
        https://stackoverflow.com/questions/8595973/truncate-to-three-decimals-in-python/8595991
        """
        n = str(n)
        return (n if not n.find('.') + 1 else n[:n.find('.') + d + 1])

    #--- calc chop times
    def _calc_chop_times(self):
        logger.debug("  -> Start calc Chop Times: length: {} raw time: {}".format(self.cfg.chops.length,self.raw.times[-1]))
        self._chop_times = None
        
        if self.raw.times[-1] <= self.cfg.chops.length:
           cps      = np.zeros([1,2],dtype=np.float32)
           cps[0,0] = 0.0
           logger.warning("---> <Raw Times> : {} smaler than <Chop Times> : {}\n\n".format(self.raw.times[-1],self._chop_times))
        else:
           n_chops,t_rest = np.divmod(self.raw.times[-1],self.cfg.chops.length)
           n_chops = int(n_chops)
           dtime   = self.cfg.chops.length + t_rest // n_chops # add rest to length
        
           cps          = np.zeros([n_chops,2],dtype=np.float32)
           cps[:,0]    += np.arange(n_chops) * dtime
           cps[0:-1,1] += cps[1:,0]
        #cps[-1,1]    = '%.3f'%( self.raw.times[-1] ) # ???? error in mne crop line 438
        # fb 01.11.2019
        
        cps[-1,1] = None #self.trunc_nd(self.raw.times[-1], 3)  # ???? error in mne crop line 438 mne error tend == or less tmax
        self._chop_times = cps
        
        logger.debug("  -> Chop Times:\n{}".format(self._chop_times))
        return self._chop_times

   #--- calc chop times from events
    def _calc_chop_times_from_events(self):
        
        logger.info("  -> Chop Times:\n{}".format(self._chop_times))
        return self._chop_times
 
    def apply_fit(self,**kwargs):
        self._clear()
        self._update_from_kwargs(**kwargs)
      #--- load config
        self._CFG.update(**kwargs)
 
      #--- init or load raw
        self._raw,self._raw_fname = jb.get_raw_obj(self.raw_fname,raw=self.raw)
      
      #--- get picks from raw
        self._picks = jb.picks.meg_nobads(self._raw)
      
      #--- chop times
        if self.cfg.chops.epocher.use:
            """ToDo use epocher information chop onset,offset"""
            pass
        else:
            self._calc_chop_times()
        
        if not isinstance(self._chop_times,(np.ndarray)):
           logger.error("---> No <chop times> defined for ICA\n" +
                        "  -> raw filename : {}\n".format(self._raw_fname))
           return None

        #--- ck for 1.filter => filter inplace:
        if self.cfg.pre_filter.run:
           filename = self._FiPre.apply(
                                 flow  = self.cfg.pre_filter.flow,
                                 fhigh = self.cfg.pre_filter.fhigh,
                                 save  = self.cfg.pre_filter.save,
                                 raw   = self.raw,
                                 picks = self._picks
                                )
           self._raw_isfiltered = True
           
        else:
           filename = self._raw_fname
        
           
        fname,fextention = op.basename(filename).rsplit('-',1) #raw.fif
        path_ica         = op.join( os.path.dirname(filename),"ica" )
        path_ica_chops   = op.join( path_ica,"chops" )
        
        mkpath(path_ica_chops,mode=0o770)
        
        msg=["---> Apply ICA => FIT ICA -> mkdirs\n  -> ica     : {}\n  -> chops   : {}\n  -> filename: {}\n  -> raw filename: {}".format(path_ica,path_ica_chops,fname,self._raw_fname),
             "  -> is filtered: {}".format(self._raw_isfiltered)]
        logger.info( "\n".join(msg))
        
        for idx in range( self._chop_times.shape[0] ) :
           
            chop = self._chop_times[idx]
            
            ica_fname = op.join(path_ica_chops,fname)
            
            if np.isnan(chop[1]):
               ica_fname = fname + ',{:06d}-{:06d}-ica.fif'.format(int(chop[0]),int(self.raw.times[-1]))
            else:
               ica_fname = fname + ',{:06d}-{:06d}-ica.fif'.format(int(chop[0]),int(chop[1]))
          
            logger.info("---> Start ICA chop: {} / {}\n".format(idx+1,self._chop_times.shape[0])+
                        " --> chop id      : {}\n".format(chop)+
                        "  -> ica fname    : {}\n".format(ica_fname)+
                        "  -> ica chop path: {}\n".format(path_ica_chops)+
                        "  -> raw filename : {}\n".format(self._raw_fname))
            
            if self._chop_times.shape[0] > 1:
               raw_chop = self._raw.copy().crop(tmin=chop[0],tmax=chop[1])
            else:
               raw_chop = self._raw
          #--- get dict from struct
            reject = self.CFG.GetDataDict(key="reject")
            
            self._ica_obj = fit_ica(raw=raw_chop,picks=self._picks,reject=reject,
                          ecg_ch=self.cfg.ecg.channel,eog_hor=self.cfg.eog.hor_ch,eog_ver=self.cfg.eog.ver_ch,
                          flow_ecg=self.cfg.ecg.flow,fhigh_ecg=self.cfg.ecg.fhigh,
                          flow_eog=self.cfg.eog.flow,fhigh_eog=self.cfg.eog.fhigh,
                          ecg_thresh=self.cfg.ecg.thresh,eog_thresh=self.cfg.eog.thresh,use_jumeg=self.cfg.ecg.use_jumeg,
                          random_state=self.cfg.random_state)
  
           #--- save ica object
            if self.cfg.save:
               self._ica_obj.save( os.path.join(path_ica_chops,ica_fname) )
            
            logger.info("---> DONE ICA chop: {} / {} -> ica filename: {}".format(idx+1,self._chop_times.shape[0],ica_fname))
            
        logger.info("---> DONE ICA FITs: {} -> raw filename: {}".format(self._chop_times.shape[0],self._raw_fname))
            #title = ('ICA decomposition using %s (took %.1fs)' % (method, fit_time))
            #self._ica_obj.plot_components(title=title)
            
            #return self._ica_obj

          # use TSV visual inspection deselect ICs
          
    def apply_transform(self,**kwargs):
        """
        ToDo
        jb.load ica from file or obj
        calc chop times
        check chops exists
        mk filename
        for chops:
            apply_ica_and_plot_performance
        
        append cleand chops to new raw
        save raw-ar
        
        :param kwargs:
        :return:
        """
        self._clear()
        self._update_from_kwargs(**kwargs)
        
        print('ICA components excluded: ',self._ica_obj.exclude)

        #clean_filt_fname = op.join(dirname,prefix_filt + ',{},ar,{}-{}-raw.fif'.format(info_filt,int(tmin),tmaxi))
        #raw_filt_chop_fname = op.join(dirname,prefix_filt + ',{},{}-{}-raw.fif'.format(info_filt,int(tmin),tmaxi))

        #######################################################################
        # apply the ICA to data and save the resulting files
        #######################################################################

            #print('Running cleaning on filtered data...')
            #clean_filt_chop = apply_ica_and_plot_performance(raw_filt_chop,ica,ecg_ch,eog_ver,
            #                                                 raw_filt_chop_fname,clean_fname=clean_filt_fname,
            #                                                 picks=picks,replace_pre_whitener=True,
            #                                                 reject=reject,save=save)

            #raw_chop_clean_filtered_list.append(clean_filt_chop)

        '''
        may try this ???
        
        from mne.decoding import UnsupervisedSpatialFilter

        from sklearn.decomposition import PCA,FastICA
        x=raw.get_data()
        ica= UnsupervisedSpatialFilter(FastICA(30), average=False)
        ica_data = ica.fit_transform(X)
        ev1 = mne.EvokedArray(np.mean(ica_data, axis=0),
                      mne.create_info(30, epochs.info['sfreq'],
                                      ch_types='eeg'), tmin=tmin)
        ev1.plot(show=False, window_title='ICA', time_unit='s')
        plt.show()
        '''
        
    def info(self):
        logger.info("Apply ICA\n"+
                    "  -> raw obj:   {}\n".format(self._raw)+
                    "  -> raw fname: {}\n".format(self._raw_fname)+
                    "  -> raw path:  {}\n".format(self._raw_path)+
                    "  -> config:    {}\n".format(self.CFG.filename))

if __name__ == "__main__":
  
  #--- init/update logger
   jumeg_logger.setup_script_logging(logger=logger)
 
   stage = "${JUMEG_TEST_DATA}/mne"
   fcfg  = "intext_config01.yaml"
   
   raw       = None
   fpath     = "211855/INTEXT01/190329_1004/6"
   path      = os.path.join(stage,fpath)
   # raw_fname = "211855_INTEXT01_190329_1004_6_c,rfDC,meeg,nr,bcc-raw.fif"
   raw_fname = "211855_INTEXT01_190329_1004_6_c,rfDC,meeg,nr,bcc,int-raw.fif"

   logger.info("JuMEG Apply ICA mne-version: {}".format(mne.__version__))
 #--
   jICA = JuMEG_PIPELINES_ICA()
   jICA.apply_fit( path=path,raw_fname=raw_fname,config=fcfg,key="ica")
