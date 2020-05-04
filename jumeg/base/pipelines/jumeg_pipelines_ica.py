#!/usr/bin/env python3
# -+-coding: utf-8 -+-

#--------------------------------------------
# Authors:
# Frank Boers      <f.boers@fz-juelich.de>
# Christian Kiefer <c.kiefer@fz-juelich.de>
#--------------------------------------------
# Date: 12.112.19
#--------------------------------------------
# License: BSD (3-clause)
#--------------------------------------------
# Updates
#--------------------------------------------

import os,os.path as op
import numpy as np
import time,datetime

from distutils.dir_util import mkpath

import mne

from jumeg.decompose.ica_replace_mean_std import ICA,apply_ica_replace_mean_std
from jumeg.jumeg_preprocessing            import get_ics_cardiac, get_ics_ocular
#---
from jumeg.base                           import jumeg_logger
from jumeg.base.jumeg_base                import jumeg_base as jb
from jumeg.base.jumeg_base_config         import JuMEG_CONFIG as jCFG
#---
from jumeg.base.pipelines.jumeg_pipelines_ica_perfromance  import JuMEG_ICA_PERFORMANCE
from jumeg.base.pipelines.jumeg_pipelines_ica_svm          import JuMEG_ICA_SVM
from jumeg.base.pipelines.jumeg_pipelines_chopper          import JuMEG_PIPELINES_CHOPPER

#---
from jumeg.filter.jumeg_mne_filter import JuMEG_MNE_FILTER

logger = jumeg_logger.get_logger()

__version__= "2020.04.23.001"


def fit_ica(raw, picks, reject, ecg_ch, eog_hor, eog_ver,
            flow_ecg, fhigh_ecg, flow_eog, fhigh_eog, ecg_thresh,
            eog_thresh, use_jumeg=True, random_state=42):
    """
    author: C.Kiefer; c.kiefer@fz-juelich.de
    
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
    
    logger.info('Start ICA FIT: init ICA object')
    ica = ICA(method='fastica', n_components=40, random_state=random_state,
              max_pca_components=None, max_iter=5000, verbose=False)
  
    logger.debug('ICA FIT: apply ICA.fit\n reject: {} \n picks: {}'.format(reject,picks))
    ica.fit(raw, picks=picks, decim=None, reject=reject, verbose=True)
    logger.info('Done ICA FIT')
    #######################################################################
    # identify bad components
    #######################################################################

    if use_jumeg:
        logger.info("JuMEG Computing scores and identifying components ...")
       #--- get ECG related components using JuMEG
        ic_ecg,sc_ecg = get_ics_cardiac(raw, ica, flow=flow_ecg, fhigh=fhigh_ecg,
                                        thresh=ecg_thresh, tmin=-0.5, tmax=0.5, name_ecg=ecg_ch,
                                        use_CTPS=True) #[0]
        ic_ecg = list(set(ic_ecg))
        ic_ecg.sort()
      
       #--- get EOG related components using JuMEG
        ic_eog = get_ics_ocular(raw, ica, flow=flow_eog, fhigh=fhigh_eog,
                                       thresh=eog_thresh, name_eog_hor=eog_hor, name_eog_ver=eog_ver,
                                       score_func='pearsonr')
        ic_eog = list(set(ic_eog))
        ic_eog.sort()

       #--- if necessary include components identified by correlation as well
        bads_list = []
        bads_list.extend( ic_ecg )
        bads_list.extend( ic_eog )
        bads_list.sort()
        ica.exclude = bads_list
        msg = ["JuMEG identified ICA components",
               "  -> ECG components: {}".format(ic_ecg),
               "  ->         scores: {}".format(sc_ecg[ic_ecg]),
               "  -> EOG components: {}".format(ic_eog)
              ]
        logger.debug("\n".join(msg))
    else:
        logger.info("MNE Computing scores and identifying components ...")
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

        # if necessary include components identified by correlation as well
        ica.exclude = highly_corr
        msg = ["MNE Highly correlated artifact components",
               "  -> ECG  : {} ".format(highly_corr_ecg),
               "  -> EOG 1: {} ".format(highly_corr_eog1),
               "  -> EOG 2: {} ".format(highly_corr_eog2)
               ]
               
        logger.debug("\n".join(msg))
   
    logger.info("done ICA FIT\n  -> excluded ICs: {}\n".format(ica.exclude))
    return ica

class JuMEG_PIPELINES_ICA(object):
    def __init__(self,**kwargs):
        super().__init__()
        
        self.PreFilter      = JuMEG_MNE_FILTER()
        self.Chopper        = JuMEG_PIPELINES_CHOPPER()
        self.ICAPerformance = JuMEG_ICA_PERFORMANCE()
        self.SVM            = JuMEG_ICA_SVM()
        
        self.useSVM         = False
        self.verbose        = False
        self.debug          = False
        self.report_key     = "ica"
        
        self._CFG           = jCFG(**kwargs)
        self._plot_dir      = None
        self._ics_found_svm = None
        
        self._clear()
        
    @property
    def ICs(self): return self._ica_obj.exclude
  
    @property
    def stage(self): return self._stage
    @stage.setter
    def stage(self,v):
        self._stage=v
             
    @property
    def path(self): return self._raw_path
    @path.setter
    def path(self,v):
        if not v: return
        if jb.isPath(v):
           self._raw_path = v
        else:
            logger.exception("!!! No such path: {}".format(v))
            
    @property
    def path_ica(self): return os.path.join(self.path,"ica")
    @property
    def path_ica_chops(self): return os.path.join(self.path_ica,"chops")

    @property
    def plot_dir(self): return os.path.join(self.path,self.cfg.plot_dir)
    
    @property
    def raw(self): return self._raw

    @property
    def raw_fname(self): return self._raw_fname

    @raw_fname.setter
    def raw_fname(self,v):
        self._raw_fname = jb.isFile(v,path=self.path)

    @property
    def picks(self): return self._picks
    
    @property
    def CFG(self): return self._CFG
    @property
    def cfg(self): return self._CFG._data


    def clear(self,objects=None):
    
        if isinstance(objects,(list)):
            while objects:
                try:
                    o = objects.pop()
                    o.close()
                    del o
                except:
                    pass
    
        self.PreFilter.clear()
        self.Chopper.clear()
        self.ICAPerformance.clear()
        self._clear()
    
    def _clear(self):
        self._start_time = time.time()
        
        self._stage     = None
        self._path      = None
        self._path_ica  = None

        self._raw       = None
        self._raw_path  = None
        self._raw_fname = None
        self._raw_isfiltered = False
        
        self._ica_obj   = None
        self._picks     = None
        
        self._filter_prefix = ""
        self._filter_fname  = ""
        
    def _update_from_kwargs(self,**kwargs):
        self._raw      = kwargs.get("raw",self._raw)
        self.path      = kwargs.get("path",self._path)
        self._stage    = kwargs.get("stage",self.stage)
        self.raw_fname = kwargs.get("raw_fname",self._raw_fname)
   
    def _set_ecg_eog_annotations(self):
        """
        finding ECG, EOG events in raw, setting events as anotations
        """
       #--- find ECG in raw
        self.ICAPerformance.ECG.find_events(raw=self.raw,**self.CFG.GetDataDict("ecg"))
       #--- find EOG in raw
        annotations = self.ICAPerformance.EOG.find_events(raw=self.raw,**self.CFG.GetDataDict("eog"))
        self.raw.set_annotations(annotations)
  
    def trunc_nd(self,n,d):
        """
        https://stackoverflow.com/questions/8595973/truncate-to-three-decimals-in-python/8595991
        """
        n = str(n)
        return (n if not n.find('.') + 1 else n[:n.find('.') + d + 1])
   
    def _initRawObj(self):
        """
        load or get RAW obj
        init & mkdir path tree  <stage>/../ica/chops
        init picks from RAW
        
        init report HDF file name
 
        """
        self._raw,self._raw_fname = jb.get_raw_obj(self.raw_fname,raw=self.raw)
    
        self._raw_path = os.path.dirname(self._raw_fname)
        if self.stage:
            self._raw_path = os.join(self.stage,self._raw_path)
        #---
        mkpath(self.path_ica_chops,mode=0o770)
    
        #--- get picks from raw
        self._picks = jb.picks.meg_nobads(self._raw)
        
    def _get_chop_name(self,raw,chop=None,extention="-ica.fif",postfix=None,fullpath=False):
        """
        raw
        chop     = None
        extention= "-ica.fif" [-raw.fif]
        postfix  = None      [ar]
        fullpath = True
                   if True: includes path in filename
        Return:
        -------
        fname chop,fname orig
        """
        fname = jb.get_raw_filename(raw)
        fname,fextention = op.basename(fname).rsplit('-',1)
        if fullpath:
           if fname.startswith(self.path_ica_chops):
              fchop = fname
           else:
              fchop = op.join(self.path_ica_chops,fname)
        else:
           fchop = os.path.basename(fname)
           
        if postfix:
           fchop +=","+postfix
        try:
           if len(chop):
              if np.isnan(chop[1]):
                 fchop += ',{:04d}-{:04d}'.format(int(chop[0]),int(self.raw.times[-1]))
              else:
                 fchop += ',{:04d}-{:04d}'.format(int(chop[0]),int(chop[1]))
        except:
            pass
        if extention:
           fchop+=extention
    
        return fchop,fname
   
    def _apply_fit(self,raw_chop=None,chop=None,idx=None):
        """
        call to jumeg fit_ica
        raw_chop = None
        chop     = None
        
        ToDo
        if not overwrite
          if ICA file exist: load ICA
          else calc ICA
        
        :return:
        ICA obj, ica-filename
        """
        self._ica_obj       = None
        self._ics_found_svm = None

        fname_ica,fname = self._get_chop_name(raw_chop,chop=None)
      
        msg=["start ICA FIT chop: {} / {}".format(idx + 1,self.Chopper.n_chops),
             " --> chop id      : {}".format(chop),
             "  -> ica fname    : {}".format(fname_ica),
             "  -> ica chop path: {}".format(self.path_ica_chops),
             "  -> raw filename : {}".format(fname)
             ]
        logger.info("\n".join(msg))
        
       #--- ck for ovewrite & ICA exist
        load_from_disk = False
        if not self.cfg.fit.overwrite:
           load_from_disk = jb.isFile(fname_ica,path=self.path_ica_chops)
       
        if load_from_disk:
           self._ica_obj,fname_ica = jb.get_raw_obj(fname_ica,path=self.path_ica_chops)
           logger.info("DONE LOADING ICA chop form disk: {}\n  -> ica filename: {}".
                       format(chop,fname_ica))
        else:
            if self.useArtifactRejection:
               with jumeg_logger.StreamLoggerSTD(label="ica fit"):
                    self._ica_obj = fit_ica(raw=raw_chop,picks=self.picks,reject=self.CFG.GetDataDict(key="reject"),
                                   ecg_ch=self.cfg.ecg.ch_name,ecg_thresh=self.cfg.ecg.thresh,
                                   flow_ecg=self.cfg.ecg.flow,fhigh_ecg=self.cfg.ecg.fhigh,
                                  #---
                                   eog_hor = self.cfg.eog.hor_ch,
                                   eog_ver = self.cfg.eog.ver_ch,
                                   flow_eog=self.cfg.eog.flow,fhigh_eog=self.cfg.eog.fhigh,
                                   eog_thresh=self.cfg.eog.thresh,
                                  #---
                                   use_jumeg=self.cfg.ecg.use_jumeg,
                                   random_state=self.cfg.random_state)
           
               self._ica_obj.exclude = list( set( self._ica_obj.exclude ) )
               
            if self.useSVM:
               if not self._ica_obj:
                  logger.info('SVM start ICA FIT: init ICA object')
                 #--- !!! ToDo put parameter in CFG file
                  self._ica_obj = ICA(method='fastica',n_components=40,random_state=42,max_pca_components=None,max_iter=5000,verbose=False)
                  self._ica_obj.fit(raw_chop,picks=self.picks,decim=None,reject=self.CFG.GetDataDict(key="reject"),
                                verbose=True)
               else:
                 logger.info('SVM ICA Obj start')
                #--- !!! do_copy = True => resample
                 self._ica_obj,_ = self.SVM.run(raw=self.raw,ICA=self._ica_obj,picks=self.picks,do_crop=False,do_copy=True)
                 logger.info('DONE SVM ICA FIT: apply ICA.fit')

        #-- save ica object
        if self.cfg.fit.save and not load_from_disk:
           logger.info("saving ICA chop: {}\n".format(idx + 1,self._chop_times.shape[0]) +
                       "  -> ica filename   : {}".format(fname_ica))
           self._ica_obj.save(os.path.join(self.path_ica_chops,fname_ica))
              
        logger.info("done ICA FIT for chop: {}\n".format(chop)+
                    "  -> raw chop filename    : {}\n".format(fname_ica)+
                    "-"*30+"\n"+
                    "  -> ICs found JuMEG/MNE  : {}\n".format(self.SVM.ICsMNE)+
                    "  -> ICs found SVM        : {}\n".format(self.SVM.ICsSVM) +
                    "  -> ICs excluded         : {}\n".format(self.ICs)+
                    "-"*30+"\n"+
                    "  -> save ica fit         : {}".format(self.cfg.fit.save)
                   )
        return self._ica_obj,fname_ica

  
    def apply_ica_artefact_rejection(self,raw,ICA,fname_raw= None,fname_clean=None,replace_pre_whitener=True,copy_raw=True,
                                     reject=None):
        """
        Applies ICA to the raw object. (ica transform)

        Parameters
        ----------
            raw : mne.io.Raw()  (raw chop)
                  Raw object ICA is applied to
            ica : ICA object
                  ICA object being applied d to the raw object
            fname_raw : str | None
                  Path for saving the raw object
            fname_clean : str | None
                  Path for saving the ICA cleaned raw object
            reject: MNE reject dict
            replace_pre_whitener : bool
                  If True, pre_whitener is replaced when applying ICA to
                  unfiltered data otherwise the original pre_whitener is used.
            copy_raw: make a copy of raw
            
        Returns
        -------
            raw_clean : mne.io.Raw()
                       Raw object after ICA cleaning
        """
        logger.info("Start ICA Transform => call <apply_ica_replace_mean_std>")
        if copy_raw:
           _raw = raw.copy()
        else:
           _raw = raw
           
        raw_clean = None
        ica = ICA.copy() # ToDo exclude copy
        
        with jumeg_logger.StreamLoggerSTD(label="ica fit"):
             raw_clean = apply_ica_replace_mean_std(_raw,ica,picks=self.picks,
                                                    reject=reject,exclude=ica.exclude,n_pca_components=None)
            
        return raw_clean


    def _update_report(self,data):
        """
        
        :param fimages:
        :return:
        """
      #--- update report config
        CFG = jCFG()
        report_config = os.path.join(self.plot_dir,os.path.basename(self.raw_fname).rsplit("_",1)[0] + "-report.yaml")
        d = None
        if not CFG.load_cfg(fname=report_config):
            d = { "ica":data }
        else:
            CFG.config["ica"] = data
        CFG.save_cfg(fname=report_config,data=d)


    def _apply(self,raw=None,ICAs=None,run_transform=False,save_ica=False,save_chops=False,save_chops_clean=False,save_clean=True):
        """
        
        :param raw             : raw filtered or unfilterd
        :param run_transform   : self.cfg.transform.run or self.cfg.transform.filtered.run
        :param ICAs            : list of ICA objs if None calc ICA fit
        :param save_ica        : save ICA obj
        :param save_chops      : self.cfg.transform.unfiltered.save_chop or self.cfg.transform.filtered.save_chop
        :param save_chops_clean: self.cfg.transform.unfiltered.save_chop_clean or self.cfg.transform.filtered.save_chop_clean
        :param save_clean      : self.cfg.transform.filtered.save or self.cfg.transform.unfiltered.save
        :return:
           raw_clean,ICA_objs
           ICAs obj list to transform with unfilterd data if self.PreFilter.isFiltered
           titles
           images as np.arry
           
        """
        raw_clean     = None
        ICA_objs      = []
        raw_chops_clean_list = []
        fimages = []
        
        for idx in range(self.Chopper.n_chops):
            chop = self.Chopper.chops[idx]
            logger.info("Start ICA FIT & Transform chop: {} / {}\n".format(idx + 1,self.Chopper.n_chops))
        
           #--- chop raw
            raw_chop = self.Chopper.copy_crop_and_chop(raw,chop)
            fname_chop,fname_raw = self._get_chop_name(raw_chop,chop=chop,extention="-raw.fif")
            jb.set_raw_filename(raw_chop,fname_chop)
    
           #--- ICA fit chop
            if ICAs:
               ICA = ICAs[idx]
            else:
               ICA,fname_ica = self._apply_fit(raw_chop=raw_chop,chop=chop,idx=idx)
               ICA_objs.append(ICA)
            
            fname_chop,_ = self._get_chop_name(raw_chop,extention="-raw.fif")
            fname_chop = os.path.join(self.path_ica_chops,fname_chop)
            
            if save_chops:
                raw_chop.save(fname_chop,overwrite=True)

            #--- ICA Transform chop
            if run_transform:
               fout = jb.get_raw_filename(raw_chop)
               raw_chops_clean_list.append(self.apply_ica_artefact_rejection(raw_chop,ICA,
                                                                             reject=self.CFG.GetDataDict(key="reject")))
             
              #--- plot performance
               txt = "ICs JuMEG/MNE: "
               if self.useSVM:
                  if self.SVM.ICsMNE:
                     txt+= ",".join( [str(i) for i in self.SVM.ICsMNE ] )
                  txt+= " SVM: {}".format(self.SVM.ICsSVM)
               else:
                  txt+= ",".join( [str(i) for i in self._ica_obj.exclude ] )
            
               # logger.info("raw chop:\n {}".format(raw_chop.annotations))
               self.ICAPerformance.plot(raw=raw_chop,raw_clean=raw_chops_clean_list[-1],verbose=True,text=txt,
                                        plot_path=self.plot_dir,
                                        fout=fout.rsplit("-",1)[0] + "-ar")
               fimages.append( self.ICAPerformance.Plot.fout )
               
               if save_chops_clean:
                  fname_clean,_ = self._get_chop_name(raw_chop,extention="-raw.fif",postfix="ar")
                  fname_clean = os.path.join(self.path_ica_chops,fname_clean)
                  raw_chops_clean_list[-1].save(fname_clean,overwrite=True)
               
            logger.info("done ICA FIT & transform chop: {} / {}\n".format(idx + 1,self.Chopper.n_chops))
        
      #--- concat & save raw chops to raw_clean
        if raw_chops_clean_list:
          
           fname_clean = fname_raw.replace("-raw.fif",",ar-raw.fif")
           if not fname_clean.endswith(",ar-raw.fif"):
              fname_clean += ",ar-raw.fif"
           
           raw_clean = self.Chopper.concat_and_save(raw_chops_clean_list,
                                                    fname       = fname_clean,
                                                    annotations = raw.annotations,
                                                    save        = save_clean)
            
           del( raw_chops_clean_list )
           
        return raw_clean,ICA_objs,fimages
        
   #==== MAIN function
    def run(self,**kwargs):
        """
        
        :param kwargs:
        :return:
        raw_unfiltered_clean,raw_filtered_clean
        
        """
        self._clear()
        self._update_from_kwargs(**kwargs)
        
       #--- load config
        kwargs["useStruct"] = True
        self._CFG.update(**kwargs )
        self.useSVM               = self.cfg.fit.use_svm
        self.useArtifactRejection = self.cfg.fit.use_artifact_rejection
        
       #--- init or load raw
        self._initRawObj()
       #--- find & store ECG/EOG events in raw.annotations
        self._set_ecg_eog_annotations()
       #--- chop times
        self.Chopper.update(raw=self.raw,length=self.cfg.chops.length,and_mask=self.cfg.chops.and_mask,
                            exit_on_error=self.cfg.chops.exit_on_error,
                            description=self.cfg.chops.description,time_window_sec=self.cfg.chops.time_window,
                            show=self.cfg.chops.show,verbose=self.verbose,debug=self.debug)
            
        msg = [
            "Apply ICA => FIT & Transform",
            "  -> filename      : {}".format(self._raw_fname),
            "  -> ica chop path : {}".format(self.path_ica_chops),
            "-" * 40,
            "  -> chops [sec]    : {}".format(self.Chopper.chops_as_string ),
            "  -> chops [indices]: {}".format(self.Chopper.indices_as_string ),
            "-" * 40
            ]
        
       #--- apply pre-filter
        if self.cfg.pre_filter.run:
           self.PreFilter.apply(
                   flow      = self.cfg.pre_filter.flow,
                   fhigh     = self.cfg.pre_filter.fhigh,
                   save      = self.cfg.pre_filter.save,
                   overwrite = self.cfg.pre_filter.overwrite,
                   raw       = self.raw.copy(),
                   picks     = self.picks,
                   annotations = self.raw.annotations.copy()
                  )
           
           msg = self.PreFilter.GetInfo(msg=msg)
    
        else:
           self.PreFilter.clear()
           
        logger.info("\n".join(msg) )
        
        ICA_objs             = None
        raw_filtered_clean   = None
        raw_unfiltered_clean = None
        
        fimages_filtered     = []
        fimages_unfiltered   = None
        
       #--- apply raw-filter ica-fit,transform, save
        if self.PreFilter.isFiltered:
           raw_filtered_clean,ICA_objs,fimages_filtered = self._apply(raw = self.PreFilter.raw,
                                                     run_transform    = self.cfg.transform.run and self.cfg.transform.filtered.run,
                                                     save_chops       = self.cfg.transform.filtered.save_chop,
                                                     save_chops_clean = self.cfg.transform.filtered.save_chop_clean,
                                                     save_clean       = self.cfg.transform.filtered.save)
           self.PreFilter.raw.close()

       #---apply transform for unfilterd data update data-mean
        raw_unfiltered_clean, _ ,fimages_unfiltered = self._apply(raw = self.raw,
                                                    ICAs = ICA_objs,
                                                    run_transform    = self.cfg.transform.run and self.cfg.transform.unfiltered.run,
                                                    save_chops       = self.cfg.transform.unfiltered.save_chop,
                                                    save_chops_clean = self.cfg.transform.unfiltered.save_chop_clean,
                                                    save_clean       = self.cfg.transform.unfiltered.save)
                           
         
        logger.info("DONE ICA FIT & Transpose\n"+
                    "  -> filename : {}\n".format( jb.get_raw_filename(raw_unfiltered_clean) )+
                    "  -> time to process :{}".format( datetime.timedelta(seconds= time.time() - self._start_time ) ))
        
            
       #--- plot
        data = { "ICA-FI-AR":None,"ICA-AR":None }
        if self.PreFilter.isFiltered:
           self.ICAPerformance.plot(raw=self.PreFilter.raw,raw_clean=raw_filtered_clean,plot_path = self.plot_dir,
                                    text=None,fout = self.PreFilter.fname.rsplit("-",1)[0] + "-ar")
           data["ICA-FI-AR"] = [self.ICAPerformance.Plot.fout,*fimages_filtered]
           
        if raw_unfiltered_clean:
           self.ICAPerformance.plot(raw=self.raw,raw_clean=raw_unfiltered_clean,verbose=True,text=None,
                                    plot_path=self.plot_dir,fout=self.raw_fname.rsplit("-",1)[0] + "-ar")
           data["ICA-AR"] = [self.ICAPerformance.Plot.fout,*fimages_unfiltered]
        
        
       #-- check data shapes orig and transformed
        shapes=[self._raw._data.shape]
        labels=["raw original"]
     
        if raw_unfiltered_clean:
           shapes.append(raw_unfiltered_clean._data.shape)
           labels.append("raw unfiltered_clean")
        if raw_filtered_clean:
           shapes.append(raw_filtered_clean._data.shape)
           labels.append("raw filtered_clean")
                      
        if not self.Chopper.compare_data_shapes(shapes,labels):
           raise ValueError(" ERROR in chop & crop data: shapes not equal\n")
           
        self._update_report(data)
        
        self.clear(objects=ICA_objs)
        
        return raw_unfiltered_clean,raw_filtered_clean

def test1():
   #--- init/update logger
    jumeg_logger.setup_script_logging(logger=logger)
    
    stage = "$JUMEG_PATH_LOCAL_DATA/exp/MEG94T/mne"
    fcfg  = os.path.join(stage,"meg94t_config01.yaml")
    fpath = "206720/MEG94T0T2/130820_1335/2/"
    
    path = os.path.join(stage,fpath)
    raw_fname = "206720_MEG94T0T2_130820_1335_2_c,rfDC,meeg,nr,bcc,int-raw.fif"


    stage = "$JUMEG_PATH_LOCAL_DATA/exp/QUATERS/mne"
    fcfg  = os.path.join(stage,"jumeg_config.yaml") #""quaters_config01.yaml")
    fpath = "210857/QUATERS01/191210_1325/1"
    path = os.path.join(stage,fpath)
    raw_fname = "210857_QUATERS01_191210_1325_1_c,rfDC,meeg,nr,bcc,int-raw.fif"

    #stage = "${JUMEG_TEST_DATA}/mne"
    #fcfg = "intext_config01.yaml"
    
    raw = None
    #fpath = "211855/INTEXT01/190329_1004/6"
    #path = os.path.join(stage,fpath)
    # raw_fname = "211855_INTEXT01_190329_1004_6_c,rfDC,meeg,nr,bcc-raw.fif"
    #raw_fname = "211855_INTEXT01_190329_1004_6_c,rfDC,meeg,nr,bcc,int-raw.fif"
    
    logger.info("JuMEG Apply ICA mne-version: {}".format(mne.__version__))
    #--
    jICA = JuMEG_PIPELINES_ICA()
    raw_unfiltered_clean,raw_filtered_clean = jICA.run(path=path,raw_fname=raw_fname,config=fcfg,key="ica")

    #raw_filtered_clean.plot(block=True)

if __name__ == "__main__":
  test1()
  
'''
    def _calc_chop_times(self):
        """
        calc chop times & indices

        Returns
        self._chop_times,self._chop_indices
        -------
        TYPE
            DESCRIPTION.

        """
        logger.debug("Start calc Chop Times: length: {} raw time: {}".format(self.cfg.chops.length,self.raw.times[-1]))
        
        self._chop_times   = None
        self._chop_indices = None
       
        #--- warn if times less than chop length
        if self.raw.times[-1] <= self.cfg.chops.length:
           logger.warning("<Raw Times> : {} smaler than <Chop Times> : {}\n\n".format(self.raw.times[-1],self._chop_times))
                       
        self._chop_times,self._chop_indices = get_chop_times_indices(self.raw.times,chop_length=self.cfg.chops.length) 
        
        if self.debug:
           logger.debug("Chop Times:\n  -> {}\n --> Indices:  -> {}".format(self._chop_times,self._chop_indices))
        
        return self._chop_times,self._chop_indices
        
       
    def _copy_crop_and_chop(self,raw,chop):
        """
        copy raw
        crop
        :param raw:
        :param chop:
        :return:
        """
        if self._chop_times.shape[0] > 1:
           raw_crop = raw.copy().crop(tmin=chop[0],tmax=chop[1])
           if self.debug:
              logger.debug("RAW Crop Annotation : {}\n  -> tmin: {} tmax: {}\n {}\n".format(jb.get_raw_filename(raw),chop[0],chop[1],raw_crop.annotations))
           return raw_crop
        return raw
'''
    