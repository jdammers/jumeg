"""

Authors:
 - Frank Boers: f.boers@fz-juelich.de
 - Jürgen Dammers: j.dammers@fz-juelich.de

"""

import os, os.path as op
import numpy as np

import mne
from mne.preprocessing import ICA

from dcnn_utils      import (logger,get_chop_times_indices, auto_label_cardiac,expandvars,
                             auto_label_ocular, get_unique_list, transform_mne_ica2data, transform_ica2data)
from dcnn_utils       import collect_source_info, add_aux_channels
from dcnn_base        import DCNN_PATH,DCNN_MEG, DCNN_ICA, DCNN_SOURCES
from dcnn_performance import PERFORMANCE_PLOT

__version__= "2020.08.11.001"

# -----------------------------------------------------
# compute downsampling frequency and final chop length
# -----------------------------------------------------
def calc_sfreq_ds(times_orig, n_samp_chop, chop_len_init=180.0):

    chop_len_init = np.float64(chop_len_init)

    # compute number of chops with length with
    t_range = times_orig.max() - times_orig.min()
    n_chops, t_rest = np.divmod(t_range, chop_len_init, dtype=np.float64)
    n_chops = int(n_chops)
    if n_chops < 2:
        chop_length = chop_len_init + t_rest
    else:
        # distribute remaining time to all chops
        chop_length = chop_len_init + t_rest // n_chops  # chop_len in s

    # compute downsampling frequency given the chop length and number of samples
    sfreq_ds = np.int64(n_samp_chop) / chop_length

    # test if number of samples per chop is as requested
    n_samp = np.round(chop_length * sfreq_ds).astype(int)
    if n_samp != n_samp_chop:
        raise ValueError("Downsampling frequency does not match the required number of samples per chop!")

    return sfreq_ds, chop_length

# -----------------------------------------------------
# filter and resampling raw data
# -----------------------------------------------------
def meg_filter_resample(raw=None, flow_ica=None, fhigh_ica=None, picks=None, n_jobs=1,
                        res_time=None, chop_len_init=180.0):
    '''
    works in place overwrite raw

    Parameters
    ----------
    raw : TYPE, optional
        DESCRIPTION. The default is None.
    flow_ica : TYPE, optional
        DESCRIPTION. The default is None.
    fhigh_ica : TYPE, optional
        DESCRIPTION. The default is None.
    picks : TYPE, optional
        DESCRIPTION. The default is None.
    n_jobs : TYPE, optional
        DESCRIPTION. The default is 1.
    res_time : TYPE, optional
        DESCRIPTION. The default is None.
    chop_len_init : TYPE, optional
        DESCRIPTION. The default is 60.0.

    Returns
    -------
    raw : TYPE
        DESCRIPTION.
    sfreq_ds : TYPE
        DESCRIPTION.
    chop_length : TYPE
        DESCRIPTION.

    '''

    # 1. apply filter prior to ICA processing
    raw.filter(flow_ica, fhigh_ica, picks=picks, n_jobs=n_jobs,
               filter_length='auto', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
               method='fir', phase='zero', fir_window='hamming')

    # 2. compute downsampling frequency given the requested number of samples per chop
    #    we will have a chop duration of about 60s
    sfreq_ds, chop_length = calc_sfreq_ds(raw.times, res_time, chop_len_init=chop_len_init)

    # 3. apply downsampling
    raw.resample(sfreq_ds, npad='auto', window='boxcar', stim_picks=None,
                 n_jobs=n_jobs, events=None)

    return raw, sfreq_ds, chop_length

# =====================================================
# gDCNN class
# =====================================================
class DCNN(object):
    """
    gDCNN class
    ----------------------------------------------
    Apply ICA and add data to training/test set
     => read MEG data and
       - optional apply noise compensation (CTF annd 4D)
       - optional apply filter and/or notch filter
     => apply ICA
        - filter MEG data
        - donwsampling (model expects a fixed number of  samples)
        - crops data into about 2-3 minutes chops
        - automatic ECG/EOG labeling (CTPS, correlation, SVM)

    """

    def __init__(self, version=None, n_jobs=1, path={}, meg={}, ica={}, sources={}, res={},
                 info=None, verbose=False):
        """
        init via config dict:
                dcnn = gDCNN(**config)

        Parameters
        ----------
        version : string, optional, default: None
        n_jobs  : int,    optional, default: 1.
        path    : dict,   optional, default: {}

        meg     : dict,   optional, default: {}
                  init MEG CLS
        ica     : dict,   optional, default: {}
                  init ICA CLS
        sources : dict,   optional, default: {}
                  init SOURCES CLS
        res     : dict,   optional, default: {}
                  keys: res_time,res_space

        Returns
        -------
        None

        """

        self.version = version
        #self.path    = path
        self.config_info = info

        if res:
            self.model = res
        else:
            self.model = {'res_time': None ,'res_space': None}

        # the following will later be saved in one file for each experiment
        self._PATH    = DCNN_PATH(**path)
        self._MEG     = DCNN_MEG(**meg)  # MEG system and data settings
        self._ICA     = DCNN_ICA(**ica)
        self._SOURCES = DCNN_SOURCES(**sources)
        self.n_jobs   = n_jobs # set n_jobs for MEG  not ICA
        self.verbose  = verbose
        self._PP      = PERFORMANCE_PLOT()

    @property
    def path(self): return self._PATH

    @property
    def meg(self): return self._MEG

    @property
    def ica(self): return self._ICA

    @property
    def sources(self): return self._SOURCES

    @property
    def n_jobs(self): return self._n_jobs

    @n_jobs.setter
    def n_jobs(self ,v):
        # set n_jobs for MEG,ICA too
        if v is None:
            v = 1
        self._n_jobs    = v
        self.meg.n_jobs = v

    @property
    def verbose(self): return self._verbose

    @verbose.setter
    def verbose(self ,v):
        self._verbose     = v
        self.path.verbose = v
        self.meg.verbose  = v
        self.ica.verbose  = v
        self.sources.verbose = v

    @property
    def PerformancePlot( self ): return self._PP

    def _meg_filter_resample(self,raw=None):
        '''
        wrapper for meg_filter_resample

        Returns
        -------
        None.

        '''

        self.ica.chop_n_times = self.model['res_time']
        if not raw:
            raw = self.meg.raw

        # apply filtering and downsampling
        raw ,sfreq_ds ,chop_length = meg_filter_resample(raw=raw ,n_jobs=self.n_jobs,
                                                       picks=self.meg.picks.chan,
                                                       res_time=self.model['res_time'],
                                                       flow_ica=self.ica.flow_ica,
                                                       fhigh_ica=self.ica.fhigh_ica,
                                                       chop_len_init=self.ica.chop_len_init
                                                       )
        self.meg.sfreq_ds     = sfreq_ds
        self.meg.chop_length  = chop_length
        self.ica.chop_length  = chop_length

        return raw

    # -----------------------------------------------------
    # - crop data into chops of 2-3 minute length to increase
    #   data size for training
    # - apply ICA on each chop
    # -----------------------------------------------------
    def _ica_apply_chops(self, raw=None):
        if not raw:
            raw = self.meg.raw

        # get chop times and indices using the fixed number of samples required per chop
        chop_times, chop_indices = get_chop_times_indices(raw.times, chop_nsamp=self.model['res_time'])
        n_chop  = len(chop_times)

        # check if we chop_n_times is larger than origial n_times
        if n_chop == 1:
            nt = chop_indices[0][1]  + 1    #
            if nt < self.ica.chop_n_times:
                self.ica.chop_n_times = nt

        # ---------------------------------------
        # crop data and apply ICA on each chop
        # ---------------------------------------
        n_samples_chop = self.ica.chop_n_times
        sfreq_ds       = self.meg.sfreq_ds
        raw_chop_list  = []
        ica_list       = []

        for ichop in range(n_chop):
            # crop data if n_chop > 1
            if n_chop > 1:
                tmin, tmax = chop_times[ichop]
                raw_chop = raw.copy().crop(tmin=tmin, tmax=tmax)

                # resample last chop to fit the number of samples required (last chop has more samples)
                if ichop + 1 == n_chop:
                    sfreq_last = (n_samples_chop - 1) / raw_chop.times[-1]
                    raw_chop.resample(sfreq_last, npad='auto', window='boxcar', stim_picks=None, events=None)
            else:
                raw_chop = raw.copy()  # here we have a single chop only
                sfreq_last = sfreq_ds

            # collect chops in list
            raw_chop_list.append(raw_chop)

            # apply ICA
            ica = ICA(method=self.ica.ica_method, fit_params=self.ica.fit_params, max_iter=400,
                      n_components=self.ica.n_components, random_state=self.ica.random_state, verbose=None)
            ica.fit(raw_chop, picks=self.meg.picks.meg, decim=None, verbose=True)
            ica_list.append(ica)

        # add chop details to ica dict - set in sub class ica.chop
        self.ica.n_chop          = n_chop
        self.ica.chop.n_samples  = n_samples_chop
        self.ica.chop.sfreq_ds   = sfreq_ds
        self.ica.chop.sfreq_last = sfreq_last   # (self.ica.chop_n_times -1) /raw_chop.times[-1]
        self.ica.chop.picks      = self.meg.picks.meg.copy()
        self.ica.chop.times      = chop_times
        self.ica.chop.indices    = chop_indices
        self.ica.chop.ica        = ica_list

        return raw_chop_list


    # -----------------------------------------------------
    #  label ICs to get cardiac and ocular ICs
    # -----------------------------------------------------
    def _ica_label_chops(self, raw_chop_list):

        n_chops      = self.ica.chop.n_chop
        self.ica.chop.exclude.clear()

        # loop across chops
        logger.info("ICA chops: {}\n list: {}".format(self.ica.chop.n_chop ,self.ica.chop.ica))

        events_ecg     = []
        events_eog_ver = []
        events_eog_hor = []
        for ichop in range(n_chops):
            raw_chop = raw_chop_list[ichop]
            msg = ["ica idx: {}".format(ichop),
                   "ica obj: {}".format(self.ica.chop.ica[ichop])]
            logger.info("\n".join(msg))

            # ica = self.ica['chop']['ica'][ichop]
            ica = self.ica.chop.ica[ichop]

            # get ICs with cardiac activity
            decg = auto_label_cardiac(raw_chop, ica, self.meg.ecg_ch, tmin=-0.4, tmax=0.4,
                                      flow=self.ica.flow_ecg, fhigh=self.ica.fhigh_ecg,
                                      thresh_ctps=self.ica.ecg_thresh_ctps, thresh_corr=self.ica.ecg_thresh_corr)


            # get ICs with ocular activity
            deog = auto_label_ocular(raw_chop, ica, self.meg.eog_ch1, name_eog_hor=self.meg.eog_ch2,
                                     flow=self.ica.flow_eog, fhigh=self.ica.fhigh_eog,
                                     thresh_corr_ver=self.ica.eog_thresh_ver, thresh_corr_hor=self.ica.eog_thresh_hor)

            # ICs to be excluded
            ic_ecg = get_unique_list(decg['ic_ctps'], decg['ic_corr'])
            ic_eog = get_unique_list(deog['ic_ver'], deog['ic_hor'])

            # check for overlapping ICs (identified as ECG & EOG)
            ic_common = list(set.intersection(set(ic_ecg), set(ic_eog)))
            for ic in ic_common:
                # determine which score is larger
                sc_ecg1 = decg['scores_ctps'][ic]  # 0: ECG
                sc_ecg2 = decg['scores_corr'][ic]  # 1: ECG
                sc_eog1 = deog['scores_ver'][ic]   # 2: EOG
                sc_eog2 = deog['scores_hor'][ic]   # 3: EOG
                ix_max = np.array(np.abs([sc_ecg1, sc_ecg2, sc_eog1, sc_eog2])).argmax()
                # group IC based on the largest abs score
                if ix_max < 2:
                    ic_eog.remove(ic)    # is ECG, thus remove IC in the list of EOG
                else:
                    ic_ecg.remove(ic)   # is EOG, thus remove IC in the list of ECG
            ica.exclude = get_unique_list(ic_ecg, ic_eog)

            # collect events from each chop
            events_ecg.append(decg['events_ecg'])
            events_eog_ver.append(deog['events_ver'])
            events_eog_hor.append(deog['events_hor'])

            # ToDo in self.ica.chop.exclude append function
            #  e.g.  self.ica.chop.exclude.update( ecg=decg,eog=deog )
            # -------------------------------
            # store info in ica.chop object
            # -------------------------------
            self.ica.chop.ica[ichop] = ica
            self.ica.chop.exclude.exclude.append( ica.exclude )       # ICs to be excluded

            # ECG ICs
            self.ica.chop.exclude.ecg.append(ic_ecg)
            self.ica.chop.exclude.ecg_ctps.append(decg['ic_ctps'])
            self.ica.chop.exclude.ecg_corr.append(decg['ic_corr'])
            # ECG scores
            self.ica.chop.exclude.ecg_ctps_scores.append(decg['scores_ctps'])
            self.ica.chop.exclude.ecg_corr_scores.append(decg['scores_corr'])

            # EOG ICs
            self.ica.chop.exclude.eog.append(list(ic_eog))
            self.ica.chop.exclude.eog_ver.append(deog['ic_ver'])
            self.ica.chop.exclude.eog_hor.append(deog['ic_hor'])
            # EOG scores
            self.ica.chop.exclude.eog_ver_scores.append(deog['scores_ver'])
            self.ica.chop.exclude.eog_hor_scores.append(deog['scores_hor'])

        # store events in sources class
        self.sources._events_ecg     = np.array(events_ecg)
        self.sources._events_eog_ver = np.array(events_eog_ver)
        self.sources._events_eog_hor = np.array(events_eog_hor)
        # store scores in sources class
        self.sources.score._ecg_ctps = np.array(self.ica.chop.exclude.ecg_ctps_scores)
        self.sources.score._ecg_corr = np.array(self.ica.chop.exclude.ecg_corr_scores)
        self.sources.score._eog_ver  = np.array(self.ica.chop.exclude.eog_ver_scores)
        self.sources.score._eog_hor  = np.array(self.ica.chop.exclude.eog_hor_scores)


    # -----------------------------------------------------
    # get ICA sources (ICA time courses)
    # -----------------------------------------------------
    def _get_source_data(self, raws=None):
            n_comp    = self.ica.n_components
            n_chop    = self.ica.chop.n_chop
            n_samples = self.ica.chop.n_samples
            picks_meg = self.meg.picks.meg
            picks_aux = self.meg.picks.aux
            n_aux = len(picks_aux)

            # init data
            _data_ica = np.zeros([n_chop, n_comp, n_samples], dtype='float64')
            _data_aux = np.zeros([n_chop, n_aux, n_samples], dtype='float64')

            # loop across chops
            for ichop in range(n_chop):
                ica = self.ica.chop.ica[ichop]
                raw = raws[ichop]

                # compute ICA sources
                data = raw._data[picks_meg]
                data, _ = ica._pre_whiten(data, raw.info, picks_meg)
                _data_ica[ichop] = ica._transform(data)

                # extract AUX data
                _data_aux[ichop] = raw._data[picks_aux]

            return _data_ica, _data_aux


    # -----------------------------------------------------
    # label ICA sources
    # -----------------------------------------------------
    def _get_source_labels(self):
            n_comp = self.ica.n_components
            n_chop = self.ica.chop.n_chop
            _labels = []
            label_default = ['other'] * n_comp
            for ichop in range(n_chop):
                labels_chop = label_default.copy()
                for ix in self.ica.chop.exclude.eog[ichop]:
                    labels_chop[ix] = 'EOG'
                for ix in self.ica.chop.exclude.ecg[ichop]:
                    labels_chop[ix] = 'ECG'
                _labels.append(labels_chop)

            return _labels


    # -----------------------------------------------------
    # update excludes based on scores
    # -----------------------------------------------------
    def _update_exclude(self):
        for ichop in range(self.ica.chop.n_chop):
            # get ICA info
            ica = self.ica.chop.ica[ichop].copy()
            label = ['ECG', 'EOG', 'EOG']
            ic_ecg_ctps  = ic_ecg_corr  = []
            ic_eogV = ic_eogH = []

            for ic in ica.exclude:
                # init relative scores
                rel_score_ecg_ctps = rel_score_ecg_corr = 0.0
                rel_score_eogV     = rel_score_eogH     = 0.0

                # compute score relative to threshold
                if self.sources.score.ecg_ctps[ichop][0] >= 0:
                    rel_score_ecg_ctps = self.sources.score.ecg_ctps[ichop][ic] / self.ica.ecg_thresh_ctps
                if self.sources.score.ecg_corr[ichop][0] >= 0:
                    rel_score_ecg_corr = self.sources.score.ecg_corr[ichop][ic] / self.ica.ecg_thresh_corr
                if self.sources.score.eog_ver[ichop][0] >= 0:
                    rel_score_eogV = self.sources.score.eog_ver[ichop][ic] / self.ica.eog_thresh_ver
                if self.sources.score.eog_hor[ichop][0] >= 0:
                    rel_score_eogH = self.sources.score.eog_hor[ichop][ic] / self.ica.eog_thresh_hor

                # get score with largest ratio
                ix = np.array([rel_score_ecg_ctps, rel_score_ecg_ctps, rel_score_eogV, rel_score_eogH]).argmax()
                if ix == 0:                  # ECG CTPS
                    ic_ecg_ctps.append(ic)
                elif ix == 1:                # ECG CORR
                    ic_ecg_corr.append(ic)
                elif ix == 2:                # EOG ver CORR
                    ic_eogV.append(ic)
                elif ix == 3:                # EOG hor CORR
                    ic_eogH.append(ic)
            # update exclude info
            self.ica.chop.exclude.ecg[ichop]      = get_unique_list(ic_ecg_ctps, ic_ecg_corr)
            self.ica.chop.exclude.ecg_ctps[ichop] = get_unique_list(ic_ecg_ctps)
            self.ica.chop.exclude.ecg_corr[ichop] = get_unique_list(ic_ecg_corr)

            self.ica.chop.exclude.eog[ichop]      = get_unique_list(ic_eogV, ic_eogH)
            self.ica.chop.exclude.eog_ver[ichop]  = get_unique_list(ic_eogV)
            self.ica.chop.exclude.eog_hor[ichop]  = get_unique_list(ic_eogH)

    # -----------------------------------------------------
    # plot ICA traces and scores for all chops
    # -----------------------------------------------------
    def plot_ica_traces(self, fname_npz):

        import matplotlib
        import matplotlib.pylab as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        backend_orig = matplotlib.get_backend()  # store original backend
        matplotlib.use('Agg')                    # switch to agg backend to not show the plot

        # settings
        figsize   = (16, 13)
        col_ecg   = 'red'
        col_eog   = 'green'
        col_other = 'lightsteelblue'
        col_kurt  = 'darkgray'
        col_aux   = 'magenta'

        figs      = []
        captions  = []
        name = op.basename(fname_npz[:-4])
        n_chop = self.ica.chop.n_chop

        # compute topo images with 'head'
        self.ica.update_topo_images_head(res=300)

        # loop across chops
        for ichop in range(n_chop):
            aux_data = self.sources.data_aux[ichop]
            if len(self.meg.picks.aux_types) > 0:
                aux_types = self.meg.picks.aux_types
            else:
                aux_types = ['aux'] * aux_data.shape[0]
            n_aux = aux_data.shape[0]
            n_k = 2  # show ICs with large kurtosis
            # calc number of ICs below threshold
            n_max = 12
            n_ecg = len(self.ica.chop.exclude.ecg[ichop])
            n_eog = len(self.ica.chop.exclude.eog[ichop])
            n_bt = n_max - n_k - n_aux - n_ecg - n_eog
            src_data, src_label, src_info, src_ics = collect_source_info(self.sources, self.ica.chop.exclude, ichop,
                                                                         n_below_thresh=n_bt, n_kurtosis=n_k)
            # add aux info
            data = np.concatenate([src_data, aux_data])
            # label = np.concatenate([src_label, aux_types])
            info = np.concatenate([src_info, aux_types])

            n_ics = len(src_ics)
            n_chan, n_times = data.shape
            twin = self.ica.chop.times[ichop]
            times = np.linspace(twin[0], twin[1], n_times)

            topo_img = self.ica.topo.images_head[ichop]   # topo head image

            fig, axlist = plt.subplots(n_chan, 1, figsize=figsize, sharex=True)
            plt.subplots_adjust(left=0.05, right=0.97, bottom=0.1, top=0.9, wspace=0.1, hspace=0.3)
            for ichan, ax in enumerate(axlist):
                # title
                if ichan == 0:
                    title = name + ' chop#%d of %d' % (ichop, n_chop)
                else:
                    title = ''
                # color
                if ichan < (n_chan - n_aux):
                    if src_label[ichan] == 'ECG':
                        col = col_ecg
                    elif src_label[ichan] == 'EOG':
                        col = col_eog
                    elif src_label[ichan] == 'other':
                        if ichan < (n_ecg + n_eog + n_bt):
                            col = col_other
                        else:
                            col = col_kurt
                else:
                    col = col_aux

                # plot topography
                divider = make_axes_locatable(ax)  # add axes for topo and time course
                ax_topo = divider.append_axes('left', size='10%', pad=0.0)
                if ichan < n_ics:
                    topo = topo_img[src_ics[ichan]]
                    ax_topo.imshow(topo, cmap='RdBu_r', interpolation='bilinear')
                    ax_topo.figure.tight_layout()
                ax_topo.axis('off')

                # plot time course
                ax.axes.set_title(title)
                ax.plot(times, data[ichan], label=info[ichan], color=col)
                ax.spines["top"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.axes.set_xlim(twin)
                ax.legend(loc=1, fontsize='medium', bbox_to_anchor=(1., 1.5))
            plt.tight_layout()
            figs.append(fig)
            # captions.append('%s ... Chop#%d' % (name[:25], ichop))
            captions.append('%s ... Chop#%d' % (name.split(",")[0], ichop))

        # switch to original backend
        matplotlib.use(backend_orig)

        return figs, captions

    def plot_artifact_performance(self):
        """
        plot artifact performance averaged on ECG and EOG onsets
        for reconstructed raw data (meg,ecg,eog,stim) from ICA

        Returns
        -------
        figs,captions
        """
        captions  = []
        self.PerformancePlot.clear_figures() # clear list of figs
        name      = op.basename( self.meg.fname.split(".",-1)[0] )

        msg=["plot artifact rejection performance:",
             "fname: {}".format(os.path.basename( self.meg.fname )),
             "path : {}".format(os.path.dirname(  self.meg.fname )),
             "-"*30,
             "name : {}".format(name)]

        logger.info("\n  -> ".join(msg))
        toffset = 0.0 # time range info

        for ichop in range( self.ica.chop.n_chop ):
            # captions.append('AR %s ... Chop#%d' % (name.split(",")[0], ichop))
            captions.append('%s ... Chop#%d' % (name.split(",")[0], ichop))
            logger.info("plot ARP ICA chop: {} caption: {}".format(ichop,captions[-1]))

            ica = self.ica.chop.ica[ichop].copy()

           # --  get data from sources
            data_ica   = self.sources.data_ica[ichop]
            data_aux   = self.sources.data_aux[ichop]
            aux_labels = self.meg.picks.aux_labels
            aux_types  = self.meg.picks.aux_types

            if data_aux.shape[0] != len(aux_labels):
                logger.exception("ERROR : channel count in aux data [{}] is not equal to counts in aux labels [{}]".
                                 format(data_aux.shape[0], len(aux_labels)))
                return

           # -- reconstruct: raw, raw_clean + add aux data
            raw_chop, raw_chop_clean = transform_ica2data(data_ica, ica)
            raw_chop, raw_chop_clean = add_aux_channels([raw_chop, raw_chop_clean], data_aux, aux_labels, aux_types)

            # -- ck for eog_ch2
            # ToDo implement lists of ECG,EOG artifacts in DCNN cls!!! not fixed channel numbers !!!
            artifacts = [ self.meg.ecg_ch,self.meg.eog_ch1 ]
            events    = [ self.sources.events_ecg[ichop],self.sources.events_eog_ver[ichop] ]

            if self.meg.eog_ch2:
                artifacts.append( self.meg.eog_ch2 )
                events.append( self.sources.events_eog_hor[ichop] )

            # -- mk ICs info
            n_ecg = len(self.ica.chop.exclude.ecg[ichop])
            n_eog = len(self.ica.chop.exclude.eog[ichop])

            trange    = "Time: {:.3f} <> {:.3f}".format(toffset,raw_chop.times[-1]+toffset) # time range info
            suptitle  = trange +"  ECG: cnts: {}  ICs: {}".format( n_ecg,",".join( [str(i) for i in self.ica.chop.exclude.ecg[ichop] ]))
            suptitle += "   "
            suptitle += "EOG: cnts: {}  ICs: {}".format( n_eog,",".join( [str(i) for i in self.ica.chop.exclude.eog[ichop] ]))

            # -- init plot
            self.PerformancePlot._fig            = None
            self.PerformancePlot.idx             = 1
            self.PerformancePlot.n_cols          = len( artifacts )
            # self.PerformancePlot.picks           = self.meg.picks.meg
            self.PerformancePlot.picks           = np.arange(len(self.meg.picks.meg))
            self.PerformancePlot.raw             = raw_chop
            self.PerformancePlot.raw_clean       = raw_chop_clean

           # -- loop for ECG,EOGs
            for idx in range( self.PerformancePlot.n_cols ):
                evt = events[idx]
                if len(evt) > 2: # must have at least 3 elements
                    self.PerformancePlot.plot(ch_name=artifacts[idx], events=evt, event_id=evt[0,2],
                                              fig_nr=ichop+1, suptitle=suptitle)
                self.PerformancePlot.idx += 1

            toffset += raw_chop.times[-1]

        return self.PerformancePlot.figures, captions

    # -----------------------------------------------------
    # get ICA sources & labels
    # -----------------------------------------------------
    def get_sources(self, raws=None):
        self.sources._data_ica, self.sources._data_aux = self._get_source_data(raws=raws)
        self.sources._labels = self._get_source_labels()


    # -----------------------------------------------------
    # get scores
    # -----------------------------------------------------
    def update_scores(self):
        self._update_exclude()
        self.sources._labels  = self._get_source_labels()


    # ---------------------------------------
    #
    # label ICA components (cardiac, ocular)
    #
    # ---------------------------------------
    def label_ica(self ,save=True):

        # make a copy
        raw = self.meg.raw.copy()

        # 1. filtering and downsampling is applied directly on raw
        #    here we also compute the chop size for cropping the data
        raw = self._meg_filter_resample(raw=raw)

        # 2. apply ica on chopped raw data (about 2-3min data chunks)
        raw_chop_list = self._ica_apply_chops(raw=raw)
        del raw   # we continue to work on raw_chop_list

        # 3. apply auto labeling ICA components
        self._ica_label_chops(raw_chop_list)  # ica.excludes are saved in object: self.ica

        # 4. compute ICA sources for each chop and component
        #    returns: IC timeseries (components), aux data and labels
        self.get_sources(raw_chop_list)

        # 5. compute topographies and images for each chop and component
        #    - ICA weights for each sensor and component: np.array([n_chop, n_chan, n_comp])
        #    - topo images for each sensor and component: np.array([n_chop, n_chan, 64, 64])
        self.ica.update_topo_data()
        self.ica.update_topo_images(res=self.model['res_space'])
        # self.ica.update_topo_images_head(res=300)  # for plotting only

        # 6. copy info to sources
        # self.sources.topo._data = self.ica.topo.data
        # self.sources.topo._images = self.ica.topo.images
        # self.sources.topo._images_head = self.ica.topo.images_head


        if save:
            fout_gdcnn = self.save_gdcnn()
        else:
            fout_gdcnn = None

        logger.info('DONE ICA and auto labeling')

        return fout_gdcnn


    # ---------------------------------------
    #
    # apply manual correction to ICA labels
    #
    # ---------------------------------------
    def labels_correct(self, save=True, path_out=None):
        # ica_check_labels(self, save=save)

        for ichop in range(self.ica.chop.n_chop):
            # get ICA info
            ica = self.ica.chop.ica[ichop].copy()
            ic_exclude = ica.exclude  # copy original selected ICs

            data_ica = self.sources.data_ica[ichop]

            # reconstruct MEG signals and create raw object
            data_meg = transform_mne_ica2data(data_ica, ica)
            raw_chop = mne.io.RawArray(data_meg, ica.info)

            # plot sources and components
            # NOTE: click on ICA number (e.g. ICA001) to add or remove the IC to the list of ica.exclude[]
            # ica.plot_components()
            ica.plot_sources(raw_chop, block=True)
            # ica.plot_overlay(raw_chop, exclude=ica.exclude, picks='mag')

            # update ICA object
            if save:
                ica.exclude = list(np.unique(ica.exclude))
                self.ica.chop.ica[ichop] = ica
                self.ica.chop.exclude.exclude[ichop] = ica.exclude  # overwrite excludes

        # 7. save results to disk
        if save:
            self.update_scores()
            fout_gdcnn = self.save_gdcnn(path_out=path_out)
            logger.info('DONE check ICA auto labeling')
            return fout_gdcnn


    # ---------------------------------------
    #
    # load gDCNN info and data
    #
    # ---------------------------------------
    def load_gdcnn(self, fname):
        """


        Parameters
        ----------
        fname : TYPE
            DESCRIPTION.

        Returns
        -------
        npz : TYPE
            DESCRIPTION.

        """

        # with open('data.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        # data = pickle.load(f)

        # load gDCNN data
        fname= expandvars(fname)
        logger.info('Start loading gDCNN results from disk ....\n  -> {}'.format(fname))
        npz = np.load(fname, allow_pickle=True)
        logger.info('Done loading gDCNN results from disk\n')

       # if self.verbose:
       #     for k, v in npz.items():
       #         logger.info("gDCNN data: {}\n{}".format(k, v))

        # init clases
        self.path.init(**npz.get('path').item())  # load dict

        self.model = npz.get('model').item()  # load dict

        # data objects cls
        self.meg.init(**npz['meg'].item())  # get dict from np.array
        self.ica.init(**npz['ica'].item())  # get dict from np.array
        self.sources.init(**npz['sources'].item())  # get dict from np.array

        return npz


    # ---------------------------------------
    #
    # save results from auto labeling of ICs in npz format
    #
    # ---------------------------------------
    def save_gdcnn(self, fnout=None, path_out=None):

        logger.info('Start saving results to disk ....')
        from os import makedirs

        if not path_out:
            path_out = self.path.data_train
            if not op.exists(path_out):
                makedirs(path_out)

        from os import makedirs
        # save results to disk
        try:
            if not fnout:
                logger.info("input fnout: ".format(fnout))
                name = op.basename(self.meg.fname).rsplit('.')[0]
            else:
                name = op.basename(fnout).rsplit('.')[0]
            fnout = os.path.join(path_out, name + '-gdcnn.npz')
            logger.info("output file: {}".format(fnout))
        except:
            logger.exception("ERROR\n  -> path_out: {}\n  -> fnout: {}".format(path_out, fnout))
            return False

        self.model['fname_gdcnn'] = fnout

        np.savez(fnout,model=self.model, path=self.path.dump(),
                 meg=self.meg.dump(), ica=self.ica.dump(), sources=self.sources.dump())

        logger.info('Done saving results to disk: {}'.format(fnout))
        return fnout


    def get_info(self):
        # get details
        # --- init dicts
        msg = ["DCNN Info",
               "  -> model:",
               "  -> {}".format(self.model)]
        logger.info("\n".join(msg))

        # -- objs
        self.path.get_info()
        self.meg.get_info()
        self.ica.get_info()
        self.sources.get_info()


