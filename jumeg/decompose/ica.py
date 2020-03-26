# ICA functions
'''
 authors:
            Juergen Dammers
            Lukas Breuer
 email:  j.dammers@fz-juelich.de

 Change history:
 21.01.2020: - changes in ica_array
                - now returns an MNE-type of ICA object (default)
                - in fastica changed default to whiten=False
             - added function to convert any ica object to MNE-type
             - name change: "activations" are now named "sources"
 06.01.2020: added whitening option in PCA
 17.10.2019: added mulitple useful functions to use ICA without MNE
 27.11.2015  created by Lukas Breuer
'''


#######################################################
#                                                     #
#              import necessary modules               #
#                                                     #
#######################################################
from scipy.stats import kurtosis
import math
import numpy as np
from sys import stdout
from scipy.linalg import pinv
from copy import deepcopy


#######################################################
#                                                     #
#   interface to perform (extended) Infomax ICA on    #
#                    a data array                     #
#                                                     #
#######################################################
def ica_array(data_orig, dim_reduction='', explainedVar=1.0,
              overwrite=None, return_ica_object=True,
              max_pca_components=None, method='infomax',
              cost_func='logcosh', weights=None, lrate=None,
              block=None, wchange=1e-16, annealdeg=60.,
              annealstep=0.9, n_subgauss=1, kurt_size=6000,
              maxsteps=200, pca=None, whiten=False, verbose=True):

    """
    interface to perform (extended) Infomax or FastICA on a data array

        Parameters
        ----------
        data_orig : array of data to be decomposed [nchan, ntsl].

        Optional Parameters
        -------------------
        dim_reduction : {'', 'AIC', 'BIC', 'GAP', 'MDL', 'MIBS', 'explVar'}
            Method for dimension selection. For further information about
            the methods please check the script 'dimension_selection.py'.
            default: dim_reduction='' --> no dimension reduction is performed
                                          as long as not the parameter
                                          'max_pca_components' is set.
        explainedVar : float
            Value between 0 and 1; components will be selected by the
            cumulative percentage of explained variance.
        overwrite : if set the data array will be overwritten
            (this saves memory)
            default: overwrite=None
        max_pca_components : int | None
            The number of components used for PCA decomposition. If None, no
            dimension reduction will be applied and max_pca_components will equal
            the number of channels supplied on decomposing data. Only of interest
            when dim_reduction=''
        method : {'fastica', 'infomax', 'extended-infomax'}
          The ICA method to use. Defaults to 'infomax'.

        whiten : bool, optional (default False)
            When True the `components_` vectors are multiplied
            by the square root of n_samples and then divided by the singular values
            to ensure uncorrelated outputs with unit component-wise variances.
            Whitening will remove some information from the transformed signal
            (the relative variance scales of the components) but can sometime
            improve the predictive accuracy of the downstream estimators by
            making their data respect some hard-wired assumptions.

        return_ica_object: bool, optional (default True)
            When True an MNE-type of ICA object is returned (including PCA) besides the sources
            When False the ICA unmixing matrix, the PCA obejct and the sources are returned


        FastICA parameter:
        -----------------------------
        cost_func : String
             Cost function to use in FastICA algorithm. Could be
             either 'logcosh', 'exp' or 'cube'.


        (Extended) Infomax parameter:
        -----------------------------
        weights : initialize weights matrix
            default: None --> identity matrix is used
        lrate : initial learning rate (for most applications 1e-3 is
            a  good start)
            --> smaller learining rates will slowering the convergence
            it merely indicates the relative size of the change in weights
            default:  lrate = 0.010d/alog(nchan^2.0)
        block : his block size used to randomly extract (in time) a chop
            of data
            default:  block = floor(sqrt(ntsl/3d))
        wchange : iteration stops when weight changes are smaller then this
            number
            default: wchange = 1e-16
        annealdeg : if angle delta is larger then annealdeg (in degree) the
            learning rate will be reduce
            default:  annealdeg = 60
        annealstep : the learning rate will be reduced by this factor:
            lrate  *= annealstep
            default:  annealstep = 0.9
        extended : if set extended Infomax ICA is performed
            default: None
        n_subgauss : extended=int
            The number of subgaussian components. Only considered for extended
            Infomax.
            default: n_subgauss=1
        kurt_size : int
            The window size for kurtosis estimation. Only considered for extended
            Infomax.
            default: kurt_size=6000
        maxsteps : maximum number of iterations to be done
            default:  maxsteps = 200


        Returns
        -------
        weights : un-mixing matrix
        pca : instance of PCA
            Returns the instance of PCA where all information about the
            PCA decomposition are stored.
        sources: ICA sources
    """

    # -------------------------------------------
    # check overwrite option
    # -------------------------------------------
    if overwrite == None:
        data = data_orig.copy()
    else:
        data = data_orig

    # -------------------------------------------
    # perform centering and whitening of the data
    #    - optional use the provided PCA object
    # -------------------------------------------
    if pca:
        # perform centering and whitening
        dmean = data.mean(axis=-1)
        stddev = np.std(data, axis=-1)
        dnorm = (data - dmean[:, np.newaxis])/stddev[:, np.newaxis]
        data = np.dot(dnorm.T, pca.components_[:max_pca_components].T)

        # update mean and standard-deviation in PCA object
        pca.mean_ = dmean
        pca.stddev_ = stddev
    else:
        if verbose:
            print("     ... perform centering and whitening ...")

        data, pca = whitening(data.T, dim_reduction=dim_reduction, npc=max_pca_components,
                              explainedVar=explainedVar, whiten=whiten)

    # -------------------------------------------
    # now call ICA algortithm
    # -------------------------------------------
    # FastICA
    if method == 'fastica':
        from sklearn.decomposition import fastica
        # By Lukas, whitening was set to True. The new default is whiten=False
        # However, whitening should not ne applied again, was whitened already (which is the default)
        _, unmixing_, sources_ = fastica(data, fun=cost_func, max_iter=maxsteps, tol=1e-4, whiten=False)
        sources = sources_.T
        weights = unmixing_

    # Infomax or extended Infomax
    else:
        if method == 'infomax':
            extended = False
        elif method == 'extended-infomax':
            extended = True
        else:
            print(">>>> WARNING: Entered ICA method not found!")
            print(">>>>          Allowed are fastica, extended-infomax and infomax")
            print(">>>>          Using now the default ICA method which is Infomax")
            extended = False

        weights = infomax(data, weights=weights, l_rate=lrate, block=block,
                          w_change=wchange, anneal_deg=annealdeg, anneal_step=annealstep,
                          extended=extended, n_subgauss=n_subgauss, kurt_size=kurt_size,
                          max_iter=maxsteps, verbose=verbose)
        sources = np.dot(weights, data.T)

    # create an MNE-Python type of ICA object
    # Note, when used with MNE functions the info dict needs
    # to be manually added to the ica object, such as, ica.info = info
    if return_ica_object:
        ica = ica_convert2mne(weights, pca, method=method)
        return ica, sources
    else:
       return weights, pca, sources


#######################################################
#                                                     #
#   interface to perform (extended) Infomax ICA on    #
#                    a data array                     #
#                                                     #
#######################################################
def infomax2data(unmixing, pca, sources, idx_zero=None):

    """
    interface to perform (extended) Infomax ICA on a data array

        Parameters
        ----------
        unmixing: the ICA un-mixing (weight) matrix
        pca : instance of PCA object
        sources : underlying sources
        idx_zero : indices of independent components (ICs) which
            should be removed
            default: idx_zero=None --> not IC is removed

        Returns
        -------
        data : back-transformed cleaned data array
    """

    # -------------------------------------------
    # check dimension of the input data
    # -------------------------------------------
    npc   = len(unmixing)
    nchan = len(pca.components_)
    ntsl  = sources.shape[1]

    # create array for principal components
    pc    = np.zeros((nchan, ntsl))

    # -------------------------------------------
    # backtransform data
    # -------------------------------------------
    mixing = pinv(unmixing)

    if idx_zero is not None:
       mixing[:, idx_zero] = 0.

    pc[:npc] = np.dot(mixing, sources)            # back-transform to PCA-space
    data = np.dot(pca.components_.T, pc)          # back-transform to data-space
    del pc                                        # delete principal components
    data = (data * pca.stddev_[:, np.newaxis]) + pca.mean_[:, np.newaxis]  # reverse normalization

    # return results
    return data


#######################################################
#                                                     #
# routine for PCA decomposition prior to ICA          #
#                                                     #
#######################################################
def whitening(data, dim_reduction='',
              npc=None, explainedVar=1.0, whiten=False):

    """
    routine to perform whitening prior to Infomax ICA application
    (whitening is based on Principal Component Analysis from the
    RandomizedPCA package from sklearn.decomposition)

        Parameters
        ----------
        X : data array [ntsl, nchan] for decomposition.
        dim_reduction : {'', 'AIC', 'BIC', 'GAP', 'MDL', 'MIBS', 'explVar'}
            Method for dimension selection. For further information about
            the methods please check the script 'dimension_selection.py'.
            default: dim_reduction='' --> no dimension reduction is performed as
                                          long as not the parameter 'npc' is set.
        npc : int | None
            The number of components used for PCA decomposition. If None, no
            dimension reduction will be applied and max_pca_components will equal
            the number of channels supplied on decomposing data. Only of interest
            when dim_reduction=''
            default: npc = None
        explainedVar : float | None
            Must be between 0 and 1. If float, the number of components
            selected matches the number of components with a cumulative
            explained variance of 'explainedVar'
            default: explainedVar = None

        whiten : bool, optional (default False)
            When True the `components_` vectors are multiplied
            by the square root of n_samples and then divided by the singular values
            to ensure uncorrelated outputs with unit component-wise variances.
            Whitening will remove some information from the transformed signal
            (the relative variance scales of the components) but can sometime
            improve the predictive accuracy of the downstream estimators by
            making their data respect some hard-wired assumptions.


        Returns
        -------
        whitened_data : data array [nchan, ntsl] of decomposed sources
        ica : instance of ICA
            Returns the instance of ICA where all information about the
            PCA decomposition are updated.
        sel : array containing the indices of the selected ICs
            (depends on the variable npc)
    """

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    from sklearn.decomposition import PCA
    from . import dimension_selection as dim_sel

    # -------------------------------------------
    # check input data
    # -------------------------------------------
    ntsl, nchan = data.shape

    if (nchan < 2) or (ntsl < nchan):
        raise ValueError('Data size too small!')

    # -------------------------------------------
    # perform PCA decomposition
    # -------------------------------------------
    X = data.copy()
    # whiten = False
    dmean = X.mean(axis=0)
    stddev = np.std(X, axis=0)
    X = (X - dmean[np.newaxis, :]) / stddev[np.newaxis, :]

    pca = PCA(n_components=None, whiten=whiten, svd_solver='auto', copy=True)

    # -------------------------------------------
    # perform whitening
    # -------------------------------------------
    whitened_data = pca.fit_transform(X)

    # -------------------------------------------
    # update PCA structure
    # -------------------------------------------
    pca.mean_ = dmean
    pca.stddev_ = stddev

    # -------------------------------------------
    # check dimension selection
    # -------------------------------------------
    if dim_reduction == 'AIC':
        npc, _ = dim_sel.aic(pca.explained_variance_)
    elif dim_reduction == 'BIC':
        npc = dim_sel.mibs(pca.explained_variance_, ntsl)
    elif dim_reduction == 'GAP':
        npc = dim_sel.gap(pca.explained_variance_)
    elif dim_reduction == 'MDL':
        _, npc = dim_sel.mdl(pca.explained_variance_)
    elif dim_reduction == 'MIBS':
        npc = dim_sel.mibs(pca.explained_variance_, ntsl)
    elif dim_reduction == 'explVar':
        npc = dim_sel.explVar(pca.explained_variance_,explainedVar)

    elif npc is None:
        npc = nchan

    # return results
    # return whitened_data[:, :(npc + 1)], pca
    return whitened_data[:, :(npc)], pca


#######################################################
#                                                     #
#          real Infomax implementation                #
#                                                     #
#######################################################
def infomax(data, weights=None, l_rate=None, block=None, w_change=1e-12,
            anneal_deg=60., anneal_step=0.9, extended=False, n_subgauss=1,
            kurt_size=6000, ext_blocks=1, max_iter=200,
            fixed_random_state=None, verbose=None):
    """
    Run the (extended) Infomax ICA decomposition on raw data
    based on the publications of Bell & Sejnowski 1995 (Infomax)
    and Lee, Girolami & Sejnowski, 1999 (extended Infomax)

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        The data to unmix.
    w_init : np.ndarray, shape (n_features, n_features)
        The initialized unmixing matrix. Defaults to None. If None, the
        identity matrix is used.
    l_rate : float
        This quantity indicates the relative size of the change in weights.
        Note. Smaller learining rates will slow down the procedure.
        Defaults to 0.010d / alog(n_features ^ 2.0)
    block : int
        The block size of randomly chosen data segment.
        Defaults to floor(sqrt(n_times / 3d))
    w_change : float
        The change at which to stop iteration. Defaults to 1e-12.
    anneal_deg : float
        The angle at which (in degree) the learning rate will be reduced.
        Defaults to 60.0
    anneal_step : float
        The factor by which the learning rate will be reduced once
        ``anneal_deg`` is exceeded:
            l_rate *= anneal_step
        Defaults to 0.9
    extended : bool
        Wheather to use the extended infomax algorithm or not. Defaults to
        True.
    n_subgauss : int
        The number of subgaussian components. Only considered for extended
        Infomax.
    kurt_size : int
        The window size for kurtosis estimation. Only considered for extended
        Infomax.
    ext_blocks : int
        The number of blocks after which to recompute Kurtosis.
        Only considered for extended Infomax.
    max_iter : int
        The maximum number of iterations. Defaults to 200.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    unmixing_matrix : np.ndarray of float, shape (n_features, n_features)
        The linear unmixing operator.
    """

    # define some default parameter
    max_weight = 1e8
    restart_fac = 0.9
    min_l_rate = 1e-10
    blowup = 1e4
    blowup_fac = 0.5
    n_small_angle = 200
    degconst = 180.0 / np.pi

    # for extended Infomax
    extmomentum = 0.5
    signsbias = 0.02
    signcount_threshold = 25
    signcount_step = 2
    if ext_blocks > 0:  # allow not to recompute kurtosis
        n_subgauss = 1  # but initialize n_subgauss to 1 if you recompute

    # check data shape
    n_samples, n_features = data.shape
    n_features_square = n_features ** 2

    # check input parameter
    # heuristic default - may need adjustment for
    # large or tiny data sets
    if l_rate is None:
        l_rate = 0.01 / math.log(n_features ** 2.0)

    if block is None:
        block = int(math.floor(math.sqrt(n_samples / 3.0)))

    if verbose:
        print('computing%sInfomax ICA' % ' Extended ' if extended is True else ' ')

    # collect parameter
    nblock = n_samples // block
    lastt = (nblock - 1) * block + 1

    # initialize training
    if weights is None:
        # initialize weights as identity matrix
        weights = np.identity(n_features, dtype=np.float64)

    BI = block * np.identity(n_features, dtype=np.float64)
    bias = np.zeros((n_features, 1), dtype=np.float64)
    onesrow = np.ones((1, block), dtype=np.float64)
    startweights = weights.copy()
    oldweights = startweights.copy()
    step = 0
    count_small_angle = 0
    wts_blowup = False
    blockno = 0
    signcount = 0

    # for extended Infomax
    if extended is True:
        signs = np.identity(n_features)
        signs.flat[slice(0, n_features * n_subgauss, n_features)]
        kurt_size = min(kurt_size, n_samples)
        old_kurt = np.zeros(n_features, dtype=np.float64)
        oldsigns = np.zeros((n_features, n_features))

    # trainings loop
    olddelta, oldchange = 1., 0.
    while step < max_iter:

        # shuffle data at each step
        if fixed_random_state:
            np.random.seed(step)  # --> permutation is fixed but differs at each step
        else:
            np.random.seed(None)

        permute = list(range(n_samples))
        np.random.shuffle(permute)

        # ICA training block
        # loop across block samples
        for t in range(0, lastt, block):
            u = np.dot(data[permute[t:t + block], :], weights)
            u += np.dot(bias, onesrow).T

            if extended is True:
                # extended ICA update
                y = np.tanh(u)
                weights += l_rate * np.dot(weights,
                                           BI - np.dot(np.dot(u.T, y), signs) -
                                           np.dot(u.T, u))
                bias += l_rate * np.reshape(np.sum(y, axis=0,
                                            dtype=np.float64) * -2.0,
                                            (n_features, 1))

            else:
                # logistic ICA weights update
                y = 1.0 / (1.0 + np.exp(-u))
                weights += l_rate * np.dot(weights,
                                           BI + np.dot(u.T, (1.0 - 2.0 * y)))
                bias += l_rate * np.reshape(np.sum((1.0 - 2.0 * y), axis=0,
                                            dtype=np.float64), (n_features, 1))

            # check change limit
            max_weight_val = np.max(np.abs(weights))
            if max_weight_val > max_weight:
                wts_blowup = True

            blockno += 1
            if wts_blowup:
                break

            # ICA kurtosis estimation
            if extended is True:

                n = np.fix(blockno / ext_blocks)

                if np.abs(n) * ext_blocks == blockno:
                    if kurt_size < n_samples:
                        rp = np.floor(np.random.uniform(0, 1, kurt_size) *
                                      (n_samples - 1))
                        tpartact = np.dot(data[rp.astype(int), :], weights).T
                    else:
                        tpartact = np.dot(data, weights).T

                    # estimate kurtosis
                    kurt = kurtosis(tpartact, axis=1, fisher=True)

                    if extmomentum != 0:
                        kurt = (extmomentum * old_kurt +
                                (1.0 - extmomentum) * kurt)
                        old_kurt = kurt

                    # estimate weighted signs
                    signs.flat[::n_features + 1] = ((kurt + signsbias) /
                                                    np.abs(kurt + signsbias))

                    ndiff = ((signs.flat[::n_features + 1] -
                              oldsigns.flat[::n_features + 1]) != 0).sum()
                    if ndiff == 0:
                        signcount += 1
                    else:
                        signcount = 0
                    oldsigns = signs

                    if signcount >= signcount_threshold:
                        ext_blocks = np.fix(ext_blocks * signcount_step)
                        signcount = 0

        # here we continue after the for
        # loop over the ICA training blocks
        # if weights in bounds:
        if not wts_blowup:
            oldwtchange = weights - oldweights
            step += 1
            angledelta = 0.0
            delta = oldwtchange.reshape(1, n_features_square)
            change = np.sum(delta * delta, dtype=np.float64)

            if verbose:
                info = "\r" if step > 0 else ""
                info = ">>> Step %4d of %4d; wchange: %1.4e\n" % (step, max_iter, change)
                stdout.write(info)
                stdout.flush()

            if step > 1:
                angledelta = math.acos(np.sum(delta * olddelta) /
                                       math.sqrt(change * oldchange))
                angledelta *= degconst

            # anneal learning rate
            oldweights = weights.copy()
            if angledelta > anneal_deg:
                l_rate *= anneal_step    # anneal learning rate
                # accumulate angledelta until anneal_deg reached l_rates
                olddelta = delta
                oldchange = change
                count_small_angle = 0  # reset count when angle delta is large
            else:
                if step == 1:  # on first step only
                    olddelta = delta  # initialize
                    oldchange = change
                count_small_angle += 1
                if count_small_angle > n_small_angle:
                    max_iter = step

            # apply stopping rule
            if step > 2 and change < w_change:
                step = max_iter
            elif change > blowup:
                l_rate *= blowup_fac

        # restart if weights blow up
        # (for lowering l_rate)
        else:
            step = 0  # start again
            wts_blowup = 0  # re-initialize variables
            blockno = 1
            l_rate *= restart_fac  # with lower learning rate
            weights = startweights.copy()
            oldweights = startweights.copy()
            olddelta = np.zeros((1, n_features_square), dtype=np.float64)
            bias = np.zeros((n_features, 1), dtype=np.float64)

            # for extended Infomax
            if extended:
                signs = np.identity(n_features)
                signs.flat[slice(0, n_features * n_subgauss, n_features)]
                oldsigns = np.zeros((n_features, n_features))

            if l_rate > min_l_rate:
                if verbose:
                    print('... lowering learning rate to %g \n... re-starting...' % l_rate)
            else:
                raise ValueError('Error in Infomax ICA: unmixing_matrix matrix'
                                 'might not be invertible!')

    # return ICA unmixing matrix
    return weights.T        # after transpose shape corresponds to [n_features, n_samples]


#######################################################
#
# ica_convert2mne:
#   - create a MNE-type of ICA object
#   - include pca object into ica object
#   - define entries as used by MNE-Python
#
#######################################################
def ica_convert2mne(unmixing, pca, info=None, method='fastica'):

    # create MNE-type of ICA object
    from mne.preprocessing.ica import ICA
    n_comp = unmixing.shape[1]

    if method == 'extended-infomax':
        ica_method = 'infomax'
        fit_params = dict(extended=True)
    else:
        ica_method = method
        fit_params = None

    ica = ICA(n_components=n_comp, method=ica_method, fit_params=fit_params)

    # add PCA object
    ica.pca = pca

    # PCA info to be used bei MNE-Python
    ica.pca_mean_ = pca.mean_
    ica.pca_components_ = pca.components_
    exp_var = pca.explained_variance_
    ica.pca_explained_variance_ = exp_var
    ica.pca_explained_variance_ratio_ = pca.explained_variance_ratio_

    # ICA info
    ica.n_components_ = n_comp
    ica.n_components = n_comp
    ica.components_ = unmixing                  # compatible with sklearn
    ica.unmixing_= ica.components_                                              # as used by sklearn
    ica.mixing_ = pinv(ica.unmixing_)                                           # as used by sklearn
    ica.unmixing_matrix_ = ica.unmixing_ / np.sqrt(exp_var[0:n_comp])[None, :]  # as used by MNE-Python
    ica.mixing_matrix_ = pinv(ica.unmixing_matrix_)                             # as used by MNE-Python
    ica._ica_names = ['ICA%03d' % ii for ii in range(n_comp)]
    ica.fun = method
    if info:
        ica.info = info

    return ica


#######################################################
#                                                     #
# ica2data: back-transformation to data space         #
#                                                     #
#######################################################
def ica2data(sources, ica, pca, idx_zero=None, idx_keep=None):

    """
    ica2data: computes back-transformation from ICA to Data space
    :param sources: shape [n_samples, n_features]
    :param ica: ICA object from sklearn.decomposition
    :param pca: PCA object from sklearn.decomposition
    :param idx_zero: list of components to remove (optional)
    :param idx_keep: list of components to remove (optional)
    :return: data re-computed from ICA sources
    """

    # n_features = pca.n_features_
    n_features = pca.n_components_             # In rare cases, n_components_ < n_features_
    n_samples, n_comp = sources.shape
    A = ica.mixing_.copy()                     # ICA mixing matrix

    # create data with full dimension
    data = np.zeros((n_samples, n_features))
    idx_all = np.arange(n_comp)

    # if idx_keep is set it will overwrite idx_zero
    if idx_keep is not None:
        idx_zero = np.setdiff1d(idx_all, idx_keep)

    # if idx_zero or idx_keep was set idx_zero is always defined
    if idx_zero is not None:
        A[:, idx_zero] = 0.0

    # --------------------------------------------------------
    # back transformation to PCA space
    #
    # identical results:
    #    data[:, :n_comp] = ica.inverse_transform(sources)
    #    data[:, :n_comp] = np.dot(sources, ica.mixing_.T)
    # --------------------------------------------------------
    data[:, :n_comp] = np.dot(sources, A.T)

    # --------------------------------------------------------
    # back transformation to Data space
    #
    # identical results:
    #    data = pca.inverse_transform(data)
    #    data =  np.dot(data, np.sqrt(pca.explained_variance_[:, np.newaxis]) *
    #                                 pca.components_) + pca.mean_
    # --------------------------------------------------------
    # back transformation to data space
    data = pca.inverse_transform(data)

    return data


#######################################################
#                                                     #
# ica2data_single_component                           #
#                                                     #
#######################################################
def ica2data_single_components(sources, ica, pca, picks=None):

    # back-transformation of single ICs to data space
    # result is of shape: [n_components, data.shape]

    n_features = pca.n_features_
    n_samples, n_comp = sources.shape

    # create data with full dimension
    data = np.zeros((n_comp, n_samples, n_features))

    # loop over all ICs
    for icomp in range(n_comp):
        data[icomp] = ica2data(sources, ica, pca, idx_keep=icomp)


    # ===========================================
    #  for comparison with MNE
    # ===========================================
    # unmixing = np.eye(n_comp)
    # unmixing[:n_comp, :n_comp] = ica.unmixing_matrix_
    # unmixing = np.dot(unmixing, pca.components_[:n_comp])
    # mixing = np.eye(n_comp)
    # mixing[:n_comp, :n_comp] = ica.mixing_matrix_
    # mixing = np.dot(pca.components_[:n_comp].T, mixing)
    #
    # proj_mat = np.dot(mixing[:, [icomp]], unmixing[[icomp], :])
    # data = np.dot(proj_mat, data_tpq.T)
    #
    # # store mean TPQ values
    # # x_mean_comp[:,icomp] = data.mean(axis=0)
    # x_mean_comp[:, icomp] = data.mean(axis=1)

    return data
