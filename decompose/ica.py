# Authors: Lukas Breuer <l.breuer@fz-juelich.de>
'''
Created on 27.11.2015

@author: lbreuer
'''


#######################################################
#                                                     #
#              import necessary modules               #
#                                                     #
#######################################################
from scipy.stats import kurtosis
import math
import numpy as np



#######################################################
#                                                     #
#   interface to perform (extended) Infomax ICA on    #
#                    a data array                     #
#                                                     #
#######################################################
def ica_array(data_orig, dim_reduction='',
              explainedVar=1.0, overwrite=None,
              max_pca_components=None, method='infomax',
              cost_func='logcosh', weights=None, lrate=None,
              block=None, wchange=1e-16, annealdeg=60.,
              annealstep=0.9, n_subgauss=1, kurt_size=6000,
              maxsteps=200, pca=None, verbose=True):

    """
    interface to perform (extended) Infomax or FastICA on a data array

        Parameters
        ----------
        data_orig : array of data to be decomposed [nchan, ntsl].
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
        activations : underlying sources
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
            print "     ... perform centering and whitening ..."

        data, pca = whitening(data.T, dim_reduction=dim_reduction, npc=max_pca_components,
                              explainedVar=explainedVar)


    # -------------------------------------------
    # now call ICA algortithm
    # -------------------------------------------
    # FastICA
    if method == 'fastica':
        from sklearn.decomposition import fastica
        _, unmixing_, sources_ = fastica(data, fun=cost_func, max_iter=maxsteps, tol=1e-4,
                                         whiten=True)
        activations = sources_.T
        weights = unmixing_

    # Infomax or extended Infomax
    else:
        if method == 'infomax':
            extended = False
        elif method == 'extended-infomax':
            extended = True
        else:
            print ">>>> WARNING: Entered ICA method not found!"
            print ">>>>          Allowed are fastica, extended-infomax and infomax"
            print ">>>>          Using now the default ICA method which is Infomax"
            extended = False

        weights = infomax(data, weights=weights, l_rate=lrate, block=block,
                          w_change=wchange, anneal_deg=annealdeg, anneal_step=annealstep,
                          extended=extended, n_subgauss=n_subgauss, kurt_size=kurt_size,
                          max_iter=maxsteps, verbose=verbose)
        activations = np.dot(weights, data.T)

    # return results
    return weights, pca, activations





#######################################################
#                                                     #
#   interface to perform (extended) Infomax ICA on    #
#                    a data array                     #
#                                                     #
#######################################################
def infomax2data(weights, pca, activations, idx_zero=None):

    """
    interface to perform (extended) Infomax ICA on a data array

        Parameters
        ----------
        weights : un-mixing matrix
        pca : instance of PCA object
        activations : underlying sources
        idx_zero : indices of independent components (ICs) which
            should be removed
            default: idx_zero=None --> not IC is removed

        Returns
        -------
        data : backtransformed cleaned data array
    """

    # -------------------------------------------
    # import necessary modules
    # -------------------------------------------
    from scipy.linalg import pinv


    # -------------------------------------------
    # check dimension of the input data
    # -------------------------------------------
    npc   = len(weights)
    nchan = len(pca.components_)
    ntsl  = activations.shape[1]

    # create array for principal components
    pc    = np.zeros((nchan, ntsl))

    # -------------------------------------------
    # backtransform data
    # -------------------------------------------
    iweights = pinv(weights)

    if idx_zero is not None:
       iweights[:, idx_zero] = 0.

    pc[:npc] = np.dot(iweights, activations)      # back-transform to PCA-space
    data     = np.dot(pca.components_.T, pc)      # back-transform to data-space
    del pc                                        # delete principal components
    data     = (data * pca.stddev_[:, np.newaxis]) + pca.mean_[:, np.newaxis]  # reverse normalization


    # return results
    return data




#######################################################
#                                                     #
# routine for PCA decomposition prior to ICA          #
#                                                     #
#######################################################
def whitening(data, dim_reduction='',
              npc=None, explainedVar=1.0):

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
    from sklearn.decomposition import RandomizedPCA
    import dimension_selection as dim_sel


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
    whiten = False
    n_components = None if dim_reduction == '' else npc
    dmean = X.mean(axis=0)
    stddev = np.std(X, axis=0)
    X = (X - dmean[np.newaxis, :]) / stddev[np.newaxis, :]

    pca = RandomizedPCA(n_components=n_components, whiten=whiten,
                        copy=True)

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
        npc, _ = dim_sel.aic_mdl(pca.explained_variance_)
    elif dim_reduction == 'BIC':
        npc = dim_sel.mibs(pca.explained_variance_, ntsl,
                           use_bic=True)
    elif dim_reduction == 'GAP':
        npc = dim_sel.gap(pca.explained_variance_)
    elif dim_reduction == 'MDL':
        _, npc = dim_sel.aic_mdl(pca.explained_variance_)
    elif dim_reduction == 'MIBS':
        npc = dim_sel.mibs(pca.explained_variance_, ntsl,
                           use_bic=False)
    elif dim_reduction == 'explVar':
        # compute explained variance manually
        explained_variance_ratio_ = pca.explained_variance_
        explained_variance_ratio_ /= explained_variance_ratio_.sum()
        npc = np.sum(explained_variance_ratio_.cumsum() <= explainedVar)
    elif npc is None:
        npc = nchan

    # return results
    return whitened_data[:, :(npc+1)], pca




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
        print 'computing%sInfomax ICA' % ' Extended ' if extended is True else ' '

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
                from sys import stdout
                info = "\r" if iter > 0 else ""
                info += ">>> Step %4d of %4d; wchange: %1.4e" % (step+1, max_iter, change)
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
                    print '... lowering learning rate to %g \n... re-starting...' % l_rate
            else:
                raise ValueError('Error in Infomax ICA: unmixing_matrix matrix'
                                 'might not be invertible!')


    # prepare return values
    return weights.T

