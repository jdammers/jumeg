# Authors: Lukas Breuer <l.breuer@fz-juelich.de>
'''
Created on 28.11.2014

@author: lbreuer
'''

# ToDo: Integrate interface to also run FastICA on a data array

#######################################################
#                                                     #
#              import necessary modules               #
#                                                     #
#######################################################
import numpy as np
from mne.preprocessing import infomax_



#######################################################
#                                                     #
#   interface to perform (extended) Infomax ICA on    #
#                    a data array                     #
#                                                     #
#######################################################
def infomax_array(data_orig, explainedVar=1.0, overwrite=None,
                  max_pca_components=None, weights=None,
                  lrate=None, block=None, wchange=1e-16,
                  annealdeg=60., annealstep=0.9, extended=None,
                  n_subgauss=1, kurt_size=6000, maxsteps=200,
                  verbose=True):

    """
    interface to perform (extended) Infomax ICA on a data array

        Parameters
        ----------
        data_orig : array of data to be decomposed [nchan, ntsl].
        explainedVar : float
            Value between 0 and 1; components will be selected by the
            cumulative percentage of explained variance.
        overwrite : if set the data array will be overwritten
            (this saves memory)
            default: overwrite=None
        max_pca_components : int | None
            The number of components used for PCA decomposition. If None, no
            dimension reduction will be applied and max_pca_components will equal
            the number of channels supplied on decomposing data.

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
        wchange : iteration stops when weight changes is smaller then this
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
    if verbose:
        print "     ... perform centering and whitening ..."
    data, pca = whitening(data, npc=max_pca_components, explainedVar=explainedVar)


    # -------------------------------------------
    # now call (extended) Infomax ICA
    # -------------------------------------------
    weights = infomax_.infomax(data, weights=weights, l_rate=lrate, block=block,
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
def whitening(data, npc=None, explainedVar=None):

    """
    routine to perform whitening prior to Infomax ICA application
    (whitening is based on Principal Component Analysis from the
    RandomizedPCA package from sklearn.decomposition)

        Parameters
        ----------
        X : data array [nchan, ntsl] for decomposition.
        npc : int | None
            The number of components used for PCA decomposition. If None, no
            dimension reduction will be applied and max_pca_components will equal
            the number of channels supplied on decomposing data.
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


    # -------------------------------------------
    # check input data
    # -------------------------------------------
    nchan, ntsl = data.shape

    if (nchan < 2) or (ntsl < nchan):
        raise ValueError('Data size too small!')


    # -------------------------------------------
    # perform PCA decomposition
    # -------------------------------------------
    X = data.copy()
    whiten = False
    n_components = npc
    dmean = X.mean(axis=-1)
    stddev = np.std(X, axis=-1)
    X = (X - dmean[:, np.newaxis]) / stddev[:, np.newaxis]


    pca = RandomizedPCA(n_components=n_components, whiten=whiten,
                        copy=True)

    full_var = np.var(X, axis=1).sum()


    # -------------------------------------------
    # perform whitening
    # -------------------------------------------
    whitened_data = pca.fit_transform(X.T)


    # -------------------------------------------
    # update PCA structure
    # -------------------------------------------
    pca.mean_ = dmean
    pca.stddev_ = stddev

    # -------------------------------------------
    # check explained variance
    # -------------------------------------------
    if explainedVar:
        # compute explained variance manually
        explained_variance_ratio_ = pca.explained_variance_ / full_var
        npc = np.sum(explained_variance_ratio_.cumsum() <= explainedVar)
    elif npc is None:
        npc = nchan

    # return results
    return whitened_data[:, :(npc+1)], pca



