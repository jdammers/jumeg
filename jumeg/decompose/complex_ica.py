# Authors: Lukas Breuer <l.breuer@fz-juelich.de>
'''
Created on 31.03.2015

@author: lbreuer
'''


#######################################################
#                                                     #
#              import necessary modules               #
#                                                     #
#######################################################
import numpy as np
import scipy as sc
import types


#######################################################
#                                                     #
#              some general functions                 #
#                                                     #
#######################################################
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  function to estimate the covariance matrix of complex
#  input data. Note that the numpy.cov() function is not
#         working properly for complex input data!
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def cov(m, y=None, rowvar=1, bias=0, ddof=None):

    """
    Estimate a covariance matrix, given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        form as that of `m`.
    rowvar : int, optional
        If `rowvar` is non-zero (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : int, optional
        Default normalization is by ``(N - 1)``, where ``N`` is the number of
        observations given (unbiased estimate). If `bias` is 1, then
        normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        .. versionadded:: 1.5
        If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
        the number of observations; this overrides the value implied by
        ``bias``. The default value is ``None``.

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.
    """

    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    if len(m.shape) == 2:
        tmp = m[0, 0]
    else:
        tmp = m[0]

    if isinstance(tmp, complex):
        dtype = complex
    else:
        dtype = float


    X = np.array(m, ndmin=2, dtype=dtype)
    if X.size == 0:
        # handle empty arrays
        return np.array(m)
    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        axis = 0
        tup = (slice(None), np.newaxis)
    else:
        axis = 1
        tup = (np.newaxis, slice(None))


    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=dtype)
        X = np.concatenate((X, y), axis)

    X -= X.mean(axis=1-axis)[tup]
    if rowvar:
        N = X.shape[1]
    else:
        N = X.shape[0]

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0
    fact = float(N - ddof)

    if not rowvar:
        return (np.dot(X.T, X.conj()) / fact).squeeze()
    else:
        return (np.dot(X, X.T.conj()) / fact).squeeze()



#######################################################
#                                                     #
#             complex ICA implementation              #
#                                                     #
#######################################################
def complex_ica(data, complex_mixing=True,
                already_normalized=False, pca_dim=0.95,
                ica_dim=200, zero_tolerance=1e-7,
                conv_eps=1e-7, max_iter=10000, lrate=0.1,
                whiten_mat=[], dewhiten_mat=[],
                cost_function='g2', envelopeICA=False,
                verbose=True, pca_only=False,
                overwrite=False):

    """
    This is simple python code for computing FastICA on
    complex valued signals

        Parameters
        ----------
        data: array of data to be decomposed [nchan, ntsl].
        complex_mixing: if mixing matrix should be real or complex
            default: complex_mixing=True
        pca_dim: the number of PCA components used to apply FourierICA.
            If pca_dim > 1 this refers to the exact number of components.
            If between 0 and 1 pca_dim refers to the variance which
            should be explained by the chosen components
            default: pca_dim=0.95
        zero_tolerance: threshold for eigenvalues to be considered (when
            applying PCA). All eigenvalues smaller than this threshold are
            discarded
            default: zero_tolerance=1e-7
        conv_eps: iteration stops when weight changes are smaller
            then this number
            default: conv_eps = 1e-16
        max_iter: maximum number od iterations used in FourierICA
            default: max_iter=10000
        lrate: initial learning rate
            default: lrate=1.0
        whiten_mat: if both whiten_mat and dewhiten_mat are set PCA is
            applied using the given matrices (i.e., no whitening and
            de-whitening matrices must be estimated). Important when
            ICASSO should be used to reduce the calculation time.
        dewhiten_mat: if both whiten_mat and dewhiten_mat are set PCA is
            applied using the given matrices (i.e., no whitening and
            de-whitening matrices must be estimated). Important when
            ICASSO should be used to reduce the calculation time.
        cost_function: can be one of the following:
            'g1': g_1(y) = 1 / (2 * np.sqrt(lrate + y))
            'g2': g_2(y) = 1 / (lrate + y)
            'g3': g_3(y) = y
        envelopeICA: if set ICA is estimated on the envelope of the
            complex input data, i.e., the mixing model is |x|=As
            default: envelopeICA=False
        verbose: bool, str, int, or None
            If not None, override default verbose level
            (see mne.verbose).
            default: verbose=True

        Returns
        -------
        W_orig: estimated de-mixing matrix
        A_orig: estimated mixing matrix
        S: decomposed signal (Fourier coefficients)
        dmean: mean of the input data (in Fourier space)
        objective: values of the objectiv function used to
            sort the input data
        whiten_mat: whitening matrix
        dewhiten_mat: dewhitening matrix
    """

    # -----------------------------------
    # copy and check input data
    # -----------------------------------
    # extract some parameter
    if not overwrite:
        origdata = data.copy()

    ntsl, nchan = data.shape


    # -----------------------------------
    # check if ICA should be estimated
    # on the envelope of the data
    # -----------------------------------
    if envelopeICA:
        complex_mixing = False
        if already_normalized:
            dmean = np.zeros((1, nchan))
            dstddev = np.ones((1, nchan))
        else:
            dmean = np.mean(data, axis=0).reshape((1, nchan))
            dstddev = np.std(data, axis=0).reshape((1, nchan))

        if isinstance(data[0, 0], complex):
            data = np.abs(data)
        else:
            print(">>> WARNING: Input data are not complex, i.e., ICA on data envelope cannot be performed.")
            print(">>>          Instead standard ICA is performed.")

        dmean_abs = np.mean(data, axis=0).reshape((1, nchan))
        dstddev_abs = np.std(data, axis=0).reshape((1, nchan))
        data = (data - np.dot(np.ones((ntsl, 1)), dmean_abs)) / np.dot(np.ones((ntsl, 1)), dstddev_abs)

    elif already_normalized:
        dmean = np.zeros((1, nchan))
        dstddev = np.ones((1, nchan))
    else:
        # -----------------------------------
        # subtract mean values from channels
        # -----------------------------------
        dmean = np.mean(data, axis=0).reshape((1, nchan))
        dstddev = np.std(data, axis=0).reshape((1, nchan))
        data = (data - np.dot(np.ones((ntsl, 1)), dmean)) / np.dot(np.ones((ntsl, 1)), dstddev)


    # -----------------------------------
    # do PCA (on data matrix)
    # -----------------------------------
    # check if whitening and de-whitening matrices
    if np.any(whiten_mat) and np.any(dewhiten_mat):
        pca_dim = whiten_mat.shape[0]
        ica_dim = pca_dim

    else:
        if nchan > 1000:
            print(">>> Launching PCA...due to data size this might take a while...")

        covmat = cov(data, rowvar=0)

        # check if complex mixing is assumed
        if not complex_mixing:
            covmat = covmat.real

        Dc, Ec = sc.linalg.eig(covmat)
        idx_sort = np.argsort(Dc)[::-1]
        Dc = Dc[idx_sort]

        if not complex_mixing:
            Dc = Dc.real
            Ec = Ec.real

        # ------------------------------
        # perform model order selection
        # ------------------------------
        # --> can either be performed by
        #     (1.) MIBS                  (self.pca_dim = None)
        #     (2.) explained variance    (0 < self.pca_dim < 1)
        #     (3.) or a fix number       (self.pca_dim > 1)
        if not pca_dim:
            from .dimension_selection import mibs
            pca_dim = mibs(Dc.real, ntsl)
        elif np.abs(pca_dim) <= 1.0:
            # estimate explained variance
            explVar = np.abs(Dc.copy())
            explVar /= explVar.sum()
            pca_dim = np.sum(explVar.cumsum() <= np.abs(pca_dim)) + 1
        else:
            pca_dim = np.abs(pca_dim)

        # checks for negativ eigenvalues
        if any(Dc[0:pca_dim] < 0):
            print(">>> WARNING: Negative eigenvalues! Reducing PCA and ICA dimension...")

        # check for eigenvalues near zero (relative to the maximum eigenvalue)
        zero_eigval = np.sum((Dc[0:pca_dim]/Dc[0]) < zero_tolerance)

        # adjust dimensions if necessary (because zero eigenvalues were found)
        pca_dim -= zero_eigval
        if pca_dim < ica_dim:
            ica_dim = pca_dim

        if verbose:
            print(">>> PCA dimension is %d and ICA dimension is %d" % (pca_dim, ica_dim))


        # construct whitening and dewhitening matrices
        Dc_sqrt = np.sqrt(Dc[0:pca_dim])
        Dc_sqrt_inv = 1.0 / Dc_sqrt
        Ec = Ec[:, idx_sort[0:pca_dim]]
        whiten_mat = np.dot(np.diag(Dc_sqrt_inv), Ec.conj().transpose())
        dewhiten_mat = np.dot(Ec, np.diag(Dc_sqrt))

    # reduce dimensions and whiten data. |Zmat_c| is the
    # main input for the iterative algorithm
    Zmat_c = np.dot(whiten_mat, data.transpose())
    Zmat_c_tr = Zmat_c.conj().transpose()     # also used in the fixed-point iteration

    # check if only PCA should be performed
    if pca_only:
        # return explained variance as objective function
        objective = np.abs(Dc.copy())
        objective /= objective.sum()
        return whiten_mat, dewhiten_mat, Zmat_c, dmean, dstddev, objective, whiten_mat, dewhiten_mat


    # ----------------------------------------------------------------
    # COMPLEX-VALUED FAST_ICA ESTIMATION
    # ----------------------------------------------------------------
    if verbose and complex_mixing:
        print("... Launching complex-valued FastICA:")
    elif verbose:
        print("... Launching FastICA:")

    # initial point, make it imaginary and unitary
    if complex_mixing:
        W_old = np.random.randn(ica_dim, pca_dim) + np.random.randn(ica_dim, pca_dim) * 1j
    else:
        W_old = np.random.randn(ica_dim, pca_dim)


    W_old = np.dot(sc.linalg.sqrtm(np.linalg.inv(np.dot(W_old, W_old.conj().transpose()))), W_old)

    # iteration start here
    for iter in range(0, max_iter):

        # compute outputs, note lack of conjugate
        Y = np.dot(W_old, Zmat_c)
        Y2 = np.abs(Y * Y.conj())

        # # compute nonlinearities
        if cost_function == 'g1':
            gY = 1.0/(2.0 * np.sqrt(lrate + Y2))
            dmv = np.sum(((2.0 * lrate + Y2)/(4.0 * (lrate + Y2)**1.5)), axis=1)

        elif cost_function == 'g3':
            gY = Y2
            dmv = np.sum((2.0 * Y2), axis=1)

        elif cost_function == 'sigmoidal':
            gY = 1.0 / (1.0 + np.exp(-Y2))
            dmv = np.sum(((1.0 + (1.0 + Y2) * np.exp(-Y2))*gY**2), axis=1)

        else:
            gY = 1.0 / (lrate + Y2)
            dmv = lrate * np.sum(gY**2, axis=1)

        # fixed-point iteration
        W_new = np.dot((Y * gY), Zmat_c_tr) - np.dot(np.diag(dmv), W_old)

        # in case we want to restrict W to be real-valued, do it here
        if complex_mixing:
            W = W_new
        else:
            W = W_new.real

        # make unitary
        W = np.dot(sc.linalg.sqrtm(np.linalg.inv(np.dot(W, W.conj().transpose()))), W)

        # check if converged
        conv_criterion = 1.0 - np.sum(np.abs(np.sum(W * W_old.conj(), axis=1)))/ica_dim
        if conv_criterion < conv_eps:
            break

        if verbose:
            from sys import stdout
            info = "\r" if iter > 0 else ""
            info += ">>> Step %4d of %4d; wchange: %1.4e" % (iter+1, max_iter, conv_criterion)
            stdout.write(info)
            stdout.flush()


        # store old value
        W_old = W


    # compute mixing matrix (in whitened space)
    A = W.conj().transpose()

    # compute source signal estimates
    S = np.dot(W, Zmat_c)

    # tell if convergence problems
    if conv_criterion > conv_eps:
        print("\nWARNING: Failed to converge, results may be wrong!")
    else:
        if verbose:
            print("\n>>> Converged!")


    # ----------------------------------------------------------------
    # SORT COMPONENTS AND TRANSFORMS TO ORIGINAL SPACE
    # ----------------------------------------------------------------
    if verbose:
        print("... Sorting components and reformatting results.")

    # compute objective function for each component
    objective = -np.mean(np.log(lrate + np.abs(S * S.conj())), axis=1)

    # sort components using the objective
    comp_order = np.argsort(objective)[::-1]
    objective = objective[comp_order]
    W = W[comp_order, :]
    A = A[:, comp_order]
    S = S[comp_order, :]

    # compute mixing and de-mixing matrix in original channel space
    # Spatial filters
    W_orig = np.dot(W, whiten_mat)

    # Spatial patterns
    A_orig = np.dot(dewhiten_mat, A)


    if verbose:
        print("... Done!")

    if not overwrite:
        data = origdata

    return W_orig, A_orig, S, dmean, dstddev, objective, whiten_mat, dewhiten_mat
