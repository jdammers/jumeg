# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.fourier_ica --------------------------------------
----------------------------------------------------------------------
 authors:
            Juergen Dammers
            Lukas Breuer
 email: j.dammers@fz-juelich.de

 Change history:
 30.10.2019: bug fix: gap, mibs, and bic now return rank not index
 17.10.2019: added rank estimation using PCA, FA in a cross-validation scenario
 25.09.2019: separated functions that were combined

----------------------------------------------------------------------
 The implementation of the following methods for automated
 selection of the optimal data dimensionality is based on
 following publications
----------------------------------------------------------------------

 A. Cichocki,  &  S. Amari, 2002. "Adaptive Blind Signal and Image
 Processing - Learning Algorithms and Applications,"
 John Wiley & Sons

 T. P. Minka, 'Automatic choice of dimensionality
 for PCA', MIT Press (2001)

 Z. He, A. Cichocki, S. Xie and K. Choi, "Detecting the number
 of clusters in n-way probabilistic clustering," IEEE Trans.
 Pattern Anal. Mach. Intell., vol. 32, pp. 2006-2021, Nov, 2010.

 M. Wax, and T. Kailath, "Detection of signals by
 information-theoretic criteria," IEEE Trans. on Acoustics,
 vol. 33, pp. 387-392, 1985.


----------------------------------------------------------------------
 Overview
----------------------------------------------------------------------

All methods are based on the eigenvalues you get from the
eigenvalue decomposition of the data covariance matrix.

All methods try to estimate the optimal data dimension:
 - aic():  Akaike's information criterion
 - bic():  Bayesian Information Criteria
 - mibs(): MInka Bayesian model Selection
 - mdl():  Minimum description length
 - gap():  probabilistic clustering
 - explVar(): explained variance
----------------------------------------------------------------------
"""

# ------------------------------------------
# import necessary modules
# ------------------------------------------
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  AIC - Akaike's information criterion
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def aic(eigenvalues):

    """
    Routine to estimate the model order using
    the Akaike's information criterion (AIC)
    For detailed information see:
    M. Wax, and T. Kailath,
    "Detection of signals by information-theoretic
    criteria," IEEE Trans. on Acoustics,
    vol. 33, pp. 387-392, 1985.

            Parameters
            ----------
            eigenvalues: eigenvalues received when applying
                PCA. Note eigenvalues must be sorted decreasing

            Returns
            -------
            aic_dim: optimal data dimension based on the AIC
                method
    """

    # ------------------------------------------
    # check input parameter
    # ------------------------------------------
    neig = len(eigenvalues)
    aic = np.ones((neig))


    # ------------------------------------------
    # loop over all eigenvalues to estimate AIC
    # ------------------------------------------
    for idx in range(1, neig):
        log_rho = np.mean(np.log(eigenvalues[idx:])) - np.log(np.mean(eigenvalues[idx:]))
        aic[idx] = -2.0 * neig * (neig - idx + 1) * log_rho + 2.0 * (idx + 1) * (2.0 * neig - idx + 1)


    # ------------------------------------------
    # get rank of minimum AIC value
    # ------------------------------------------
    aic_dim = aic[1:].argmin() + 1

    return aic_dim


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  BIC - Bayesian Information Criteria
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def bic(eigenvalues, n_samples):

    """
    Routine to estimate the Baysian Information Criteria

            Parameters
            ----------
            eigenvalues: eigenvalues received when applying
                PCA. Note eigenvalues must be sorted decreasing
            n_samples: number of samples/ time slices used to
                estimate the covariance matrix for PCA

            Returns
            -------
            bic: optimal data dimension
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from math import gamma

    # ------------------------------------------
    # set variables to be confirm with notation
    # in Chichocki and Amari, 'Adaptive Blind
    # Signal And Image Processing', (2006), p.93
    # ------------------------------------------
    N = n_samples
    m = len(eigenvalues)
    bic_val = np.zeros(m)
    log_pi = np.log(np.pi)
    log_2pi = np.log(2.0 * np.pi)
    log_N = np.log(N)

    # ------------------------------------------
    # loop over all possible ranks
    # ------------------------------------------
    for n in range(1, m):

        # ------------------------------------------
        # define some variables for BIC
        #------------------------------------------
        sigma = np.mean(eigenvalues[n:])
        d_n = m*n - 0.5*n*(n+1)
        p_n = -n * np.log(2.0)
        A_n = 0.0
        prod_lambda = np.sum(np.log(eigenvalues[:n]))
        eigenvalues_tmp = eigenvalues.copy()
        eigenvalues_tmp[n:] = sigma


        # ------------------------------------------
        # estimate p_n and A_n
        # ------------------------------------------
        # loop over n
        for idx in range(n):
            p_n += np.log(gamma(0.5*(m-idx))) - (0.5*(m-idx) * log_pi)

            for j in range(idx+1, m):
                A_n += np.log(eigenvalues_tmp[idx] - eigenvalues_tmp[j]) +\
                       np.log(eigenvalues_tmp[j]) + np.log(eigenvalues_tmp[idx]) + \
                       log_N + np.log(eigenvalues[idx]-eigenvalues[j])

        # ------------------------------------------
        # estimate the BIC value
        # ------------------------------------------
        bic_val[n] = - 0.5 * N * prod_lambda - N * (m-n) * np.log(sigma) - 0.5*(d_n+n) * log_N


    # ------------------------------------------
    # get rank of maximum BIC value
    # ------------------------------------------
    max_bic = bic_val[1:].argmax() + 1

    return max_bic


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  MIBS - MInka Bayesian model Selection
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mibs(eigenvalues, n_samples):

    """
    Routine to estimate the MInka Bayesian model
    Selection (MIBS) value as introduced in:
    T. P. Minka, 'Automatic choice of dimensionality
    for PCA', MIT Press (2001)

    Note: For numerical stability here ln(MIBS) is
    estimated instead of MIBS

            Parameters
            ----------
            eigenvalues: eigenvalues received when applying
                PCA. Note eigenvalues must be sorted decreasing
            n_samples: number of samples/ time slices used to
                estimate the covariance matrix for PCA

            Returns
            -------
            mibs: optimal data dimension
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from math import gamma

    # ------------------------------------------
    # set variables to be confirm with notation
    # in Chichocki and Amari, 'Adaptive Blind
    # Signal And Image Processing', (2006), p.93
    # ------------------------------------------
    N = n_samples
    m = len(eigenvalues)
    mibs_val = np.zeros(m)
    log_pi = np.log(np.pi)
    log_2pi = np.log(2.0 * np.pi)
    log_N = np.log(N)

    # ------------------------------------------
    # loop over all possible ranks
    # ------------------------------------------
    for n in range(1, m):

        # ------------------------------------------
        # define some variables for MIBS
        #------------------------------------------
        sigma = np.mean(eigenvalues[n:])
        d_n = m*n - 0.5*n*(n+1)
        p_n = -n * np.log(2.0)
        A_n = 0.0
        prod_lambda = np.sum(np.log(eigenvalues[:n]))
        eigenvalues_tmp = eigenvalues.copy()
        eigenvalues_tmp[n:] = sigma


        # ------------------------------------------
        # estimate p_n and A_n
        # ------------------------------------------
        # loop over n
        for idx in range(n):
            p_n += np.log(gamma(0.5*(m-idx))) - (0.5*(m-idx) * log_pi)

            for j in range(idx+1, m):
                A_n += np.log(eigenvalues_tmp[idx] - eigenvalues_tmp[j]) +\
                       np.log(eigenvalues_tmp[j]) + np.log(eigenvalues_tmp[idx]) + \
                       log_N + np.log(eigenvalues[idx]-eigenvalues[j])

        # ------------------------------------------
        # estimation of MIBS
        # ------------------------------------------
        mibs_val[n] = p_n - 0.5 * N * prod_lambda - N * (m-n) * np.log(sigma) - \
                      0.5 * A_n + 0.5*(d_n+n) * log_2pi - 0.5 * n * log_N


    # ------------------------------------------
    # get rank of maximum MIBS value
    # ------------------------------------------
    max_mibs = mibs_val[1:].argmax() + 1

    return max_mibs


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  MDL - Minimum description length
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mdl(eigenvalues):

    """
    Routine to estimate the model order using the
    minimum description length (MDL) criterion.
    For detailed information see:
    M. Wax, and T. Kailath,
    "Detection of signals by information-theoretic
    criteria," IEEE Trans. on Acoustics,
    vol. 33, pp. 387-392, 1985.

            Parameters
            ----------
            eigenvalues: eigenvalues received when applying
                PCA. Note eigenvalues must be sorted decreasing

            Returns
            -------
            mdl_dim: optimal data dimension based on the MDL
                method
    """

    # ------------------------------------------
    # check input parameter
    # ------------------------------------------
    neig = len(eigenvalues)
    mdl = np.ones((neig))


    # ------------------------------------------
    # loop over all eigenvalues to estimate MDL
    # ------------------------------------------
    for idx in range(1, neig):
        log_rho = np.mean(np.log(eigenvalues[idx:])) - np.log(np.mean(eigenvalues[idx:]))
        mdl[idx] = -1.0 * neig * (neig - idx + 1) * log_rho + 0.5 * (idx + 1) * (2.0 * neig - idx + 1) * np.log(neig)


    # ------------------------------------------
    # get rank of minimum MDL value
    # ------------------------------------------
    mdl_dim = mdl[1:].argmin() + 1

    return mdl_dim


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  GAP
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def gap(eigenvalues):

    """
    Routine to estimate the model order using
    the GAP value as introduced in:
    Z. He, A. Cichocki, S. Xie and K. Choi,
    "Detecting the number of clusters in n-way
    probabilistic clustering,"
    IEEE Trans. Pattern Anal. Mach. Intell.,
    vol. 32, pp. 2006-2021, Nov, 2010.


            Parameters
            ----------
            eigenvalues: eigenvalues received when applying
                PCA. Note eigenvalues must be sorted decreasing

            Returns
            -------
            pca_dim: optimal data dimension
    """

    # ------------------------------------------
    # check input parameter
    # ------------------------------------------
    neig = len(eigenvalues)
    gap_values = np.ones((neig))


    # ------------------------------------------
    # loop over all eigenvalues
    # ------------------------------------------
    for idx in range(0, (neig-2)):
        temp = np.mean(eigenvalues[idx+1:])
        gap_values[idx] = (eigenvalues[idx+1] - temp) / (eigenvalues[idx] - temp)  # np.var(eigenvalues[idx+1:])/np.var(eigenvalues[idx:])


    # ------------------------------------------
    # get rank of maximum GAP value
    # ------------------------------------------
    pca_dim = gap_values.argmin() + 1


    return pca_dim



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Explained Variance
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def explVar(eigenvalues, explainedVar=0.95):

    """
    Routine to estimate the model order using
    the explained variance

            Parameters
            ----------
            eigenvalues: eigenvalues received when applying
                PCA. Note eigenvalues must be sorted decreasing

             explainedVar: for which we would like to know the
                number of components

            Returns
            -------
            pca_dim: optimal data dimension
    """

    explained_variance_ratio = (eigenvalues/eigenvalues.sum()).cumsum()
    pca_dim = np.sum(explained_variance_ratio <= explainedVar)

    return pca_dim


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# estimate rank from largest PCA score using cross-validation
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def pca_rank_cv(data, n_comp_list, cv=5, whiten=True):
    """
    estimate rank from largest PCA score using cross-validation
     - data: must be of shape [n_chan, n_times] = [n_features, n_samples]
     - whiten: applies rank estimation on whitened data (default: True)
    based on: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    """
    pca = PCA(svd_solver='auto', whiten=whiten)
    pca_scores = []
    for n in n_comp_list:
        pca.n_components = np.int(n)
        pca_scores.append(np.mean(cross_val_score(pca, data.T, cv=cv)))
    n_components_pca = n_comp_list[np.argmax(pca_scores)]

    return n_components_pca


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# estimate rank from largest Factor Analysis score using cross-validation
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fa_rank_cv(data, n_comp_list, cv=5):
    """
    estimates rank from largest Factor Analysis score using cross-validation
     - data: must be of shape [n_chan, n_times] = [n_features, n_samples]
             if rank estimation should be performed on whitened data, you need
             to apply whitening before
    based on: https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    """
    fa = FactorAnalysis()
    fa_scores = []
    for n in n_comp_list:
        fa.n_components = np.int(n)
        fa_scores.append(np.mean(cross_val_score(fa, data.T, cv=cv)))
    n_components_fa = n_comp_list[np.argmax(fa_scores)]

    return n_components_fa



