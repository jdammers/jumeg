# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.fourier_ica --------------------------------------
----------------------------------------------------------------------
 author     : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 13.11.2015
 version    : 1.0

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

 mibs() --> Routine to estimate the MInka Bayesian model
    Selection (MIBS) value; can also be used to perform
    model order selection based on the Bayesian Information
    Criterion (BIC)

 gap() --> Routine to estimate the model order using
    the GAP value

 aic_mdl() --> Routine to estimate the model order using
    the Akaike's information criterion (AIC) or the
    minimum description length (MDL) criterion.

----------------------------------------------------------------------
"""

# ------------------------------------------
# import necessary modules
# ------------------------------------------
import numpy as np


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  method to estimate the optimal data dimension for ICA
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def mibs(eigenvalues, n_samples, use_bic=False):

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
            use_bic: if set the BIC-method is used instead
                of MIBS to estimate the optimal dimension

            Returns
            -------
            pca_dim: optimal data dimension
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
    bic_val = np.zeros(m)
    log_pi = np.log(np.pi)
    log_2pi = np.log(2.0 * np.pi)
    log_N = np.log(N)

    # ------------------------------------------
    # loop over all possible ranks
    # ------------------------------------------
    for n in range(1, m):

        # ------------------------------------------
        # define some variables for MIBS and BIC
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
        # estimate the MIBS/BIC value
        # ------------------------------------------
        mibs_val[n] = p_n - 0.5 * N * prod_lambda - N * (m-n) * np.log(sigma) - \
                      0.5 * A_n + 0.5*(d_n+n) * log_2pi - 0.5 * n * log_N
        bic_val[n] = - 0.5 * N * prod_lambda - N * (m-n) * np.log(sigma) - 0.5*(d_n+n) * log_N


    # ------------------------------------------
    # get index of maximum MIBS/BIC value
    # ------------------------------------------
    max_bic = bic_val.argmax()
    max_mibs = mibs_val.argmax()

    if use_bic:
        pca_dim = max_bic
    else:
        pca_dim = max_mibs

    return pca_dim



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  method to estimate the optimal data dimension for ICA
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
    # get index of maximum GAP value
    # ------------------------------------------
    pca_dim = gap_values.argmin()


    return pca_dim




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  method to estimate the optimal data dimension for ICA
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def aic_mdl(eigenvalues):

    """
    Routine to estimate the model order using
    the Akaike's information criterion (AIC) or the
    minimum description length (MDL) criterion. For
    detailed information see:
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
            mdl_dim: optimal data dimension based on the MDL
                method
    """

    # ------------------------------------------
    # check input parameter
    # ------------------------------------------
    neig = len(eigenvalues)
    aic = np.ones((neig))
    mdl = np.ones((neig))


    # ------------------------------------------
    # loop over all eigenvalues to estimate AIC
    # and MDL values
    # ------------------------------------------
    for idx in range(1, neig):
        log_rho = np.mean(np.log(eigenvalues[idx:])) - np.log(np.mean(eigenvalues[idx:]))
        aic[idx] = -2.0 * neig * (neig - idx + 1) * log_rho + 2.0 * (idx + 1) * (2.0 * neig - idx + 1)
        mdl[idx] = -1.0 * neig * (neig - idx + 1) * log_rho + 0.5 * (idx + 1) * (2.0 * neig - idx + 1) * np.log(neig)


    # ------------------------------------------
    # get index of minimum AIC/MDL value
    # ------------------------------------------
    aic_dim = aic[1:].argmin() + 1
    mdl_dim = mdl[1:].argmin() + 1

    return aic_dim, mdl_dim
