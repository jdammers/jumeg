#-*- coding: utf-8 -*-
'''utility functions for causality analysis'''

import math
import numpy as np


def _tsdata_to_var(X, p):
    """
    Calculate coefficients and recovariance and noise covariance of
    the optimized model order.
    ref: http://users.sussex.ac.uk/~lionelb/MVGC/html/tsdata_to_var.html
    Parameters
    ----------
    X: narray, shape (n_sources, n_times, n_epochs)
          The data to estimate the model order for.
    p: int, the optimized model order.
    Returns
    ----------
    A: array, coefficients of the specified model
    SIG:array, recovariance of this model
    E:  array, noise covariance of this model
    """
    n, m, N = X.shape
    p1 = p + 1
    A = np.nan
    SIG = np.nan
    E = np.nan
    q1n = p1 * n
    I = np.eye(n)
    XX = np.zeros((n, p1, m + p, N))
    for k in range(p1):
        XX[:, k, k:k + m, :] = X
    AF = np.zeros((n, q1n))
    AB = np.zeros((n, q1n))
    k = 1
    kn = k * n
    M = N * (m - k)
    kf = list(range(0, kn))
    kb = list(range(q1n - kn, q1n))
    XF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
    XB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
    CXF = np.linalg.cholesky(XF.dot(XF.T)).T
    CXB = np.linalg.cholesky(XB.dot(XB.T)).T
    AF[:, kf] = np.linalg.solve(CXF.T, I)
    AB[:, kb] = np.linalg.solve(CXB.T, I)

    del p1, XF, XB, CXF, CXB

    while k <= p:

        print('morder', k)

        tempF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
        af = AF[:, kf]
        EF = af.dot(tempF)

        del af, tempF

        tempB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
        ab = AB[:, kb]
        EB = ab.dot(tempB)

        del ab, tempB

        CEF = np.linalg.cholesky(EF.dot(EF.T)).T
        CEB = np.linalg.cholesky(EB.dot(EB.T)).T
        R = np.dot(np.linalg.solve(CEF.T, EF.dot(EB.T)), np.linalg.inv(CEB))

        del EB, CEF, CEB

        RF = np.linalg.cholesky(I - R.dot(R.T)).T
        RB = np.linalg.cholesky(I - (R.T).dot(R)).T
        k = k + 1
        kn = k * n
        M = N * (m - k)
        kf = np.arange(kn)
        kb = list(range(q1n - kn, q1n))
        AFPREV = AF[:, kf]
        ABPREV = AB[:, kb]
        AF[:, kf] = np.linalg.solve(RF.T, AFPREV - R.dot(ABPREV))
        AB[:, kb] = np.linalg.solve(RB.T, ABPREV - R.T.dot(AFPREV))

        del RF, RB, ABPREV
    E = np.linalg.solve(AFPREV[:, :n], EF)

    del EF, AFPREV

    SIG = (E.dot(E.T)) / (M - 1)
    E = np.reshape(E, (n, m - p, N), order='F')
    temp = np.linalg.solve(-AF[:, :n], AF[:, n:])

    del AF

    A = np.reshape(temp, (n, n, p), order='F')
    return A, SIG, E


def _erfcc(x):
    """Whiteness test. Complementary error function."""
    z = abs(x)
    t = 1. / (1. + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (.37409196 +
                     t * (.09678418 + t * (-.18628806 + t * (.27886807 +
                     t * (-1.13520398 + t * (1.48851587 + t * (-.82215223 +
                                                               t * .17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r


def _normcdf(x, mu, sigma):
    t = x - mu
    y = 0.5 * _erfcc(-t / (sigma * math.sqrt(2.0)))
    if y > 1.0:
        y = 1.0
    return y


def _durbinwatson(X, E):
    n, m = X.shape
    dw = np.sum(np.diff(E, axis=0) ** 2) / np.sum(E ** 2, axis=0)
    A = np.dot(X, X.T)
    from scipy.signal import lfilter
    B = lfilter(np.array([-1, 2, -1]), 1, X.T, axis=0)
    temp = X[:, [0, m - 1]] - X[:, [1, m - 2]]
    B[[0, m - 1], :] = temp.T
    D = np.dot(B, np.linalg.pinv(A))
    C = X.dot(D)
    nu1 = 2 * (m - 1) - np.trace(C)
    nu2 = 2 * (3 * m - 4) - 2 * np.trace(B.T.dot(D)) + np.trace(C.dot(C))
    mu = nu1 / (m - n)
    sigma = math.sqrt(2./((m - n) * (m - n + 2)) * (nu2 - nu1 * mu))
    pval = _normcdf(dw, mu, sigma)
    pval = 2 * min(pval, 1-pval)
    return dw, pval


def _whiteness(X, E):
    """
    Durbin-Watson test for whiteness (no serial correlation) of
    VAR residuals.
    Prarameters
    -----------
    X: array
      Multi-trial time series data.
    E: array
      Residuals time series.

    Returns
    -------
    dw: array
       Vector of Durbin-Watson statistics.
    pval: array
         Vector of p-values.
    """
    n, m, N = X.shape
    dw = np.zeros(n)
    pval = np.zeros(n)
    for i in range(n):
        Ei = np.squeeze(E[i, :, :])
        e_a, e_b = Ei.shape
        tempX = np.reshape(X, (n, m * N), order='F')
        tempE = np.reshape(Ei, (e_a * e_b), order='F')
        dw[i], pval[i] = _durbinwatson(tempX, tempE)
    return dw, pval


def _consistency(X, E):
    '''
    Consistency test. [1]

    Parameters
    -----------
    X: array
      Multi-trial time series data.
    E: array
      Residuals time series.
    Returns
    -------
    cons: float
    consistency test measurement.

    [1] Ding, M. et al "Short-window spectral analysis of cortical
        event-related potentials by adaptive multivariate autoregressive
        modeling: data preprocessing, model validation, and variability
        assessment." (2000), Biol. Cybern., vol. 83, 35-45
    '''
    n, m, N = X.shape
    p = m - E.shape[1]
    X = X[:, p:m, :]
    n1, m1, N1 = X.shape
    X = np.reshape(X, (n1, m1 * N1), order='F')
    E = np.reshape(E, (n1, m1 * N1), order='F')
    s = N * (m - p)
    Y = X - E
    Rr = X.dot(X.T) / (s - 1)
    Rs = Y.dot(Y.T) / (s - 1)
    cons = 1 - np.linalg.norm(Rs - Rr, 2) / np.linalg.norm(Rr, 2)
    return cons

#
# def compute_morder(X):
#     '''Compute a suitable model order for the given data.
#
#     Parameters
#     X: Data array (trials, channels, samples)
#
#     Returns
#     morder: Model order
#
#     Note: The array will be reshaped to channels, samples, trials.
#     '''
#     print('Reshaping to - (sources, samples, trials)')
#     X = X.transpose(1, 2, 0)
#     print('data is of shape ', X.shape)
#
#     print('computing the model order..')
#     p_max = 0
#     # the shape is (sources, samples, trials)
#     n, m, N = X.shape
#     if p_max == 0:
#         p_max = m - 1
#     q = p_max
#     q1 = q + 1
#     XX = np.zeros((n, q1, m + q, N))
#     for k in xrange(q1):
#         XX[:, k, k:k + m, :] = X
#     q1n = q1 * n
#     bic = np.empty((q, 1))
#     bic.fill(np.nan)
#     I = np.identity(n)
#
#     # initialise recursion
#     AF = np.zeros((n, q1n))  # forward AR coefficients
#     AB = np.zeros((n, q1n))  # backward AR coefficients
#
#     k = 1
#     kn = k * n
#     M = N * (m - k)
#     kf = range(0, kn)
#     kb = range(q1n - kn, q1n)
#     XF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
#     XB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
#
#     CXF = np.linalg.cholesky(XF.dot(XF.T)).T
#     CXB = np.linalg.cholesky(XB.dot(XB.T)).T
#     AF[:, kf] = np.linalg.solve(CXF.T, I)
#     AB[:, kb] = np.linalg.solve(CXB.T, I)
#
#     while k <= q - 1:
#         print('model order = %d' % k)
#         tempF = np.reshape(XX[:, 0:k, k:m, :], (kn, M), order='F')
#         af = AF[:, kf]
#         EF = af.dot(tempF)
#         tempB = np.reshape(XX[:, 0:k, k - 1:m - 1, :], (kn, M), order='F')
#         ab = AB[:, kb]
#         EB = ab.dot(tempB)
#         CEF = np.linalg.cholesky(EF.dot(EF.T)).T
#         CEB = np.linalg.cholesky(EB.dot(EB.T)).T
#         R = np.dot(np.linalg.solve(CEF.T, EF.dot(EB.T)), np.linalg.inv(CEB))
#         CRF = np.linalg.cholesky(I - R.dot(R.T)).T
#         CRB = np.linalg.cholesky(I - (R.T).dot(R)).T
#         k = k + 1
#         kn = k * n
#         M = N * (m - k)
#         kf = np.arange(kn)
#         kb = range(q1n - kn, q1n)
#         AFPREV = AF[:, kf]
#         ABPREV = AB[:, kb]
#         AF[:, kf] = np.linalg.solve(CRF.T, AFPREV - R.dot(ABPREV))
#         AB[:, kb] = np.linalg.solve(CRB.T, ABPREV - R.T.dot(AFPREV))
#         E = np.linalg.solve(AF[:, :n], AF[:, kf]).dot(np.reshape(XX[:, :k, k:m,
#                                                       :], (kn, M), order='F'))
#         DSIG = np.linalg.det((E.dot(E.T)) / (M - 1))
#         i = k - 1
#         K = i * n * n
#         L = -(M / 2) * math.log(DSIG)
#         bic[i - 1] = -2 * L + K * math.log(M)
#
#     # morder = np.nanmin(bic), np.nanargmin(bic) + 1
#     morder = np.nanargmin(bic) + 1
#
#     return morder
#


def do_mvar_evaluation(X, morder, whit_max=3., whit_min=1., thr_cons=0.8):
    '''
    Fit MVAR model to data using scot and do some basic checks.

    X: array (trials, channels, samples)
    morder: the model order

    Returns:
    (is_white, consistency, is_stable)
    '''
    print('starting checks and MVAR fitting...')
    # tsdata_to_var from MVGC requires sources x samples x trials
    # X is of shape trials x sources x samples (which is what ScoT uses)
    A, SIG, E = _tsdata_to_var(X.transpose(1, 2, 0), morder)
    del A, SIG

    whi = False
    dw, pval = _whiteness(X.transpose(1, 2, 0), E)
    if np.all(dw < whit_max) and np.all(dw > whit_min):
        whi = True
    cons = _consistency(X.transpose(1, 2, 0), E)
    del dw, pval, E

    from scot.var import VAR
    mvar = VAR(morder)
    mvar.fit(X)  # scot func which requires shape trials x sources x samples
    is_st = mvar.is_stable()
    if cons < thr_cons or is_st is False or whi is False:
        print('ERROR: Model order not ideal - check parameters !!')

    return str(whi), cons, str(is_st)


def prepare_causality_matrix(cau, surr, freqs, nfft, sfreq, surr_thresh=95):
    '''Prepare a final causality matrix after averaging across frequency bands.

    1. Average across freq bands.
    2. Zero the diagonals.
    3. Apply threshold based on surrogate.
    4. Return the final causality matrix.

    Parameters
    cau: array (n_nodes, n_nodes, n_freqs)
    surr: array (n_surr, n_nodes, n_nodes, n_freqs)
    nfft: the nfft parameter
    sfreq: Sampling frequency
    freqs: Frequency bands
        e.g. [(4, 8), (8, 12), (12, 18), (18, 30), (30, 45)]
    surr_thresh: Percentile value of surrogate to be applied as threshold.

    Return
    cau_array: array (n_bands, n_nodes, n_nodes)
    '''
    n_nodes, nfft = cau.shape[0], cau.shape[-1]
    delta_F = sfreq / float(2 * nfft)
    n_freqs = len(freqs)
    n_surr = surr.shape[0]

    cau_con = []  # contains list of avg. band matrices
    max_surrs, max_cons = [], []
    diag_ind = np.diag_indices(n_nodes)

    for flow, fhigh in freqs:
        print(('flow: %d, fhigh: %d' % (flow, fhigh)))
        fmin, fmax = int(flow / delta_F), int(fhigh / delta_F)
        cau_band = np.mean(cau[:, :, fmin:fmax+1], axis=-1)
        surr_band = np.mean(surr[:, :, :, fmin:fmax+1], axis=-1)

        # make the diagonals zero
        cau_band[diag_ind] = 0.
        for n in range(n_surr):
            surr_band[n][diag_ind] = 0.

        max_surrs.append(np.max(surr_band))
        max_cons.append(np.max(cau_band))
        print(max_cons)

        # get the threshold from the surrogates
        if surr_thresh:
            print(('applying %dth percentile threshold from surrogate' % surr_thresh))
            con_thresh = np.percentile(surr_band[surr_band != 0], surr_thresh)
            # apply threshold on the caus matrix
            cau_band[cau_band < con_thresh] = 0.
        else:
            con_thresh = None

        # do some sanity checks
        assert cau_band.any(), 'cau_band has zeros only !!'

        # save the final caus matrix
        cau_con.append(cau_band)

    return np.array(cau_con), np.array(max_cons), np.array(max_surrs)


def make_frequency_bands(cau, freqs, sfreq):
    '''Average connectivity/causality matrix across given frequency bands.

    Parameters
    cau: array (n_nodes, n_nodes, n_freqs)
    sfreq: Sampling frequency
    freqs: Frequency bands
        e.g. [(4, 8), (8, 12), (12, 18), (18, 30), (30, 45)]

    Return
    cau_array: array (n_bands, n_nodes, n_nodes)
    '''
    n_nodes, nfft = cau.shape[0], cau.shape[-1]
    delta_F = sfreq / float(2 * nfft)
    n_freqs = len(freqs)

    cau_con = []  # contains list of avg. band matrices
    diag_ind = np.diag_indices(n_nodes)

    for flow, fhigh in freqs:
        print(('flow: %d, fhigh: %d' % (flow, fhigh)))
        fmin, fmax = int(flow / delta_F), int(fhigh / delta_F)
        cau_band = np.mean(cau[:, :, fmin:fmax+1], axis=-1)

        # make the diagonals zero
        cau_band[diag_ind] = 0.

        # do some sanity checks
        assert cau_band.any(), 'cau_band has zeros only !!'

        # save the final caus matrix
        cau_con.append(cau_band)

    return np.array(cau_con)


def compute_order(X, m_max, verbose=True):
    """Estimate AR order with BIC

    Parameters
    ----------
    X : ndarray, shape (trials, n_channels, n_samples)

    m_max : int
        The maximum model order to test

    Reference
    ---------
    [1] provides the equation:BIC(m) = 2*log[det(Σ)]+ 2*(p**2)*m*log(N*n*m)/(N*n*m),
    Σ is the noise covariance matrix, p is the channels, N is the trials, n
    is the n_samples, m is model order.

    [1] Mingzhou Ding, Yonghong Chen. Granger Causality: Basic Theory and Application
    to Neuroscience.Elsevier Science, 7 February 2008.

    URL: https://gist.github.com/dongqunxi/b23d1679b9bffa8e458c11f93bd8d6ff


    Returns
    -------
    o_m : int
        Estimated order
    bic : ndarray, shape (m_max + 1,)
        The BIC for the orders from 1 to m_max.
    """
    import scot
    from scipy import linalg

    N, p, n = X.shape
    bic = []
    for m in range(m_max):
        mvar = scot.var.VAR(m+1)
        mvar.fit(X)
        sigma = mvar.rescov
        m_bic = np.log(linalg.det(sigma))
        m_bic += (p ** 2) * m * np.log(N*n) / (N*n)
        bic.append(m_bic)
        if verbose:
            print(('model order: %d, BIC value: %.2f' %(m+1, bic[m])))

    o_m = np.argmin(bic) + 1
    return o_m, bic


def compute_causal_outflow_inflow(caus):
    '''
    Given a causality matrix of shape (n_bands, n_nodes, n_nodes),
    the function returns the normalized causal outflow and inflow across the
    nodes.

    Input
    caus: ndarray | shape (n_bands, n_nodes, n_nodes)

    Output
    c_outflow: ndarray | shape (n_bands, n_nodes)
        Normalised causal outflow.
    c_inflow: ndarray | shape (n_bands, n_nodes)
        Normalised causal inflow.
    '''
    n_bands, n_nodes, _ = caus.shape
    c_outflow = np.zeros((n_bands, n_nodes))
    c_inflow = np.zeros((n_bands, n_nodes))

    for band in range(n_bands):
        band_ = caus[band]
        # causal outflow per ROI
        c_out_sums = band_.sum(axis=1)  # sum across columns (j's)
        c_outflow[band] = c_out_sums / np.max(c_out_sums)
        # causal inflow per ROI
        c_in_sums = band_.sum(axis=0)  # sum across rows (i's)
        c_inflow[band] = c_in_sums / np.max(c_in_sums)
    return c_outflow, c_inflow
