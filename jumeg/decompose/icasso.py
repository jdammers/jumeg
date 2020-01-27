# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.fourier_ica --------------------------------------
----------------------------------------------------------------------
 author     : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 09.11.2016
 version    : 1.2

----------------------------------------------------------------------
 This simple implementation of ICASSO is based on the following
 publication:
----------------------------------------------------------------------

 J. Himberg, A. Hyvaerinen, and F. Esposito. 'Validating the
 independent components of neuroimaging time-series via
 clustering and visualization', Neuroimage, 22:3(1214-1222), 2004.

 Should you use this code, we kindly request you to cite the
 aforementioned publication.

 <http://research.ics.aalto.fi/ica/icasso/about+download.shtml
  DOWNLOAD ICASSO from here>

----------------------------------------------------------------------
 Overview
----------------------------------------------------------------------

Perform ICASSO estimation. ICASSO is based on running ICA
multiple times with slightly different conditions and
clustering the obtained components. Note, here FourierICA
is applied

 1. Runs ICA with given parameters M times on data X.
 2. Clusters the estimates and computes other statistics.
 3. Returns (and visualizes) the best estimates.

----------------------------------------------------------------------
 How to use ICASSO?
----------------------------------------------------------------------

from jumeg.decompose import icasso

icasso_obj = = JuMEG_icasso()
W, A, quality, fourier_ica_obj = icasso_obj.fit(fn_raw, stim_name='STI 013',
                                                event_id=1, tmin_stim=-0.5,
                                                tmax_stim=0.5, flow=4.0, fhigh=34.0)

--> for further comments we refer directly to the functions or to
    fourier_ica_test.py

----------------------------------------------------------------------
"""

# ------------------------------------------
# import necessary modules
# ------------------------------------------
import numpy as np


########################################################
#                                                      #
#                  JuMEG_icasso class                  #
#                                                      #
########################################################
class JuMEG_icasso(object):

    def __init__(self, ica_method='fourierica', average=False, nrep=50,
                 fn_inv=None, src_loc_method='dSPM', snr=1.0,
                 morph2fsaverage=True, stim_name=None, event_id=1,
                 flow=4.0, fhigh=34.0, tmin_win=0.0, tmax_win=1.0,
                 pca_dim=None, dim_reduction='MDL', conv_eps=1e-9,
                 max_iter=2000, tICA=False, lrate=1.0, cost_function=None,
                 decim_epochs=False):
        """
        Generate ICASSO object.

            Parameters
            ----------
            ica_method: string which ICA method should be used
                default: ica_method='FourierICA'
            average: should ICA be performed on data averaged above
                subjects?
                default: average=False
            nrep: number of repetitions ICA should be performed
                default: nrep=50
            fn_inv: file name of inverse operator. If given
                FourierICA is applied on data transformed to
                source space
            src_loc_method: method used for source localization.
                Only of interest if 'fn_inv' is set
                default: src_loc_method='dSPM'
            snr: signal-to-noise ratio for performing source
                localization
                default: snr=1.0
            morph2fsaverage: should data be morphed to the
                'fsaverage' brain?
                default: morph2fsaverage=True
            stim_name: string which contains the name of the
                stimulus channel. Only necessary if ICA should
                be applied to evoked data.
            event_id: integer of list of integer containing the
                event IDs which should be used to generate epochs
                default: event_id=1
            flow: lower frequency border for estimating the optimal
                de-mixing matrix using FourierICA
                default: flow=4.0
            fhigh: upper frequency border for estimating the optimal
                de-mixing matrix using FourierICA
                default: fhigh=34.0
                Note: here default flow and fhigh are choosen to
                   contain:
                   - theta (4-7Hz)
                   - low (7.5-9.5Hz) and high alpha (10-12Hz),
                   - low (13-23Hz) and high beta (24-34Hz)
            tmin_win: time of interest prior to stimulus onset.
                Important for generating epochs to apply FourierICA
                default=0.0
            tmax_win: time of interest after stimulus onset.
                Important for generating epochs to apply FourierICA
                default=1.0
            dim_reduction: {'', 'AIC', 'BIC', 'GAP', 'MDL', 'MIBS', 'explVar'}
                Method for dimension selection. For further information about
                the methods please check the script 'dimension_selection.py'.
            pca_dim: Integer. The number of components used for PCA
                decomposition.
            conv_eps: iteration stops when weight changes are smaller
                then this number
                default: conv_eps = 1e-9
            max_iter: integer containing the maximal number of
                iterations to be performed in ICA estimation
                default: max_iter=2000
            tICA: bool if temporal ICA should be applied (and not)
                FourierICA
                default: tICA=False
            lrate: float containg the learning rate which should be
                used in the applied ICA algorithm
                default: lrate=1.0
            cost_function: string containg the cost-function to
                use in the appled ICA algorithm. For further information
                look in fourier_ica.py
                default: cost_funtion=None
            decim_epochs: integer. If set the number of epochs used
                to estimate the optimal demixing matrix is decimated
                to the given number.
                default: decim_epochs=False


            Returns
            -------
            object: ICASSO object
        """

        self._ica_method = ica_method
        self.average = average
        self._nrep = nrep
        self.fn_inv = fn_inv
        self.src_loc_method = src_loc_method
        self.snr = snr
        self.morph2fsaverage = morph2fsaverage
        self.whitenMat = []                     # whitening matrix
        self.dewhitenMat = []                   # de-whitening matrix
        self.W_est = []                         # de-mixing matrix
        self.A_est = []                         # mixing matrix
        self.dmean = []                         # data mean
        self.dstd = []                          # data standard-deviation
        self.stim_name = stim_name
        self.event_id = event_id
        self.flow = flow
        self.fhigh = fhigh
        self._sfreq = 0.0
        self.tmin_win = tmin_win
        self.tmax_win = tmax_win

        # ICA parameter
        self.conv_eps = conv_eps                # stopping threshold
        self.max_iter = max_iter
        self.lrate = lrate                      # learning rate for the ICA algorithm
        self.tICA = tICA                        # should temporal ICA be performed?
        self.pca_dim = pca_dim
        self.dim_reduction= dim_reduction
        self.cost_function = cost_function
        self.decim_epochs = decim_epochs


        # make sure to chose meaningful parameters
        # when not FourierICA is used
        if self.ica_method != 'fourierica':
            if conv_eps == 1e-9:
                self.conv_eps = 1e-12           # stopping threshold

            if max_iter == 2000:
                self.max_iter = 200

            if lrate == 1:
                self.lrate = None               # learning rate for the ICA algorithm




    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get maximum number of repetitions
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_nrep(self, nrep):
        self._nrep = nrep

    def _get_nrep(self):
        return int(self._nrep)

    nrep = property(_get_nrep, _set_nrep)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get ICA method
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_ica_method(self, ica_method):
        possible_methods = ['extended-infomax', 'fastica',
                            'fourierica', 'infomax']
        if ica_method in possible_methods:
            self._ica_method = ica_method
        else:
            print('WARNING: chosen ICA method does not exist!')
            print('Must be one of the following methods: ', possible_methods)
            print('But your choice was: ', ica_method)
            print('Programm stops!')
            import pdb
            pdb.set_trace()

    def _get_ica_method(self):
        return self._ica_method

    ica_method = property(_get_ica_method, _set_ica_method)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # estimate linkage between components
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _linkage(self, dis):

        # initialize some variables
        dlen, dim = dis.shape
        Md = dis.copy()
        Md += np.diag(np.ones(dlen)*np.inf)


        # ------------------------------------------
        # estimate clusters
        # ------------------------------------------
        # --> each vector is at first in its own cluster
        Z = np.zeros((dlen-1, 3)) + np.NaN
        clusters = np.arange(dlen)
        Cdist = Md.copy()


        for idx in np.arange(dlen-1):

            d_min = np.min(Cdist)
            if np.isinf(d_min):
                break                                    # no more connected clusters
            else:
                min_idx = np.argmin(np.min(Cdist, axis=0))

            c1 = np.argmin(Cdist[:, min_idx])            # cluster1
            c2 = clusters[min_idx]                       # cluster2

            # combine the two clusters
            c1_inds = (clusters == c1).nonzero()[0]      # vectors belonging to c1
            c2_inds = (clusters == c2).nonzero()[0]      # vectors belonging to c2
            c_inds = np.concatenate((c1_inds, c2_inds))  # members of the new cluster
            nc_inds = len(c_inds)

            # find bigger cluster
            if len(c2_inds) > len(c1_inds):
                c, k = c2, c1
            else:
                c, k = c1, c2

            clusters[c_inds] = c                         # update cluster info
            Z[idx, :] = [c, k, d_min]                    # save info into Z


            # ------------------------------------------
            # update cluster distances
            # ------------------------------------------
            # remove the subclusters from the cdist table
            for idxC in c_inds:
                Cdist[idxC, c_inds] = np.Inf             # distance of clusters to its members = Inf

            k_inds = c_inds[c_inds != c]                 # vector of the smallest cluster
            Cdist[k_inds, :] = np.Inf                    # set distance of the subcluster to
            Cdist[:, k_inds] = np.Inf                    # other clusters = Inf

            # update the distance of this cluster to the other clusters
            idxC = (clusters != c).nonzero()[0]

            if len(idxC) > 0:
                cl = np.unique(clusters[idxC])

                for l in cl:

                    o_inds = (clusters == l).nonzero()[0]  # indices belonging to cluster k
                    no_inds = len(o_inds)
                    vd = np.zeros((nc_inds, no_inds))
                    for ivd in range(nc_inds):
                        vd[ivd, :] = Md[c_inds[ivd], o_inds]

                    vd = vd.flatten()
                    idxvd = np.isfinite(vd).nonzero()[0]
                    nidxvd = len(idxvd)
                    sd = np.Inf if nidxvd == 0 else np.sum(vd[idxvd])/nidxvd
                    Cdist[c, l] = sd
                    Cdist[l, c] = sd

        last = Z[idx, 0]
        if np.isnan(last):
            last = Z[idx-1, 0]
            rest = np.setdiff1d(np.unique(clusters), last)
            Z[idx:dlen-2, 0] = rest.transpose()
            Z[idx:dlen-2, 1] = last
            Z[idx:dlen-2, 2] = np.Inf
            idx -= 1
        else:
            rest = []


        # ------------------------------------------
        # return values
        # ------------------------------------------
        # calculate the order of the samples
        order = np.array([last])

        # go through the combination matrix from top to down
        for k in range(idx, -1, -1):
            c_var = Z[k, 0]
            k_var = np.array([Z[k, 1]])
            idx_var = np.where(order == c_var)[0]

            if len(idx_var) == 0:
                order = np.concatenate((k_var, order))
            else:
                order = np.concatenate((order[:idx_var[0]], k_var, order[idx_var[0]:]))

        order = np.concatenate((rest, order))[::-1]

        # to maintain compatibility with Statistics Toolbox, the values
        # in Z must be yet transformed so that they are similar to the
        # output of the LINKAGE function
        Zs = Z.copy()
        current_cluster = np.array(list(range(dlen)))
        iter_stop = len(Z[:, 0])
        for idx in range(iter_stop):
            Zs[idx, 0] = current_cluster[int(Z[idx, 0])]
            Zs[idx, 1] = current_cluster[int(Z[idx, 1])]
            current_cluster[int(Z[idx, 0])] = dlen + idx
            current_cluster[int(Z[idx, 1])] = dlen + idx

        return Zs, order



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # estimate similarities
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _corrw(self):

        # get some dimension information
        npc = int(self.W_est[0].shape[0])
        nchan = int(self.W_est[0].shape[1])
        ntimes = int(len(self.W_est))

        # save estimated demixing matrices W in one matrix
        weight = np.zeros((ntimes*npc, nchan), dtype=np.complex)
        for idx in range(ntimes):
            weight[(idx*npc):((idx+1)*npc), :] = self.W_est[idx]

        weight = np.dot(weight, self.dewhitenMat)

        # normalize rows to unit length
        weight_norm = np.abs(np.sqrt(np.sum(weight*weight.conj(), axis=1))).reshape((npc*ntimes, 1))
        weight /= np.repeat(weight_norm, npc, axis=1)

        # compute similarities
        similarities = np.abs(np.dot(weight, weight.conj().transpose()))
        similarities[similarities > 1] = 1
        similarities[similarities < 0] = 0

        return similarities



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # generate partitions from Z
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _z_to_partition(self, Z):

        nz = Z.shape[0] + 1
        C = np.zeros((nz, nz))
        C[0, :] = np.arange(nz)

        for ic in range(1, nz):
            C[ic, :] = C[ic-1, :]
            idx = (C[ic, :] == Z[ic-1, 0]) + (C[ic, :] == Z[ic-1, 1])
            C[ic, idx == 1] = nz - 1 + ic

        for ic in range(nz):
            uniqC = np.unique(C[ic, :])
            newidx = []
            for elemC in C[ic, :]:
                newidx = np.concatenate((newidx, (uniqC == elemC).nonzero()[0]))

            C[ic, :] = newidx

        idx = list(range(nz-1, -1, -1))
        partition = C[idx, :]

        return partition



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # compute cluster statistics
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _clusterstat(self, S, partitions):

        # number of clusters
        Ncluster = int(np.max(partitions) + 1)

        # initialize dictionary
        stat = {'internal_sum': np.zeros(Ncluster) * np.NaN,
                'internal_min': np.zeros(Ncluster) * np.NaN,
                'internal_avg': np.zeros(Ncluster) * np.NaN,
                'internal_max': np.zeros(Ncluster) * np.NaN,
                'external_sum': np.zeros(Ncluster) * np.NaN,
                'external_min': np.zeros(Ncluster) * np.NaN,
                'external_avg': np.zeros(Ncluster) * np.NaN,
                'external_max': np.zeros(Ncluster) * np.NaN,
                'between_min': np.zeros((Ncluster, Ncluster)),
                'between_avg': np.zeros((Ncluster, Ncluster)),
                'between_max': np.zeros((Ncluster, Ncluster))}

        for cluster in range(Ncluster):
            thisPartition = np.where(partitions == cluster)[0]
            nthisPartition = len(thisPartition)
            S_ = np.zeros((nthisPartition, nthisPartition))
            for i in range(nthisPartition):
                S_[i, :] = S[thisPartition[i], thisPartition]

            S_[list(range(nthisPartition)), list(range(nthisPartition))] = np.NaN
            S_ = S_[np.isfinite(S_)]

            if len(S_) > 0:
                stat['internal_sum'][cluster] = np.sum(S_)
                stat['internal_min'][cluster] = np.min(S_)
                stat['internal_avg'][cluster] = np.mean(S_)
                stat['internal_max'][cluster] = np.max(S_)

            if Ncluster > 1:
                cthisPartition = np.where(partitions != cluster)[0]
                ncthisPartition = len(cthisPartition)
                S_ = np.zeros((nthisPartition, ncthisPartition))
                for i in range(nthisPartition):
                    S_[i, :] = S[thisPartition[i], cthisPartition]

                stat['external_sum'][cluster] = np.sum(S_)
                stat['external_min'][cluster] = np.min(S_)
                stat['external_avg'][cluster] = np.mean(S_)
                stat['external_max'][cluster] = np.max(S_)

            for i in range(Ncluster):
                Pi = np.where(i == partitions)[0]
                for j in range(i+1, Ncluster):
                    Pj = np.where(j == partitions)[0]
                    d_ = np.zeros((len(Pi), len(Pj)))
                    for iPi in range(len(Pi)):
                        d_[iPi, :] = S[Pi[iPi], Pj]

                    stat['between_min'][i, j] = np.min(d_)
                    stat['between_avg'][i, j] = np.mean(d_)
                    stat['between_max'][i, j] = np.max(d_)


        stat['between_min'] += stat['between_min'].transpose()
        stat['between_avg'] += stat['between_avg'].transpose()
        stat['between_max'] += stat['between_max'].transpose()

        return stat



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # estimate the R-index as defined in
    # Levine, E., Domany, E., 2001. 'Resampling method for
    # unsupervised estimation of cluster validity'.
    # Neural Comput. 13 (11), 2573-2593.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _rindex(self, dissimilarities, partitions, verbose=True):

        nPart = partitions.shape[0]

        # number of clusters in each partition
        Ncluster = np.max(partitions, axis=1)
        ri = np.zeros(nPart)

        if verbose:
            print(">>> Computing R-index...")

        for k in range(nPart):
            hist, bin_edges = np.histogram(partitions[k, :], bins=np.arange(1, Ncluster[k]+2))
            if any(hist == 1):
                # contains one-item clusters (index very doubtful)
                ri[k] = np.NaN
            elif Ncluster[k] == 0:
                # degenerate partition (all in the same cluster)
                ri[k] = np.NaN
            else:
                # compute cluster statistics
                stat = self._clusterstat(dissimilarities, partitions[k, :])
                between = stat['between_avg']
                between[list(range(len(between))), list(range(len(between)))] = np.Inf
                internal = stat['internal_avg'].transpose()
                ri[k] = np.mean(internal/np.min(between, axis=0))

        return ri



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # estimate clusters
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _cluster(self, verbose=True):

        # ------------------------------------------
        # compute dissimilarities
        # ------------------------------------------
        similarities = self._corrw()
        dissimilarities = 1.0 - similarities


        # ------------------------------------------
        # generate partitions
        # ------------------------------------------
        Z, order = self._linkage(dissimilarities)
        partitions = self._z_to_partition(Z)


        # ------------------------------------------
        # compute cluster validity
        # ------------------------------------------
        npc = int(self.W_est[0].shape[0])
        indexR = self._rindex(dissimilarities, partitions[:npc, :], verbose=verbose)


        return Z, order, partitions, indexR, dissimilarities, similarities



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # estimate curve that decreases from v0 to vn with a
    # rate that is somewhere between linear and 1/t
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _potency_curve(self, v0, vn, t):

        return v0 * ((1.0*vn/v0)**(np.arange(t)/(t-1.0)))



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # compute principal coordinates (using linear
    # Metric Multi-Dimensional Scaling)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _mmds(self, D):

        nD = D.shape[0]
        # square dissimilarities
        D2 = D**2

        # center matrix
        Z = np.identity(nD) - np.ones((nD, nD))/(1.0 * nD)

        # double centered inner product
        B = -0.5 * np.dot(Z, np.dot(D2, Z))

        # SVD
        U, sing, V = np.linalg.svd(B)

        # coordinates
        X = np.dot(U, np.diag(np.sqrt(sing)))

        return X



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # projects data vectors using Curvilinear Component
    # Analysis
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _cca(self, D, P, epochs, Mdist, alpha0, lambda0):

        # check input data
        noc, dim = D.shape
        noc_x_1 = np.zeros(noc, dtype=np.int)
        me = np.zeros(dim)
        st = np.zeros(dim)

        # estimate mean and standard-deviation
        for i in range(dim):
            idx = np.where(np.isfinite(D[:, i]))[0]
            me[i] = np.mean(D[idx, i])
            st[i] = np.std(D[idx, i])

        # replace unknown projections in initial
        # projection with known values
        inds = np.where(np.isnan(P))[0]
        if len(inds):
            P[inds] = np.random.rand(len(inds))

        dummy, odim = P.shape
        odim_x_1 = np.ones((odim, 1), dtype=np.int)

        # training length
        train_len = int(epochs * noc)
        # random sample order
        sample_inds = np.floor(noc * np.random.rand(train_len))

        # mutual distances
        nMdist = Mdist.shape[0]
        if nMdist == 1:
            Mdist = np.repeat(1, noc)

        if nMdist != noc:
            print(">>> ERROR: Mutual distance matrix size and data set size do not match!")
            import pdb
            pdb.set_trace()

        # alpha and lambda
        Alpha = self._potency_curve(alpha0, alpha0/100.0, train_len)
        Lambda = self._potency_curve(lambda0, 0.01, train_len)

        # action
        for i in range(train_len):

            ind = int(sample_inds[i])            # sample index
            dx = Mdist[:, ind]                   # mutual distance in the input space
            known = np.where(np.isfinite(dx))[0]
            nknown = len(known)

            if nknown > 0:
                y = P[ind, :].reshape(1, odim)        # sample vector's projection
                dyy = P[known, :] - y[noc_x_1[known], :]
                dy = np.sqrt(np.dot(dyy**2, odim_x_1))
                dy[dy == 0] = 1.0                     # to get ride of div-by-zero's
                fy = np.exp(-dy/Lambda[i]) * (dx[known].reshape(nknown, 1)/dy - 1.0)
                P[known, :] += Alpha[i] * fy[:, np.zeros(odim, dtype=np.int)] * dyy

        # set projections of totally unknown vectors as unknown
        unknown = np.where(np.isnan(D))[0]
        if len(unknown) > 0:
            D_tmp = D.copy()
            D_tmp[unknown] = 1
            unknown = np.where(np.sum(D_tmp, axis=1) == dim)[0]
            if len(unknown) > 0:
                P[unknown, :] = np.NaN

        return P



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # to project points on plane so that Euclidean distances
    # between the projected points correspond to the
    # similarity matrix between IC estimates
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _projection(self, dis, verbose=True):

        # initialize some parameter
        outputDim = 2    # we project onto a plane
        alpha = 0.7
        epochs = 75.0
        radius = np.max([self.nrep/20.0, 10])
        s2d = 'sqrtsim2dis'

        # perform similarity-to-dissimilarity transformation
        D = np.abs(np.sqrt(dis))
        nD = D.shape[0]

        if verbose:
            print(">>> Perform projection to plane...")

        # start from MMDS (linear Metric Multi-Dimensional Scaling)
        init_proj = self._mmds(D)
        init_proj = init_proj[:, :outputDim]
        dummy = np.random.rand(nD, outputDim)

        proj = self._cca(dummy, init_proj, epochs, D, alpha, radius)

        return proj



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # to get the index of the component in the center
    # of each cluster
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _idx2centrotypes(self, P, similarities, mode='partition'):

        if mode == 'index':
            nsim = len(P)
            similarity = np.zeros((nsim, nsim))
            for i in range(nsim):
                similarity[i, :] = similarities[P[i], P]

            idx_one = np.argmax(np.sum(similarity, axis=0))
            centro_idx = P[idx_one]

        elif mode == 'partition':
            Ncluster = int(np.max(P) + 1)
            centro_idx = np.zeros(Ncluster, dtype=np.int)
            for i in range(Ncluster):
                idx = np.where(P == i)[0]
                centro_idx[i] = self._idx2centrotypes(idx, similarities, mode='index')

        else:
            print(">>> ERROR: Unknown operation mode!")
            import pdb
            pdb.set_trace()

        return centro_idx



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # get optimal demixing matrix W
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _getW(self, centro_idx):

        import types

        nW = len(self.W_est)
        npc, nchan = self.W_est[0].shape
        npc = int(npc)
        nchan = int(nchan)

        if isinstance(self.W_est[0][0, 0], complex):
            allW = np.zeros((nW * npc, nchan), dtype=np.complex)
        else:
            allW = np.zeros((nW * npc, nchan))


        for iw in range(nW):
            allW[iw*npc:(iw+1)*npc, :] = self.W_est[iw]

        return allW[centro_idx, :]



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # method to estimate the quality of a cluster
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _cluster_quality(self, partition, simililarities, mode='mean'):

        Ncluster = np.max(partition)
        stat = self._clusterstat(simililarities, partition)

        # compute score
        if mode == 'minmax':
            internal = stat['internal_min']
            external = stat['external_max']
        elif mode == 'mean':
            internal = stat['internal_avg']
            external = stat['external_avg']
        else:
            print(">>> ERROR: Unrecognized score function!")
            import pdb
            pdb.set_trace()

        internal[np.isnan(internal)] = 0
        external[np.isnan(external)] = 0
        score = np.abs(internal - external)

        return score



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # to compute the stability (quality) indices of the
    # estimated clusters
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _stability(self, partitions, similarities, L=None):

        # check input parameter
        npc = int(self.W_est[0].shape[0])
        if L == None: L = npc-1

        Ncluster = list(range(L))
        NofEstimates = np.zeros(L, dtype=np.int)
        partition = partitions[L, :]

        for i in Ncluster:
            idx = np.where(partition == i)[0]
            NofEstimates[i] = len(idx)

        # compute cluster quality index
        Iq = self._cluster_quality(partition, similarities, mode='mean')

        return Iq



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # get optimal (de-)mixing matrix
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _get_results(self, partitions, similarities, L=None, sort_results=True):

        # check input parameter
        npc = int(self.W_est[0].shape[0])
        if L == None: L = npc-1

        if L < 0 or L > npc:
            print(">>> WARNING: Number of requested estimate clusters out of range!")
            print(">>> Setting number of clusters to %d" % npc)
            L = npc

        # get indices of ICs in the cluster centers
        centro_idx = self._idx2centrotypes(partitions[L, :], similarities, mode='partition')

        # get optimal demixing matrix
        W = self._getW(centro_idx)
        Iq = self._stability(partitions, similarities, L=L)

        if sort_results:
            idx_sort = np.argsort(Iq)[::-1]
            Iq = Iq[idx_sort]
            W = W[idx_sort, :]

        A = np.linalg.pinv(W)

        return A, W, Iq


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # prepare data for applying the fit routine
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def prepare_data_for_fit(self, fn_raw, stim_name=None,
                             stim_delay=0, tmin_stim=0.0, tmax_stim=1.0,
                             flow=4.0, fhigh=34.0,
                             event_id=1, resp_id=None, corr_event_picking=None,
                             hamming_data=True, remove_outliers=True,
                             fn_inv=None, contrast_id=[],
                             baseline=(None, None), averaged_epochs=False,
                             decim_epochs=False, interpolate_bads=True,
                             unfiltered=False, verbose=True):

        '''
        Routine to prepare the data for ICA application. Preparation
        includes epoch generation,  transformation to Fourier space
        (if desired) and source localization applied to single
        epochs.

            Parameters
            ----------
            fn_raw: filename of the input data (expect fif-file).
            stim_name:  name of the stimulus channel. Note, for
                applying FourierCIA data are chopped around stimulus
                onset. If not set data are chopped in overlapping
                windows
                default: stim_names=None
            stim_delay: delay of stimulus presentation in milliseconds
                default: stim_delay=0
            tmin_stim: time of interest prior to stimulus onset.
                Important for generating epochs to apply FourierICA
                default = 0.0
            tmax_stim: time of interest after stimulus onset.
                Important for generating epochs to apply FourierICA
                default = 1.0
            flow: lower frequency border for estimating the optimal
                de-mixing matrix using FourierICA
                default: flow=4.0
            fhigh: upper frequency border for estimating the optimal
                de-mixing matrix using FourierICA
                default: fhigh=34.0
                Note: here default flow and fhigh are choosen to
                   contain:
                   - theta (4-7Hz)
                   - low (7.5-9.5Hz) and high alpha (10-12Hz),
                   - low (13-23Hz) and high beta (24-34Hz)
            event_id: ID of the event of interest to be considered in
                the stimulus channel. Only of interest if 'stim_name'
                is set
                default: event_id=1
            resp_id: Response ID for correct event estimation. Note:
                Must be in the order corresponding to the 'event_id'
                default: resp_id=None
            corr_event_picking: if set should contain the complete python
                path and name of the function used to identify only the
                correct events
            hamming_data: if set a hamming window is applied to each
                epoch prior to Fourier transformation
                default: hamming_data=True
            remove_outliers: If set outliers are removed from the Fourier
                transformed data.
                Outliers are defined as windows with large log-average power (LAP)

                     LAP_{c,t}=log \sum_{f}{|X_{c,tf}|^2

                where c, t and f are channels, window time-onsets and frequencies,
                respectively. The threshold is defined as |mean(LAP)+3 std(LAP)|.
                This process can be bypassed or replaced by specifying a function
                handle as an optional parameter.
                remove_outliers=False
            fn_inv: file name of inverse operator. If given
                FourierICA is applied on data transformed to
                source space
            contrast_id:  If set FourierICA is applied to contrast epochs
                between events in event_id and events in contrast_id.
                is set
                default: contrast_id=[]
            baseline: If set baseline correction is applied to epochs prior to
                ICA estimation.
            averaged_epochs: Should epochs be averaged before
                FourierICA application? Note, averaged data require
                less memory!
                default: average=False
            decim_epochs: if set the number of epochs will be reduced (per
                subject) to that number for the estimation of the demixing matrix.
                Note: the epochs were chosen randomly from the complete set of
                epochs.
            interpolate_bads: if set bad channels are interpolated (using the
                mne routine raw.interpolate_bads()).
            unfiltered: bool
                If true data are not filtered to a certain frequency range when
                Fourier transformation is applied
                default: unfiltered=False
            verbose: bool, str, int, or None
                If not None, override default verbose level
                (see mne.verbose).
                default: verbose=True

            Returns
            -------
            meg_data: array
                2D array containg the MEg data used for FourierICA estimation
            src_loc_data: array
                3D array containing the source localization
                data used for FourierICA estimation
                (nfreq x nepochs x nvoxel)
            vertno: list
                list containing two arrays with the order
                of the vertices.
            data_already_stft: boolean
                'True' if data are transformed to Fourier space, otherwise
                'False'
            events: list
                list containing the indices of all events used to generate the
                epochs for applying FourierICA
            sfreq: float
                sampling frequency of the data
            meg_channels: list
                list containing the name of all MEG channels used for FourierICA
        '''


        # ------------------------------------------
        # import necessary module
        # ------------------------------------------
        from .fourier_ica import apply_stft, stft_source_localization
        from mne import find_events, pick_types
        from mne.io import Raw

        # ------------------------------------------
        # prepare data to apply FourierICA
        # ------------------------------------------
        meg_raw = Raw(fn_raw, preload=True)
        # interpolate bad channels
        if interpolate_bads:
            meg_raw.interpolate_bads()

        meg_channels = pick_types(meg_raw.info, meg=True, eeg=False,
                                  eog=False, stim=False, exclude='bads')
        meg_data = meg_raw._data[meg_channels, :]
        sfreq = meg_raw.info['sfreq']

        # check if ICASSO should be applied
        # to evoked or resting state data
        if stim_name:
            events_all = find_events(meg_raw, stim_channel=stim_name, consecutive=True,
                                     shortest_event=1)

            # check if there is a stimulus delay
            if stim_delay:
                stim_delay_tsl = int(np.round(stim_delay * meg_raw.info['sfreq']/1000.0))
                events_all[:, 0] += stim_delay_tsl

            # check if only correct events should be chosen
            if corr_event_picking:
                if isinstance(corr_event_picking, str):
                    import importlib
                    mod_name, func_name = corr_event_picking.rsplit('.', 1)
                    mod = importlib.import_module(mod_name)
                    func = getattr(mod, func_name)
                    resp_name = 'STI 013' if stim_name == 'STI 014' else 'STI 014'
                    response = find_events(meg_raw, stim_channel=resp_name, consecutive=True,
                                           shortest_event=1)
                    if np.any(resp_id):
                        events_all, _ = func(events_all, response, sfreq, event_id, resp_id)
                    else:
                        events_all, _ = func(events_all, response, sfreq, event_id)

                else:
                    print(">>> ERROR: 'corr_event_picking' should be a string containing the complete python")
                    print(">>>          path and name of the function used to identify only the correct events!")
                    import pdb
                    pdb.set_trace()


            if np.any(contrast_id):
                contrast_events = events_all[events_all[:, 2] == contrast_id, 0]

            if not isinstance(event_id, (list, tuple)):
                event_id = [event_id]

            for idx, event in enumerate(event_id):
                if idx == 0:
                    events = events_all[events_all[:, 2] == event, :]
                else:
                    events = np.concatenate((events, events_all[events_all[:, 2] == event, :]))

            if not self.tICA:
                events = events[:, 0]

        else:
            events = []


        if self.tICA and not fn_inv:
            print(">>> ERROR: For applying temporal ICA in source space the file name ")
            print("           of the inverse operator is required!")
            import pdb
            pdb.set_trace()


        # ------------------------------------------
        # check if ICA should be applied in source
        # space
        # ------------------------------------------
        if fn_inv:

            # ------------------------------------------
            # check if temporal ICA should be applied
            # on data transformed to source space
            # --> note: here data are not transformed
            #           to Fourier space
            # ------------------------------------------
            if self.tICA:

                # -------------------------------------------
                # check if all necessary parameters are set
                # -------------------------------------------
                if not stim_name:
                    print(">>> ERROR: For applying temporal ICA in source space a stimulus name is required!")
                    import pdb
                    pdb.set_trace()


                # -------------------------------------------
                # generate epochs around stimulus onset
                # -------------------------------------------
                from mne import Epochs
                epoch_data = Epochs(meg_raw, events, event_id,
                                    tmin_stim, tmax_stim,
                                    picks=meg_channels, baseline=baseline,
                                    proj=False, verbose=False)

                if averaged_epochs:
                    X = epoch_data.average().data.transpose()
                    X = X.reshape([X.shape[0], 1, X.shape[1]])
                else:
                    X = epoch_data.get_data().transpose([2, 0, 1])


            # ------------------------------------------
            # FourierICA is applied on data transformed
            # to source space
            # ------------------------------------------
            else:
                # -------------------------------------------
                # transform data to STFT
                # -------------------------------------------
                # print out some information
                if verbose:
                     print(">>> transform data to Fourier space...")

                win_length_sec = tmax_stim - tmin_stim
                X, _ = apply_stft(meg_data, events=events, tpre=tmin_stim,
                                  sfreq=sfreq, flow=flow, fhigh=fhigh,
                                  win_length_sec=win_length_sec,
                                  hamming_data=hamming_data,
                                  remove_outliers=remove_outliers,
                                  baseline=baseline,
                                  decim_epochs=decim_epochs,
                                  unfiltered=unfiltered,
                                  verbose=verbose)

                if np.any(contrast_id):
                    X_contrast, _ = apply_stft(meg_data, events=contrast_events,
                                               tpre=tmin_stim, sfreq=sfreq,
                                               flow=flow, fhigh=fhigh,
                                               win_length_sec=win_length_sec,
                                               hamming_data=hamming_data,
                                               remove_outliers=remove_outliers,
                                               baseline=baseline,
                                               decim_epochs=decim_epochs,
                                               verbose=verbose)


            # -------------------------------------------
            # perform source localization
            # -------------------------------------------
            # print out some information
            if verbose:
                 print(">>> estimate inverse solution...")

            src_loc_data, vertno = stft_source_localization(X, fn_inv,
                                                            method=self.src_loc_method,
                                                            morph2fsaverage=self.morph2fsaverage,
                                                            snr=self.snr)

            if np.any(contrast_id):
                src_loc_data_contrast, _ = stft_source_localization(X_contrast, fn_inv,
                                                                    method=self.src_loc_method,
                                                                    morph2fsaverage=self.morph2fsaverage,
                                                                    snr=self.snr)
                del _
                n_epochs = np.min([src_loc_data.shape[1], src_loc_data_contrast.shape[1]])
                events = events[:n_epochs]
                src_loc_data = src_loc_data[:, :n_epochs, :] - src_loc_data_contrast[:, :n_epochs, :]
                

            data_already_stft = True
            meg_data = X


        # ------------------------------------------
        # FourierICA would be applied on
        # data in the sensor space
        # ------------------------------------------
        else:
            data_already_stft = False
            vertno = None
            src_loc_data = None


        return meg_data, src_loc_data, vertno, data_already_stft, events, sfreq, meg_channels





    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # perform ICASSO based FourierICA signal decomposition
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fit_tICA(self, ica_data, verbose=True):

        # ------------------------------------------
        # import necessary module
        # ------------------------------------------
        from .ica import ica_array
        from scipy.linalg import pinv


        # ------------------------------------------
        # print out some information
        # ------------------------------------------
        if verbose:
            print(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<")
            print(">>>      Performing %s estimation" % self.ica_method)
            print(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<")


        # ------------------------------------------
        # initialize some data
        # ------------------------------------------
        pca = None


        # ------------------------------------------
        # perform ICASSO ICA
        # ------------------------------------------
        for irep in range(self.nrep):
            weights, pca, activations = ica_array(ica_data,
                                              return_ica_object=False,
                                              overwrite=None, pca=pca,
                                              max_pca_components=self.pca_dim,
                                              method=self.ica_method,
                                              cost_func=self.cost_function,
                                              weights=None, lrate=self.lrate,
                                              wchange=self.conv_eps,
                                              maxsteps=self.max_iter,
                                              verbose=verbose)

            if irep == 0:
                self.whitenMat = pca.components_
                self.dewhitenMat = pinv(pca.components_)
                self.dmean = pca.mean_
                self.dstd = pca.stddev_

            # save results in structure
            W_orig = np.dot(weights, self.whitenMat)
            A_orig = np.dot(self.dewhitenMat, pinv(weights))
            self.W_est.append(W_orig)
            self.A_est.append(A_orig)

            # print out some information
            if verbose and self.nrep > 1:
                print(">>> Running %s number %d of %d done" % (self.ica_method, irep+1, self.nrep))

                if irep == 0:
                    print("..... %s parameter:" % self.ica_method)
                    print(".....")
                    print("..... Stopping threshold: %d" % self.conv_eps)
                    print("..... Maximal number of iterations: %d" % self.max_iter)
                    print("..... Learning rate: %d" % self.lrate)
                    print("..... Number of independent components: %d" % self.pca_dim)
                    print(".....")





    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # perform ICASSO based FourierICA signal decomposit ion
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _fit_FourierICA(self, ica_data, events, sfreq,
                        complex_mixing=True, hamming_data=False,
                        remove_outliers=False, envelopeICA=False,
                        normalized=True,  data_already_stft=False,
                        verbose=True):


        # ------------------------------------------
        # import necessary module
        # ------------------------------------------
        from .fourier_ica import JuMEG_fourier_ica


        # ------------------------------------------
        # generate FourierICA object
        # ------------------------------------------
        if verbose:
            print(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<")
            print(">>>      Performing FourierICA estimation")
            print(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<")

        win_length_sec = self.tmax_win - self.tmin_win
        fourier_ica_obj = JuMEG_fourier_ica(events=events, tpre=self.tmin_win,
                                            flow=self.flow, fhigh=self.fhigh,
                                            sfreq=sfreq,
                                            win_length_sec=win_length_sec,
                                            remove_outliers=remove_outliers,
                                            hamming_data=hamming_data,
                                            complex_mixing=complex_mixing,
                                            pca_dim=self.pca_dim,
                                            max_iter=self.max_iter,
                                            conv_eps=self.conv_eps,
                                            cost_function=self.cost_function,
                                            envelopeICA=envelopeICA,
                                            lrate=self.lrate,
                                            decim_epochs=self.decim_epochs)


        # ------------------------------------------
        # initialize some data
        # ------------------------------------------
        whitenMat = []
        dewhitenMat = []


        # ------------------------------------------
        # perform ICASSO ICA
        # ------------------------------------------
        for irep in range(self.nrep):

            # apply FourierICA
            if self.nrep == 1:
                verbose_fourierICA = verbose
            else:
                verbose_fourierICA = False


            W_orig, A_orig, S_FT, Smean, Sstddev, objective, whitenMat, \
            dewhitenMat = fourier_ica_obj.fit(ica_data, whiten_mat=whitenMat,
                                              dewhiten_mat=dewhitenMat,
                                              data_already_stft=data_already_stft,
                                              data_already_normalized=normalized,
                                              verbose=verbose_fourierICA)


            if irep == 0:
                self.whitenMat = whitenMat
                self.dewhitenMat = dewhitenMat
                self.dmean = Smean
                self.dstd = Sstddev

            # save results in structure
            self.W_est.append(W_orig)
            self.A_est.append(A_orig)

            # print out some information
            if verbose and self.nrep > 1:
                print(">>> Running FourierICA number %d of %d done" % (irep+1, self.nrep))

                if irep == 0:
                    str_hamming_window = "True" if fourier_ica_obj.hamming_data else "False"
                    str_complex_mixing = "True" if fourier_ica_obj.complex_mixing else "False"
                    print("..... Fourier ICA parameter:")
                    print(".....")
                    print("..... Sampling frequency set to: %d" % fourier_ica_obj.sfreq)
                    print("..... Start of frequency band set to: %d" % fourier_ica_obj.flow)
                    print("..... End of frequency band set to: %d" % fourier_ica_obj.fhigh)
                    print("..... Using hamming window: %s" % str_hamming_window)
                    print("..... Assume complex mixing: %s" % str_complex_mixing)
                    print("..... Number of independent components: %d" % fourier_ica_obj.ica_dim)
                    print(".....")

        return fourier_ica_obj




    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # perform ICASSO based ICA signal decomposition
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fit(self, fn_raw, ica_method=None, average=False, stim_name=None,
            event_id=None, stim_delay=0, corr_event_picking=None,
            tmin_win=None, tmax_win=None, flow=None, fhigh=None,
            dim_reduction=None, pca_dim=None,
            max_iter=None, conv_eps=None, complex_mixing=True,
            hamming_data=False, remove_outliers=False,
            envelopeICA=False, fn_inv=None, cost_function=None,
            contrast_id=[], baseline=(None, None),
            decim_epochs=False, interpolate_bads=True, verbose=True):

        """
        Perform ICASSO estimation. ICASSO is based on running ICA
        multiple times with slightly different conditions and
        clustering the obtained components. Note, here as default
        FourierICA is applied.


            Parameters
            ----------
            fn_raw: filename of the input data (expect fif-file).
            ica_method: Steing containing the information which ICA
                method should be applied. You can choose between
                'extended-infomax', 'fastica', 'infomax' and
                'fourierica'
                default: ica_method='fourierica'
            average: Should data be averaged across subjects before
                FourierICA application? Note, averaged data require
                less memory!
                default: average=False
            stim_name: name of the stimulus channel. Note, for
                applying FourierCIA data are chopped around stimulus
                onset. If not set data are chopped in overlapping
                windows
                default: stim_names=None
            event_id: Id of the event of interest to be considered in
                the stimulus channel. Only of interest if 'stim_name'
                is set
                default: event_id=1
            stim_delay: delay of stimulus presentation in milliseconds
                default: stim_delay=0
            corr_event_picking: if set should contain the complete python
                path and name of the function used to identify only the
                correct events
            tmin_win: time of interest prior to stimulus onset.
                Important for generating epochs to apply FourierICA
                default = 0.0
            tmax_win: time of interest after stimulus onset.
                Important for generating epochs to apply FourierICA
                default = 1.0
            flow: lower frequency border for estimating the optimal
                de-mixing matrix using FourierICA
                default: flow=4.0
            fhigh: upper frequency border for estimating the optimal
                de-mixing matrix using FourierICA
                default: fhigh=34.0
                Note: here default flow and fhigh are choosen to
                   contain:
                   - theta (4-7Hz)
                   - low (7.5-9.5Hz) and high alpha (10-12Hz),
                   - low (13-23Hz) and high beta (24-34Hz)
            dim_reduction: {'', 'AIC', 'BIC', 'GAP', 'MDL', 'MIBS', 'explVar'}
                Method for dimension selection. For further information about
                the methods please check the script 'dimension_selection.py'.
            pca_dim: The number of PCA components used to apply FourierICA.
                If pca_dim > 1 this refers to the exact number of components.
                If between 0 and 1 pca_dim refers to the variance which
                should be explained by the chosen components
                default: pca_dim=0.9
            max_iter: maximum number od iterations used in FourierICA
                default: max_iter=2000
            conv_eps: iteration stops when weight changes are smaller
                then this number
                default: conv_eps = 1e-9
            complex_mixing: if mixing matrix should be real or complex
                default: complex_mixing=True
            hamming_data: if set a hamming window is applied to each
                epoch prior to Fourier transformation
                default: hamming_data=False
            remove_outliers: If set outliers are removed from the Fourier
                transformed data.
                Outliers are defined as windows with large log-average power (LAP)

                     LAP_{c,t}=log \sum_{f}{|X_{c,tf}|^2

                where c, t and f are channels, window time-onsets and frequencies,
                respectively. The threshold is defined as |mean(LAP)+3 std(LAP)|.
                This process can be bypassed or replaced by specifying a function
                handle as an optional parameter.
                remove_outliers=False
            envelopeICA: if set ICA is estimated on the envelope
                of the Fourier transformed input data, i.e., the
                mixing model is |x|=As
                default: envelopeICA=False
            fn_inv: file name of inverse operator. If given
                FourierICA is applied on data transformed to
                source space
            cost_function: which cost-function should be used in the complex
                ICA algorithm
                'g1': g_1(y) = 1 / (2 * np.sqrt(lrate + y))
                'g2': g_2(y) = 1 / (lrate + y)
                'g3': g_3(y) = y
            contrast_id:  If set FourierICA is applied to contrast epochs
                between events in event_id and events in contrast_id.
                is set
                default: contrast_id=[]
            baseline: If set baseline correction is applied to epochs prior to
                ICA estimation.
            decim_epochs: if set the number of epochs will be reduced (per
                subject) to that number for the estimation of the demixing matrix.
                Note: the epochs were chosen randomly from the complete set of
                epochs.
            interpolate_bads: if set bad channels are interpolated (using the
                mne routine raw.interpolate_bads()).
            verbose: bool, str, int, or None
                If not None, override default verbose level
                (see mne.verbose).
                default: verbose=True

            Returns
            -------
            W: estimated optimal de-mixing matrix
            A: estimated mixing matrix
            Iq: quality index of the clustering between
                components belonging to one cluster
                (between 0 and 1; 1 refers to small clusters,
                i.e., components in one cluster have a highly similar)
            fourier_ica_obj: FourierICA object. For further information
                please have a look into the FourierICA routine
        """

        # ------------------------------------------
        # import necessary module
        # ------------------------------------------
        from mne import set_log_level

        # set log level to 'WARNING'
        set_log_level('WARNING')

        # ------------------------------------------
        # check input parameter
        # ------------------------------------------
        if ica_method:
            self.ica_method = ica_method

        if average:
            self.average = average

        if fn_inv:
            self.fn_inv = fn_inv

        if cost_function:
            self.cost_function = cost_function

        if dim_reduction:
            self.dim_reduction = dim_reduction

        if pca_dim:
            self.pca_dim = pca_dim

        if stim_name:
            self.stim_name = stim_name

        if event_id:
            self.event_id = event_id

        if tmin_win:
            self.tmin_win = tmin_win

        if tmax_win:
            self.tmax_win = tmax_win

        if flow:
            self.flow = flow

        if fhigh:
            self.fhigh = fhigh

        if max_iter:
            self.max_iter = max_iter

        if conv_eps:
            self.conv_eps = conv_eps

        if decim_epochs:
            self.decim_epochs = decim_epochs


        # ------------------------------------------
        # check which ICA algorithm should be
        # applied
        # ------------------------------------------
        if self.ica_method in  ['extended-infomax', 'fastica', 'infomax']:
            self.tICA = True

            if not self.cost_function in ['logcosh', 'exp', 'cube']:
                self.cost_function = 'logcosh'

        elif self.ica_method == 'fourierica':
            self.tICA = False
        else:
            print('WARNING: chosen ICA method does not exist!')
            print('Programm stops!')
            import pdb
            pdb.set_trace()



       # ------------------------------------------
        # prepare data to apply ICASSO
        # ------------------------------------------
        # check if fn_raw is a list, i.e., group FourierICA
        # should be applied
        if isinstance(fn_raw, list):

            # test if FourierICA should be applied
            if self.ica_method != 'fourierica':
                print(">>> NOTE: When using temporal group ICA it is recommended " \
                      "to use ICA based on averaged datasets")
                print(">>> Parameters are set for group ICA!")
                average_epochs = True
                self.average = False
            else:
                average_epochs = False


            # loop over all files
            for idx, fnraw in enumerate(fn_raw):
                meg_data_cur, src_loc, vert, data_already_stft, events, sfreq, picks = \
                    self.prepare_data_for_fit(fnraw, stim_name=self.stim_name,
                                              tmin_stim=self.tmin_win, tmax_stim=self.tmax_win,
                                              flow=self.flow, fhigh=self.fhigh, event_id=self.event_id,
                                              corr_event_picking=corr_event_picking, stim_delay=stim_delay,
                                              fn_inv=self.fn_inv[idx], hamming_data=hamming_data,
                                              remove_outliers=remove_outliers,
                                              contrast_id=contrast_id, baseline=baseline,
                                              averaged_epochs=average_epochs,
                                              decim_epochs=self.decim_epochs,
                                              interpolate_bads=interpolate_bads,
                                              verbose=verbose)

                # normalize source data
                fftsize, nwindows, nvoxel = src_loc.shape
                nrows_Xmat_c = fftsize*nwindows
                src_loc = src_loc.reshape((nrows_Xmat_c, nvoxel), order='F')
                dmean = np.mean(src_loc, axis=0)
                dstddev = np.std(src_loc, axis=0)


                # ---------------------------------
                # save all data in one matrix
                # ---------------------------------
                if self.average:

                    if self.ica_method == 'fourierica':
                        if idx == 0:
                            nfn_raw = len(fn_raw)
                            src_loc_data = np.zeros((nrows_Xmat_c, nvoxel), dtype=np.complex)
                            meg_data = np.zeros((fftsize, nwindows, 248), dtype=np.complex)
                            nwindows_min = nwindows

                        # check if result arrays must be reduced
                        if nwindows_min > nwindows:
                            nwindows_min = nwindows
                            src_loc_data = src_loc_data[:(nwindows_min*fftsize), :]
                            meg_data = meg_data[:, :nwindows_min, :]

                        src_loc_data += (src_loc[:(nwindows_min*fftsize), :] - dmean[np.newaxis, :]) / \
                                        (dstddev[np.newaxis, :]*nfn_raw)
                        meg_data[:, :, picks] += (meg_data_cur[:, :nwindows_min, :]/nfn_raw)
                else:
                    if idx == 0:
                        nfn_raw = len(fn_raw)
                        src_loc_data = np.zeros((nfn_raw*nrows_Xmat_c, nvoxel), dtype=np.complex)
                        meg_data = np.zeros((fftsize, nfn_raw*nwindows, 248), dtype=np.complex)

                    src_loc_data[(idx*nrows_Xmat_c):((idx+1)*nrows_Xmat_c), :] = \
                        (src_loc - dmean[np.newaxis, :]) / dstddev[np.newaxis, :]
                    meg_data[:, (idx*nwindows):((idx+1)*nwindows), picks] = meg_data_cur


                # ---------------------------------
                # free some me
                # ---------------------------------
                del meg_data_cur, src_loc, dmean, dstddev



            normalized = True

        else:
            meg_data, src_loc_data, vertno, data_already_stft, events, sfreq, picks = \
                self.prepare_data_for_fit(fn_raw, stim_name=self.stim_name,
                                          tmin_stim=self.tmin_win, tmax_stim=self.tmax_win,
                                          flow=self.flow, fhigh=self.fhigh, event_id=self.event_id,
                                          stim_delay=stim_delay, corr_event_picking=corr_event_picking,
                                          fn_inv=self.fn_inv, hamming_data=hamming_data,
                                          remove_outliers=remove_outliers, baseline=baseline,
                                          decim_epochs=self.decim_epochs, interpolate_bads=interpolate_bads,
                                          verbose=verbose)
            normalized = False

        self._sfreq = sfreq
        # ------------------------------------------
        # check if PCA dimension is set...if not
        # use MIBS to estimate the dimension
        # ------------------------------------------
        if not self.pca_dim:

            # import some modules
            from .complex_ica import cov
            from scipy.linalg import eigh
            from .dimension_selection import aic, mdl, mibs, bic, gap

            # concatenate STFT for consecutive windows in each channel
            fftsize, nwindows, nchan = meg_data.shape
            nrows_Xmat_c = fftsize*nwindows
            Xmat_c = meg_data.reshape((nrows_Xmat_c, nchan), order='F')

            covmat = cov(Xmat_c, rowvar=0)
            Dc, Ec = eigh(covmat.real)
            idx_sort = np.argsort(Dc.real)[::-1]
            Dc = Dc[idx_sort].real
            ntsl = Xmat_c.shape[0]


            if self.dim_reduction == 'AIC':
                pca_dim, _ = aic(Dc)
            elif self.dim_reduction == 'BIC':
                pca_dim = bic(Dc, ntsl)
            elif self.dim_reduction == 'MIBS':
                pca_dim = mibs(Dc, ntsl)
            elif self.dim_reduction =='GAP':
                pca_dim = gap(Dc)
            else:  # self.dim_reduction == 'MDL'
                _, pca_dim = mdl(Dc)


            if pca_dim > 60:
                print("WARNING: You have %d PCA components!" % (pca_dim))
                print("Using now explained variance...")
                explVar = np.abs(Dc.copy())
                explVar /= explVar.sum()
                pca_dim = np.sum(explVar.cumsum() <= 0.9) + 1
                print("Dimension is now: %d components!" % (pca_dim))

            self.pca_dim = pca_dim
            del Xmat_c, covmat, Ec, idx_sort, Dc, ntsl, _


        # ------------------------------------------
        # check if ICA should be applied in sensor
        # or source space
        # ------------------------------------------
        if self.fn_inv:
            ica_data = src_loc_data
        else:
            ica_data = meg_data.copy()


        # ------------------------------------------
        # check which ICA algorithm should be
        # applied
        # ------------------------------------------
        if self.ica_method in ['extended-infomax', 'fastica', 'infomax']:
            self.fit_tICA(ica_data.real.T, verbose=verbose)
            fourier_ica_obj = None

        elif self.ica_method == 'fourierica':
            fourier_ica_obj = self._fit_FourierICA(ica_data, events, sfreq,
                                                   complex_mixing=complex_mixing,
                                                   hamming_data=hamming_data,
                                                   remove_outliers=remove_outliers,
                                                   envelopeICA=envelopeICA,
                                                   normalized=normalized,
                                                   data_already_stft=data_already_stft,
                                                   verbose=verbose)
        else:
            print('WARNING: chosen ICA method does not exist!')
            print('Programm stops!')
            import pdb
            pdb.set_trace()


        # ------------------------------------------
        # perform cluster analysis
        # ------------------------------------------
        if self.nrep == 1:
            if verbose:
                print(">>>")
                print(">>> No clustering required as only one ICASSO repetition was performed...")

            W = self.W_est[0]
            A = self.A_est[0]
            Iq = np.zeros(W.shape[0])
        else:
            if verbose:
                print(">>>")
                print(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<")
                print(">>>        Performing cluster analysis         <<<")
                print(">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<")

            Z, order, partitions, indexR, dis, sim = self._cluster()
            proj = self._projection(dis)
            A, W, Iq = self._get_results(partitions, sim)


        # ------------------------------------------
        # return results
        # ------------------------------------------
        return W, A, Iq, fourier_ica_obj

