# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.fourier_ica --------------------------------------
----------------------------------------------------------------------
 autor      : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 31.03.2015
 version    : 1.0

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

 1. Runs FourierICA with given parameters M times on data X.
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

    def __init__(self, nrep=50):
        """
        Generate ICASSO object.

            Parameters
            ----------
            nrep: number of repetitions ICA should be performed
                default: nrep=50

            Returns
            -------
            object: ICASSO object
        """

        self._nrep = nrep     # number of repetitions
        self.whitenMat = []
        self.dewhitenMat = []
        self.W_est = []
        self.A_est = []


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # set/get maximum number of repetitions
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _set_nrep(self, nrep):
        self._nrep = nrep

    def _get_nrep(self):
        return int(self._nrep)

    nrep = property(_get_nrep, _set_nrep)



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # estimate linkage between components
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def linkage(self, dis):

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

            # import pdb
            # pdb.set_trace()

            if len(idx_var) == 0:
                order = np.concatenate((k_var, order))
            else:
                order = np.concatenate((order[:idx_var[0]], k_var, order[idx_var[0]:]))

        order = np.concatenate((rest, order))[::-1]

        # to maintain compatibility with Statistics Toolbox, the values
        # in Z must be yet transformed so that they are similar to the
        # output of the LINKAGE function
        Zs = Z.copy()
        current_cluster = np.array(range(dlen))
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
    def corrw(self):

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
    def z_to_partition(self, Z):

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

        idx = range(nz-1, -1, -1)
        partition = C[idx, :]

        return partition



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # compute cluster statistics
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def clusterstat(self, S, partitions):

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

            S_[range(nthisPartition), range(nthisPartition)] = np.NaN
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
    def rindex(self, dissimilarities, partitions, verbose=True):

        nPart = partitions.shape[0]

        # number of clusters in each partition
        Ncluster = np.max(partitions, axis=1)
        ri = np.zeros(nPart)

        if verbose:
            print ">>> Computing R-index..."

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
                stat = self.clusterstat(dissimilarities, partitions[k, :])
                between = stat['between_avg']
                between[range(len(between)), range(len(between))] = np.Inf
                internal = stat['internal_avg'].transpose()
                ri[k] = np.mean(internal/np.min(between, axis=0))

        return ri



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # estimate clusters
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def cluster(self, verbose=True):

        # ------------------------------------------
        # compute dissimilarities
        # ------------------------------------------
        similarities = self.corrw()
        dissimilarities = 1.0 - similarities


        # ------------------------------------------
        # generate partitions
        # ------------------------------------------
        Z, order = self.linkage(dissimilarities)
        partitions = self.z_to_partition(Z)


        # ------------------------------------------
        # compute cluster validity
        # ------------------------------------------
        npc = int(self.W_est[0].shape[0])
        indexR = self.rindex(dissimilarities, partitions[:npc, :], verbose=verbose)


        return Z, order, partitions, indexR, dissimilarities, similarities



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # estimate curve that decreases from v0 to vn with a
    # rate that is somewhere between linear and 1/t
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def potency_curve(self, v0, vn, t):

        return v0 * ((1.0*vn/v0)**(np.arange(t)/(t-1.0)))



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # compute principal coordinates (using linear
    # Metric Multi-Dimensional Scaling)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mmds(self, D):

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
    def cca(self, D, P, epochs, Mdist, alpha0, lambda0):

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
            print ">>> ERROR: Mutual distance matrix size and data set size do not match!"
            import pdb
            pdb.set_trace()

        # alpha and lambda
        Alpha = self.potency_curve(alpha0, alpha0/100.0, train_len)
        Lambda = self.potency_curve(lambda0, 0.01, train_len)

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
    def projection(self, dis, verbose=True):

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
            print ">>> Perform projection to plane..."

        # start from MMDS (linear Metric Multi-Dimensional Scaling)
        init_proj = self.mmds(D)
        init_proj = init_proj[:, :outputDim]
        dummy = np.random.rand(nD, outputDim)

        proj = self.cca(dummy, init_proj, epochs, D, alpha, radius)

        return proj



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # to get the index of the component in the center
    # of each cluster
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def idx2centrotypes(self, P, similarities, mode='partition'):

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
                centro_idx[i] = self.idx2centrotypes(idx, similarities, mode='index')

        else:
            print ">>> ERROR: Unknown operation mode!"
            import pdb
            pdb.set_trace()

        return centro_idx



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # get optimal demixing matrix W
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def getW(self, centro_idx):

        import types

        nW = len(self.W_est)
        npc, nchan = self.W_est[0].shape
        npc = int(npc)
        nchan = int(nchan)

        if isinstance(self.W_est[0][0, 0], types.ComplexType):
            allW = np.zeros((nW * npc, nchan), dtype=np.complex)
        else:
            allW = np.zeros((nW * npc, nchan))


        for iw in range(nW):
            allW[iw*npc:(iw+1)*npc, :] = self.W_est[iw]

        return allW[centro_idx, :]



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # method to estimate the quality of a cluster
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def cluster_quality(self, partition, simililarities, mode='mean'):

        Ncluster = np.max(partition)
        stat = self.clusterstat(simililarities, partition)

        # compute score
        if mode == 'minmax':
            internal = stat['internal_min']
            external = stat['external_max']
        elif mode == 'mean':
            internal = stat['internal_avg']
            external = stat['external_avg']
        else:
            print ">>> ERROR: Unrecognized score function!"
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
    def stability(self, partitions, similarities, L=None):

        # check input parameter
        npc = int(self.W_est[0].shape[0])
        if L == None: L = npc-1

        Ncluster = range(L)
        NofEstimates = np.zeros(L, dtype=np.int)
        partition = partitions[L, :]

        for i in Ncluster:
            idx = np.where(partition == i)[0]
            NofEstimates[i] = len(idx)

        # compute cluster quality index
        Iq = self.cluster_quality(partition, similarities, mode='mean')

        return Iq



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # get optimal (de-)mixing matrix
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_results(self, partitions, similarities, L=None, sort_results=True):

        # check input parameter
        npc = int(self.W_est[0].shape[0])
        if L == None: L = npc-1

        if L < 0 or L > npc:
            print ">>> WARNING: Number of requested estimate clusters out of range!"
            print ">>> Setting number of clusters to %d" % npc
            L = npc

        # get indices of ICs in the cluster centers
        centro_idx = self.idx2centrotypes(partitions[L, :], similarities, mode='partition')

        # get optimal demixing matrix
        W = self.getW(centro_idx)
        Iq = self.stability(partitions, similarities, L=L)

        if sort_results:
            idx_sort = np.argsort(Iq)[::-1]
            Iq = Iq[idx_sort]
            W = W[idx_sort, :]

        A = np.linalg.pinv(W)

        return A, W, Iq



    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # perform ICASSO based ICA signal decomposition
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fit(self, fn_raw, stim_name=None, event_id=1,
            tmin_stim=0.0, tmax_stim=1.0, flow=4.0, fhigh=34.0,
            pca_dim=0.90, max_iter=10000, conv_eps=1e-16,
            verbose=True):

        """
        Perform ICASSO estimation. ICASSO is based on running ICA
        multiple times with slightly different conditions and
        clustering the obtained components. Note, here FourierICA
        is applied


            Parameters
            ----------
            fn_raw: filename of the input data (expect fif-file).
            stim_name: name of the stimulus channel. Note, for
                applying FourierCIA data are chopped around stimulus
                onset. If not set data are chopped in overlapping
                windows
                default: stim_names=None
            event_id: Id of the event of interest to be considered in
                the stimulus channel. Only of interest if 'stim_name'
                is set
                default: event_id=1
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
            pca_dim: The number of PCA components used to apply FourierICA.
                If pca_dim > 1 this refers to the exact number of components.
                If between 0 and 1 pca_dim refers to the variance which
                should be explained by the chosen components
                default: pca_dim=0.9
            max_iter: maximum number od iterations used in FourierICA
                default: max_iter=10000
            conv_eps: iteration stops when weight changes are smaller
                then this number
                default: conv_eps = 1e-16
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
                i.e., components in one cluster a highly similar)
            fourier_ica_obj: FourierICA object. For further information
                please have a look into the FourierICA routine
        """

        # ------------------------------------------
        # import necessary module
        # ------------------------------------------
        from fourier_ica import JuMEG_fourier_ica
        from mne import find_events, pick_types, set_log_level
        from mne.io import Raw


        # set log level to 'WARNING'
        set_log_level('WARNING')

        # ------------------------------------------
        # prepare data to apply FourierICA
        # ------------------------------------------
        meg_raw = Raw(fn_raw, preload=True)
        meg_channels = pick_types(meg_raw.info, meg=True, eeg=False,
                                  eog=False, stim=False, exclude='bads')
        meg_data = meg_raw._data[meg_channels, :]

        if stim_name:
            events = find_events(meg_raw, stim_channel=stim_name, consecutive=True)
            events = events[events[:, 2] == event_id, 0]
        else:
            events = []


        # ------------------------------------------
        # generate FourierICA object
        # ------------------------------------------
        if verbose:
            print ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<"
            print ">>>      Performing FourierICA estimation      <<<"
            print ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<"

        win_length_sec = tmax_stim - tmin_stim
        fourier_ica_obj = JuMEG_fourier_ica(events=events, tpre=tmin_stim,
                                            flow=flow, fhigh=fhigh,
                                            win_length_sec=win_length_sec,
                                            remove_outliers=True,
                                            hamming_data=True,
                                            complex_mixing=True,
                                            pca_dim=pca_dim,
                                            max_iter=max_iter,
                                            conv_eps=conv_eps)


        # ------------------------------------------
        # perform ICASSO ICA
        # ------------------------------------------
        for irep in range(self.nrep):
            # apply FourierICA
            W_orig, A_orig, _, _, _, whitenMat, dewhitenMat = fourier_ica_obj.fit(meg_data.copy(), verbose=False)

            if irep == 0:
                self.whitenMat = whitenMat
                self.dewhitenMat = dewhitenMat

            # save results in structure
            self.W_est.append(W_orig)
            self.A_est.append(A_orig)

            # print out some information
            if verbose:
                print ">>> Running FourierICA number %d of %d done" % (irep+1, self.nrep)

                if irep == 0:
                    str_hamming_window = "True" if fourier_ica_obj.hamming_data else "False"
                    str_complex_mixing = "True" if fourier_ica_obj.complex_mixing else "False"
                    print "..... Fourier ICA parameter:"
                    print "....."
                    print "..... Sampling frequency set to: %d" % fourier_ica_obj.sfreq
                    print "..... Start of frequency band set to: %d" % fourier_ica_obj.flow
                    print "..... End of frequency band set to: %d" % fourier_ica_obj.fhigh
                    print "..... Using hamming window: %s" % str_hamming_window
                    print "..... Assume complex mixing: %s" % str_complex_mixing
                    print "..... Number of independent components: %d" % fourier_ica_obj.ica_dim
                    print "....."


        # ------------------------------------------
        # perform cluster analysis
        # ------------------------------------------
        if verbose:
            print ">>>"""
            print ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<"
            print ">>>        Performing cluster analysis         <<<"
            print ">>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<"

        Z, order, partitions, indexR, dis, sim = self.cluster()
        proj = self.projection(dis)
        A, W, Iq = self.get_results(partitions, sim)


        # ------------------------------------------
        # return results
        # ------------------------------------------
        return W, A, Iq, fourier_ica_obj

