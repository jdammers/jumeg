"""
====================
Jumeg MFT Functions.
====================
"""

# Author: Eberhard Eich
# License: BSD (3-clause)

import os
import numpy as np
import scipy
import scipy.linalg
import time

import mne
from mne.io.constants import FIFF

TINY = 1.e-38

########################################################################
# Compare model values to measured values
#  relerr = sum((m(P_kj*a_k)-meas_j)^2,j=1,n_meas)/sum(meas_j^2,j=1,n_meas)
#  rdmerr = sqrt( sum( (mest_i/||mest|| - meas_i/||meas||)^2, i) )
#  mag = sqrt(sum( (mest_i)^2, i)) / sqrt( sum( meas_i^2, i))
#
########################################################################
def compare_est_exp(pmat, acoeff, mexp):
    """Compare estimated to measured values

    Parameters
    ----------
    pmat: 2-dim np-array
    acoeff: np-array with coefficients
    mexp: np-array with measured values

    Returns
    -------
    with mest_j = P_kj*a_k for j=1,n_meas
    relerr = sum((mest_j-meas_j)^2,j=1,n_meas) / sum(mest_j^2,j=1,n_meas)
    rdmerr = sqrt( sum( (mest_j/||mest|| - mexp_j/||mexp||)^2, j=1,n_meas) )
    mag    = ||mest|| / ||mexp||
    or (-1.,-1.,-1.) in case of error
    """
    mest = np.zeros(pmat.shape[0])
    for k in xrange(pmat.shape[0]):
        mest[k] = np.dot(pmat[:, k], acoeff)
    sumestsq = np.dot(mest, mest)
    sumexpsq = np.dot(mexp, mexp)
    if sumexpsq<TINY or sumestsq < TINY:
        return (-1., -1., -1.)
    estnorm = np.sqrt(sumestsq)
    expnorm = np.sqrt(sumexpsq)
    mdiff = mest - mexp
    rdmdiff = mest/estnorm - mexp/expnorm
    rdmerr = np.sqrt(np.dot(rdmdiff, rdmdiff))
    sumdiffsq = np.dot(mdiff, mdiff)
    return (sumdiffsq/sumestsq, rdmerr, estnorm/expnorm)


########################################################################
# Calculate cdm from current-density-vectors with cut
#  cutcdm = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0) /
#           sum(|cdv[i]|, all i w/ |cdv(i)|>cut*cdvmax)
#  cutjlong = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0),
#                         all i w/ |cdv(i)|>cut*cdvmax)
########################################################################
def calc_cdm_w_cut(cdv, cdvcut):
    """Calculate cdm from current-density-vectors with cut

    Estimate cdm, the current directionality measure, using the direction
    of the maximum current and jlong, the total current in that direction
     cutcdm = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0) /
              sum(|cdv[i]|, all i w/ |cdv(i)|>cut*cdvmax)
     cutjlong = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0),
                            all i w/ |cdv(i)|>cut*cdvmax)

    Parameters
    ----------
    cdv: np-array w current density vectors
    cdvcut: cut for cdv-norms [0,1) of |cdvmax|

    Returns
    -------
    (cutcdm, cutjlong)
    or -1 in case of error
    """
    if len(cdv.shape) == 1:
        cdvecs = np.reshape(cdv, (cdv.shape[0]/3,3))
    elif len(cdv.shape) == 2 and cdv.shape[1] == 3:
        cdvecs = cdv
    else:
        raise ValueError(">>>>> Illegal array shape in call")
    if cdvcut<0. or cdvcut>=1.:
        raise ValueError(">>>>> cdvcut must be in [0,1)")

    cdvsq = np.sum(cdvecs**2,axis=1)
    icdvmax = cdvsq.argmax()
    cdvmaxsq = cdvsq[icdvmax]
    cdvlimsq = cdvcut*cdvcut*cdvmaxsq
    cdvsq *= (cdvsq >= cdvlimsq)
    icdvpos = np.nonzero(cdvsq)[0]
    dsumpabs = np.sum(np.sqrt(cdvsq[icdvpos]))
    cdvmaxvec = cdvecs[icdvmax,:]
    sprod = np.dot(cdvecs[icdvpos,:],cdvmaxvec)
    dsumplong = sprod[sprod > 0].sum() / np.sqrt(cdvmaxsq)
    return (dsumplong/dsumpabs, dsumplong)


########################################################################
# Calculate cdm from current-density-vectors with cut
#  cutcdm = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0) /
#           sum(|cdv[i]|, all i w/ |cdv(i)|>cut*cdvmax)
#  cutjlong = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0),
#                         all i w/ |cdv(i)|>cut*cdvmax)
########################################################################
def fit_cdm_w_cut(cdv, cdvcut):
    """Calculate cdm from current-density-vectors with cut

    Fit cdm, the current directionality measure, using the direction
    of the maximum current as initial value and scan directions on
    the unit sphere for improvement.
    Returns also jlong, the total current in that direction
     cutcdm = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0) /
              sum(|cdv[i]|, all i w/ |cdv(i)|>cut*cdvmax)
     cutjlong = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0),
                            all i w/ |cdv(i)|>cut*cdvmax)

    Parameters
    ----------
    cdv: np-array w current density vectors
    cdvcut: cut for cdv-norms [0,1) of |cdvmax|

    Returns
    -------
    (cutcdm, cutjlong)
    or -1 in case of error
    """
    if len(cdv.shape) == 1:
        cdvecs = np.reshape(cdv, (cdv.shape[0]/3,3))
    elif len(cdv.shape) == 2 and cdv.shape[1] == 3:
        cdvecs = cdv
    else:
        raise ValueError(">>>>> Illegal array shape in call")
    if cdvcut<0. or cdvcut>=1.:
        raise ValueError(">>>>> cdvcut must be in [0,1)")

    nphi = 20
    ntheta = 11
    cdvsq = np.sum(cdvecs**2,axis=1)
    icdvmax = cdvsq.argmax()
    cdvmaxsq = cdvsq[icdvmax]
    cdvlimsq = cdvcut*cdvcut*cdvmaxsq
    cdvsq *= (cdvsq >= cdvlimsq)
    icdvpos = np.nonzero(cdvsq)[0]
    dsumpabs = np.sum(np.sqrt(cdvsq[icdvpos]))
    cdvmaxvec = cdvecs[icdvmax,:]
    sprod = np.dot(cdvecs[icdvpos,:],cdvmaxvec)
    dsumplong = sprod[sprod > 0].sum() / np.sqrt(cdvmaxsq)

    for iphi in xrange(nphi):
        aphi = np.pi*float(2*iphi)/float(nphi)
        for itheta in xrange(ntheta):
            atheta = np.pi*float(itheta)/float(ntheta)
            cdvtstvec = [np.cos(aphi)*np.sin(atheta),np.sin(aphi)*np.sin(atheta),np.cos(atheta)]
            sprod = np.dot(cdvecs[icdvpos,:],cdvtstvec)
            dsumplongtst = sprod[sprod > 0].sum()
            if dsumplongtst > dsumplong:
                dsumplong = dsumplongtst

    return (dsumplong/dsumpabs, dsumplong)


########################################################################
# Calculate cdm from current-density-vectors with cut
#  cutcdm = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0) /
#           sum(|cdv[i]|, all i w/ |cdv(i)|>cut*cdvmax)
########################################################################
def scan_cdm_w_cut(cdv, cdvcut):
    """Calculate cdm from current-density-vectors with cut

    Find cdm, the current directionality measure, scanning for the
    best cdv-direction.
     cutcdm = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0) /
              sum(|cdv[i]|, all i w/ |cdv(i)|>cut*cdvmax)

    Parameters
    ----------
    cdv: np-array w current density vectors
    cdvcut: cut for cdv-norms [0,1) of |cdvmax|

    Returns
    -------
    or -1 in case of error
    """
    if len(cdv.shape) == 1:
        cdvecs = np.reshape(cdv, (cdv.shape[0]/3,3))
    elif len(cdv.shape) == 2 and cdv.shape[1] == 3:
        cdvecs = cdv
    else:
        raise ValueError(">>>>> Illegal array shape in call")
    if cdvcut<0. or cdvcut>=1.:
        raise ValueError(">>>>> cdvcut must be in [0,1)")

    cdvsq = np.sum(cdvecs**2,axis=1)
    icdvmax = cdvsq.argmax()
    cdvmaxsq = cdvsq[icdvmax]
    cdvlimsq = cdvcut*cdvcut*cdvmaxsq
    cdvsq *= (cdvsq >= cdvlimsq)
    icdvpos = np.nonzero(cdvsq)[0]
    dsumpabs = np.sum(np.sqrt(cdvsq[icdvpos]))
    cdvmaxvec = cdvecs[icdvmax,:]
    sprod = np.dot(cdvecs[icdvpos,:],cdvmaxvec)
    dsumplong = sprod[sprod > 0].sum() / np.sqrt(cdvmaxsq)

    for ivec in icdvpos:
        cdvtstvec = cdvecs[ivec,:]
        sprod = np.dot(cdvecs[icdvpos,:],cdvtstvec)
        dsumplongtst = sprod[sprod > 0].sum() / np.sqrt(cdvsq[ivec])
        if dsumplongtst > dsumplong:
            dsumplong = dsumplongtst

    return dsumplong/dsumpabs


def apply_mft(fwdname, datafile, evocondition=None, meg='mag',
              exclude='bads', mftpar=None, subject=None, save_stc=True,
              verbose=False):
    """ Apply MFT to specified data set.

    Parameters
    ----------
    fwdname: name of forward solution file
    datafile: name of datafile (ave or raw)
    evocondition: condition in case of evoked input file
    meg: meg-channels to pick ['mag']
    exclude: meg-channels to exclude ['bads']
    mftpar: dictionary with parameters for MFT algorithm
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    verbose: control variable for verbosity
             False='CRITICAL','WARNING',True='INFO','DEBUG'
             or 'chatty'='verbose' (>INFO,<DEBUG)

    Returns
    -------
    qualmft: dictionary with relerr,rdmerr,mag-arrays
    stcdata: stc with ||cdv|| at fwdmag['source_rr']
             (type corresponding to forward solution)
    """
    twgbl0 = time.time()
    tcgbl0 = time.clock()

    if mftpar is None:
        mftpar = { 'prbfct':'uniform', 'prbcnt':None, 'prbhw':None,
                   'iter':8, 'currexp':1,
                   'regtype':'PzetaE', 'zetareg':1.00,
                   'solver':'lu', 'svrelcut':5.e-4 }

    if mftpar['solver'] == 'svd':
        use_svd = True
        use_lud = False
        svrelcut = mftpar['svrelcut']
    elif mftpar['solver'] == 'lu' or mftpar['solver'] == 'ludecomp':
        use_lud = True
        use_svd = False
    else:
        raise ValueError(">>>>> mftpar['solver'] must be either 'svd' or 'lu[decomp]'")

    if mftpar['prbcnt'] == None and mftpar['prbhw'] == None:
        prbcnt = np.array([0.0,0.0,0.0],ndmin=2)
        prbdhw = np.array([0.0,0.0,0.0],ndmin=2)
    else:
        prbcnt = np.reshape(mftpar['prbcnt'],(len(mftpar['prbcnt'].flatten())/3,3))
        prbdhw = np.reshape(mftpar['prbhw'],(len(mftpar['prbhw'].flatten())/3,3))
    if prbcnt.shape != prbdhw.shape:
        raise ValueError(">>>>> mftpar['prbcnt'] and mftpar['prbhw'] must have same size")

    verbosity = 1
    if verbose == False or verbose == 'CRITICAL':
        verbosity = -1
    elif verbose == 'WARNING':
        verbosity = 0
    elif verbose == 'chatty' or verbose == 'verbose':
        verbose = 'INFO'
        verbosity = 2
    elif verbose == 'DEBUG':
        verbosity = 3

    if verbosity >= 0:
        print "meg-channels     = ",meg
        print "exclude-channels = ",exclude
        print "mftpar['iter'    ] = ",mftpar['iter']
        print "mftpar['regtype' ] = ",mftpar['regtype']
        print "mftpar['zetareg' ] = ",mftpar['zetareg']
        print "mftpar['solver'  ] = ",mftpar['solver']
        print "mftpar['svrelcut'] = ",mftpar['svrelcut']
        print "mftpar['prbfct'  ] = ",mftpar['prbfct']
        print "mftpar['prbcnt'  ] = ",mftpar['prbcnt']
        print "mftpar['prbhw'   ] = ",mftpar['prbhw']
        if mftpar['prbcnt'] != None or mftpar['prbhw'] != None:
            for icnt in xrange(prbcnt.shape[0]):
                print "  pos(prbcnt[%d])   = " % (icnt+1), prbcnt[icnt]
                print "  dhw(prbdhw[%d])   = " % (icnt+1), prbdhw[icnt]

    # Msg will be written by mne.read_forward_solution()
    fwd = mne.read_forward_solution(fwdname, verbose=verbose)
    # Block off fixed_orientation fwd-s for now:
    if fwd['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
        raise ValueError(">>>>> apply_mft() cannot handle fixed-orientation fwd-solutions")

    # Select magnetometer channels:
    fwdmag = mne.io.pick.pick_types_forward(fwd, meg=meg, ref_meg=False,
                                            eeg=False, exclude=exclude)
    lfmag = fwdmag['sol']['data']

    n_sens, n_loc = lfmag.shape
    if verbosity >= 2:
        print "Leadfield size : n_sen x n_loc = %d x %d" % (n_sens,n_loc)

    if datafile.rfind('-ave.fif') > 0 or datafile.rfind('-ave.fif.gz') > 0:
        if verbosity >= 0:
            print "Reading evoked data from %s" % datafile
        if evocondition is None:
            #indatinfo = mne.io.read_info(datafile)
            indathndl = mne.read_evokeds(datafile,
                                         baseline=(None, 0), verbose=verbose)
            if len(indathndl) > 1:
                raise ValueError(">>>>> need to specify a condition for this datafile. Aborting-")
            picks = mne.io.pick.pick_types(indathndl[0].info, meg=meg, ref_meg=False,
                                           eeg=False, stim=False, exclude=exclude)
            data = indathndl[0].data[picks,:]
        else:
            indathndl = mne.read_evokeds(datafile, condition=evocondition,
                                         baseline=(None, 0), verbose=verbose)
            #if len(indathndl) > 1:
            #    raise ValueError(">>>>> need to specify a condition for this datafile. Aborting-")
            picks = mne.io.pick.pick_types(indathndl.info, meg=meg, ref_meg=False,
                                           eeg=False, stim=False, exclude=exclude)
            data = indathndl.data[picks,:]
    elif datafile.rfind('-raw.fif') > 0 or datafile.rfind('-raw.fif.gz') > 0:
        if verbosity >= 0:
            print "Reading raw data from %s" % datafile
        indathndl = mne.io.Raw(datafile, preload=True, verbose=verbose)
        picks = mne.io.pick.pick_types(indathndl.info, meg=meg, ref_meg=False,
                                       eeg=False, stim=False, exclude=exclude)
        data = indathndl._data[picks,:]
    else:
        raise ValueError(">>>>> datafile is neither 'ave' nor 'raw'. Aborting-")
    if verbosity >= 3:
        print "data.shape = ",data.shape
    if n_sens != data.shape[0]:
        raise ValueError(">>>>> Mismatch in #channels for forward (%d) and data (%d) files. Aborting." %
                         (n_sens, data.shape[0]))

    tptotwall = 0.
    tptotcpu  = 0.
    nptotcall = 0
    tltotwall = 0.
    tltotcpu  = 0.
    nltotcall = 0
    tpcdmwall = 0.
    tpcdmcpu  = 0.
    npcdmcall = 0
    if verbosity >= 1:
        print "########## Calculate initial prob-dist:"
    tw0 = time.time()
    tc0 = time.clock()
    if mftpar['prbfct'] == 'Gauss':
        wtmp = np.zeros(n_loc/3)
        for icnt in xrange(prbcnt.shape[0]):
            testdiff = fwdmag['source_rr']-prbcnt[icnt,:]
            testdiff = testdiff/prbdhw[icnt,:]
            testdiff = testdiff*testdiff
            testsq = np.sum(testdiff,1)
            wtmp  += np.exp(-testsq)
        wdist0 = wtmp/(np.sum(wtmp)*np.sqrt(3.))
    elif mftpar['prbfct'] == 'flat' or mftpar['prbfct'] == 'uniform':
        if verbosity >= 2:
            print "Setting w=const !"
        wdist0 = np.ones(n_loc/3)/(float(n_loc)/np.sqrt(3.))
    else:
        raise ValueError(">>>>> mftpar['prbfct'] must be 'Gauss' or 'uniform'/'flat'")
    wdist3 = np.repeat(wdist0,3)
    if verbosity >= 3:
        wvecnorm = np.sum(np.sqrt(np.sum(np.reshape(wdist3, (wdist3.shape[0]/3,3))**2,axis=1)))
        print "sum(||wvec(i)||) = ",wvecnorm
    tc1 = time.clock()
    tw1 = time.time()
    if verbosity >= 1:
        print "calc(wdist0) took %.3f" % (1000.*(tc1-tc0)),"ms (%.3f s walltime)" % (tw1-tw0)

    if verbosity >= 1:
        print "########## Calculate P-matrix, incl. weights:"
    tw0 = time.time()
    tc0 = time.clock()
    wdist3rt = np.repeat(np.sqrt(wdist0),3)
    lfw = lfmag*wdist3rt
    pmat0 = np.einsum('ik,jk->ij',lfw,lfw)
    tc1 = time.clock()
    tw1 = time.time()
    tptotwall += (tw1-tw0)
    tptotcpu  += (tc1-tc0)
    nptotcall += 1
    if verbosity >= 1:
        print "calc(lf*w*lf.T) took ", 1000.*(tc1-tc0), "ms (%.3f s walltime)" % (tw1-tw0)

    # Normalize P:
    pmax = np.amax([np.abs(np.amax(pmat0)),np.abs(np.amin(pmat0))])
    if verbosity >= 3:
        print "pmax(init) = ",pmax
    pscalefct = 1.
    while pmax > 1.0:
       pmax      /= 2.
       pscalefct /= 2.
    while pmax < 0.5:
       pmax      *= 2.
       pscalefct *= 2.
    #print ">>>>> Keeping scale factor eq 1"
    #pscalefct = 1.
    pmat0 = pmat0*pscalefct
    if verbosity >= 3:
        print "pmax(fin.) = ",np.amax([np.abs(np.amax(pmat0)),np.abs(np.amin(pmat0))])

    # Regularize P:
    if mftpar['regtype'] == 'PzetaE':
        zetatrp = mftpar['zetareg']*np.trace(pmat0)/float(pmat0.shape[0])
        if verbosity >= 3:
            print "Use PzetaE-regularization with zeta*tr(P)/ncol(P) = %12.5e" % zetatrp
        ptilde0 = pmat0 + zetatrp*np.identity(pmat0.shape[0])
    elif mftpar['regtype'] == 'classic' or mftpar['regtype'] == 'PPzetaP':
        zetatrp = mftpar['zetareg']*np.trace(pmat0)/float(pmat0.shape[0])
        if verbosity >= 3:
            print "Use PPzetaP-regularization with zeta*tr(P)/ncol(P) = %12.5e" % zetatrp
        ptilde0 = np.dot(pmat0,pmat0) + zetatrp*pmat0
    else:
        raise ValueError(">>>>> mftpar['regtype'] must be 'PzetaE' or 'classic''")

    # decompose:
    if use_lud is True:
        LU0,P0 = scipy.linalg.lu_factor(ptilde0)
        #rhstmp = np.zeros([LU0.shape[1]])
        #xtmp = np.empty([LU0.shape[1]])
        #xtmp = scipy.linalg.lu_solve((LU0,P0),rhstmp)
        if verbosity >= 3:
            # Calculate condition number:
            #(sign, lndetbf) = np.linalg.slogdet(ptilde0)
            lndettr = np.sum(np.log(np.abs(np.diag(LU0))))
            #print "lndet(ptilde0) = %8.3f =?= %8.3f = sum(log(|diag(LU0)|))" % (lndetbf,lndettr)
            # log(prod(a_i, i=1,n)) for a_i = sqrt(sum(ptilde0_ij^2, j=1,n))
            denom = np.sum(np.log(np.sqrt(np.sum(ptilde0*ptilde0,axis=0))))
            lncondno = lndettr - denom
            print "ln(condno) = %8.3f, K_H = 10^(%8.3f) = %8.3f" % (lncondno,lncondno/np.log(10.),np.exp(lncondno))
            print "(K_H < 0.01 : bad, K_H > 0.1 : good)"

    if use_svd is True:
        U, s, V = np.linalg.svd(ptilde0,full_matrices=True)
        dtmp = s.max()*svrelcut
        s *= (abs(s)>=dtmp)
        sinv = [1./s[k] if s[k]!=0. else 0. for k in  xrange(ptilde0.shape[0])]
        if verbosity >= 2:
            print ">>> With rel-cutoff=%e   %d out of %d SVs remain" % \
                  (svrelcut,np.array(np.nonzero(sinv)).shape[1],len(sinv))
        if verbosity >= 3:
            stat = np.allclose(ptilde0, np.dot(U,np.dot(np.diag(s),V)))
            print ">>> Testing svd-result: %s" % stat
            if not stat:
                print "    (Maybe due to SV-cutoff?)"
            print ">>> Setting ptildeinv=(U diag(sinv) V).tr"
        ptilde0inv = np.transpose(np.dot(U,np.dot(np.diag(sinv),V)))
        if verbosity >= 3:
            stat = np.allclose(np.identity(ptilde0.shape[0]),np.dot(ptilde0inv,ptilde0))
            if stat:
                print ">>> Testing ptilde0inv-result (shld be unit-matrix): ok"
            else:
                print ">>> Testing ptilde0inv-result (shld be unit-matrix): failed"
                print np.transpose(np.dot(ptilde0inv,ptilde0))
                print ">>>"

    if verbosity >= 1:
        print "########## Create stc data and qual data arrays:"
    qualdata = { 'relerr':np.zeros(data.shape[1]), 'rdmerr':np.zeros(data.shape[1]), 'mag':np.zeros(data.shape[1]) }
    stcdata = np.zeros([n_loc/3, data.shape[1]])
    stcdata1 = [np.zeros([s['nuse'], data.shape[1]]) for s in fwdmag['src']]
    stcinds = np.zeros((n_loc/3, 2), dtype=int)
    stcinds1 = np.zeros((n_loc/3), dtype=int)
    offsets = np.append([0], [s['nuse'] for s in fwdmag['src']])
    iblck = -1
    nmatch = 0
    for s in fwdmag['src']:
        iblck = iblck + 1
        for kvert0 in  xrange(s['nuse']):
            kvert1 = offsets[iblck] + kvert0
            if np.all(np.equal(fwdmag['source_rr'][kvert1],s['rr'][s['vertno'][kvert0]])):
                stcinds[kvert1][0] = iblck
                stcinds[kvert1][1] = kvert0
                nmatch = nmatch + 1
    if verbosity >= 3:
        print "Found %d matches in creating source_rr/rr index table." % nmatch

    if verbosity >= 2:
        print "Reading slices of data to calc. cdv:"
        if data.shape[1]>1000:
            print " "
    for islice in xrange(data.shape[1]):
        wdist  = np.copy(wdist0)
        wdist3 = np.repeat(wdist,3)
        pmat = np.copy(pmat0)
        ptilde = np.copy(ptilde0)
        if use_svd is True:
            ptildeinv = np.copy(ptilde0inv)
        if use_lud is True:
            LU = np.copy(LU0)
            P = np.copy(P0)

        slice = pscalefct*data[:,islice]
        if mftpar['regtype'] == 'PzetaE':
            mtilde = np.copy(slice)
        else:
            mtilde = np.dot(pmat,slice)

        acoeff = np.empty([ptilde.shape[0]])
        if use_svd is True:
            for irow in xrange(ptilde.shape[0]):
                acoeff[irow] = np.dot(ptildeinv[irow,:],mtilde)
        if use_lud is True:
            acoeff = scipy.linalg.lu_solve((LU,P),mtilde)

        cdv = np.zeros(n_loc)
        cdvnorms = np.zeros(n_loc/3)
        for krow in xrange(lfmag.shape[0]):
            lfwtmp = lfmag[krow,:]*wdist3
            cdv += acoeff[krow]*lfwtmp

        tlw0 = time.time()
        tlc0 = time.clock()
        for mftiter in xrange(mftpar['iter']):
            # MFT iteration loop:

            cdvecs = np.reshape(cdv, (cdv.shape[0]/3,3))
            cdvnorms = np.sqrt(np.sum(cdvecs**2,axis=1))

            wdist = np.power(cdvnorms,mftpar['currexp'])*wdist0
            wdistsum = np.sum(wdist)
            wdist = wdist/wdistsum
            wdist3 = np.repeat(wdist,3)

            # Calculate new P-matrix, incl. weights:
            tw0 = time.time()
            tc0 = time.clock()
            wdist3rt = np.repeat(np.sqrt(pscalefct*wdist),3)
            lfw = lfmag*wdist3rt
            pmat = np.einsum('ik,jk->ij',lfw,lfw)

            tc1 = time.clock()
            tw1 = time.time()
            tptotwall += (tw1-tw0)
            tptotcpu  += (tc1-tc0)
            nptotcall += 1

            # Regularize P:
            if mftpar['regtype'] == 'PzetaE':
                ptilde = pmat + zetatrp*np.identity(pmat.shape[0])
            else:
                ptilde = np.dot(pmat,pmat) + zetatrp*pmat

            # decompose:
            if use_svd is True:
                U, s, V = np.linalg.svd(ptilde,full_matrices=True)
                dtmp = s.max()*svrelcut
                s *= (abs(s)>=dtmp)
                sinv = [1./s[k] if s[k]!=0. else 0. for k in  xrange(ptilde.shape[0])]
                ptildeinv = np.transpose(np.dot(U,np.dot(np.diag(sinv),V)))
                for irow in xrange(ptilde.shape[0]):
                    acoeff[irow] = np.dot(ptildeinv[irow,:],mtilde)
            if use_lud is True:
                LU,P = scipy.linalg.lu_factor(ptilde)
                acoeff = scipy.linalg.lu_solve((LU,P),mtilde)

            cdv = np.einsum('ji,i,j->i',lfmag,wdist3,acoeff)

        tc1 = time.clock()
        tw1 = time.time()
        tltotwall += (tw1-tlw0)
        tltotcpu  += (tc1-tlc0)
        nltotcall += 1
        cdvecs = np.reshape(cdv, (cdv.shape[0]/3,3))
        cdvnorms = np.sqrt(np.sum(cdvecs**2,axis=1))
        # (relerr,rdmerr,mag) = compare_est_exp(ptilde,acoeff,mtilde)
        (relerr, rdmerr, mag) = compare_est_exp(pmat, acoeff, slice)
        qualdata['relerr'][islice] = relerr
        qualdata['rdmerr'][islice] = rdmerr
        qualdata['mag'][islice] = mag

        tc0 = time.clock()
        tw0 = time.time()
        tc1 = time.clock()
        tw1 = time.time()
        tpcdmwall += (tw1-tw0)
        tpcdmcpu  += (tc1-tc0)
        npcdmcall += 1

        # Write final cdv to file:
        for iloc in xrange(n_loc/3):
            #stcdata1[stcinds[iloc][0]][stcinds[iloc][1],islice] = cdvnorms[iloc]
            stcdata[iloc, islice] = cdvnorms[iloc]
        del wdist
        if verbosity >= 2 and islice>0 and islice%1000==0:
            print "\r%6d out of %6d slices done." % (islice,data.shape[1])
    if verbosity >= 2 and data.shape[1]>1000:
        print "Done."

    vertices = [s['vertno'] for s in fwdmag['src']]
    tstep = 1./indathndl.info['sfreq']
    tmin = indathndl.times[0]
    if len(vertices) == 1:
        stc_mft = mne.VolSourceEstimate(stcdata, vertices=fwdmag['src'][0]['vertno'],
                                        tmin=tmin, tstep=tstep, subject=subject)
    else:
        vertices = [s['vertno'] for s in fwdmag['src']]
        stc_mft = mne.SourceEstimate(stcdata, vertices=vertices,
                                     tmin=tmin, tstep=tstep, subject=subject)

    stcdatamft = stc_mft.data
    print "##### Results:"
    for islice in xrange(data.shape[1]):
        print "slice=%4d: relerr=%9.3e rdmerr=%9.3e mag=%5.3f cdvmax=%9.2e" % \
              (islice, qualdata['relerr'][islice], qualdata['rdmerr'][islice], qualdata['mag'][islice],\
               np.amax(stcdatamft[:, islice]))
        #     (islice,qualdata[0,islice],qualdata[1,islice],qualdata[2,islice], \
        #     np.amax([np.amax(sb[:,islice]) for sb in stcdatamft]))
    stat = np.allclose(stcdata, stcdatamft, atol=0., rtol=1e-07)
    if stat:
        print "stcdata from mft-SR and old calc agree."
    else:
        print "stcdata from mft-SR and old calc DIFFER."

    if save_stc:
        # save Surface stc.
        print "##### Trying to save stc:"
        stcmft_fname = os.path.join(os.path.dirname(datafile),
                                    os.path.basename(datafile).split('-')[0]) + "mft"
        print "stcmft basefilename: %s" % stcmft_fname
        stc_mft.save(stcmft_fname, verbose=True)
        print "##### done."

    write_tab_files = True
    if write_tab_files:
        time_idx = np.argmax(np.max(stcdata, axis=0))
        tabfilenam = 'testtab.dat'
        print "##### Creating %s with |cdv(time_idx=%d)|" % (tabfilenam, time_idx)
        tabfile = open(tabfilenam, mode='w')
        cdvnmax = np.max(stcdata[:, time_idx])
        tabfile.write("# time_idx = %d\n" % time_idx)
        tabfile.write("# max amplitude = %11.4e\n" % cdvnmax)
        tabfile.write("#  x/mm    y/mm    z/mm     |cdv|   index\n")
        for ipnt in xrange(n_loc/3):
            copnt = 1000.*fwdmag['source_rr'][ipnt]
            tabfile.write(" %7.2f %7.2f %7.2f %11.4e %5d\n" % \
                          (copnt[0], copnt[1], copnt[2], stcdata[ipnt, time_idx], ipnt))
        tabfile.close()

        tabfilenam = 'testwtab.dat'
        print "##### Creating %s with wdist0" % tabfilenam
        tabfile = open(tabfilenam,mode='w')
        tabfile.write("# time_idx = %d\n" % time_idx)
        for icnt in xrange(prbcnt.shape[0]):
            cocnt = 1000.*prbcnt[icnt,:]
            tabfile.write("# center  %7.2f %7.2f %7.2f\n" % (cocnt[0], cocnt[1], cocnt[2]))

        tabfile.write("# max value = %11.4e\n" % np.max(wdist0))
        tabfile.write("#  x/mm    y/mm    z/mm    wdist0   index")
        for icnt in xrange(prbcnt.shape[0]):
            tabfile.write("  d_%d/mm" % (icnt+1))
        tabfile.write("\n")
        for ipnt in xrange(n_loc/3):
            copnt = 1000.*fwdmag['source_rr'][ipnt]
            tabfile.write(" %7.2f %7.2f %7.2f %11.4e %5d" %\
                          (copnt[0],copnt[1],copnt[2],wdist0[ipnt],ipnt))
            for icnt in xrange(prbcnt.shape[0]):
                cocnt = 1000.*prbcnt[icnt,:]
                dist = np.sqrt(np.dot((copnt-cocnt),(copnt-cocnt)))
                tabfile.write("  %7.2f" % dist)
            tabfile.write("\n")
        tabfile.close()

    twgbl1 = time.time()
    tcgbl1 = time.clock()
    if verbosity >= 1:
        print "calc(lf*w*lf.T) took   total  %9.2f s CPU-time (%9.2f s walltime)" % (tptotcpu,tptotwall)
        print "calc(lf*w*lf.T) took per call %9.2fms CPU-time (%9.2fms walltime)" % \
                               (1000.*tptotcpu/float(nptotcall),1000.*tptotwall/float(nptotcall))
        print "iteration-loops took   total  %9.2f s CPU-time (%9.2f s walltime)" % (tltotcpu,tltotwall)
        print "iteration-loops took per call %9.2fms CPU-time (%9.2fms walltime)" % \
                               (1000.*tltotcpu/float(nltotcall),1000.*tltotwall/float(nltotcall))
        print "Total mft-call  took   total  %9.2f s CPU-time (%9.2f s walltime)" % ((tcgbl1-tcgbl0),(twgbl1-twgbl0))

    return (fwdmag, qualdata, stc_mft)
