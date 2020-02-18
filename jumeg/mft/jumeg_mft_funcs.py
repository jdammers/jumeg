"""
====================
Jumeg MFT Functions.
====================
"""

# Author: Eberhard Eich
# Version: 190118
# License: BSD (3-clause)

import os
import numpy as np
import scipy
import scipy.linalg
import time
import warnings

import mne
from mne.io.constants import FIFF

TINY = 1.e-38
LOG_ITERWDIST = False


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
    for k in range(pmat.shape[0]):
        mest[k] = np.dot(pmat[:, k], acoeff)
    sumestsq = np.dot(mest, mest)
    sumexpsq = np.dot(mexp, mexp)
    if sumexpsq < TINY or sumestsq < TINY:
        return (-1., -1., -1.)
    estnorm = np.sqrt(sumestsq)
    expnorm = np.sqrt(sumexpsq)
    mdiff = mest - mexp
    rdmdiff = mest / estnorm - mexp / expnorm
    rdmerr = np.sqrt(np.dot(rdmdiff, rdmdiff))
    sumdiffsq = np.dot(mdiff, mdiff)
    return (sumdiffsq / sumestsq, rdmerr, estnorm / expnorm)


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
    (Somewhat sloppy, but fast)

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
        cdvecs = np.reshape(cdv, (cdv.shape[0] // 3, 3))
    elif len(cdv.shape) == 2 and cdv.shape[1] == 3:
        cdvecs = cdv
    else:
        raise ValueError(">>>>> Illegal array shape in call")
    if cdvcut < 0. or cdvcut >= 1.:
        raise ValueError(">>>>> cdvcut must be in [0,1)")

    cdvsq = np.sum(cdvecs ** 2, axis=1)
    icdvmax = cdvsq.argmax()
    cdvmaxsq = cdvsq[icdvmax]
    cdvlimsq = cdvcut * cdvcut * cdvmaxsq
    cdvsq *= (cdvsq >= cdvlimsq)
    icdvpos = np.nonzero(cdvsq)[0]
    dsumpabs = np.sum(np.sqrt(cdvsq[icdvpos]))
    cdvmaxvec = cdvecs[icdvmax, :]
    sprod = np.dot(cdvecs[icdvpos, :], cdvmaxvec)
    dsumplong = sprod[sprod > 0].sum() / np.sqrt(cdvmaxsq)
    return (dsumplong / dsumpabs, dsumplong)


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
    (Slow, but rather predictable)

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
        cdvecs = np.reshape(cdv, (cdv.shape[0] // 3, 3))
    elif len(cdv.shape) == 2 and cdv.shape[1] == 3:
        cdvecs = cdv
    else:
        raise ValueError(">>>>> Illegal array shape in call")
    if cdvcut < 0. or cdvcut >= 1.:
        raise ValueError(">>>>> cdvcut must be in [0,1)")

    nphi = 20
    ntheta = 11
    cdvsq = np.sum(cdvecs ** 2, axis=1)
    icdvmax = cdvsq.argmax()
    cdvmaxsq = cdvsq[icdvmax]
    cdvlimsq = cdvcut * cdvcut * cdvmaxsq
    cdvsq *= (cdvsq >= cdvlimsq)
    icdvpos = np.nonzero(cdvsq)[0]
    dsumpabs = np.sum(np.sqrt(cdvsq[icdvpos]))
    cdvmaxvec = cdvecs[icdvmax, :]
    sprod = np.dot(cdvecs[icdvpos, :], cdvmaxvec)
    dsumplong = sprod[sprod > 0].sum() / np.sqrt(cdvmaxsq)

    for iphi in range(nphi):
        aphi = np.pi * float(2 * iphi) / float(nphi)
        for itheta in range(ntheta):
            atheta = np.pi * float(itheta) / float(ntheta)
            cdvtstvec = [np.cos(aphi) * np.sin(atheta), np.sin(aphi) * np.sin(atheta), np.cos(atheta)]
            sprod = np.dot(cdvecs[icdvpos, :], cdvtstvec)
            dsumplongtst = sprod[sprod > 0].sum()
            if dsumplongtst > dsumplong:
                dsumplong = dsumplongtst

    return (dsumplong / dsumpabs, dsumplong)


########################################################################
# Calculate cdm from current-density-vectors with cut
#  cutcdm = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0) /
#           sum(|cdv[i]|, all i w/ |cdv(i)|>cut*cdvmax)
########################################################################
def scan_cdm_w_cut(cdv, cdvcut):
    """Calculate cdm from current-density-vectors with cut

    Find cdm, the current directionality measure, scanning for the
    best cdv-direction.
    Returns also jlong, the total current in that direction
     cutcdm = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0) /
              sum(|cdv[i]|, all i w/ |cdv(i)|>cut*cdvmax)
     cutjlong = sum(cdvmax^*cdv[i], i with cdvmax^*cdv[i]>0),
                            all i w/ |cdv(i)|>cut*cdvmax)
    (following pure doctrine, but timing not very predictable)

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
        cdvecs = np.reshape(cdv, (cdv.shape[0] // 3, 3))
    elif len(cdv.shape) == 2 and cdv.shape[1] == 3:
        cdvecs = cdv
    else:
        raise ValueError(">>>>> Illegal array shape in call")
    if cdvcut < 0. or cdvcut >= 1.:
        raise ValueError(">>>>> cdvcut must be in [0,1)")

    cdvsq = np.sum(cdvecs ** 2, axis=1)
    if cdvsq.shape[0] == 0:
        print(">>>Warning>> scan_cdm_w_cut(): cdvsq-array empty.")
        return (0., 0.)
    icdvmax = cdvsq.argmax()
    cdvmaxsq = cdvsq[icdvmax]
    cdvlimsq = cdvcut * cdvcut * cdvmaxsq
    cdvsq *= (cdvsq >= cdvlimsq)
    icdvpos = np.nonzero(cdvsq)[0]
    dsumpabs = np.sum(np.sqrt(cdvsq[icdvpos]))
    cdvmaxvec = cdvecs[icdvmax, :]
    sprod = np.dot(cdvecs[icdvpos, :], cdvmaxvec)
    dsumplong = sprod[sprod > 0].sum() / np.sqrt(cdvmaxsq)

    for ivec in icdvpos:
        cdvtstvec = cdvecs[ivec, :]
        sprod = np.dot(cdvecs[icdvpos, :], cdvtstvec)
        dsumplongtst = sprod[sprod > 0].sum() / np.sqrt(cdvsq[ivec])
        if dsumplongtst > dsumplong:
            dsumplong = dsumplongtst

    return (dsumplong / dsumpabs, dsumplong)


########################################################################
# Calculate total current from current-density-vectors with cut
#  cutjtot = sum(|cdv[i]|, all i w/ |cdv(i)|>cut*cdvmax)
########################################################################
def calc_jtotal_w_cut(cdv, cdvcut):
    """Calculate total current from current-density-vectors with cut

    Returns
     cutjtot = sum(|cdv[i]|, i w/ |cdv(i)|>cut*cdvmax)

    Parameters
    ----------
    cdv: np-array w current density vectors
    cdvcut: cut for cdv-norms [0,1) of |cdvmax|

    Returns
    -------
    cutjtot
    or -1 in case of error
    """
    if len(cdv.shape) == 1:
        cdvecs = np.reshape(cdv, (cdv.shape[0] // 3, 3))
    elif len(cdv.shape) == 2 and cdv.shape[1] == 3:
        cdvecs = cdv
    else:
        raise ValueError(">>>>> Illegal array shape in call")
    if cdvcut < 0. or cdvcut >= 1.:
        raise ValueError(">>>>> cdvcut must be in [0,1)")

    cdvsq = np.sum(cdvecs ** 2, axis=1)
    if cdvsq.shape[0] == 0:
        print(">>>Warning>> scan_cdm_w_cut(): cdvsq-array empty.")
        return (0.)
    icdvmax = cdvsq.argmax()
    cdvmaxsq = cdvsq[icdvmax]
    cdvlimsq = cdvcut * cdvcut * cdvmaxsq
    cdvsq *= (cdvsq >= cdvlimsq)
    icdvpos = np.nonzero(cdvsq)[0]
    dsumpabs = np.sum(np.sqrt(cdvsq[icdvpos]))
    return (dsumpabs)


########################################################################
# Apply MFT to the specified data for the specified parameters
#  and return stcs with length of current density vectors for
#  src-points and time slices.
#
########################################################################
def apply_mft(fwdspec, dataspec, evocondition=None, meg='mag',
              exclude='bads', mftpar=None, iterlist=None,
              calccdm=None, cdmcut=0., cdmlabels=None,
              subject=None, save_stc=True, verbose=False):
    """ Apply MFT to specified data set.

    Parameters
    ----------
    fwdspec: name of forward solution file or forward object
    dataspec: name of datafile (ave, epo, or raw)
              or evoked-, epoch-, or raw-datahandle
              Caveat: can only handle one epoch at a time
    evocondition: condition in case of evoked input file
    meg: meg-channels to pick ['mag']
    exclude: meg-channels to exclude ['bads']
    mftpar: dictionary with parameters for MFT algorithm
    iterlist: list | None
              optional list of iterations, for which data are returned
              its highest entry must match mftpar['iter']
    calccdm : str | None
              where str can be 'all', 'both', 'left', 'right'
    cdmcut : (rel.) cut to use in cdm-calculations [0.]
    cdmlabels: list of labels to analyse
               entries for 'cdmlabels', 'jlglabels', 'jtotlabels'
               in qualdata are returned, containing cdm,
               longitudinal and total current for each label.
    subject : str | None
        The subject name. While not necessary, it is safer to set the
        subject parameter to avoid analysis errors.
    save_stc : Boolean controlling saving stcdata  to a file named
               as datafile with 'raw','epo', 'ave' replaced by 'mft'
               Requires dataspec with a filename
    verbose: control variable for verbosity
             False='CRITICAL','WARNING',True='INFO','DEBUG'
             or 'chatty'='verbose' (>INFO,<DEBUG)

    Returns
    fwdmag: forward object restricted to (selected) meg-channels
    qualdata: dictionary with relerr,rdmerr,mag-arrays and
              cdm-arrays (if requested),
              always refers to highest iteration
    stcdata: stc with ||cdv|| at fwdmag['source_rr']
             (type corresponding to forward solution)
             list of stcs in case of iterlist!=None
    """
    twgbl0 = time.time()
    tcgbl0 = time.clock()

    # Use mftparm as local copy of mftpar to keep that ro.
    mftparm = {}
    if mftpar:
        mftparm.update(mftpar)
        mftparm['solver'] = mftparm['solver'].lower()
        mftparm['prbfct'] = mftparm['prbfct'].lower()
        mftparm['regtype'] = mftparm['regtype'].lower()

    mftparm.setdefault('iter', 8)
    mftparm.setdefault('currexp', 1)
    mftparm.setdefault('prbfct', 'uniform')
    mftparm.setdefault('prbcnt')
    mftparm.setdefault('prbhw')
    mftparm.setdefault('prbxfm')
    mftparm.setdefault('prblog')
    mftparm.setdefault('regtype', 'PzetaE')
    mftparm.setdefault('zetareg', 1.00)
    mftparm.setdefault('solver', 'lu')
    mftparm.setdefault('svrelcut', 5.e-4)

    if mftparm['solver'] == 'svd':
        use_svd = True
        use_lud = False
        svrelcut = mftparm['svrelcut']
    elif mftparm['solver'] == 'lu' or mftparm['solver'] == 'ludecomp':
        use_lud = True
        use_svd = False
    else:
        raise ValueError(">>>>> mftpar['solver'] must be either 'svd' or 'lu[decomp]'")

    if mftparm['prbfct'] == 'gauss':
        if not isinstance(mftparm['prbcnt'], np.ndarray) or mftparm['prbcnt'].shape[-1] != 3 or \
                not isinstance(mftparm['prbhw'], np.ndarray) or mftparm['prbhw'].shape[-1] != 3 or \
                not mftparm['prbhw'].all():
            raise ValueError(">>>>> 'prbfct'='Gauss' requires 'prbcnt' and 'prbhw' entries")
        if mftparm['prbxfm'] is None:
            prbxfm = np.identity(4)
        elif isinstance(mftparm['prbxfm'], np.ndarray) and mftparm['prbxfm'].shape == (4, 4):
            prbxfm = mftparm['prbxfm']
        else:
            raise ValueError(">>>>> 'prbxfm' must specify a xfm (or None)")
    elif mftparm['prbfct'] != 'uniform' and \
            mftparm['prbfct'] != 'flat' and \
            mftparm['prbfct'] != 'random':
        raise ValueError(">>>>> unrecognized keyword for 'prbfct'")
    if mftparm['prbcnt'] is None and mftparm['prbhw'] is None:
        prbcnt = np.array([0.0, 0.0, 0.0], ndmin=2)
        prbdhw = np.array([0.0, 0.0, 0.0], ndmin=2)
    else:
        prbcnt = np.reshape(mftparm['prbcnt'], (len(mftparm['prbcnt'].flatten()) // 3, 3))
        prbdhw = np.reshape(mftparm['prbhw'], (len(mftparm['prbhw'].flatten()) // 3, 3))
    if prbcnt.shape != prbdhw.shape:
        raise ValueError(">>>>> mftpar['prbcnt'] and mftpar['prbhw'] must have same size")

    verbosity = 1
    if verbose is False or verbose == 'CRITICAL':
        verbosity = -1
    elif verbose == 'WARNING':
        verbosity = 0
    elif verbose == 'chatty' or verbose == 'verbose':
        verbose = 'INFO'
        verbosity = 2
    elif verbose == 'DEBUG':
        verbosity = 3

    if verbosity >= 0:
        print("meg-channels     = ", meg)
        print("exclude-channels = ", exclude)
        print("mftpar['iter'    ] = ", mftparm['iter'])
        print("mftpar['currexp' ] = ", mftparm['currexp'])
        print("mftpar['regtype' ] = ", mftparm['regtype'])
        print("mftpar['zetareg' ] = ", mftparm['zetareg'])
        print("mftpar['solver'  ] = ", mftparm['solver'])
        print("mftpar['svrelcut'] = ", mftparm['svrelcut'])
        print("mftpar['prbfct'  ] = ", mftparm['prbfct'])
        print("mftpar['prbcnt'  ] = ", mftparm['prbcnt'])
        print("mftpar['prbhw'   ] = ", mftparm['prbhw'])
        if mftparm['prbcnt'] is not None or mftparm['prbhw'] is not None:
            for icnt in range(prbcnt.shape[0]):
                print("  pos(prbcnt[%d])   = " % (icnt + 1), prbcnt[icnt])
                print("  dhw(prbdhw[%d])   = " % (icnt + 1), prbdhw[icnt])
        if mftparm['prbxfm'] is None:
            print("mftpar['prbxfm'  ] = None")
        else:
            print("mftpar['prbxfm'  ] = ", mftparm['prbxfm'])
        if isinstance(mftparm['prblog'], str):
            print("mftpar['prblog'  ] = ", mftparm['prblog'])
        if calccdm:
            print("calccdm = '%s' with rel. cut = %5.2f" % (calccdm, cdmcut))
    if calccdm and (cdmcut < 0. or cdmcut >= 1.):
        raise ValueError(">>>>> cdmcut must be in [0,1)")
    if LOG_ITERWDIST:
        print(">>> Usg w_{n+1} ~ |J_n|^p * w_n")

    if isinstance(iterlist, list) and len(iterlist) > 0:
        if max(iterlist) != mftparm['iter']:
            raise ValueError(
                ">>>>> apply_mft() if an iterlist is specified, its highest entry must match mftpar['iter']")

    if isinstance(fwdspec, str):
        # Msg will be written by mne.read_forward_solution()
        fwd = mne.read_forward_solution(fwdspec, verbose=verbose)
    elif isinstance(fwdspec, dict):
        if 'info' not in fwdspec or 'sol' not in fwdspec:
            raise ValueError(">>>>> apply_mft() first argument not indicating a fwd-solution")
        fwd = fwdspec
    else:
        raise ValueError(">>>>> apply_mft() first argument not indicating a fwd-solution")
    # Block off fixed_orientation fwd-s for now:
    if fwd['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
        raise ValueError(">>>>> apply_mft() cannot handle fixed-orientation fwd-solutions")

    # Select magnetometer channels:
    fwdmag = mne.io.pick.pick_types_forward(fwd, meg=meg, ref_meg=False,
                                            eeg=False, exclude=exclude)
    lfmag = fwdmag['sol']['data']
    del fwd

    n_sens, n_loc = lfmag.shape
    n_srcspace = len([s['vertno'] for s in fwdmag['src']])
    if verbosity >= 2:
        print("Leadfield size : n_sen x n_loc = %d x %d" % (n_sens, n_loc))
        print("Number of source spaces = %d" % n_srcspace)

    if cdmlabels is not None:
        if verbosity >= 1:
            print("########## Searching for label(s) in source space(s)...")
        tc0 = time.clock()
        tw0 = time.time()

    numcdmlabels = 0
    labvrtstot = 0
    labvrtsusd = 0
    if cdmlabels is not None:
        invmri_head_t = mne.transforms.invert_transform(fwdmag['info']['mri_head_t'])
        mrsrcpnt = np.zeros(fwdmag['source_rr'].shape)
        mrsrcpnt = mne.transforms.apply_trans(invmri_head_t['trans'],
                                              fwdmag['source_rr'])
        offsets = [0]
        for s in fwdmag['src']:
            offsets = np.append(offsets, [offsets[-1] + s['nuse']])
        labinds = []
        ilab = 0
        for label in cdmlabels:
            ilab = ilab + 1

            labvrts = []
            # Find src corresponding to this label (match by position)
            # (Assume surface-labels are in head-cd, vol-labels in MR-cs)
            isrc = 0
            for s in fwdmag['src']:
                isrc += 1

                labvrts = label.get_vertices_used(vertices=s['vertno'])
                numlabvrts = len(labvrts)
                if numlabvrts == 0:
                    continue
                if not np.all(s['inuse'][labvrts]):
                    print("isrc = %d: label='%s' (np.all(s['inuse'][labvrts])=False)" % (isrc, label.name))
                    continue
                #    labindx: indices of used label-vertices in this src-space + offset2'source_rr'
                # iinlabused: indices of used label-vertices in this label
                labindx = np.searchsorted(s['vertno'], labvrts) + offsets[isrc - 1]
                iinlabused = np.searchsorted(label.vertices, labvrts)
                if s['type'] == 'surf':
                    if not np.allclose(mrsrcpnt[labindx, :], label.pos[iinlabused]):
                        continue  # mismatch
                else:
                    if not np.allclose(fwdmag['source_rr'][labindx, :], label.pos[iinlabused]):
                        continue  # mismatch
                if verbosity >= 1:
                    print("%3d %30s %7s: %5d verts %4d used" % \
                          (ilab, label.name, label.hemi, len(label.vertices), numlabvrts))
                break  # from src-space-loop

            if len(labvrts) > 0:
                labvrtstot += len(label.vertices)
                labvrtsusd += len(labvrts)
                labinds.append(labindx)
                numcdmlabels = len(labinds)
            else:
                warnings.warn('NO vertex found for label \'%s\' in any source space' % label.name)
        if verbosity >= 1:
            print("--> sums: %5d verts %4d used" % (labvrtstot, labvrtsusd))
            tc1 = time.clock()
            tw1 = time.time()
            print("prep. labels took %.3f" % (1000. * (tc1 - tc0)), "ms (%.3f s walltime)" % (tw1 - tw0))

    if isinstance(dataspec, str):
        if dataspec.rfind('-ave.fif') > 0 or dataspec.rfind('-ave.fif.gz') > 0:
            if verbosity >= 0:
                print("Reading evoked data from %s" % dataspec)
            datafile = dataspec
            if evocondition is None:
                # indatinfo = mne.io.read_info(dataspec)
                indathndl = mne.read_evokeds(dataspec,
                                             baseline=(None, 0), verbose=verbose)
                if len(indathndl) > 1:
                    raise ValueError(">>>>> need to specify a condition for this datafile. Aborting-")
                indathndl = indathndl[0]
            else:
                indathndl = mne.read_evokeds(dataspec, condition=evocondition,
                                             baseline=(None, 0), verbose=verbose)
                # if len(indathndl) > 1:
                #    raise ValueError(">>>>> need to specify a condition for this datafile. Aborting-")
            picks = mne.io.pick.pick_types(indathndl.info, meg=meg, ref_meg=False,
                                           eeg=False, stim=False, exclude=exclude)
            data = indathndl.data[picks, :]
        elif dataspec.rfind('-raw.fif') > 0 or dataspec.rfind('-raw.fif.gz') > 0:
            if verbosity >= 0:
                print("Reading raw data from %s" % dataspec)
            if evocondition is not None:
                raise ValueError(">>>>> evocondition must not be specified with raw data. Aborting-")
            indathndl = mne.io.Raw(dataspec, preload=True, verbose=verbose)
            picks = mne.io.pick.pick_types(indathndl.info, meg=meg, ref_meg=False,
                                           eeg=False, stim=False, exclude=exclude)
            data = indathndl._data[picks, :]
        elif dataspec.rfind('-epo.fif') > 0 or dataspec.rfind('-epo.fif.gz') > 0:
            if verbosity >= 0:
                print("Reading epoched data from %s" % dataspec)
            if evocondition is not None:
                raise ValueError(">>>>> evocondition must not be specified with epoch data. Aborting-")
            indathndl = mne.read_epochs(dataspec, preload=True, verbose=verbose)
            picks = mne.io.pick.pick_types(indathndl.info, meg=meg, ref_meg=False,
                                           eeg=False, stim=False, exclude=exclude)
            if len(indathndl._data.shape) == 3 and indathndl._data.shape[0] > 1:
                raise ValueError(">>>>> apply_mft() cannot handle more than one epoch at a time")
            data = indathndl._data[0, picks, :]
        else:
            raise ValueError(">>>>> datafile is neither 'ave' nor 'epo' nor 'raw'. Aborting-")
    elif isinstance(dataspec, dict):
        print("##### Check how we got here #####")
        if 'info' not in dataspec:  # or not type(dataspec)==mne.evoked.Evoked:
            raise ValueError(">>>>> apply_mft() second argument not indicating a data-handle")
        indathndl = dataspec
        if save_stc:
            raise ValueError(">>>>> saving stc to file only possible with named imput datafile. Aborting-")
    elif type(dataspec) == mne.evoked.Evoked or type(dataspec) == mne.evoked.EvokedArray:
        if isinstance(dataspec.info, dict):
            if isinstance(evocondition, str) and not dataspec.comment == evocondition:
                raise ValueError(">>>>> apply_mft() mismatch of dataspec and specified condition")
            indathndl = dataspec
        else:
            raise ValueError(">>>>> apply_mft() second argument not indicating evoked data-handle")
        picks = mne.io.pick.pick_types(indathndl.info, meg=meg, ref_meg=False,
                                       eeg=False, stim=False, exclude=exclude)
        data = indathndl.data[picks, :]
        if save_stc:
            raise ValueError(">>>>> saving stc to file only possible with named imput datafile. Aborting-")
    elif type(dataspec) == mne.epochs.Epochs or type(dataspec) == mne.epochs.EpochsFIF:
        if isinstance(dataspec.info, dict):
            indathndl = dataspec
        else:
            raise ValueError(">>>>> apply_mft() second argument not indicating epochs data-handle")
        picks = mne.io.pick.pick_types(indathndl.info, meg=meg, ref_meg=False,
                                       eeg=False, stim=False, exclude=exclude)
        if len(indathndl._data.shape) == 3 and indathndl._data.shape[0] > 1:
            raise ValueError(">>>>> apply_mft() cannot handle more than one epoch at a time")
        data = indathndl._data[0, picks, :]
        if save_stc:
            raise ValueError(">>>>> saving stc to file only possible with named imput datafile. Aborting-")
        if evocondition is not None:
            raise ValueError(">>>>> evocondition must not be specified with epoch data. Aborting-")
    elif type(dataspec) == list and type(dataspec[0]) == mne.evoked.Evoked:
        if len(dataspec) > 1:
            if isinstance(evocondition, str):
                match = [dataspec[k].comment == evocondition for k in range(len(dataspec))]
                if len(np.where(match)[0]) == 1:
                    indathndl = dataspec[np.where(match)[0][0]]
                    print("> Found evoked data for condition ", indathndl.comment)
                else:
                    raise ValueError(">>>>> apply_mft() condition not found or unique in dataspec")
            else:
                raise ValueError(">>>>> apply_mft() cannot handle list of evokeds with length>1 wo condition")
        elif isinstance(dataspec.info, dict):
            indathndl = dataspec[0]
        else:
            raise ValueError(">>>>> apply_mft() second argument not indicating evoked data-handle")
        picks = mne.io.pick.pick_types(indathndl.info, meg=meg, ref_meg=False,
                                       eeg=False, stim=False, exclude=exclude)
        data = indathndl.data[picks, :]
        if save_stc:
            raise ValueError(">>>>> saving stc to file only possible with named imput datafile. Aborting-")
    elif type(dataspec) == mne.io.fiff.raw.Raw:
        indathndl = dataspec
        picks = mne.io.pick.pick_types(indathndl.info, meg=meg, ref_meg=False,
                                       eeg=False, stim=False, exclude=exclude)
        data = indathndl._data[picks, :]
        if evocondition is not None:
            raise ValueError(">>>>> evocondition must not be specified with raw data. Aborting-")
        if save_stc:
            raise ValueError(">>>>> saving stc to file only possible with named imput datafile. Aborting-")
    else:
        raise ValueError(">>>>> apply_mft() second argument not indicating a data-handle")
    if verbosity >= 3:
        print("data.shape = ", data.shape)
    # Test if datafile and fwd-soln might be compatible:
    if n_sens != data.shape[0]:
        raise ValueError(">>>>> Mismatch in #channels for forward (%d) and data (%d) files. Aborting." %
                         (n_sens, data.shape[0]))
    else:
        mismatch = False
        for isens in range(n_sens):
            if not np.allclose(indathndl.info['chs'][picks[isens]]['loc'], \
                               fwdmag['info']['chs'][isens]['loc']):
                print(">>>>> position mismatch data-forward_soln for '%s'" % \
                      indathndl.info['chs'][isens]['ch_name'])
                mismatch = True
        if mismatch is True:
            raise ValueError(">>>>> Mismatch in channel-geometry for forward and data files. Aborting.")

    tptotwall = 0.
    tptotcpu = 0.
    nptotcall = 0
    tltotwall = 0.
    tltotcpu = 0.
    nltotcall = 0
    tpcdmwall = 0.
    tpcdmcpu = 0.
    npcdmcall = 0
    if verbosity >= 1:
        print("########## Calculate initial prob-dist:")
    tw0 = time.time()
    tc0 = time.clock()
    if mftparm['prbfct'] == 'gauss':
        wtmp = np.zeros(n_loc // 3)
        for icnt in range(prbcnt.shape[0]):
            testdiff = fwdmag['source_rr'] - prbcnt[icnt, :]
            if mftparm['prbxfm'] is not None:
                tmpvecs = mne.transforms.apply_trans(prbxfm, testdiff, move=False)
                testdiff = tmpvecs
            testdiff = testdiff / prbdhw[icnt, :]
            testdiff = testdiff * testdiff
            testsq = np.sum(testdiff, 1)
            wtmp += np.exp(-testsq)
        wdist0 = wtmp / (np.sum(wtmp) * np.sqrt(3.))
    elif mftparm['prbfct'] == 'flat' or mftparm['prbfct'] == 'uniform':
        if verbosity >= 2:
            print("Setting initial w=const !")
        wdist0 = np.ones(int(n_loc / 3)) / (float(n_loc) / np.sqrt(3.))
    elif mftparm['prbfct'] == 'random':
        if verbosity >= 2:
            print("Setting initial w=random !")
        wdist0 = np.random.random_sample((n_loc / 3,))
        wdist0 /= (np.sum(wdist0) / np.sqrt(3.))
    else:
        raise ValueError(">>>>> mftpar['prbfct'] must be 'Gauss' or 'uniform'/'flat'")
    wdist3 = np.repeat(wdist0, 3)
    if verbosity >= 3:
        wvecnorm = np.sum(np.sqrt(np.sum(np.reshape(wdist3, (wdist3.shape[0] / 3, 3)) ** 2, axis=1)))
        print("sum(||wvec(i)||) = ", wvecnorm)
    if verbosity >= 2 and mftparm['prbfct'] == 'random':
        wvecnorm = np.sum(np.sqrt(np.sum(np.reshape(wdist3, (wdist3.shape[0] / 3, 3)) ** 2, axis=1)))
        print("sum(||wvec(i)||) = ", wvecnorm)
    tc1 = time.clock()
    tw1 = time.time()
    if isinstance(mftparm['prblog'], str):
        wdisttabfile = open(mftparm['prblog'], mode='w')
        wdisttabfile.write("# Initial probabillty distribution in head-CS\n")
        wdisttabfile.write("#\n")
        wdisttabfile.write("# index  x/mm    y/mm    z/mm     prob\n")
        for ipos in range(n_loc // 3):
            copnt = 1000. * fwdmag['source_rr'][ipos, :]
            wdisttabfile.write(" %5d  %7.2f %7.2f %7.2f  %12.5e\n" % \
                               (ipos, copnt[0], copnt[1], copnt[2], wdist0[ipos]))
        wdisttabfile.write("# sum(prob) = %6.4f\n" % np.sum(wdist0))
        wdisttabfile.close()
        if verbosity >= 1:
            print("Initial prob-dist written to ", mftparm['prblog'])
    if verbosity >= 1:
        print("calc(wdist0) took %.3f" % (1000. * (tc1 - tc0)), "ms (%.3f s walltime)" % (tw1 - tw0))

    if verbosity >= 1:
        print("########## Calculate P-matrix, incl. weights:")
    tw0 = time.time()
    tc0 = time.clock()
    # Split the leadfield matrix into submatrices for speed
    # cf. https://www.johndcook.com/blog/2018/08/31/how-fast-can-you-multiply-matrices/
    #     https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
    #     (A of size n x m, B of size m x p  with max(n, m, p) = m
    rtwd3 = np.repeat(np.sqrt(wdist0), 3)
    ibsize = 1024
    ilastseg = n_loc - n_loc % ibsize
    lfw = lfmag[:, ilastseg:] * rtwd3[ilastseg:]
    pmat0 = np.einsum('ik,jk->ij', lfw, lfw)
    for iseg in range(n_loc // ibsize):
        lfw = lfmag[:, iseg * ibsize:iseg * ibsize + ibsize] * rtwd3[iseg * ibsize:iseg * ibsize + ibsize]
        pmat0 += np.einsum('ik,jk->ij', lfw, lfw)
    # lfw = lfmag*np.repeat(np.sqrt(wdist0),3)
    # pmat1 = np.einsum('ik,jk->ij',lfw,lfw)
    # print "allclose(pmat0,pmat1) = ", np.allclose(pmat0,pmat1)
    # # Avoiding sqrt is expensive!
    # # pmat0 = np.einsum('ik, k, jk -> ij', lfmag, wdist3, lfmag)
    tc1 = time.clock()
    tw1 = time.time()
    tptotwall += (tw1 - tw0)
    tptotcpu += (tc1 - tc0)
    nptotcall += 1
    if verbosity >= 1:
        print("calc(lf*w*lf.T) took ", 1000. * (tc1 - tc0), "ms (%.3f s walltime)" % (tw1 - tw0))

    # Normalize P:
    pmax = np.amax([np.abs(np.amax(pmat0)), np.abs(np.amin(pmat0))])
    if verbosity >= 3:
        print("pmax(init) = ", pmax)
    pscalefct = 1.
    while pmax > 1.0:
        pmax /= 2.
        pscalefct /= 2.
    while pmax < 0.5:
        pmax *= 2.
        pscalefct *= 2.
    # print ">>>>> Keeping scale factor eq 1"
    # pscalefct = 1.
    pmat0 = pmat0 * pscalefct
    if verbosity >= 3:
        print("pmax(fin.) = ", np.amax([np.abs(np.amax(pmat0)), np.abs(np.amin(pmat0))]))

    # Regularize P:
    if mftparm['regtype'] == 'pzetae':
        zetatrp = mftparm['zetareg'] * np.trace(pmat0) / float(pmat0.shape[0])
        if verbosity >= 3:
            print("Use PzetaE-regularization with zeta*tr(P)/ncol(P) = %12.5e" % zetatrp)
        ptilde0 = pmat0 + zetatrp * np.identity(pmat0.shape[0])
    elif mftparm['regtype'] == 'classic' or mftparm['regtype'] == 'ppzetap':
        zetatrp = mftparm['zetareg'] * np.trace(pmat0) / float(pmat0.shape[0])
        if verbosity >= 3:
            print("Use PPzetaP-regularization with zeta*tr(P)/ncol(P) = %12.5e" % zetatrp)
        ptilde0 = np.dot(pmat0, pmat0) + zetatrp * pmat0
    else:
        raise ValueError(">>>>> mftpar['regtype'] must be 'PzetaE' or 'PPzetaP'/'classic''")

    # decompose:
    if use_lud is True:
        LU0, P0 = scipy.linalg.lu_factor(ptilde0)
        # rhstmp = np.zeros([LU0.shape[1]])
        # xtmp = np.empty([LU0.shape[1]])
        # xtmp = scipy.linalg.lu_solve((LU0,P0),rhstmp)
        if verbosity >= 2:
            # Calculate condition number:
            # (sign, lndetbf) = np.linalg.slogdet(ptilde0)
            lndettr = np.sum(np.log(np.abs(np.diag(LU0))))
            # print "lndet(ptilde0) = %8.3f =?= %8.3f = sum(log(|diag(LU0)|))" % (lndetbf,lndettr)
            # log(prod(a_i, i=1,n)) for a_i = sqrt(sum(ptilde0_ij^2, j=1,n))
            denom = np.sum(np.log(np.sqrt(np.sum(ptilde0 * ptilde0, axis=0))))
            lncondno = lndettr - denom
            print("ln(condno) = %8.3f, K_H = 10^(%8.3f) = %8.3f" % (lncondno, lncondno / np.log(10.), np.exp(lncondno)))
            print("(K_H < 0.01 : bad, K_H > 0.1 : good)")

    if use_svd is True:
        U, s, V = np.linalg.svd(ptilde0, full_matrices=True)
        if verbosity >= 2:
            print(">>> SV range %e ... %e" % (np.amax(s), np.amin(s)))
        dtmp = s.max() * svrelcut
        s *= (abs(s) >= dtmp)
        sinv = [1. / s[k] if s[k] != 0. else 0. for k in range(ptilde0.shape[0])]
        if verbosity >= 2:
            print(">>> With rel-cutoff=%e   %d out of %d SVs remain" % \
                  (svrelcut, np.array(np.nonzero(sinv)).shape[1], len(sinv)))
        if verbosity >= 3:
            stat = np.allclose(ptilde0, np.dot(U, np.dot(np.diag(s), V)))
            print(">>> Testing svd-result: %s" % stat)
            if not stat:
                print("    (Maybe due to SV-cutoff?)")
            print(">>> Setting ptildeinv=(U diag(sinv) V).tr")
        ptilde0inv = np.transpose(np.dot(U, np.dot(np.diag(sinv), V)))
        if verbosity >= 3:
            stat = np.allclose(np.identity(ptilde0.shape[0]), np.dot(ptilde0inv, ptilde0))
            if stat:
                print(">>> Testing ptilde0inv-result (shld be unit-matrix): ok")
            else:
                print(">>> Testing ptilde0inv-result (shld be unit-matrix): failed")
                print(np.transpose(np.dot(ptilde0inv, ptilde0)))
                print(">>>")

    if verbosity >= 1:
        print("########## Create stc data and qual data arrays:")
    qualdata = {'relerr': np.zeros(data.shape[1]), 'rdmerr': np.zeros(data.shape[1]),
                'mag': np.zeros(data.shape[1])}
    if calccdm is not None:
        if verbosity >= 0 and \
                n_srcspace == 1 and (calccdm == 'left' or calccdm == 'right'):
            print(">>>Warning>> cdm-results may differ from what you expect.")
        ids = data.shape[1]
        if calccdm == 'all':
            (qualdata['cdmall'], qualdata['jlgall']) = (np.zeros(ids), np.zeros(ids))
            (qualdata['cdmleft'], qualdata['jlgleft']) = (np.zeros(ids), np.zeros(ids))
            (qualdata['cdmright'], qualdata['jlgright']) = (np.zeros(ids), np.zeros(ids))
        elif calccdm == 'both':
            (qualdata['cdmleft'], qualdata['jlgleft']) = (np.zeros(ids), np.zeros(ids))
            (qualdata['cdmright'], qualdata['jlgright']) = (np.zeros(ids), np.zeros(ids))
        elif calccdm == 'left':
            (qualdata['cdmleft'], qualdata['jlgleft']) = (np.zeros(ids), np.zeros(ids))
        elif calccdm == 'right':
            (qualdata['cdmright'], qualdata['jlgright']) = (np.zeros(ids), np.zeros(ids))
        elif calccdm == 'glob':
            (qualdata['cdmall'], qualdata['jlgall']) = (np.zeros(ids), np.zeros(ids))
        if 'cdmleft' in qualdata:
            fwdlhinds = np.where(fwdmag['source_rr'][:, 0] < 0.)[0]
        if 'cdmright' in qualdata:
            fwdrhinds = np.where(fwdmag['source_rr'][:, 0] > 0.)[0]
    if cdmlabels is not None and numcdmlabels > 0:
        qualdata['cdmlabels'] = np.zeros((numcdmlabels, data.shape[1]))
        qualdata['jlglabels'] = np.zeros((numcdmlabels, data.shape[1]))
        qualdata['jtotlabels'] = np.zeros((numcdmlabels, data.shape[1]))

    del fwdmag

    if isinstance(iterlist, list) and len(iterlist) > 0:
        stcdatalist = []
        for i in range(len(iterlist)):
            stcdatalist.append(np.zeros([n_loc // 3, data.shape[1]]))
    else:
        stcdatalist = [np.zeros([n_loc // 3, data.shape[1]])]

    if verbosity >= 2:
        print("Reading %d slices of data to calc. cdv:" % data.shape[1])
        if data.shape[1] > 1000:
            print(" ")
    for islice in range(data.shape[1]):
        wdist = np.copy(wdist0)
        wdist3 = np.repeat(wdist, 3)
        pmat = np.copy(pmat0)
        ptilde = np.copy(ptilde0)
        if use_svd is True:
            ptildeinv = np.copy(ptilde0inv)
        if use_lud is True:
            LU = np.copy(LU0)
            P = np.copy(P0)

        slice = pscalefct * data[:, islice]
        if mftparm['regtype'] == 'pzetae':
            mtilde = np.copy(slice)
        else:
            mtilde = np.dot(pmat, slice)

        acoeff = np.empty([ptilde.shape[0]])
        if use_svd is True:
            for irow in range(ptilde.shape[0]):
                acoeff[irow] = np.dot(ptildeinv[irow, :], mtilde)
        if use_lud is True:
            acoeff = scipy.linalg.lu_solve((LU, P), mtilde)

        cdv = np.zeros(n_loc)
        cdvnorms = np.zeros(n_loc // 3)
        for krow in range(lfmag.shape[0]):
            lfwtmp = lfmag[krow, :] * wdist3
            cdv += acoeff[krow] * lfwtmp
        if isinstance(iterlist, list) and len(iterlist) > 0 and 0 in iterlist:
            iil = iterlist.index(0)
            cdvecs = np.reshape(cdv, (cdv.shape[0] // 3, 3))
            cdvnorms = np.sqrt(np.sum(cdvecs ** 2, axis=1))
            for iloc in range(n_loc // 3):
                stcdatalist[iil][iloc, islice] = cdvnorms[iloc]

        tlw0 = time.time()
        tlc0 = time.clock()
        for mftiter in range(mftparm['iter']):
            # MFT iteration loop:

            cdvecs = np.reshape(cdv, (cdv.shape[0] // 3, 3))
            cdvnorms = np.sqrt(np.sum(cdvecs ** 2, axis=1))

            if LOG_ITERWDIST:
                wdist = np.power(cdvnorms, mftparm['currexp']) * wdist
            else:
                wdist = np.power(cdvnorms, mftparm['currexp']) * wdist0
            wdistsum = np.sum(wdist)
            wdist = wdist / wdistsum
            wdist3 = np.repeat(wdist, 3)

            # Calculate new P-matrix, incl. weights:
            tw0 = time.time()
            tc0 = time.clock()
            rtwd3 = np.repeat(np.sqrt(pscalefct * wdist), 3)
            ilastseg = n_loc - n_loc % ibsize
            lfw = lfmag[:, ilastseg:] * rtwd3[ilastseg:]
            pmat = np.einsum('ik,jk->ij', lfw, lfw)
            for iseg in range(n_loc // ibsize):
                lfw = lfmag[:, iseg * ibsize:iseg * ibsize + ibsize] * rtwd3[iseg * ibsize:iseg * ibsize + ibsize]
                pmat += np.einsum('ik,jk->ij', lfw, lfw)
            # lfw = lfmag*np.repeat(np.sqrt(pscalefct*wdist),3)
            # pmat = np.einsum('ik,jk->ij',lfw,lfw)

            tc1 = time.clock()
            tw1 = time.time()
            tptotwall += (tw1 - tw0)
            tptotcpu += (tc1 - tc0)
            nptotcall += 1

            # Regularize P:
            if mftparm['regtype'] == 'pzetae':
                ptilde = pmat + zetatrp * np.identity(pmat.shape[0])
            else:
                ptilde = np.dot(pmat, pmat) + zetatrp * pmat

            # decompose:
            if use_svd is True:
                U, s, V = np.linalg.svd(ptilde, full_matrices=True)
                dtmp = s.max() * svrelcut
                s *= (abs(s) >= dtmp)
                sinv = [1. / s[k] if s[k] != 0. else 0. for k in range(ptilde.shape[0])]
                ptildeinv = np.transpose(np.dot(U, np.dot(np.diag(sinv), V)))
                for irow in range(ptilde.shape[0]):
                    acoeff[irow] = np.dot(ptildeinv[irow, :], mtilde)
            if use_lud is True:
                LU, P = scipy.linalg.lu_factor(ptilde)
                acoeff = scipy.linalg.lu_solve((LU, P), mtilde)

            cdv = np.einsum('ji,i,j->i', lfmag, wdist3, acoeff)

            if isinstance(iterlist, list) and len(iterlist) > 0 and mftiter + 1 in iterlist:
                iil = iterlist.index(mftiter + 1)
                cdvecs = np.reshape(cdv, (cdv.shape[0] // 3, 3))
                cdvnorms = np.sqrt(np.sum(cdvecs ** 2, axis=1))
                for iloc in range(n_loc // 3):
                    stcdatalist[iil][iloc, islice] = cdvnorms[iloc]

        tc1 = time.clock()
        tw1 = time.time()
        tltotwall += (tw1 - tlw0)
        tltotcpu += (tc1 - tlc0)
        nltotcall += 1
        cdvecs = np.reshape(cdv, (cdv.shape[0] // 3, 3))
        cdvnorms = np.sqrt(np.sum(cdvecs ** 2, axis=1))
        # (relerr,rdmerr,mag) = compare_est_exp(ptilde,acoeff,mtilde)
        (relerr, rdmerr, mag) = compare_est_exp(pmat, acoeff, slice)
        qualdata['relerr'][islice] = relerr
        qualdata['rdmerr'][islice] = rdmerr
        qualdata['mag'][islice] = mag

        tc0 = time.clock()
        tw0 = time.time()
        if 'cdmall' in qualdata:
            (qualdata['cdmall'][islice], qualdata['jlgall'][islice]) = scan_cdm_w_cut(cdv, cdmcut)
        if 'cdmleft' in qualdata:
            (qualdata['cdmleft'][islice], qualdata['jlgleft'][islice]) = \
                scan_cdm_w_cut(cdvecs[fwdlhinds, :], cdmcut)
        if 'cdmright' in qualdata:
            (qualdata['cdmright'][islice], qualdata['jlgright'][islice]) = \
                scan_cdm_w_cut(cdvecs[fwdrhinds, :], cdmcut)
        if 'cdmlabels' in qualdata:
            for ilab in range(numcdmlabels):
                (qualdata['cdmlabels'][ilab, islice], qualdata['jlglabels'][ilab, islice]) = \
                    scan_cdm_w_cut(cdvecs[labinds[ilab], :], cdmcut)
                qualdata['jtotlabels'][ilab, islice] = \
                    calc_jtotal_w_cut(cdvecs[labinds[ilab], :], cdmcut)
        tc1 = time.clock()
        tw1 = time.time()
        tpcdmwall += (tw1 - tw0)
        tpcdmcpu += (tc1 - tc0)
        npcdmcall += 1

        # Write final cdv to the stcdatalist:
        if not isinstance(iterlist, list) or len(iterlist) == 0:
            for iloc in range(n_loc // 3):
                stcdatalist[0][iloc, islice] = cdvnorms[iloc]
        del wdist
        del pmat
        del ptilde
        if use_svd is True:
            del ptildeinv
        if use_lud is True:
            del LU
            del P
        if verbosity >= 2 and islice > 0 and islice % 1000 == 0:
            print("\r%6d out of %6d slices done." % (islice, data.shape[1]))
    if verbosity >= 2 and data.shape[1] > 1000:
        print("Done.")

    # Restore access to fwdmag:
    if isinstance(fwdspec, str):
        # Msg will be written by mne.read_forward_solution()
        fwd = mne.read_forward_solution(fwdspec, verbose=verbose)
    elif isinstance(fwdspec, dict):
        if 'info' not in fwdspec or 'sol' not in fwdspec:
            raise ValueError(">>>>> apply_mft() first argument not indicating a fwd-solution")
        fwd = fwdspec
    else:
        raise ValueError(">>>>> apply_mft() first argument not indicating a fwd-solution")
    fwdmag = mne.io.pick.pick_types_forward(fwd, meg=meg, ref_meg=False,
                                            eeg=False, exclude=exclude)
    del fwd

    vertices = [s['vertno'] for s in fwdmag['src']]
    tstep = 1. / indathndl.info['sfreq']
    tmin = indathndl.times[0]
    stc_mftlist = []
    for istclist in range(len(stcdatalist)):
        if len(vertices) == 1:
            stc_mft = mne.VolSourceEstimate(stcdatalist[istclist], vertices=fwdmag['src'][0]['vertno'],
                                            tmin=tmin, tstep=tstep, subject=subject)
        elif len(vertices) == 2:
            vertices = [s['vertno'] for s in fwdmag['src']]
            stc_mft = mne.SourceEstimate(stcdatalist[istclist], vertices=vertices,
                                         tmin=tmin, tstep=tstep, subject=subject)
        else:
            vertices = np.concatenate(([s['vertno'] for s in fwdmag['src']]))
            stc_mft = mne.VolSourceEstimate(stcdatalist[istclist], vertices=vertices,
                                            tmin=tmin, tstep=tstep, subject=subject)
        stc_mftlist.append(stc_mft)
    stcdatamft = stc_mftlist[-1].data
    if isinstance(iterlist, list) and len(iterlist) > 0:
        # Set stcdatamft to the highest iteration
        stc_mft = stc_mftlist[iterlist.index(max(iterlist))]
        stcdatamft = stc_mft.data

    print("##### Results:")
    for islice in range(data.shape[1]):
        print("slice=%4d: relerr=%9.3e rdmerr=%9.3e mag=%5.3f cdvmax=%9.2e" % \
              (islice, qualdata['relerr'][islice], qualdata['rdmerr'][islice], qualdata['mag'][islice], \
               np.amax(stcdatamft[:, islice])))
    iil = 0
    if isinstance(iterlist, list) and len(iterlist) > 0:
        iil = iterlist.index(max(iterlist))

    stat = np.allclose(stcdatalist[iil], stcdatamft, atol=0., rtol=1e-07)
    if stat:
        print("stcdata from mft-SR and old calc agree.")
    else:
        print("stcdata from mft-SR and old calc DIFFER.")

    if save_stc:
        # save Surface stc.
        print("##### Trying to save stc:")
        stcmft_fname = os.path.join(os.path.dirname(datafile),
                                    os.path.basename(datafile).split('-')[0]) + "mft"
        print("stcmft basefilename: %s" % stcmft_fname)
        stc_mft.save(stcmft_fname, verbose=True)
        print("##### done.")

    twgbl1 = time.time()
    tcgbl1 = time.clock()
    if verbosity >= 1:
        print("calc(lf*w*lf.T) took   total  %9.2f s CPU-time (%9.2f s walltime)" % (tptotcpu, tptotwall))
        print("calc(lf*w*lf.T) took per call %9.2fms CPU-time (%9.2fms walltime)" % \
              (1000. * tptotcpu / float(nptotcall), 1000. * tptotwall / float(nptotcall)))
        print("scan_cdm calls  took   total  %9.2f s CPU-time (%9.2f s walltime)" % (tpcdmcpu, tpcdmwall))
        print("scan_cdm calls  took per call %9.2fms CPU-time (%9.2fms walltime)" % \
              (1000. * tpcdmcpu / float(npcdmcall), 1000. * tpcdmwall / float(npcdmcall)))
        print("iteration-loops took   total  %9.2f s CPU-time (%9.2f s walltime)" % (tltotcpu, tltotwall))
        print("iteration-loops took per call %9.2fms CPU-time (%9.2fms walltime)" % \
              (1000. * tltotcpu / float(nltotcall), 1000. * tltotwall / float(nltotcall)))
        print("Total mft-call  took   total  %9.2f s CPU-time (%9.2f s walltime)" % (
            (tcgbl1 - tcgbl0), (twgbl1 - twgbl0)))
    if isinstance(iterlist, list) and len(iterlist) > 0:
        return (fwdmag, qualdata, stc_mftlist)
    else:
        return (fwdmag, qualdata, stc_mftlist[0])
