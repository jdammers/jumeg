"""
====================
Jumeg MFT Plotting.
====================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

from mne import SourceEstimate, VolSourceEstimate
from mne.transforms import invert_transform, apply_trans


def plot_global_cdv_dist(stcdata, fwdmag=None):
    """
    Plot global cdv-distribution at time of max |cdv|
    Parameters
    ----------
    stcdata: stc with ||cdv|| (point sequence as in fwdmag['source_rr'])
    fwdmag:  (opt) forward solution (to colorize srcs)
    """
    print("##### Plot global cdv-distribution at time of max |cdv|:")
    time_idx = np.argmax(np.max(stcdata, axis=0))
    fig = plt.figure()
    plt.xlim((0, stcdata.shape[0] + 100))
    plt.ylim((-0.1 * np.max(stcdata[:, time_idx]),
              1.1 * np.max(stcdata[:, time_idx])))
    if fwdmag is None:
        cdvnmax = stcdata[:, time_idx]
        plt.plot(range(cdvnmax.shape[0]), cdvnmax)
    else:
        offsets = [0]
        for s in fwdmag['src']:
            offsets = np.append(offsets, [offsets[-1] + s['nuse']])
        for isrc in range(len(fwdmag['src'])):
            cdvnmax = stcdata[offsets[isrc]:offsets[isrc + 1], time_idx]
            plt.plot(range(offsets[isrc], offsets[isrc + 1]), cdvnmax)
    plt.xlabel('n (index in fwd-solution)')
    plt.ylabel('|cdv(t_i=%d|' % time_idx)
    if fwdmag is None:
        plt.title('|cdv(src)| at fixed time for src-space(s)', loc='center')
    else:
        plt.title('|cdv(src)| at fixed time for (colored) src-space(s)', loc='center')
    plt.savefig('testfig_cdvgtfixed.png')
    plt.close()


def plot_visualize_mft_sources(fwdmag, stcdata, tmin, tstep,
                               subject, subjects_dir):
    """
    Plot the MFT sources at time point of peak.
    Parameters
    ----------
    fwdmag:  forward solution
    stcdata: stc with ||cdv|| (point sequence as in fwdmag['source_rr'])
    tmin, tstep, subject: passed to mne.SourceEstimate()
    """
    print("##### Attempting to plot:")
    # cf. decoding/plot_decoding_spatio_temporal_source.py
    vertices = [s['vertno'] for s in fwdmag['src']]
    if len(vertices) == 1:
        vertices = [fwdmag['src'][0]['vertno'][fwdmag['src'][0]['rr'][fwdmag['src'][0]['vertno']][:, 0] <= -0.],
                    fwdmag['src'][0]['vertno'][fwdmag['src'][0]['rr'][fwdmag['src'][0]['vertno']][:, 0] > -0.]]
    elif len(vertices) > 2:
        warnings.warn('plot_visualize_mft_sources(): Cannot handle more than two sources spaces')
        return

    stc_feat = SourceEstimate(stcdata, vertices=vertices,
                              tmin=tmin, tstep=tstep, subject=subject)
    itmaxsum = np.argmax(np.sum(stcdata, axis=0))
    twmin = tmin + tstep * float(itmaxsum - stcdata.shape[1] / 20)
    twmax = tmin + tstep * float(itmaxsum + stcdata.shape[1] / 20)
    for ihemi, hemi in enumerate(['lh', 'rh', 'both']):
        brain = stc_feat.plot(surface='white', hemi=hemi, subjects_dir=subjects_dir,
                              transparent=True, clim='auto')
        # use peak getter to move visualization to the time point of the peak
        print("Restricting peak search to [%fs, %fs]" % (twmin, twmax))
        if hemi == 'both':
            brain.show_view('parietal')
            vertno_max, time_idx = stc_feat.get_peak(hemi=None, time_as_index=True,
                                                     tmin=twmin, tmax=twmax)
        else:
            brain.show_view('lateral')
            vertno_max, time_idx = stc_feat.get_peak(hemi=hemi, time_as_index=True,
                                                     tmin=twmin, tmax=twmax)
        print("hemi=%s: setting time_idx=%d" % (hemi, time_idx))
        brain.set_data_time_index(time_idx)
        if hemi == 'lh' or hemi == 'rh':
            # draw marker at maximum peaking vertex
            brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
                           scale_factor=0.6)

        if len(fwdmag['src']) > ihemi:
            fwds = fwdmag['src'][ihemi]
            comax = fwds['rr'][vertno_max]
            print("hemi=%s: vertno_max=%d, time_idx=%d fwdmag['src'][%d]['rr'][vertno_max] = " % \
                  (hemi, vertno_max, time_idx, ihemi), comax)

            offsets = np.append([0], [s['nuse'] for s in fwdmag['src']])
            if hemi == 'lh':
                ifoci = [np.nonzero([stcdata[0:offsets[1], time_idx] >= 0.25 * np.max(stcdata[:, time_idx])][0])]
            elif len(fwdmag['src']) > 1:
                ifoci = [np.nonzero([stcdata[offsets[1]:, time_idx] >= 0.25 * np.max(stcdata[:, time_idx])][0])]
            vfoci = fwds['vertno'][ifoci[0][0]]
            cfoci = fwds['rr'][vfoci]
            print("Coords  of %d sel. vfoci: " % cfoci.shape[0])
            print(cfoci)
            print("vfoci: ")
            print(vfoci)
            print("brain.geo[%s].coords[vfoci] : " % hemi)
            print(brain.geo[hemi].coords[vfoci])

            mrfoci = np.zeros(cfoci.shape)
            invmri_head_t = invert_transform(fwdmag['info']['mri_head_t'])
            mrfoci = apply_trans(invmri_head_t['trans'], cfoci, move=True)
            print("mrfoci: ")
            print(mrfoci)

            # Just some blops along the coordinate axis:
            # This will not yield reasonable results w an inflated brain.
            # bloblist = np.zeros((300,3))
            # for i in xrange(100):
            #    bloblist[i,0] = float(i)
            #    bloblist[i+100,1] = float(i)
            #    bloblist[i+200,2] = float(i)
            # mrblobs = apply_trans(invmri_head_t['trans'], bloblist, move=True)
            # brain.add_foci(mrblobs, coords_as_verts=False, hemi=hemi, color='yellow', scale_factor=0.3)
        brain.save_image('testfig_map_%s.png' % hemi)
        brain.close()


def plot_cdv_distribution(fwdmag, stcdata):
    """
    Plot cdv-distribution.
    Parameters
    ----------
    fwdmag:  forward solution
    stcdata: stc with ||cdv|| (point sequence as in fwdmag['source_rr'])
    """
    print("##### Plot cdv-distribution:")
    maxxpnt = np.max([len(s['vertno']) for s in fwdmag['src']])
    time_idx = np.argmax(np.max(stcdata, axis=0))
    fig = plt.figure()
    plt.xlim((0, maxxpnt + 100))
    plt.ylim((-0.1 * np.max(stcdata[:, time_idx]),
              1.1 * np.max(stcdata[:, time_idx])))
    offsets = [0]
    for s in fwdmag['src']:
        offsets = np.append(offsets, [offsets[-1] + s['nuse']])
    for isrc, s in enumerate(fwdmag['src']):
        cdvnmax = stcdata[offsets[isrc]:offsets[isrc + 1], time_idx]
        plt.plot(range(cdvnmax.shape[0]), np.sort(cdvnmax))
        plt.xlabel('index in amplitude-sorted list')
        plt.ylabel('|cdv(t_i=%d|' % time_idx)

    plt.title('sorted |cdv(src)| at fixed time for src-space(s)', loc='center')
    plt.savefig('testfig_cdvtfixed.png')
    plt.close()


def plot_max_amplitude_data(fwdmag, stcdata, tmin, tstep, subject, method='mft'):
    """
    Plot max(|cdv(src)|) vs. time and src-space(s)
    Parameters
    ----------
    fwdmag: forward solution
    stcdata: stc with ||cdv|| (point sequence as in fwdmag['source_rr'])
    tmin, tstep, subject: passed to mne.VolSourceEstimate()
    method: used as y-axis label
    """
    print("##### Attempting to plot max. amplitude data:")
    fig = plt.figure()
    offsets = [0]
    for s in fwdmag['src']:
        offsets = np.append(offsets, [offsets[-1] + s['nuse']])
    for isrc, s in enumerate(fwdmag['src']):
        stc = VolSourceEstimate(stcdata[offsets[isrc]:offsets[isrc + 1], :], vertices=s['vertno'],
                                tmin=tmin, tstep=tstep, subject=subject)
        # View activation time-series
        plt.xlim((1e3 * np.min(stc.times), 1e3 * np.max(stc.times)))
        plt.ylim((0, np.max(stcdata)))
        plt.plot(1e3 * stc.times, np.max(stc.data, axis=0))
        #        label=(('lh', 'rh'))[isrc])
        plt.xlabel('time (ms)')
        plt.ylabel('%s value' % method)
        # plt.savefig('testfig'+"{0:0=2d}".format(isrc)+'.png')
    plt.title('max(|cdv(src)|) vs. time and src-space(s)', loc='center')
    plt.savefig('testfig_cdvmaxsrc.png')
    plt.close()


def plot_max_cdv_data(stc_mft, lhmrinds, rhmrinds):
    """
    Plot max CDV data.
    Parameters
    ----------
    stcdata: stc with ||cdv|| (point sequence as in fwdmag['source_rr'])
    """
    print("##### Attempting to plot max. cdv data:")
    fig = plt.figure()
    stcdata = stc_mft.data
    plt.plot(1e3 * stc_mft.times, np.max(stcdata[lhmrinds[0], :], axis=0), 'r', label='lh')
    plt.plot(1e3 * stc_mft.times, np.max(stcdata[rhmrinds[0], :], axis=0), 'g', label='rh')
    plt.plot(1e3 * stc_mft.times, np.max(stcdata, axis=0), 'b', label='all')
    plt.xlabel('time (ms)')
    plt.ylabel('max(|cdv|) value')
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig('testfig_cdvmax')
    plt.close()


def plot_cdvsum_data(stc_mft, lhmrinds, rhmrinds):
    """
    Plot cdvsum data.
    """
    print("##### Attempting to cdvsum data:")
    fig = plt.figure()
    stcdata = stc_mft.data
    plt.plot(1e3 * stc_mft.times, np.sum(stcdata[lhmrinds[0], :], axis=0), 'r', label='lh')
    plt.plot(1e3 * stc_mft.times, np.sum(stcdata[rhmrinds[0], :], axis=0), 'g', label='rh')
    plt.plot(1e3 * stc_mft.times, np.sum(stcdata, axis=0), 'b', label='all')
    plt.xlabel('time (ms)')
    plt.ylabel('sum(|cdv|) value')
    plt.legend(loc='upper right', fontsize=10)
    plt.savefig('testfig_cdvsum')
    plt.close()


def plot_quality_data(qualmft, stc_mft):
    """
    Plot quality data.
    """
    print("##### Attempting to plot quality data:")
    fig = plt.figure()
    # relerrscal = pow(10,-int(np.log10(np.max(qualmft['relerr'][:]))))
    scalexp = -int(np.log10(np.max(qualmft['relerr'][:]))) - 1
    relerrscal = pow(10, scalexp)
    ls = '-'
    # if fwdname.rfind('vol')>0: ls = '--'
    plt.ylim((0, 1.05))
    plt.plot(1e3 * stc_mft.times, np.ones(stc_mft.times.shape), 'k:', linewidth=1)
    plt.plot(1e3 * stc_mft.times, relerrscal * qualmft['relerr'][:], 'r' + ls, label='relerr')
    plt.plot(1e3 * stc_mft.times, qualmft['rdmerr'][:], 'g' + ls, label='rdmerr')
    plt.plot(1e3 * stc_mft.times, qualmft['mag'][:], 'b' + ls, label='mag')
    plt.xlabel('time (ms)')
    plt.ylabel('r: 10^%d*relerr, g: rdmerr, b: mag' % scalexp)
    plt.legend(loc='center right', fontsize=10)
    plt.savefig('testfig_qual')
    plt.close()


def plot_cdm_data(qualmft, stc_mft, cdmlabels=None, selmaxjlong=False, outfile=None):
    """
    Plot CDM data.
    """
    if cdmlabels is not None and 'cdmlabels' in qualmft:
        if selmaxjlong:
            if 'jlglabels' in qualmft:
                jlgmaxv = np.max(qualmft['jlglabels'], axis=1)
            else:
                warnings.warn('plot_cdm_data(): qualmft-arg has no jlglabels-key')
                return

        print("##### Attempting to plot cdm data for labels:")
        fig = plt.figure()
        plt.ylim((0, 1.05))
        cdmmean = np.mean(qualmft['cdmlabels'], axis=1)
        icdmtop = np.zeros(len(cdmmean), dtype=np.int32)
        # Don't consider tiny labels:
        # for ilab, label in enumerate(cdmlabels):
        #    if len(label.vertices) < 5000:
        #        cdmmean[ilab] = 0.
        # Restrict to largest cdmmean-s:
        if len(cdmmean) > 10:
            if selmaxjlong:
                for itop in range(10):
                    itmp = np.argmax(jlgmaxv)
                    icdmtop[itmp] = 1
                    jlgmaxv[itmp] = 0.
            else:
                for itop in range(10):
                    itmp = np.argmax(cdmmean)
                    icdmtop[itmp] = 1
                    cdmmean[itmp] = 0.
        else:
            icdmtop = np.ones(len(cdmmean), dtype=np.int32)
        if len(cdmmean) > 10:
            if selmaxjlong:
                print("Labels with largest max(jlong):")
            else:
                print("Labels with largest avg(cdm):")
        else:
            print("Labels with largest avg(cdm):")
        for ilab, label in enumerate(cdmlabels):
            if icdmtop[ilab] > 0:
                print("%3d %30s %6s: avg(cdm) = %7.4f" % \
                      (ilab + 1, label.name, label.hemi, np.mean(qualmft['cdmlabels'][ilab, :])))
                # plt.plot(1e3 * stc_mft.times, qualmft['cdmlabels'][ilab,:], color=label.color, label='_none')
                if label.color is not None:
                    plt.plot(1e3 * stc_mft.times, qualmft['cdmlabels'][ilab, :], color=label.color,
                             label="%2d" % (ilab + 1))
                else:
                    plt.plot(1e3 * stc_mft.times, qualmft['cdmlabels'][ilab, :], label="%2d" % (ilab + 1))
        plt.xlabel('time (ms)')
        plt.ylabel('cdm value')
        plt.legend(loc='lower left', fontsize=10)
        if outfile is not None:
            plt.title(outfile, loc='center')
            plt.savefig(outfile + 'label')
        else:
            plt.savefig('testfig_labelcdm')
        plt.close()

    if 'cdmall' not in qualmft and 'cdmright' not in qualmft and \
            'cdmleft' not in qualmft:
        return
    print("##### Attempting to plot cdm data:")
    fig = plt.figure()
    plt.ylim((0, 1.05))
    if 'cdmleft' in qualmft:
        plt.plot(1e3 * stc_mft.times, qualmft['cdmleft'][:], 'r', label='lh')
    if 'cdmright' in qualmft:
        plt.plot(1e3 * stc_mft.times, qualmft['cdmright'][:], 'g', label='rh')
    if 'cdmall' in qualmft:
        plt.plot(1e3 * stc_mft.times, qualmft['cdmall'][:], 'b', label='all')
    plt.xlabel('time (ms)')
    plt.ylabel('cdm value')
    plt.legend(fontsize=10)
    if outfile is not None:
        plt.title(outfile, loc='center')
        plt.savefig(outfile)
    else:
        plt.savefig('testfig_cdm')
    plt.close()


def plot_jlong_data(qualmft, stc_mft, outfile=None):
    """
    Plot jlong label data.
    """
    if 'jlgall' not in qualmft and 'jlgright' not in qualmft and \
            'jlgleft' not in qualmft:
        return
    print("##### Attempting to plot jlong data:")
    fig = plt.figure()
    # plt.ylim((0,1.05))
    if 'jlgleft' in qualmft:
        plt.plot(1e3 * stc_mft.times, qualmft['jlgleft'][:], 'r', label='lh')
    if 'jlgright' in qualmft:
        plt.plot(1e3 * stc_mft.times, qualmft['jlgright'][:], 'g', label='rh')
    if 'jlgall' in qualmft:
        plt.plot(1e3 * stc_mft.times, qualmft['jlgall'][:], 'b', label='all')
    plt.xlabel('time (ms)')
    plt.ylabel('jlong value')
    plt.legend(fontsize=10)
    if outfile is not None:
        plt.title(outfile, loc='center')
        plt.savefig(outfile)
    else:
        plt.savefig('testfig_jlong')
    plt.close()


def plot_jlong_labeldata(qualmft, stc_mft, cdmlabels, outfile=None):
    """
    Plot jlong label data.
    """
    if cdmlabels is None or 'jlglabels' not in qualmft \
            or 'cdmlabels' not in qualmft:
        return

    print("##### Attempting to plot jlong data for labels:")
    fig = plt.figure()
    jlgmaxv = np.max(qualmft['jlglabels'], axis=1)
    ijlgtop = np.zeros(len(jlgmaxv), dtype=np.int32)
    # Don't consider tiny labels:
    jlgglbmax = np.max(jlgmaxv)
    # for ilab, label in enumerate(cdmlabels):
    #    if len(label.vertices) < 5000:
    #        jlgmaxv[ilab] = 0.
    # Restrict to largest jlgmax-s:
    if len(jlgmaxv) > 10:
        for itop in range(10):
            itmp = np.argmax(jlgmaxv)
            ijlgtop[itmp] = 1
            jlgmaxv[itmp] = 0.
    else:
        ijlgtop = np.ones(len(jlgmaxv), dtype=np.int32)
    plt.ylim((0, 1.05 * jlgglbmax))
    print("Labels with largest max(jlong):")
    for ilab, label in enumerate(cdmlabels):
        if ijlgtop[ilab] > 0:
            print("%3d %30s %6s: max(jlong) = %12.4e" % \
                  (ilab + 1, label.name, label.hemi, np.max(qualmft['jlglabels'][ilab, :])))
            # plt.plot(1e3 * stc_mft.times, qualmft['jlglabels'][ilab,:], color=label.color, label='_none')
            if label.color is not None:
                plt.plot(1e3 * stc_mft.times, qualmft['jlglabels'][ilab, :], color=label.color,
                         label="%2d" % (ilab + 1))
            else:
                plt.plot(1e3 * stc_mft.times, qualmft['jlglabels'][ilab, :], label="%2d" % (ilab + 1))
    plt.xlabel('time (ms)')
    plt.ylabel('jlong value')
    plt.legend(loc='upper left', fontsize=10)
    if outfile is not None:
        plt.title(outfile, loc='center')
        plt.savefig(outfile)
    else:
        plt.savefig('testfig_labeljlong')
    plt.close()


def plot_jtotal_labeldata(qualmft, stc_mft, cdmlabels, outfile=None):
    """
    Plot jtotal label data.
    """
    if cdmlabels is None or 'jtotlabels' not in qualmft \
            or 'cdmlabels' not in qualmft:
        return

    print("##### Attempting to plot jtotal data for labels:")
    fig = plt.figure()
    jtotmaxv = np.max(qualmft['jtotlabels'], axis=1)
    ijtottop = np.zeros(len(jtotmaxv), dtype=np.int32)
    # Don't consider tiny labels:
    jtotglbmax = np.max(jtotmaxv)
    # for ilab, label in enumerate(cdmlabels):
    #    if len(label.vertices) < 5000:
    #        jtotmaxv[ilab] = 0.
    # Restrict to largest jtotmax-s:
    if len(jtotmaxv) > 10:
        for itop in range(10):
            itmp = np.argmax(jtotmaxv)
            ijtottop[itmp] = 1
            jtotmaxv[itmp] = 0.
    else:
        ijtottop = np.ones(len(jtotmaxv), dtype=np.int32)
    plt.ylim((0, 1.05 * jtotglbmax))
    print("Labels with largest max(jtotal):")
    for ilab, label in enumerate(cdmlabels):
        if ijtottop[ilab] > 0:
            print("%3d %30s %6s: max(jtotal) = %12.4e" % \
                  (ilab + 1, label.name, label.hemi, np.max(qualmft['jtotlabels'][ilab, :])))
            # plt.plot(1e3 * stc_mft.times, qualmft['jtotlabels'][ilab,:], color=label.color, label='_none')
            if label.color is not None:
                plt.plot(1e3 * stc_mft.times, qualmft['jtotlabels'][ilab, :], color=label.color,
                         label="%2d" % (ilab + 1))
            else:
                plt.plot(1e3 * stc_mft.times, qualmft['jtotlabels'][ilab, :], label="%2d" % (ilab + 1))
    plt.xlabel('time (ms)')
    plt.ylabel('jtotal value')
    plt.legend(loc='upper left', fontsize=10)
    if outfile is not None:
        plt.title(outfile, loc='center')
        plt.savefig(outfile)
    else:
        plt.savefig('testfig_labeljtot')
    plt.close()
