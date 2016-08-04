"""
====================
Jumeg MFT Plotting.
====================
"""

import numpy as np
import matplotlib.pyplot as plt

from nilearn.plotting import plot_stat_map
from nilearn.image import index_img

from mne import SourceEstimate, VolSourceEstimate
from mne.transforms import invert_transform, apply_trans

def plot_global_cdv_dist(stcdata):
    '''
    Plot global cdv-distribution at time of max |cdv|
    '''
    print "##### Plot global cdv-distribution at time of max |cdv|:"
    time_idx = np.argmax(np.max(stcdata, axis=0))
    fig = plt.figure()
    plt.xlim((0, stcdata.shape[0]+100))
    plt.ylim((-0.1*np.max(stcdata[:, time_idx]),
               1.1*np.max(stcdata[:, time_idx])))
    cdvnmax = stcdata[:, time_idx]
    print "cdvnmax.shape=", cdvnmax.shape
    plt.plot(xrange(cdvnmax.shape[0]), cdvnmax)
    plt.xlabel('n')
    plt.ylabel('|cdv(t_i=%d|' % time_idx)
    plt.savefig('testfig_cdvgtfixed.png')
    plt.close()


def plot_visualize_mft_sources(fwdmag, stcdata, tmin, tstep,
                               subject, subjects_dir):
    '''
    Plot the MFT sources at time point of peak.
    '''
    print "##### Attempting to plot:"
    # cf. decoding/plot_decoding_spatio_temporal_source.py
    vertices = [s['vertno'] for s in fwdmag['src']]
    if len(vertices) == 1:
        vertices = [fwdmag['src'][0]['vertno'][fwdmag['src'][0]['rr'][fwdmag['src'][0]['vertno']][:, 0] <= -0.],
                    fwdmag['src'][0]['vertno'][fwdmag['src'][0]['rr'][fwdmag['src'][0]['vertno']][:, 0] > -0.]]

    stc_feat = SourceEstimate(stcdata, vertices=vertices,
                              tmin=-0.2, tstep=tstep, subject=subject)
    for hemi in ['lh', 'rh']:
        brain = stc_feat.plot(surface='white', hemi=hemi, subjects_dir=subjects_dir,
                              transparent=True, clim='auto')
        brain.show_view('lateral')
        # use peak getter to move visualization to the time point of the peak
        tmin = 0.095
        tmax = 0.10
        print "Restricting peak search to [%fs, %fs]" % (tmin, tmax)
        if hemi == 'both':
            vertno_max, time_idx = stc_feat.get_peak(hemi='rh', time_as_index=True,
                                                     tmin=tmin, tmax=tmax)
        else:
            vertno_max, time_idx = stc_feat.get_peak(hemi=hemi, time_as_index=True,
                                                     tmin=tmin, tmax=tmax)
        if hemi == 'lh':
            comax = fwdmag['src'][0]['rr'][vertno_max]
            print "hemi=%s: vertno_max=%d, time_idx=%d fwdmag['src'][0]['rr'][vertno_max] = " %\
                  (hemi, vertno_max, time_idx), comax
        elif len(fwdmag['src']) > 1:
            comax = fwdmag['src'][1]['rr'][vertno_max]
            print "hemi=%s: vertno_max=%d, time_idx=%d fwdmag['src'][1]['rr'][vertno_max] = " %\
                  (hemi, vertno_max, time_idx), comax

        print "hemi=%s: setting time_idx=%d" % (hemi, time_idx)
        brain.set_data_time_index(time_idx)
        # draw marker at maximum peaking vertex
        brain.add_foci(vertno_max, coords_as_verts=True, hemi=hemi, color='blue',
                       scale_factor=0.6)
        offsets = np.append([0], [s['nuse'] for s in fwdmag['src']])
        if hemi == 'lh':
            ifoci = [np.nonzero([stcdata[0:offsets[1],time_idx]>=0.25*np.max(stcdata[:,time_idx])][0])]
            vfoci = fwdmag['src'][0]['vertno'][ifoci[0][0]]
            cfoci = fwdmag['src'][0]['rr'][vfoci]
            print "Coords  of %d sel. vfoci: " % cfoci.shape[0]
            print cfoci
            print "vfoci: "
            print vfoci
            print "brain.geo['lh'].coords[vfoci] : "
            print brain.geo['lh'].coords[vfoci]
        elif len(fwdmag['src']) > 1:
            ifoci = [np.nonzero([stcdata[offsets[1]:,time_idx]>=0.25*np.max(stcdata[:,time_idx])][0])]
            vfoci = fwdmag['src'][1]['vertno'][ifoci[0][0]]
            cfoci = fwdmag['src'][1]['rr'][vfoci]
            print "Coords  of %d sel. vfoci: " % cfoci.shape[0]
            print cfoci
            print "vfoci: "
            print vfoci
            print "brain.geo['rh'].coords[vfoci] : "
            print brain.geo['rh'].coords[vfoci]

        mrfoci = np.zeros(cfoci.shape)
        invmri_head_t = invert_transform(fwdmag['info']['mri_head_t'])
        mrfoci = apply_trans(invmri_head_t['trans'],cfoci, move=True)
        print "mrfoci: "
        print mrfoci

        # Just some blops:
        bloblist = np.zeros((300,3))
        for i in xrange(100):
            bloblist[i,0] = float(i)
            bloblist[i+100,1] = float(i)
            bloblist[i+200,2] = float(i)
        mrblobs = apply_trans(invmri_head_t['trans'], bloblist, move=True)
        brain.save_image('testfig_map_%s.png' % hemi)
        brain.close()


def plot_cdv_distribution(fwdmag, stcdata):
    '''
    Plot cdv-distribution.
    '''
    print "##### Plot cdv-distribution:"
    maxxpnt = np.max([len(s['vertno']) for s in fwdmag['src']])
    iblck = -1
    time_idx = np.argmax(np.max(stcdata, axis=0))
    fig = plt.figure()
    plt.xlim((0, maxxpnt + 100))
    plt.ylim((-0.1 * np.max(stcdata[:, time_idx]),
              1.1 * np.max(stcdata[:, time_idx])))
    offsets = np.append([0], [s['nuse'] for s in fwdmag['src']])
    print "offsets = ",offsets
    for s in fwdmag['src']:
        iblck = iblck + 1
        cdvnmax = stcdata[offsets[iblck]:offsets[iblck]+offsets[iblck+1], time_idx]
        print "cdvnmax.shape=", cdvnmax.shape
        plt.plot(xrange(cdvnmax.shape[0]), np.sort(cdvnmax))
        plt.xlabel('n')
        plt.ylabel('|cdv(t_i=%d|' % time_idx)

    plt.savefig('testfig_cdvtfixed.png')
    plt.close()


def plot_max_amplitude_data(fwdmag, stcdata, tmin, tstep, subject, method='mft'):
    print "##### Attempting to plot max. amplitude data:"
    fig = plt.figure()
    iblck = -1
    offsets = np.append([0], [s['nuse'] for s in fwdmag['src']])
    for s in fwdmag['src']:
        iblck = iblck + 1
        stc = VolSourceEstimate(stcdata[offsets[iblck]:offsets[iblck]+offsets[iblck+1],:], vertices=s['vertno'],
                                    tmin=tmin, tstep=tstep, subject=subject)
        # View activation time-series
        plt.xlim((1e3*np.min(stc.times), 1e3*np.max(stc.times)))
        plt.ylim((0, np.max(stcdata)))
        plt.plot(1e3 * stc.times, np.max(stc.data, axis=0),
                 label=(('lh', 'rh'))[iblck])
        plt.xlabel('time (ms)')
        plt.ylabel('%s value' % method)
        plt.savefig('testfig'+"{0:0=2d}".format(iblck)+'.png')
    plt.close()


def plot_max_cdv_data(stc_mft, lhmrinds, rhmrinds):
    ''' Plot max CDV data.
    '''
    print "##### Attempting to plot max. cdv data:"
    fig = plt.figure()
    stcdata = stc_mft.data
    plt.plot(1e3 * stc_mft.times, np.max(stcdata[lhmrinds[0],:], axis=0), label='lh')
    plt.plot(1e3 * stc_mft.times, np.max(stcdata[rhmrinds[0],:], axis=0), label='rh')
    plt.plot(1e3 * stc_mft.times, np.max(stcdata, axis=0), label='all')
    plt.xlabel('time (ms)')
    plt.ylabel('max(|cdv|) value')
    plt.legend()
    plt.savefig('testfig_cdvmax')
    plt.close()


def plot_cdvsum_data(stc_mft, lhmrinds, rhmrinds):
    '''Plot cdvsum data.
    '''
    print "##### Attempting to cdvsum data:"
    fig = plt.figure()
    stcdata = stc_mft.data
    plt.plot(1e3 * stc_mft.times, np.sum(stcdata[lhmrinds[0],:],axis=0),label='lh')
    plt.plot(1e3 * stc_mft.times, np.sum(stcdata[rhmrinds[0],:],axis=0),label='rh')
    plt.plot(1e3 * stc_mft.times, np.sum(stcdata,axis=0),label='all')
    plt.xlabel('time (ms)')
    plt.ylabel('sum(|cdv|) value')
    plt.legend()
    plt.savefig('testfig_cdvsum')
    plt.close()


def plot_quality_data(qualmft, stc_mft):
    '''Plot quality data.
    '''
    print "##### Attempting to plot quality data:"
    fig = plt.figure()
    # relerrscal = pow(10,-int(np.log10(np.max(qualmft['relerr'][:]))))
    scalexp = -int(np.log10(np.max(qualmft['relerr'][:])))-1
    relerrscal = pow(10,scalexp)
    ls = '-'
    #if fwdname.rfind('vol')>0: ls = '--'
    plt.ylim((0,1.05))
    plt.plot(1e3 * stc_mft.times, relerrscal*qualmft['relerr'][:],'r'+ls, label='relerr')
    plt.plot(1e3 * stc_mft.times, qualmft['rdmerr'][:], 'g'+ls, label='rdmerr')
    plt.plot(1e3 * stc_mft.times, qualmft['mag'][:], 'b'+ls, label='mag')
    plt.xlabel('time (ms)')
    plt.ylabel('r: 10^%d*relerr, g: rdmerr, b: mag' % scalexp)
    plt.legend(loc='center right')
    plt.savefig('testfig_qual')
    plt.close()


# TODO cdmdata computation to be added into apply_mft
def plot_cdm_data(stc_mft, cdmdata):
    '''Plot CDM data.
    '''
    print "##### Attempting to plot cdm data:"
    fig = plt.figure()
    plt.ylim((0,1.05))
    plt.plot(1e3 * stc_mft.times, cdmdata[0, :], 'r', label='lh')
    plt.plot(1e3 * stc_mft.times, cdmdata[1, :], 'g', label='rh')
    plt.plot(1e3 * stc_mft.times, cdmdata[2, :], 'b', label='all')
    # plt.plot(1e3 * stc_mft.times, cdmdata[3,:],'m',label='lh,fit')
    # plt.plot(1e3 * stc_mft.times, cdmdata[4,:],'c',label='rh,fit')
    # plt.plot(1e3 * stc_mft.times, cdmdata[5,:],'k',label='all,fit')
    plt.xlabel('time (ms)')
    plt.ylabel('cdm value')
    plt.legend()
    plt.savefig('testfig_cdm')
    plt.close()


# TODO jlngdata computation to be added into apply_mft
def plot_jlong_data(stc_mft, jlngdata):
    print "##### Attempting to plot jlong data:"
    fig = plt.figure()
    # plt.ylim((0,1.05))
    plt.plot(1e3 * stc_mft.times, jlngdata[0, :], 'r', label='lh')
    plt.plot(1e3 * stc_mft.times, jlngdata[1, :], 'g', label='rh')
    plt.plot(1e3 * stc_mft.times, jlngdata[2, :], 'b', label='all')
    plt.xlabel('time (ms)')
    plt.ylabel('j_long value')
    plt.legend()
    plt.savefig('testfig_jlong')
    plt.close()
