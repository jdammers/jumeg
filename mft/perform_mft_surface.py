"""
====================
Jumeg MFT example.
====================
"""

from mne.datasets import sample
from jumeg_mft_funcs import apply_mft
import jumeg_mft_plot

checkresults = True

data_path = sample.data_path()
subject = 'sample'
subjects_dir = data_path + '/subjects'
fwdname = data_path + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
evoname = data_path + '/MEG/sample/sample_audvis-ave.fif'
evocondition = 'Left Auditory'
rawname = data_path + '/MEG/sample/sample_audvis_10s-raw.fif'
t1_fname = subjects_dir + '/' + 'sample/mri/T1.mgz'

# Set up pick list: MEG - bad channels
want_meg = 'mag'
want_ref = False
want_eeg = False
want_stim = False
exclude = 'bads'
include = []

print "########## MFT parameter:"
# mftpar = { 'prbfct':'Gauss',
#           'prbcnt':np.array([[-1.039, 0.013,0.062],[-0.039, 0.013,0.062]]),
#           'prbhw':np.array([[0.040,0.040,0.040],[0.040,0.040,0.040]]) }
mftpar = {'prbfct': 'uniform',
           'prbcnt': None,
           'prbhw': None}
mftpar.update({'iter': 8, 'currexp': 1.0})
mftpar.update({'regtype': 'PzetaE', 'zetareg': 1.00})
# mftpar.update({ 'regtype':'classic', 'zetareg':1.0})
mftpar.update({'solver': 'lu', 'svrelcut': 5.e-4})

print "mftpar['prbcnt'  ] = ",mftpar['prbcnt']
print "mftpar['prbhw'   ] = ",mftpar['prbhw']
print "mftpar['iter'    ] = ",mftpar['iter']
print "mftpar['regtype' ] = ",mftpar['regtype']
print "mftpar['zetareg' ] = ",mftpar['zetareg']
print "mftpar['solver'  ] = ",mftpar['solver']
print "mftpar['svrelcut'] = ",mftpar['svrelcut']
cdmcut = 0.10
print "cdmcut = ", cdmcut

print "##########################"
print "##### Calling apply_mft()"
print "##########################"
fwdmag, qualmft, stc_mft = apply_mft(fwdname, evoname, evocondition=evocondition,
                                     subject=subject, meg=want_meg,
                                     mftpar=mftpar, verbose='verbose')

stcdata = stc_mft.data

print " "
print "########## Some geo-numbers:"
lhinds = np.where(fwdmag['source_rr'][:, 0] <= 0.)
rhinds = np.where(fwdmag['source_rr'][:, 0] > 0.)
print "> Discriminating lh/rh by sign of fwdmag['source_rr'][:,0]:"
print "> lhinds[0].shape[0] = ", lhinds[0].shape[0], " rhinds[0].shape[0] = ", rhinds[0].shape[0]
invmri_head_t = mne.transforms.invert_transform(fwdmag['info']['mri_head_t'])
mrsrc = np.zeros(fwdmag['source_rr'].shape)
mrsrc = mne.transforms.apply_trans(invmri_head_t['trans'], fwdmag['source_rr'], move=True)
lhmrinds = np.where(mrsrc[:, 0] <= 0.)
rhmrinds = np.where(mrsrc[:, 0] > 0.)
print "> Discriminating lh/rh by sign of fwdmag['source_rr'][:,0] in MR coords:"
print "> lhmrinds[0].shape[0] = ", lhmrinds[0].shape[0], " rhmrinds[0].shape[0] = ", rhmrinds[0].shape[0]

# plotting routines
jumeg_mft_plot.jumeg_mft_plot.plot_global_cdv_dist(stcdata)
jumeg_mft_plot.plot_visualize_mft_sources(fwdmag, stcdata)
jumeg_mft_plot.plot_cdv_distribution(fwdmag, stc_data)
jumeg_mft_plot.plot_max_amplitude_data(fwdmag, stcdata)
jumeg_mft_plot.plot_max_cdv_data(stcdatamft, lhmrinds, rhmrinds)
jumeg_mft_plot.plot_cdvsum_data(stcdatamft, lhmrinds, rhmrinds)
jumeg_mft_plot.plot_quality_data(qualmft, stf_mft)

# jumeg_mft_plot.plot_cdm_data(stc, cdmdata)
# jumeg_mft_plot.plot_jlong_data(stc, jlngdata)

print_transforms = False
if print_transforms:
    print "##### Transforms:"
    print "fwdmag['info']['mri_head_t']:"
    print fwdmag['info']['mri_head_t']
    invmri_head_t = mne.transforms.invert_transform(fwdmag['info']['mri_head_t'])
    print "Inverse of fwdmag['info']['mri_head_t']:"
    print invmri_head_t

# stc_feat.save will create lh/rh-stc-s [md5-]equal to stc.save 00/01 above.
stcfname = os.path.join(os.path.dirname(evoname),
                        os.path.basename(evoname).split('-')[0]) + "tst"
print "stc basefilename: %s" % stcfname
stc_feat.save(stcfname, verbose=True)

write_tab_files = False
if write_tab_files:
    tabfilenam = 'testtab.dat'
    print "##### Creating %s with |cdv(time_idx=%d)|" % (tabfilenam, time_idx)
    tabfile = open(tabfilenam, mode='w')
    cdvnmax = np.max(stcdata[:, time_idx])
    tabfile.write("# time_idx = %d\n" % time_idx)
    tabfile.write("# max amplitude = %11.4e\n" % cdvnmax)
    tabfile.write("#  x/mm    y/mm    z/mm     |cdv|   index\n")
    for ipnt in xrange(n_loc/3):
        copnt = 1000.*fwdmag['source_rr'][ipnt]
        tabfile.write(" %7.2f %7.2f %7.2f %11.4e %5d\n" %\
                      (copnt[0],copnt[1],copnt[2],stcdata[ipnt,time_idx],ipnt))
    tabfile.close()

    tabfilenam = 'testwtab.dat'
    print "##### Creating %s with wdist0" % tabfilenam
    tabfile = open(tabfilenam,mode='w')
    tabfile.write("# time_idx = %d\n" % time_idx)
    for icnt in xrange(prbcnt.shape[0]):
        cocnt = 1000.*prbcnt[icnt,:]
        tabfile.write("# center  %7.2f %7.2f %7.2f\n" % (cocnt[0],cocnt[1],cocnt[2]))

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

print "Done."
