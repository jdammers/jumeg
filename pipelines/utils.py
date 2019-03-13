import mne
import os
import os.path as op
import pickle

from jumeg.jumeg_suggest_bads import suggest_bads
from jumeg.jumeg_interpolate_bads import interpolate_bads


def noise_reduction(dirname, raw_fname, denoised_fname, refnotch):
    """
    Apply the noise reducer to the raw file and save the result.

    Parameters:
    -----------
    dirname : str
        Path to the directory where the raw file is stored.
    raw_fname : str
        File name of the raw file.
    denoised_fname : str
        File name under which the denoised raw file is saved.
    refnotch : list
        List of frequencies for which a notch filter is applied.

    Returns:
    --------
    None
    """

    from jumeg.jumeg_noise_reducer import noise_reducer, plot_denoising

    # read the raw file
    raw = mne.io.Raw(op.join(dirname, raw_fname), preload=True)

    # apply noise reducer thrice to reference channels with different freq parameters
    # the nr-raw.fif are rewritten
    # low pass filter for freq below 5 hz
    raw_nr = noise_reducer(raw_fname, raw=raw, reflp=5., return_raw=True)

    raw.close()

    raw_nr = noise_reducer(raw_fname, raw=raw_nr, refhp=0.1, noiseref=['RFG ...'], return_raw=True)

    # notch filter to remove power line noise
    raw_nr = noise_reducer(raw_fname, raw=raw_nr, refnotch=refnotch,
                           fnout=op.join(dirname, denoised_fname),
                           return_raw=True)

    raw_nr.close()

    # plot final plotting
    plot_name = denoised_fname.rsplit('-raw.fif')[0] + '-plot'
    plot_denoising([op.join(dirname, raw_fname), op.join(dirname, denoised_fname)],
                   n_jobs=2, fnout=op.join(dirname, plot_name), show=False)


def init_bads_dict(bads_dict_fname):
    if op.isfile(bads_dict_fname):
        bads_dict_file = open(bads_dict_fname, mode='r')
        bads_dict = pickle.load(bads_dict_file)
        bads_dict_file.close()

    else:
        bads_dict = dict()

    return bads_dict


def save_bads_dict(bads_dict, bads_dict_fname):
    bads_dict_file = open(bads_dict_fname, mode='w')
    pickle.dump(bads_dict, bads_dict_file)
    bads_dict_file.close()


def interpolate_bads_batch(subject_list, subjects_dir, bads_dict_fname):
    """
    Scan the subject directories for files ending in ',nr-raw.fif' and
    ',nr-empty.fif' and interpolate the bad channels.

    Parameters:
    -----------
    subject_list : list of str
        List of all subject IDs.
    subjects_dir : str
        Path to the subjects directory.
    bads_dict_fname : str
        Name for the file storing the dictionary containing
        a list of bad channels for each raw file.

    Returns:
    --------
    """

    bads_dict = init_bads_dict(bads_dict_fname)

    for subj in subject_list:
        dirname = op.join(subjects_dir, subj)
        sub_file_list = os.listdir(dirname)
        for raw_fname in sub_file_list:

            if raw_fname.endswith('meeg,nr-raw.fif'):
                bcc_fname = op.join(dirname, raw_fname.split('/')[-1].split('-raw.fif')[0] + ',bcc-raw.fif')
                _interpolate_bads(raw_fname, bcc_fname, bads_dict, bads_dict_fname, dirname)

            if raw_fname.endswith('rfDC,nr-empty.fif'):
                bcc_fname = op.join(dirname, raw_fname.split('/')[-1].split('-empty.fif')[0] + ',bcc-empty.fif')
                _interpolate_bads(raw_fname, bcc_fname, bads_dict, bads_dict_fname, dirname)


def _interpolate_bads(raw_fname, bcc_fname, bads_dict, bads_dict_fname, dirname):

    bads_dict_key = bcc_fname.split('/')[-1]

    if not op.isfile(bcc_fname):

        raw = mne.io.Raw(op.join(dirname, raw_fname), preload=True)
        if bads_dict_key not in bads_dict:
            #  automatically suggest bad channels and plot results for visual inspection
            marked, raw = suggest_bads(raw, show_raw=True)
            bads_dict[bads_dict_key] = marked
        else:
            marked = bads_dict[bads_dict_key]
            raw.info['bads'] = marked
        # Interpolate bad channels using jumeg
        raw_bcc = interpolate_bads(raw, origin=None, reset_bads=True)
        raw_bcc.plot(block=True)
        raw_bcc.save(bcc_fname, overwrite=True)
        save_bads_dict(bads_dict, bads_dict_fname)
