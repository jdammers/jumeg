import mne
import os
import os.path as op
import pickle

from jumeg.jumeg_suggest_bads import suggest_bads
from jumeg.jumeg_interpolate_bads import interpolate_bads as jumeg_interpolate_bads


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

    reflp = 5.
    refhp = 0.1
    noiseref_hp = ['RFG ...']

    from jumeg.jumeg_noise_reducer import noise_reducer, plot_denoising

    subj = op.basename(raw_fname).split('_')[0]
    config_dict_fname = op.join(op.dirname(raw_fname), subj + '_config_dict.pkl')

    # read the raw file
    raw = mne.io.Raw(op.join(dirname, raw_fname), preload=True)

    # apply noise reducer thrice to reference channels with different freq parameters
    # the nr-raw.fif are rewritten
    # low pass filter for freq below 5 hz
    raw_nr = noise_reducer(raw_fname, raw=raw, reflp=reflp, return_raw=True)

    raw.close()

    raw_nr = noise_reducer(raw_fname, raw=raw_nr, refhp=refhp, noiseref=noiseref_hp, return_raw=True)

    # notch filter to remove power line noise
    raw_nr = noise_reducer(raw_fname, raw=raw_nr, refnotch=refnotch,
                           fnout=op.join(dirname, denoised_fname),
                           return_raw=True)

    raw_nr.close()

    # plot final plotting
    plot_name = denoised_fname.rsplit('-raw.fif')[0] + '-plot'
    plot_denoising([op.join(dirname, raw_fname), op.join(dirname, denoised_fname)],
                   n_jobs=2, fnout=op.join(dirname, plot_name), show=False)

    # save config file
    nr_dict = dict()
    nr_dict['reflp'] = reflp
    nr_dict['refhp'] = refhp
    nr_dict['refnotch'] = refnotch
    nr_dict['output_file'] = denoised_fname

    save_config_file(config_dict_fname, process='noise_reducer',
                     input_fname=raw_fname, process_config_dict=nr_dict)


def save_config_file(config_dict_fname, process, input_fname, process_config_dict):
    """

    Parameters:
    -----------
    config_dict_fname : str
        Name under which the config dict is to be saved.
    process : str
        Name of the process, e.g., 'noise_reducer'.
    input_fname : str
        Name of the input raw file.
    process_config_dict : dict
        Dictionary containing all the settings used in the process.

    Returns:
    --------
    None
    """

    config_dict = init_dict(config_dict_fname)

    try:
        config_dict[process][input_fname] = process_config_dict
    except KeyError:
        # dict does not exist create first
        config_dict[process] = dict()
        config_dict[process][input_fname] = process_config_dict

    save_dict(config_dict, config_dict_fname)


def init_dict(bads_dict_fname):
    if op.isfile(bads_dict_fname):
        bads_dict_file = open(bads_dict_fname, mode='r')
        bads_dict = pickle.load(bads_dict_file)
        bads_dict_file.close()

    else:
        bads_dict = dict()

    return bads_dict


def save_dict(bads_dict, bads_dict_fname):
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

    bads_dict = init_dict(bads_dict_fname)

    for subj in subject_list:
        dirname = op.join(subjects_dir, subj)
        sub_file_list = os.listdir(dirname)
        for raw_fname in sub_file_list:

            if raw_fname.endswith('meeg,nr-raw.fif'):
                bcc_fname = op.join(dirname, raw_fname.split('/')[-1].split('-raw.fif')[0] + ',bcc-raw.fif')
                interpolate_bads(raw_fname, bcc_fname, bads_dict, bads_dict_fname, dirname)

            if raw_fname.endswith('rfDC,nr-empty.fif'):
                bcc_fname = op.join(dirname, raw_fname.split('/')[-1].split('-empty.fif')[0] + ',bcc-empty.fif')
                interpolate_bads(raw_fname, bcc_fname, bads_dict, bads_dict_fname, dirname)


def interpolate_bads(raw_fname, bcc_fname, dirname):

    ib_dict = dict()

    subj = op.basename(raw_fname).split('_')[0]
    config_dict_fname = op.join(op.dirname(raw_fname), subj + '_config_dict.pkl')

    if not op.isfile(bcc_fname):

        raw = mne.io.Raw(op.join(dirname, raw_fname), preload=True)
        # automatically suggest bad channels and plot results for visual inspection
        marked, raw = suggest_bads(raw, show_raw=True)

        ib_dict['suggested_bads'] = raw.info['bads']

        # Interpolate bad channels using jumeg
        raw_bcc = jumeg_interpolate_bads(raw, origin=None, reset_bads=True)

        # check if everything looks good
        raw_bcc.plot(block=True)
        raw_bcc.save(bcc_fname, overwrite=True)

        ib_dict['bads_visual_inspection'] = raw_bcc.info['bads']
        ib_dict['output_file'] = bcc_fname

        save_config_file(config_dict_fname, process='interpolate_bads',
                         input_fname=raw_fname, process_config_dict=ib_dict)
