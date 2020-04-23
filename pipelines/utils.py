import mne
import os
import os.path as op
import pickle
import fnmatch
import errno

from jumeg.jumeg_suggest_bads import suggest_bads
from jumeg.jumeg_interpolate_bads import interpolate_bads as jumeg_interpolate_bads


def reset_directory(path=None):
    """
    check whether the directory exits, if yes, recreate the directory
    ----------
    path : the target directory.
    """
    import shutil
    isexists = os.path.exists(path)
    if isexists:
        shutil.rmtree(path)
    os.makedirs(path)


def set_directory(path=None):
    """
    check whether the directory exits, if no, create the directory
    ----------
    path : the target directory.

    """
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)


def find_files(rootdir='.', pattern='*'):
    """
    Looks for all files in the root directory matching the file
    name pattern.

    Parameters:
    -----------
    rootdir : str
        Path to the directory to be searched.
    pattern : str
        File name pattern to be looked for.

    Returns:
    --------
    files : list
        List of file names matching the pattern.
    """

    files = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    files = sorted(files)

    return files


def noise_reduction(dirname, raw_fname, denoised_fname, nr_cfg, state_space_fname):
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
    nr_cfg: dict
        Dict containing the noise reducer specific settings from
        the config file.
    state_space_fname : str
        Second half of the name under which the state space dict is to
        be saved, e.g., subj + state_space_name.

    Returns:
    --------
    None
    """
    refnotch = nr_cfg['refnotch']
    reflp = nr_cfg['reflp']
    refhp = nr_cfg['refhp']
    noiseref_hp = nr_cfg['noiseref_hp']

    from jumeg.jumeg_noise_reducer import noise_reducer, plot_denoising

    subj = op.basename(raw_fname).split('_')[0]
    ss_dict_fname = op.join(op.dirname(raw_fname), subj + state_space_fname)

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
                   n_jobs=1, fnout=op.join(dirname, plot_name), show=False)

    # save config file
    nr_dict = nr_cfg.copy()
    nr_dict['input_file'] = op.join(dirname, raw_fname)
    nr_dict['process'] = 'noise_reducer'
    nr_dict['output_file'] = op.join(dirname, denoised_fname)

    save_state_space_file(ss_dict_fname, process_config_dict=nr_dict)


def save_state_space_file(config_dict_fname, process_config_dict):
    """

    Parameters:
    -----------
    ss_dict_fname : str
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

    output_file = process_config_dict['output_file']

    config_dict[output_file] = process_config_dict

    save_dict(config_dict, config_dict_fname)


def init_dict(dict_fname):
    if op.isfile(dict_fname):
        dict_file = open(dict_fname, mode='rb')
        dictionary = pickle.load(dict_file)
        dict_file.close()

    else:
        dictionary = dict()

    return dictionary


def save_dict(bads_dict, bads_dict_fname):
    bads_dict_file = open(bads_dict_fname, mode='wb')
    pickle.dump(bads_dict, bads_dict_file)
    bads_dict_file.close()


def interpolate_bads_batch(subject_list, subjects_dir, state_space_fname):
    """
    Scan the subject directories for files ending in ',nr-raw.fif' and
    ',nr-empty.fif' and interpolate the bad channels.

    Parameters:
    -----------
    subject_list : list of str
        List of all subject IDs.
    recordings_dir : str
        Path to the subjects directory.
    state_space_fname : str
        Second half of the name under which the state space dict is to
        be saved, e.g., subj + state_space_name.

    Returns:
    --------
    """

    for subj in subject_list:
        dirname = op.join(subjects_dir, subj)
        sub_file_list = os.listdir(dirname)
        for raw_fname in sub_file_list:

            if raw_fname.endswith('meeg,nr-raw.fif'):
                bcc_fname = op.join(dirname, raw_fname.split('/')[-1].split('-raw.fif')[0] + ',bcc-raw.fif')
                interpolate_bads(raw_fname, bcc_fname, dirname, state_space_fname)

            if raw_fname.endswith('rfDC,nr-empty.fif'):
                bcc_fname = op.join(dirname, raw_fname.split('/')[-1].split('-empty.fif')[0] + ',bcc-empty.fif')
                interpolate_bads(raw_fname, bcc_fname, dirname, state_space_fname)


def interpolate_bads(raw_fname, bcc_fname, dirname, state_space_fname):

    ib_dict = dict()

    subj = op.basename(raw_fname).split('_')[0]
    ss_dict_fname = op.join(op.dirname(raw_fname), subj + state_space_fname)

    if not op.isfile(bcc_fname):

        raw = mne.io.Raw(op.join(dirname, raw_fname), preload=True)
        # automatically suggest bad channels and plot results for visual inspection
        marked, raw = suggest_bads(raw, show_raw=True)

        ib_dict['bads_channels'] = raw.info['bads']

        # Interpolate bad channels using jumeg
        raw_bcc = jumeg_interpolate_bads(raw, origin=None, reset_bads=True)

        # check if everything looks good
        raw_bcc.plot(block=True)
        raw_bcc.save(bcc_fname, overwrite=True)

        ib_dict['input_file'] = raw_fname
        ib_dict['process'] = 'interpolate_bads'
        ib_dict['output_file'] = bcc_fname

        save_state_space_file(ss_dict_fname, process_config_dict=ib_dict)


def mksubjdirs(subjects_dir, subj):
    """
    Create the directories required by freesurfer.

    Parameters:
    -----------
    recordings_dir : str
        Path to the subjects directory.
    subj : str
        ID of the subject.

    Returns:
    --------
    None
    """

    # make list of folders to create

    folders_to_create = ['bem', 'label', 'morph', 'mpg', 'mpg', 'mri', 'rgb',
                         'scripts', 'stats', 'surf', 'tiff', 'tmp', 'touch']

    mri_subfolders = ['aseg', 'brain', 'filled', 'flash', 'fsamples', 'norm',
                      'orig', 'T1', 'tmp', 'transforms', 'wm']

    for count in range(0,len(mri_subfolders)):
        mri_subfolders[count] = os.path.join('mri', mri_subfolders[count])

    folders_to_create.extend(mri_subfolders)

    # create folders
    for folder in folders_to_create:
        dirname_prep = os.path.join(subjects_dir, subj, folder)

        try:
            os.makedirs(dirname_prep)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(dirname_prep):
                pass
            else:
                raise