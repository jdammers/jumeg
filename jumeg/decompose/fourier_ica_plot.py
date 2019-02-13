# Authors: Lukas Breuer <l.breuer@fz-juelich.de>

"""
----------------------------------------------------------------------
--- jumeg.decompose.fourier_ica_plot ---------------------------------
----------------------------------------------------------------------
 autor      : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 17.11.2016
 version    : 1.1

----------------------------------------------------------------------
 This is a simple implementation to plot the results achieved by
 applying FourierICA
----------------------------------------------------------------------
"""
#######################################################
#                                                     #
#          plotting functions for FourierICA          #
#                                                     #
#######################################################
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Simple function to adjust axis in plots
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def adjust_spines(ax, spines, labelsize=10):


    """
    Simple function to adjust axis in plots

        Parameters
        ----------
        ax: axis object
            Plot object which should be adjusted
        spines: list of strings ['bottom', 'left']
            Name of the axis which should be adjusted
        labelsize: integer
            Font size for the x- and y-axis labels
    """

    for loc, spine in list(ax.spines.items()):
        if loc in spines:
            spine.set_position(('outward', 4))  # outward by 4 points
            # spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# function to generate automatically combined labels
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_combined_labels(subject='fsaverage', subjects_dir=None,
                        parc='aparc.a2009s'):

    """
    Helper function to combine labels automatically
    according to previous studies.

        Parameters
        ----------
        subject: string containing the subjects name
            default: subject='fsaverage'
        subjects_dir: Subjects directory. If not given the
            system variable SUBJECTS_DIR is used
            default: subjects_dir=None
        parc: name of the parcellation to use for reading
            in the labels
            default: parc='aparc.a2009s'

        Return
        ------
        label_keys: names of the new labels
        labels: list containing the combined labels
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from mne import read_labels_from_annot
    import numpy as np
    from os.path import join


    # ------------------------------------------
    # define labels based on previous studies
    # ------------------------------------------
    # to get more information about the label names and their
    # locations check the following publication:
    # Destrieux et al. (2010), Automatic parcellation of human
    # cortical gyri and sulci using standard anatomical nomenclature,
    # NeuroImage, DOI: 10.1016/j.neuroimage.2010.06.010
    label_combinations = {
        'auditory': ['G_temp_sup-G_T_transv', 'G_temp_sup-Plan_polar',
                     'Lat_Fis-post'],
        'broca': ['G_front_inf-Opercular', 'G_front_inf-Triangul',
                  'Lat_Fis-ant-Vertical'],
        'cingulate': ['G_cingul-Post-dorsal', 'G_cingul-Post-ventral',
                      'G_and_S_cingul-Ant', 'G_and_S_cingul-Mid-Ant',
                      'G_and_S_cingul-Mid-Post', 'S_pericallosal',
                      'cingul-Post-ventral'],
        'frontal': ['G_and_S_frontomargin', 'G_and_S_transv_frontopol',
                    'G_front_inf-Orbital', 'G_front_middle',
                    'G_front_sup', 'G_orbital',
                    'G_rectus', 'G_subcallosal',
                    'Lat_Fis-ant-Horizont', 'S_front_inf',
                    'S_front_middle', 'S_front_sup',
                    'S_orbital_lateral', 'S_orbital-H_Shaped',
                    'S_suborbital'],
        'gustatory': ['G_and_S_subcentral'],
        'insula': ['S_circular_insula_ant', 'S_circular_insula_inf',
                   'S_circular_insula_sup', 'G_Ins_lg_and_S_cent_ins',
                   'G_insular_short'],
        'motor': ['G_precentral', 'S_precentral-sup-part',
                  'S_precentral-inf-part', 'S_central'],
        'olfactory': ['S_temporal_transverse'],
        'somatosensory': ['G_postcentral', 'S_postcentral'],
        'somatosensory associated': ['G_and_S_paracentral', 'G_pariet_inf-Angular',
                                     'G_parietal_sup', 'S_cingul-Marginalis',
                                     'S_intrapariet_and_P_trans'],
        'temporal': ['G_oc-temp_lat-fusifor', 'G_oc-temp_med-Parahip',
                     'G_temp_sup-Plan_polar', 'G_temporal_inf',
                     'G_temporal_middle', 'G_temp_sup-Lateral',
                     'Pole_temporal', 'S_collat_transv_ant',
                     'S_oc-temp_lat', 'S_oc-temp_med_and_Lingual',
                     'S_temporal_inf', 'S_temporal_sup'],
        'vision': ['G_and_S_occipital_inf', 'G_occipital_middle',
                   'G_oc-temp_med-Lingual', 'S_collat_transv_post',
                   'S_oc_sup_and_transversal', 'S_occipital_ant',
                   'S_oc_middle_and_Lunatus'],
        'visual': ['G_cuneus', 'G_precuneus',
                   'S_calcarine', 'S_parieto_occipital',
                   'G_occipital_sup', 'Pole_occipital',
                   'S_subparietal'],
        'wernicke': ['G_pariet_inf-Supramar', 'G_temp_sup-Plan_tempo',
                     'S_interm_prim-Jensen']
    }
    label_keys = list(label_combinations.keys())
    labels = []


    # ------------------------------------------
    # combine labels
    # ------------------------------------------
    # loop over both hemispheres
    for hemi in ['lh', 'rh']:
        # read all labels in
        labels_all = read_labels_from_annot(subject, parc=parc, hemi=hemi,
                                            surf_name='inflated',
                                            subjects_dir=subjects_dir,
                                            verbose=False)

        # loop over all labels to extract label names
        label_names = []
        for label in labels_all:
            label_names.append(label.name)

        # ------------------------------------------
        # now generate labels based on previous
        # studies
        # ------------------------------------------
        # loop over all previously defined labels
        for label_key in label_keys:
            # get name of all labels related to the current one
            label_members = label_combinations[label_key]
            label_members = [x+'-'+hemi for x in label_members]

            # check which labels we need for the current one
            idx_labels_want = np.where(np.in1d(label_names, label_members))[0]
            labels_want = [labels_all[i] for i in idx_labels_want]

            # combine labels
            label_new = np.sum(labels_want)
            label_new.name = label_key + '-' + hemi

            # fill the surface between sources
            label_new.values.fill(1.0)
            label_new.smooth(subject=subject, subjects_dir=subjects_dir)

            # save new label
            fnout = join(subjects_dir, subject, 'label',
                         hemi + '.' + label_key + '.label')
            label_new.save(fnout)
            labels.append(label_new)


    return label_keys, labels




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# function to get the anatomical label to a given vertex
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_anat_label_name(vertex, hemi, labels=None, subject='fsaverage',
                        subjects_dir=None, parc='aparc.a2009s'):

    """
    Helper function to get to a given vertex the
    name of the anatomical label

        Parameters
        ----------
        vertex: integer containing the vertex number
        hemi: string containing the information in which
            hemisphere the vertex is located. Should be
             either 'lh' or 'rh'
        labels: labels to use for checking. If not given
            the labels are read from the subjects directory
            default: labels=None
        subject: string containing the subjects name
            default: subject='fsaverage'
        subjects_dir: Subjects directory. If not given the
            system variable SUBJECTS_DIR is used
            default: subjects_dir=None
        parc: name of the parcellation to use for reading
            in the labels
            default: parc='aparc.a2009s'


        Return
        ------
        name: string containing the name of the anatomical
            label related to the given vertex
    """


    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from mne import read_labels_from_annot
    import numpy as np


    # ------------------------------------------
    # check input parameter
    # ------------------------------------------
    # check if labels are given or must be read
    if not labels:
        labels = read_labels_from_annot(subject, parc=parc, hemi=hemi,
                                        surf_name='inflated',
                                        subjects_dir=subjects_dir,
                                        verbose=False)

    # ------------------------------------------
    # loop over labels to find corresponding
    # label
    # ------------------------------------------
    name = ''

    for label in labels:

        if label.hemi == hemi:
            # get vertices of current label
            label_vert = np.in1d(np.array(vertex), label.vertices)
            if label_vert:
                name = label.name
                break

    if name == '':
        name = 'unknown-' + hemi

    return name




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# function to get the MNI-coordinate(s) to a given
# FourierICA component
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_mni_coordinates(A_orig,
                        subject='fsaverage', subjects_dir=None,
                        parc='aparc.a2009s', percentile=97,
                        combine_labels=True):

    """
    Helper function to get the MNI-coordinate(s) to a given
    FourierICA component. The selection if a component has
    activation in both hemispheres or only in one is made
    like follows: estimate for each component an activation
    threshold based on the given percentile. Next, estimate
    the total number of voxels in the component which are
    above the estimated threshold. Now check if at least 20%
    of the total number of voxels above threshold are in each
    hemisphere. If yes both hemispheres are marked as active,
    otherwise only one.

        Parameters
        ----------
        A_orig:  array
            2D-mixing-array (nvoxel, ncomp) estimated
            when applying FourierICA
        subject: string containing the subjects name
            default: subject='fsaverage'
        subjects_dir: Subjects directory. If not given the
            system variable SUBJECTS_DIR is used
            default: subjects_dir=None
        parc: name of the parcellation to use for reading
            in the labels
            default: parc='aparc.a2009s'
        percentile: integer
            value between 0 and 100 used to set a lower
            limit for the shown intensity range of the
            spatial plots
        combine_labels: if set labels are combined automatically
            according to previous studies
            default: combine_labels=True

        Return
        ------
        mni_coords: dictionary
            The dictionary contains two elements: 'rh' and 'lh',
            each of which containing a list with the MNI
            coordinates as string.
            Note, each list contains the same number of
            elements as components are given. If there is no MNI
            coordinate for a component an empty string is used, e.g.
            for two components
            {'rh': ['(37.55, 1.58, -21.71)', '(44.78, -10.41, 27.89)'],
             'lh': ['(-39.43, 5.60, -27.80)', '']}
        hemi_loc_txt: list
            containing for each FourierICA component to which region
            it spatially belongs ('left', 'right' or 'both')
        classification: dictionary
            classification object. It is a dictionary containing
            two sub-dictionaries 'lh' and 'rh' (for left and
            right hemisphere). In both sub-dictionaries the
            information about the groups is stored, i.e. a
            group/region name + the information which components
            are stored in this group (as indices). An example
            for 6 components might look like this:
            {'rh': {'somatosensory': [1, 3], 'cingulate': [4, 5]},
             'lh': {'somatosensory': [1, 2], 'cingulate': [0, 5]}}
        labels: list of strings
            names of the labels which are involved in this data set
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from mne import vertex_to_mni
    import numpy as np
    from os import environ
    import types


    # -------------------------------------------
    # check input parameter
    # -------------------------------------------
    if not subjects_dir:
        subjects_dir = environ.get('SUBJECTS_DIR')


    # -------------------------------------------
    # generate spatial profiles
    # (using magnitude and phase)
    # -------------------------------------------
    if isinstance(A_orig[0, 0], complex):
        A_orig_mag = np.abs(A_orig)
    else:
        A_orig_mag = A_orig


    # -------------------------------------------
    # set some parameters
    # -------------------------------------------
    nvoxel, ncomp = A_orig_mag.shape
    nvoxel_half = int(nvoxel / 2)
    hemi = ['lh', 'rh']
    hemi_names = ['left ', 'right', 'both ']
    hemi_indices = [[0, nvoxel_half], [nvoxel_half, -1]]
    hemi_loc_txt = np.array(['     '] * ncomp)
    hemi_loc = np.zeros(ncomp)


    # -------------------------------------------
    # generate structures to save results
    # -------------------------------------------
    # generate dictionary to save MNI coordinates
    mni_coords = {'rh': [''] * ncomp, 'lh': [''] * ncomp}


    # ------------------------------------------
    # check if labels should be combined
    # automatically
    # ------------------------------------------
    if combine_labels:
        label_names, labels = get_combined_labels(subject=subject,
                                                  subjects_dir=subjects_dir,
                                                  parc=parc)

        # generate empty classification dictionary
        class_keys = label_names[:]
        class_keys.append('unknown')
        classification = {'lh': {key: [] for key in class_keys},
                          'rh': {key: [] for key in class_keys}}
    # if not generate empty variables
    else:
        label_names, labels = None, None
        classification = {}


    # ------------------------------------------
    # loop over all components
    # ------------------------------------------
    for icomp in range(ncomp):

        # ------------------------------------------
        # extract maxima in the spatial profile of
        # the current component separately for both
        # hemispheres
        # ------------------------------------------
        idx_ver_max_lh = np.argmax(A_orig_mag[:nvoxel_half, icomp])
        idx_ver_max_rh = np.argmax(A_orig_mag[nvoxel_half:, icomp])

        # ------------------------------------------
        # check for both maxima if they are
        # significant
        # ------------------------------------------
        # set some paremeter
        threshold = np.percentile(A_orig_mag[:, icomp], percentile)
        nidx_above = len(np.where(A_orig_mag[:, icomp] > threshold)[0])
        cur_label_name = []

        # loop over both hemispheres
        for idx_hemi, idx_vertex_max in enumerate([idx_ver_max_lh, idx_ver_max_rh]):

            # get the number of vertices above the threshold
            # in the current hemisphere
            nidx_above_hemi = len(np.where(A_orig_mag[hemi_indices[idx_hemi][0]:hemi_indices[idx_hemi][1],
                                           icomp] > threshold)[0])

            # check if at least 20% of all vertices above the threshold
            # are in the current hemisphere
            if nidx_above_hemi * 5 > nidx_above:
                # get MNI-coordinate
                mni_coord = vertex_to_mni(idx_vertex_max, idx_hemi, subject,
                                          subjects_dir=subjects_dir)[0]

                # store results in structures
                mni_coords[hemi[idx_hemi]][icomp] = \
                    '(' + ', '.join(["%2.2f" % x for x in mni_coord]) + ')'

                # store hemisphere information
                hemi_loc[icomp] += idx_hemi + 1.0

                # ------------------------------------------
                # get MNI-coordinate to vertex as well as
                # the name of the corresponding anatomical
                # label
                # ------------------------------------------
                anat_name = get_anat_label_name(idx_vertex_max, hemi[idx_hemi],
                                                subject=subject, subjects_dir=subjects_dir,
                                                parc=parc, labels=labels)
                cur_label_name.append(anat_name[:-3])
            else:
                cur_label_name.append(' ')


        # ------------------------------------------
        # check which results must be saved
        # ------------------------------------------
        if combine_labels:

            # check if activation was found in both hemispheres
            # --> if not we can directly save the results
            if ' ' in cur_label_name:
                # adjust classification dictionary
                if cur_label_name[0] == ' ':
                    classification[hemi[1]][cur_label_name[1]].append(icomp)
                else:
                    classification[hemi[0]][cur_label_name[0]].append(icomp)

            # --> otherwise we have to make sure that we group the
            #     component only into one region
            else:
                # check if both vertices are in the same anatomical location
                # --> then we have no problem
                if cur_label_name[0] == cur_label_name[1]:
                    classification[hemi[0]][cur_label_name[0]].append(icomp)
                    classification[hemi[1]][cur_label_name[1]].append(icomp)
                else:
                    # check if we have an unknown region being involved
                    # --> if yes chose the other one
                    if cur_label_name[0] == 'unknown':
                        classification[hemi[1]][cur_label_name[1]].append(icomp)
                        hemi_loc[icomp], mni_coords[hemi[0]][icomp] = 2, ''
                    elif cur_label_name[1] == 'unknown':
                        classification[hemi[0]][cur_label_name[0]].append(icomp)
                        hemi_loc[icomp], mni_coords[hemi[1]][icomp] = 1, ''
                    # otherwise chose the region with the strongest vertex
                    else:
                        if A_orig_mag[idx_ver_max_lh, icomp] > A_orig_mag[idx_ver_max_rh, icomp]:
                            classification[hemi[0]][cur_label_name[0]].append(icomp)
                            hemi_loc[icomp], mni_coords[hemi[1]][icomp] = 1, ''

                        else:
                            classification[hemi[1]][cur_label_name[1]].append(icomp)
                            hemi_loc[icomp], mni_coords[hemi[0]][icomp] = 2, ''



    # ------------------------------------------
    # adjust hemi_loc_txt if activity was found
    # in both hemispheres
    # ------------------------------------------
    for idx, hemi_name in enumerate(hemi_names):
        idx_change = np.where(hemi_loc == (idx + 1.0))[0]
        hemi_loc_txt[idx_change] = hemi_name


    # ------------------------------------------
    # adjust label_names to only contain regions
    # being involved in processing the current
    # data
    # ------------------------------------------
    labels = []
    for cur_hemi in hemi:
        for key in label_names:
            if classification[cur_hemi][key]:
                labels.append(key)

    labels = np.unique(labels).tolist()

    return mni_coords, hemi_loc_txt, classification, labels




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# helper function to check if classification was
# performed prior to plotting
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _check_classification(classification, ncomp):

    """
    Helper function to check if classification was
    performed prior to plotting

        Parameters
        ----------
        classification: dictionary
            classification object from the group_ica_object.
            It is a dictionary containing two sub-dictionaries
            'lh' and 'rh' (for left and right hemisphere). In
            both sub-dictionaries the information about the
            groups is stored, i.e. a group/region name + the
            information which components are stored in this
            group
        ncomp: integer
            number of components

        Return
        ------
        keys: list containing the group names
        key_borders: list containing the group borders, i.e.
            the information where to plot a new group name
        idx_sort: array containing the plotting order of
            the components, i.e. components beloning to one
            group are plotted together
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    import numpy as np


    # ------------------------------------------
    # check if classification was done
    # ------------------------------------------
    key_borders = []
    if np.any(classification):

        # initialize empty lists
        idx_sort = []
        keys_hemi = list(classification.keys())

        # sort keys
        keys = list(classification[keys_hemi[0]].keys())
        keys.sort(key=lambda v: v.upper())

        # set 'unknown' variables to the end
        keys.remove('unknown')
        keys.append('unknown')

        # remove keys with empty entries
        keys_want = []
        for key in keys:
            if classification[keys_hemi[0]][key] or\
                    classification[keys_hemi[1]][key]:
                keys_want.append(key)

        # loop over all keys
        for key in keys_want:
            # get indices to each class
            idx_lh = classification[keys_hemi[0]][key]
            idx_rh = classification[keys_hemi[1]][key]

            # get indices of components in both hemispheres
            idx_both = np.intersect1d(idx_lh, idx_rh)

            # get indices of components only in right hemisphere
            idx_only_rh = np.setdiff1d(idx_rh, idx_lh)

            # get indices of components only in left hemisphere
            idx_only_lh = np.setdiff1d(idx_lh, idx_rh)

            # add components to list of sorted indices
            idx_all = np.concatenate((idx_both, idx_only_rh, idx_only_lh))
            idx_sort += idx_all.tolist()
            key_borders.append(len(idx_all))

        # add first border and estimate cumulative sum to
        # have the right borders
        key_borders = np.insert(key_borders, 0, 1)
        key_borders = np.cumsum(key_borders)[:-1]


    # ------------------------------------------
    # if classification was not performed set
    # some default values
    # ------------------------------------------
    else:
        idx_sort = np.arange(ncomp)
        keys_want = []


    return keys_want, key_borders, idx_sort




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# helper function to handle time courses for plotting
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _get_temporal_envelopes(fourier_ica_obj, W_orig, temporal_envelope=[],
                            src_loc_data=[], tICA=False, global_scaling=True,
                            win_length_sec=None, tpre=None, flow=None):

    """
    Helper function to check if classification was
    performed prior to plotting

        Parameters
        ----------
        fourier_ica_obj: FourierICA object generated
            when applying jumeg.decompose.fourier_ica
        W_orig: array
            2D-demixing-array (ncomp x nvoxel) estimated
            when applying FourierICA
        temporal_envelope: list of arrays containing
            the temporal envelopes. If the temporal
            envelopes are already given here z-scoring
            and mean estimation is performed
        src_loc_data: array
            3D array containing the source localization
            data used for FourierICA estimation
            (nfreq x nepochs x nvoxel). Only necessary
            if temporal_envelope is not given.
        tICA: bool
            If set we know that temporal ICA was applied
            when estimating the FourierICA, i.e. when
            generating the temporal-envelopes the data
            must not be transformed from the Fourier
            domain to the time-domain
        global_scaling: bool
            If set all temporal-envelopes are globally
            scaled. Otherwise each component is scaled
            individually
        win_length_sec: float or None
            Length of the epoch window in seconds
        tpre: float or None
            Lower border (in seconds) of the time-window
            used for generating/showing the epochs. If
            'None' the value stored in 'fourier_ica_obj'
            is used
        flow: float, integer or None
            Lower frequency border for generating the
            temporal-envelope. If 'None' the frequency
            border stored in 'fourier_ica_obj' is used

        Return
        ------
        temporal_envelope_mean: list containing the 2D
            arrays of the mean temporal envelopes
            of the components
        temporal_envelope: list containing the 3D
            arrays of the temporal envelopes of the
            components. Necessary for estimating the
            spectral profiles
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from mne.baseline import rescale
    import numpy as np
    from scipy import fftpack


    # -------------------------------------------
    # check input parameter
    # -------------------------------------------
    if tpre == None:
        tpre = fourier_ica_obj.tpre
    if flow == None:
        flow = fourier_ica_obj.flow
    if not win_length_sec:
        win_length_sec = fourier_ica_obj.win_length_sec

    # estimate some simple parameter
    sfreq = fourier_ica_obj.sfreq
    ncomp, nvoxel = W_orig.shape

    win_ntsl = int(np.floor(sfreq * win_length_sec))
    startfftind = int(np.floor(flow * win_length_sec))


    # -------------------------------------------
    # check if temporal envelope is already
    # given or should be estimated
    # -------------------------------------------
    if temporal_envelope == []:

        # -------------------------------------------
        # check if 'src_loc_data' is given...
        # if not throw an error
        # -------------------------------------------
        if src_loc_data == []:
            print(">>> ERROR: You have to provide either the 'temporal_envelope' or")
            print(">>> 'src_loc_data'. Otherwise no temporal information can be plotted!")
            import pdb
            pdb.set_trace()


        # -------------------------------------------
        # get independent components
        # -------------------------------------------
        nfreq, nepochs, nvoxel = src_loc_data.shape
        act = np.zeros((ncomp, nepochs, nfreq), dtype=np.complex)
        if tICA:
            win_ntsl = nfreq

        temporal_envelope = np.zeros((nepochs, ncomp, win_ntsl))
        fft_act = np.zeros((ncomp, win_ntsl), dtype=np.complex)

        # loop over all epochs to get time-courses from
        # source localized data by inverse FFT
        for iepoch in range(nepochs):

            # normalize data
            src_loc_zero_mean = (src_loc_data[:, iepoch, :] - np.dot(np.ones((nfreq, 1)), fourier_ica_obj.dmean)) / \
                                np.dot(np.ones((nfreq, 1)), fourier_ica_obj.dstd)

            act[:ncomp, iepoch, :] = np.dot(W_orig, src_loc_zero_mean.transpose())
            #act[ncomp:, iepoch, :] = np.dot(W_orig, src_loc_zero_mean.transpose())

            if tICA:
                temporal_envelope[iepoch, :, :] = act[:, iepoch, :].real
            else:
                # -------------------------------------------
                # generate temporal profiles
                # -------------------------------------------
                # apply inverse STFT to get temporal envelope
                fft_act[:, startfftind:(startfftind + nfreq)] = act[:, iepoch, :]
                temporal_envelope[iepoch, :, :] = fftpack.ifft(fft_act, n=win_ntsl, axis=1).real



    # -------------------------------------------
    # average temporal envelope
    # -------------------------------------------
    if not isinstance(temporal_envelope, list):
        temporal_envelope = [[temporal_envelope]]

    ntemp = len(temporal_envelope)
    temporal_envelope_mean = np.empty((ntemp, 0)).tolist()
    times = (np.arange(win_ntsl) / sfreq + tpre)


    # -------------------------------------------
    # perform baseline correction
    # -------------------------------------------
    for itemp in range(ntemp):
        for icomp in range(ncomp):
            temporal_envelope[itemp][0][:, icomp, :] = rescale(temporal_envelope[itemp][0][:, icomp, :],
                                                               times, (None, 0), 'zscore')


    # -------------------------------------------
    # estimate mean from temporal envelopes
    # -------------------------------------------
    for itemp in range(ntemp):
        temporal_envelope_mean[itemp].append(np.mean(temporal_envelope[itemp][0], axis=0)[:, 5:-5])


    # -------------------------------------------
    # check if global scaling should be used
    # -------------------------------------------
    # if not scale each component separately between -0.5 and 0.5
    if not global_scaling:
        for icomp in range(ncomp):
            min_val = np.min([temporal_envelope_mean[0][0][icomp, :], temporal_envelope_mean[1][0][icomp, :]])
            max_val = np.max([temporal_envelope_mean[0][0][icomp, :], temporal_envelope_mean[1][0][icomp, :]])
            scale_fact = 1.0 / (max_val - min_val)

            for itemp in range(ntemp):
                temporal_envelope_mean[itemp][0][icomp, :] = np.clip(
                    scale_fact * temporal_envelope_mean[itemp][0][icomp, :]
                    - scale_fact * min_val - 0.5, -0.5, 0.5)

    # if global scaling should be used, scale all
    # data between -0.5 and 0.5
    else:
        # scale temporal envelope between -0.5 and 0.5
        min_val = np.min(temporal_envelope_mean)
        max_val = np.max(temporal_envelope_mean)
        scale_fact = 1.0 / (max_val - min_val)
        for itemp in range(ntemp):
            temporal_envelope_mean[itemp][0] = np.clip(scale_fact * temporal_envelope_mean[itemp][0]
                                                       - scale_fact * min_val - 0.5, -0.5, 0.5)


    return temporal_envelope_mean, temporal_envelope




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# helper function to handle spatial profiles for plotting
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _get_spatial_profiles(A_orig, keys, idx_text, vertno=[],
                          subject='fsaverage', subjects_dir=None,
                          labels=None, classification={}, percentile=97,
                          mni_coord=[], add_foci=False, fnout=None):

    """
    Helper function to get/generate the spatial
    profiles of the FourierICA components for
    plotting

        Parameters
        ----------
        A_orig: array
            2D-mixing-array (nvoxel, ncomp) estimated
            when applying FourierICA
        keys: list containing the group names
        idx_text: list containing the information in which
            brain hemisphere a component is mainly
            located (could be either 'both', 'left', 'right'
            or ' ' if no classification was performed before
            plotting)
        vertno: list
            list containing two arrays with the order
            of the vertices. If not given it will be
            generated in this routine
        subject: string
            string containing the subjects ID
        subjects_dir: string
            string containing the subjects directory
            path
        labels: list of strings
            names of the labels which should be plotted.
            Note, the prefix 'lh.' and the suffix '.label'
            are automatically added
        classification: dictionary
            classification object from the group_ica_object.
            It is a dictionary containing two sub-dictionaries
            'lh' and 'rh' (for left and right hemisphere). In
            both sub-dictionaries the information about the
            groups is stored, i.e. a group/region name + the
            information which components are stored in this
            group
        percentile: integer
            value between 0 and 100 used to set a lower
            limit for the shown intensity range of the
            spatial plots
        mni_coord: list of strings
            if given the MNI coordinates are plotted
            beneath the spatial profiles
        add_foci: bool
            if True and the MNI coordinates are given
            a foci is plotted at the position of the
            MNI coordinate
        fnout: string or None
            if labels and classification is given the
            output filename of the brain plot containing
            all labels. If 'None' the results are not stored


        Return
        ------
        temp_plot_dir: string
            directory where the spatial profiles are
            stored
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from matplotlib import gridspec as grd
    from matplotlib import pyplot as plt
    from mayavi import mlab
    from mne.source_estimate import _make_stc
    import numpy as np
    from os import environ, makedirs
    from os.path import exists, join
    import re
    from scipy import misc
    from surfer import set_log_level
    import types

    # set log level to 'WARNING'
    set_log_level('CRITICAL')

    import mayavi
    mayavi.mlab.options.offscreen = True


    # -------------------------------------------
    # create temporary directory to save plots
    # of spatial profiles
    # -------------------------------------------
    temp_plot_dir = join(subjects_dir, subject, 'temp_plots')
    if not exists(temp_plot_dir):
        makedirs(temp_plot_dir)


    # -------------------------------------------
    # generate spatial profiles
    # (using magnitude and phase)
    # -------------------------------------------
    if not subjects_dir:
        subjects_dir = environ.get('SUBJECTS_DIR')

    if isinstance(A_orig[0, 0], complex):
        A_orig_mag = np.abs(A_orig)
    else:
        A_orig_mag = A_orig

    nvoxel, ncomp = A_orig_mag.shape


    # -------------------------------------------
    # check if vertno is given, otherwise
    # generate it
    # -------------------------------------------
    if not np.any(vertno):
        vertno = [np.arange(nvoxel/2), np.arange(nvoxel/2)]


    # -------------------------------------------
    # check if labels should be plotted and if
    # classification was already performed
    # --> if yes define some colors for the
    #     labels
    # -------------------------------------------
    if labels and classification:
        colors = ['green', 'red', 'cyan', 'yellow', 'mediumblue',
                  'magenta', 'chartreuse', 'indigo', 'sandybrown',
                  'slateblue', 'purple', 'lightpink', 'springgreen',
                  'orange', 'sienna', 'cadetblue', 'crimson',
                  'maroon', 'powderblue', 'deepskyblue', 'olive']



    # -------------------------------------------
    # loop over all components to generate
    # spatial profiles
    # -------------------------------------------
    for icomp in range(ncomp):

        # -------------------------------------------
        # plot spatial profile
        # -------------------------------------------
        # generate stc-object from current component
        A_cur = A_orig_mag[:, icomp]
        src_loc = _make_stc(A_cur[:, np.newaxis], vertices=vertno, tmin=0, tstep=1,
                            subject=subject)

        # define current range (Xth percentile)
        fmin = np.percentile(A_cur, percentile)
        fmax = np.max(A_cur)
        fmid = 0.5 * (fmin + fmax)
        clim = {'kind': 'value',
                'lims': [fmin, fmid, fmax]}

        # plot spatial profiles
        brain = src_loc.plot(surface='inflated', hemi='split', subjects_dir=subjects_dir,
                             config_opts={'cortex': 'bone'}, views=['lateral', 'medial'],
                             time_label=' ', colorbar=False, clim=clim)

        # check if foci should be added to the plot
        if add_foci and np.any(mni_coord):
            for i_hemi in ['lh', 'rh']:
                mni_string = mni_coord[i_hemi][icomp]

                # if 'mni_string' is not empty (it might be empty if activity
                # can only be found in one hemisphere) plot a foci
                if mni_string != "":
                    mni_float = list(map(float, re.findall("[-+]?\d*\.\d+|\d+", mni_string)))
                    brain.add_foci(mni_float, coords_as_verts=False, hemi=i_hemi, color='chartreuse',
                                   scale_factor=1.5, map_surface='white')


        # -------------------------------------------
        # check if labels should be plotted
        # -------------------------------------------
        if labels and classification:

            # import module to read in labels
            from mne import read_label

            # get path to labels
            dir_labels = join(subjects_dir, subject, 'label')

            # identify in which group the IC is classified
            hemi = 'rh' if idx_text[icomp] == 'right' else 'lh'


            # read in the corresponding label
            for idx_key, key in enumerate(keys):
                if icomp in classification[hemi][key]:
                    label_name = ".%s.label" % key
                    color = colors[idx_key]
                    break

            # loop over both hemispheres to read the label in and plot it
            hemi = ['lh', 'rh'] if idx_text[icomp] == 'both ' else [hemi]
            for hemi_cur in hemi:
                label = read_label(join(dir_labels, hemi_cur + label_name), subject=subject)
                brain.add_label(label, borders=False, hemi=hemi_cur, color=color, alpha=0.1)
                brain.add_label(label, borders=True, hemi=hemi_cur, color=color)

        # save results
        fn_base = "IC%02d_spatial_profile.png" % (icomp+1)
        fnout_img = join(temp_plot_dir, fn_base)
        brain.save_image(fnout_img)

        # close mlab figure
        mlab.close(all=True)


    # -------------------------------------------
    # also generate one plot with all labels
    # -------------------------------------------
    if labels and classification:

        # set clim in a way that no activity can be seen
        # (Note: we only want to see the labels)
        clim = {'kind': 'value',
                'lims': [fmax, 1.5 * fmax, 2.0 * fmax]}

        # generate plot
        brain = src_loc.plot(surface='inflated', hemi='split', subjects_dir=subjects_dir,
                             config_opts={'cortex': 'bone'}, views=['lateral', 'medial'],
                             time_label=' ', colorbar=False, clim=clim, background='white')

        # loop over all labels
        for idx_key, key in enumerate(keys):
            label_name = ".%s.label" % key
            color = colors[idx_key]

            # loop over both hemispheres in order to plotting the labels
            for hemi in ['lh', 'rh']:
                label = read_label(join(dir_labels, hemi + label_name), subject=subject)
                brain.add_label(label, borders=False, hemi=hemi, color=color, alpha=0.6)

        # save results
        if fnout:
            fnout_img = '%s_labels.png' % fnout
            brain.save_image(fnout_img)

        # close mlab figure
        mlab.close(all=True)


        # -------------------------------------------
        # now adjust the label plot appropriate
        # -------------------------------------------
        # read spatial profile image
        spat_tmp = misc.imread(fnout_img)

        # rearrange image
        x_size, y_size, _ = spat_tmp.shape
        x_half, y_half = x_size / 2, y_size / 2
        x_frame, y_frame = int(0.11 * x_half), int(0.01 * y_half)
        spatial_profile = np.concatenate((spat_tmp[x_frame:(x_half - x_frame), y_frame:(y_half - y_frame), :],
                                          spat_tmp[(x_half + x_frame):-x_frame, y_frame:(y_half - y_frame), :],
                                          spat_tmp[(x_half + x_frame):-x_frame, (y_half + y_frame):-y_frame, :],
                                          spat_tmp[x_frame:(x_half - x_frame), (y_half + y_frame):-y_frame, :]),
                                         axis=1)

        # plot image
        plt.ioff()
        fig = plt.figure('Labels plots', figsize=(17, 3))
        gs = grd.GridSpec(1, 30, wspace=0.00001, hspace=0.00001,
                          left=0.0, right=1.0, bottom=0.0, top=1.0)

        # set plot position and plot image
        p1 = fig.add_subplot(gs[0, 0:26])
        p1.imshow(spatial_profile)
        adjust_spines(p1, [])

        # add label names
        keys_fac = 0.8/len(keys)
        keys_split = 0
        p_text = fig.add_subplot(gs[0, 26:30])
        keys_sort_idx = np.argsort(keys)
        for idx_key in range(len(keys)):
            key = keys[keys_sort_idx[idx_key]]
            # check if string should be split
            if len(key) > 21 and ' ' in key:
                p_text.text(0.0, 0.9-keys_fac*(idx_key+keys_split), key.split()[0]+'-',
                            fontsize=13, color=colors[keys_sort_idx[idx_key]])
                keys_split += 1
                p_text.text(0.0, 0.9-keys_fac*(idx_key+keys_split), key.split()[1],
                            fontsize=13, color=colors[keys_sort_idx[idx_key]])
            else:
                p_text.text(0.0, 0.9-keys_fac*(idx_key+keys_split), key, fontsize=13,
                            color=colors[keys_sort_idx[idx_key]])
            adjust_spines(p_text, [])

            plt.savefig(fnout_img, dpi=300)

        # close plot and set plotting back to screen
        plt.close('FourierICA plots')
        plt.ion()

    mayavi.mlab.options.offscreen = False
    return temp_plot_dir




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# helper function to get spectral profiles for plotting
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _get_spectral_profile(temporal_envelope, tpre,
                          sfreq, flow, fhigh,
                          bar_plot=False,
                          use_multitaper=False):

    """
    Helper function to get the spectral-profile of the
    temporal-envelopes of the FourierICA components
    for plotting

        Parameters
        ----------
        temporal_envelope: list of arrays containing
            the temporal envelopes.
        tpre: float
            Lower border (in seconds) of the time-window
            used for generating/showing the epochs. If
            'None' the value stored in 'fourier_ica_obj'
            is used
        sfreq: float
            Sampling frequency of the data
        flow: float or integer
            Lower frequency range for time frequency analysis
        fhigh: float or integer
            Upper frequency range for time frequency analysis
        bar_plot: boolean
            if set the number of time points for time-frequency
            estimation is reduced in order to save memory and
            computing-time
        use_multitaper: boolean
            If set 'multitaper' is usewd for time frequency
            analysis, otherwise 'stockwell'

        Return
        ------
        average_power_all: list containing the averaged
            frequency power of all components
        freqs: array containing the frequencies used to
            calculate the frequency power
        vmin: lower frequency range for plotting
        vmax: upper frequency range for plotting
    """

    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from mne.baseline import rescale
    from mne.time_frequency._stockwell import _induced_power_stockwell
    import numpy as np


    # ------------------------------------------
    # define some parameter
    # ------------------------------------------
    ntemp = len(temporal_envelope)
    ncomp = temporal_envelope[0][0].shape[1]
    win_ntsl = temporal_envelope[0][0].shape[-1]
    average_power_all = np.empty((ntemp, 0)).tolist()
    vmin = np.zeros(ncomp)
    vmax = np.zeros(ncomp)

    # define some time parameter
    times = np.arange(win_ntsl) / sfreq + tpre
    idx_start = np.argmin(np.abs(times - tpre))
    idx_end = np.argmin(np.abs(times - (tpre + win_ntsl/sfreq)))

    if bar_plot:
        decim = 10
    else:
        decim = 1

    # ------------------------------------------
    # loop over all time courses, i.e.
    # conditions, and all components
    # ------------------------------------------
    for itemp in range(ntemp):
        for icomp in range(ncomp):

            # extract some information from the temporal_envelope
            nepochs = temporal_envelope[itemp][0].shape[0]

            # ------------------------------------------
            # perform time frequency analysis
            # ------------------------------------------
            # prepare data for frequency analysis
            data_stockwell = temporal_envelope[itemp][0][:, icomp, idx_start:idx_end].\
                reshape((nepochs, 1, idx_end-idx_start))
            data_stockwell = data_stockwell.transpose([1, 0, 2])

            # mirror data to reduce transient frequencies
            data_stockwell = np.concatenate((data_stockwell[:, :, 50:0:-1],
                                             data_stockwell, data_stockwell[:, :, -1:-51:-1]), axis=-1)

            n_fft = data_stockwell.shape[-1]


            # check if 'multitaper' or 'stockwell' should be
            # used for time-frequency analysis
            if use_multitaper:
                from mne.time_frequency.tfr import _compute_tfr

                n_cycle = 3.0
                if (10.0 * n_cycle*sfreq)/(2.0 * np.pi * flow) > n_fft:
                    flow *=  ((10.0 * n_cycle*sfreq)/(2.0 * np.pi * flow))/n_fft
                    flow = np.ceil(flow)

                freqs = np.arange(flow, fhigh)
                power_data = _compute_tfr(data_stockwell, freqs, sfreq=sfreq, use_fft=True,
                                          n_cycles=n_cycle, zero_mean=True, decim=decim,
                                          output='power', method='multitaper',
                                          time_bandwidth=10)
            else:
                power_data, _, freqs = _induced_power_stockwell(data_stockwell, sfreq=sfreq, fmin=flow,
                                                                fmax=fhigh, width=0.6, decim=1, n_fft=n_fft,
                                                                return_itc=False, n_jobs=4)

            # perform baseline correction (and remove mirrored parts from data)
            power_data = rescale(power_data[:, :, int(50/decim):-int(50/decim)],
                                 times[idx_start:idx_end][0:-1:decim], (None, 0), 'mean')
            average_power = np.mean(power_data, axis=0)


            # ------------------------------------------
            # store all frequency data in one list
            # ------------------------------------------
            average_power_all[itemp].append(average_power)


            # ------------------------------------------
            # estimate frequency thresholds for plotting
            # ------------------------------------------
            vmax[icomp] = np.max((np.percentile(average_power, 98), vmax[icomp]))
            vmin[icomp] = np.min((np.percentile(average_power, 2), vmin[icomp]))

            if np.abs(vmax[icomp]) > np.abs(vmin[icomp]):
                vmin[icomp] = - np.abs(vmax[icomp])
            else:
                vmax[icomp] = np.abs(vmin[icomp])

    return average_power_all, freqs, vmin, vmax




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot results when Fourier ICA was applied in the
# source space
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
def plot_results_src_space(fourier_ica_obj, W_orig, A_orig,
                           src_loc_data=[], temporal_envelope=[],                   # parameter for temporal profiles
                           tpre=None, win_length_sec=None, tICA=False,
                           vertno=[], subject='fsaverage', subjects_dir=None,       # parameter for spatial profiles
                           percentile=97, add_foci=True, classification={},
                           mni_coords=[], labels=None,
                           flow=None, fhigh=None, bar_plot=False,                   # parameter for spectral profiles
                           global_scaling=True, ncomp_per_plot=13, fnout=None,      # general plotting parameter
                           temp_profile_names=[]):


    """
    Generate plot containing all results achieved by
    applying FourierICA in source space, i.e., plot
    spatial and spectral profiles.

        Parameters
        ----------
        fourier_ica_obj:  FourierICA object generated
            when applying jumeg.decompose.fourier_ica
        W_orig: array
            2D-demixing-array (ncomp x nvoxel) estimated
            when applying FourierICA
        A_orig:  array
            2D-mixing-array (nvoxel, ncomp) estimated
            when applying FourierICA


        **** parameter for temporal profiles ****

        src_loc_data: array
            3D array containing the source localization
            data used for FourierICA estimation
            (nfreq x nepochs x nvoxel). Only necessary
            if temporal_envelope is not given.
            default: src_loc_data=[]
        temporal_envelope: list of arrays containing
            the temporal envelopes. If not given the
            temporal envelopes are estimated here based
            on the 'src_loc_data'
            default: temporal_envelope=[]
        tpre: float
            Lower border (in seconds) of the time-window
            used for generating/showing the epochs. If
            'None' the value stored in 'fourier_ica_obj'
            is used
        win_length_sec: float or None
            Length of the epoch window in seconds.  If
            'None' the value stored in 'fourier_ica_obj'
            is used
        tICA: boolean
            should be True if temporal ICA was applied
            default: tICA=False


        **** parameter for spatial profiles ****

        vertno: list
            list containing two arrays with the order
            of the vertices. If list is empty it will be
            automatically generated
            default: vertno=[]
        subject: string
            subjects ID
            default: subject='fsaverage'
        subjects_dir: string or None
            string containing the subjects directory
            path
            default: subjects_dir=None --> system variable
            SUBJETCS_DIR is used
        percentile: integer
            value between 0 and 100 used to set a lower
            limit for the shown intensity range of the
            spatial plots
            default: percentile=97
        add_foci: bool
            if True and the MNI coordinates are given
            a foci is plotted at the position of the
            MNI coordinate
            default: add_foci=True
        classification: dictionary
            classification object from the group_ica_object.
            It is a dictionary containing two sub-dictionaries
            'lh' and 'rh' (for left and right hemisphere). In
            both sub-dictionaries the information about the
            groups is stored, i.e. a group/region name + the
            information which components are stored in this
            group
            default: classification={}
        mni_coords: list of strings
            if given the MNI coordinates are plotted
            beneath the spatial profiles
            default: mni_coords=[]
        labels: list of strings
            names of the labels which should be plotted.
            Note, the prefix 'lh.' and the suffix '.label'
            are automatically added
            default: labels=None


        **** parameter for spectral profiles ****

        flow: float or integer
            Lower frequency range for time frequency analysis
        fhigh: float or integer
            Upper frequency range for time frequency analysis
        bar_plot: boolean
            If set the results of the time-frequency analysis
            are shown as bar plot. This option is recommended
            when FourierICA was applied to resting-state data
            default: bar_plot=False


        **** general plotting parameter ****

        global_scaling: bool
            If set spatial, spectral and temporal profiles
            are globally scaled. Otherwise each component
            is scaled individually
            default: global_scaling=True
        ncomp_per_plot: integer
            number of components per plot
        fnout: string
            default: fnout=None
        temp_profile_names: list of string
            The list should have the same number of elements
            as conditions were used to generate the temporal
            envelopes. The names given here are used as headline
            for the temporal profiles in the plot
            default: temp_profile_name=[]
    """


    # ------------------------------------------
    # import necessary modules
    # ------------------------------------------
    from matplotlib import pyplot as plt
    from matplotlib import gridspec as grd
    from matplotlib.colors import Normalize
    import numpy as np
    from os import remove, rmdir
    from os.path import exists, join
    from scipy import misc


    # -------------------------------------------
    # check input parameter
    # -------------------------------------------
    if tpre == None:
        tpre = fourier_ica_obj.tpre
    if flow == None:
        flow = fourier_ica_obj.flow
    if not fhigh:
        fhigh = fourier_ica_obj.fhigh
    if not win_length_sec:
        win_length_sec = fourier_ica_obj.win_length_sec

    # check if either 'src_loc_data' or
    # 'temporal_envelope' is given, otherwise stop
    if src_loc_data == [] and temporal_envelope == []:
        print(">>> ERROR: you have either to provide the variable")
        print(">>>        'src_loc_data' or 'temporal_envelope'.")
        import pdb
        pdb.set_trace()


    # estimate/set some simple parameter
    sfreq = fourier_ica_obj.sfreq
    win_ntsl = int(np.floor(sfreq * win_length_sec))
    ncomp, nvoxel = W_orig.shape
    ylim_temp = [-0.55, 0.55]
    time_range = [tpre, tpre + win_length_sec]


    # -------------------------------------------
    # get temporal envelopes, or rather check if
    # temporal envelopes already exist or must
    # be calculated
    # -------------------------------------------
    temporal_envelope_mean, temporal_envelope = \
        _get_temporal_envelopes(fourier_ica_obj, W_orig, temporal_envelope=temporal_envelope,
                                src_loc_data=src_loc_data, tICA=tICA, global_scaling=global_scaling,
                                win_length_sec=win_length_sec, tpre=tpre, flow=flow)
    ntemp = len(temporal_envelope)


    # -------------------------------------------
    # get MNI-coordinates of the FourierICA
    # components
    # -------------------------------------------
    if not classification and not mni_coords and not labels:
        mni_coords, hemi_loc_txt, classification, labels = \
            get_mni_coordinates(A_orig, subject=subject, subjects_dir=subjects_dir,
                                percentile=percentile)

    # otherwise we only have to get the 'hemi_loc_txt' variable
    else:
        hemi_loc = np.array([int(i != '') for i in mni_coords['lh']])
        hemi_loc += np.array([2*int(i != '') for i in mni_coords['rh']])
        hemi_loc_txt = np.array(['     '] * len(hemi_loc))
        for idx, hemi_name in enumerate(['left ', 'right', 'both ']):
            idx_change = np.where(hemi_loc == (idx + 1.0))[0]
            hemi_loc_txt[idx_change] = hemi_name


    # check if classification was performed prior to plotting
    keys, key_borders, idx_sort = _check_classification(classification, ncomp)


    # -------------------------------------------
    # get spatial profiles of all components
    # Note: This will take a while
    # -------------------------------------------
    temp_plot_dir = _get_spatial_profiles(A_orig, keys, hemi_loc_txt, vertno=vertno,
                                          subject=subject, subjects_dir=subjects_dir,
                                          labels=labels, classification=classification,
                                          percentile=percentile, mni_coord=mni_coords,
                                          add_foci=add_foci, fnout=fnout)


    # -------------------------------------------
    # get spectral profiles of all components
    # Note: This will take a while
    # -------------------------------------------
    average_power_all, freqs, vmin, vmax = \
        _get_spectral_profile(temporal_envelope, tpre, sfreq, flow, fhigh, bar_plot=bar_plot)

    # check if bar plot should be used
    # --> if yes estimate histogram data and normalize results
    if bar_plot:
        # generate an array to store the results
        freq_heights = np.zeros((ntemp, ncomp, len(freqs)))

        # loop over all conditions
        for i_power, average_power in enumerate(average_power_all):
            freq_heights[i_power, :, :] = np.sum(np.abs(average_power), axis=2)

        # normalize to a range between 0 and 1
        freq_heights /= np.max(freq_heights)



    # ------------------------------------------
    # now generate plot containing spatial,
    # spectral and temporal profiles
    # ------------------------------------------
    # set some general parameter
    plt.ioff()
    nimg = int(np.ceil(ncomp/(1.0*ncomp_per_plot)))
    idx_key = 0
    nplot = list(range(ncomp_per_plot, nimg*ncomp_per_plot, ncomp_per_plot))
    nplot.append(ncomp)


    # generate image and its layout for plotting
    fig = plt.figure('FourierICA plots', figsize=(14 + ntemp * 8, 34))
    n_keys = len(key_borders) if len(key_borders) > 0 else 1
    gs = grd.GridSpec(ncomp_per_plot * 20 + n_keys * 10, 10 + ntemp * 8, wspace=0.1, hspace=0.05,
                      left=0.04, right=0.96, bottom=0.04, top=0.96)


    # ------------------------------------------
    # loop over the estimated number of images
    # ------------------------------------------
    for iimg in range(nimg):

        # clear figure (to start with a white image in each loop)
        plt.clf()

        # estimate how many plots on current image
        istart_plot = int(ncomp_per_plot * iimg)

        # set idx_class parameter
        idx_class = 1 if key_borders == [] else 0


        # ------------------------------------------
        # loop over all components which should be
        # plotted on the current image
        # ------------------------------------------
        for icomp in range(istart_plot, nplot[iimg]):

            # ----------------------------------------------
            # check if key_boarders is set and should be
            # written on the image
            # ----------------------------------------------
            if (icomp == istart_plot and key_borders != []) or \
                    ((icomp + 1) in key_borders):

                # adjust key-index
                if (icomp + 1) in key_borders:
                    idx_key += 1

                # add sub-plot with 'key_text'
                p_text = fig.add_subplot(gs[20 * (icomp - istart_plot) + idx_class * 10: \
                    20 * (icomp - istart_plot) + 8 + idx_class * 10, 0:10])
                p_text.text(0, 0, keys[idx_key-1], fontsize=25)
                adjust_spines(p_text, [])

                # adjust idx_class parameter
                idx_class += 1


            # ----------------------------------------------
            # plot spatial profiles
            # ----------------------------------------------
            # read spatial profile image
            fn_base = "IC%02d_spatial_profile.png" % (idx_sort[icomp] + 1)
            fnin_img = join(temp_plot_dir, fn_base)
            spat_tmp = misc.imread(fnin_img)
            remove(fnin_img)

            # rearrange image
            x_size, y_size, _ = spat_tmp.shape
            x_half, y_half = x_size / 2, y_size / 2
            x_frame, y_frame = int(0.11 * x_half), int(0.01 * y_half)
            spatial_profile = np.concatenate((spat_tmp[x_frame:(x_half - x_frame), y_frame:(y_half - y_frame), :],
                                              spat_tmp[(x_half + x_frame):-x_frame, y_frame:(y_half - y_frame), :],
                                              spat_tmp[(x_half + x_frame):-x_frame, (y_half + y_frame):-y_frame, :],
                                              spat_tmp[x_frame:(x_half - x_frame), (y_half + y_frame):-y_frame, :]),
                                             axis=1)

            # set plot position and plot image
            p1 = fig.add_subplot(
                gs[20 * (icomp - istart_plot) + idx_class * 10:20 * (icomp - istart_plot) + 15 + idx_class * 10, 0:10])
            p1.imshow(spatial_profile)

            # set some plotting options
            p1.yaxis.set_ticks([])
            p1.xaxis.set_ticks([])
            y_name = "IC#%02d" % (idx_sort[icomp] + 1)
            p1.set_ylabel(y_name, fontsize=18)


            # ----------------------------------------------
            # if given write MNI coordinates under the image
            # ----------------------------------------------
            if np.any(mni_coords):
                # left hemisphere
                plt.text(120, 360, mni_coords['lh'][int(idx_sort[int(icomp)])], color="black",
                         fontsize=18)
                # right hemisphere
                plt.text(850, 360, mni_coords['rh'][int(idx_sort[int(icomp)])], color="black",
                         fontsize=18)

            # add location information of the component
            # --> if located in 'both', 'left' or 'right' hemisphere
            plt.text(-220, 100, hemi_loc_txt[int(idx_sort[int(icomp)])], color="red",
                     fontsize=25, rotation=90)


            # ----------------------------------------------
            # temporal/spectral profiles
            # ----------------------------------------------
            # loop over all time courses
            for itemp in range(ntemp):

                # ----------------------------------------------
                # if given plot a headline above the time
                # courses of each condition
                # ----------------------------------------------
                if icomp == istart_plot and len(temp_profile_names):
                    # add a sub-plot for the text
                    p_text = fig.add_subplot(gs[(idx_class - 1) * 10: 6 + (idx_class - 1) * 12,
                                             (itemp) * 8 + 11:(itemp + 1) * 8 + 9])
                    # plot the text and adjust spines
                    p_text.text(0, 0, "               " + temp_profile_names[itemp], fontsize=30)
                    adjust_spines(p_text, [])


                # set plot position
                if bar_plot:
                    p2 = plt.subplot(
                        gs[20 * (icomp - istart_plot) + idx_class * 11:20 * (icomp - istart_plot) + 13 + idx_class * 10,
                        itemp * 8 + 11:(itemp + 1) * 8 + 9])
                else:
                    p2 = plt.subplot(
                        gs[20 * (icomp - istart_plot) + idx_class * 10:20 * (icomp - istart_plot) + 15 + idx_class * 10,
                        itemp * 8 + 11:(itemp + 1) * 8 + 9])


                # extract temporal plotting information
                times = (np.arange(win_ntsl) / sfreq + tpre)[5:-5]
                idx_start = np.argmin(np.abs(times - time_range[0]))
                idx_end = np.argmin(np.abs(times - time_range[1]))


                # ----------------------------------------------
                # plot spectral profile
                # ----------------------------------------------
                # check if global scaling should be used
                if global_scaling:
                    vmin_cur, vmax_cur = np.min(vmin), np.max(vmax)
                else:
                    vmin_cur, vmax_cur = vmin[icomp], vmax[icomp]

                # show spectral profile
                if bar_plot:
                    plt.bar(freqs, freq_heights[itemp, int(idx_sort[icomp]), :], width=1.0, color='cornflowerblue')
                    plt.xlim(flow, fhigh)
                    plt.ylim(0.0, 1.0)

                    # set some parameter
                    p2.set_xlabel("freq. [Hz]")
                    p2.set_ylabel("ampl. [a.u.]")

                    # ----------------------------------------------
                    # plot temporal profile on the some spot
                    # ----------------------------------------------
                    ax = plt.twiny()
                    ax.set_xlabel("time [s]")
                    ax.plot(times[idx_start:idx_end], 0.5+temporal_envelope_mean[itemp][0][int(idx_sort[icomp]), idx_start:idx_end],
                            color='red', linewidth=3.0)
                    ax.set_xlim(times[idx_start], times[idx_end])
                    ax.set_ylim(0.0, 1.0)

                else:
                    average_power = average_power_all[itemp][int(idx_sort[icomp])]
                    extent = (times[idx_start], times[idx_end], freqs[0], freqs[-1])
                    p2.imshow(average_power, extent=extent, aspect="auto", origin="lower",
                              picker=False, cmap='RdBu_r', vmin=vmin_cur,
                              vmax=vmax_cur)

                    # set some parameter
                    p2.set_xlabel("time [s]")
                    p2.set_ylabel("freq. [Hz]")


                    # ----------------------------------------------
                    # plot temporal profile on the some spot
                    # ----------------------------------------------
                    ax = plt.twinx()
                    ax.set_xlim(times[idx_start], times[idx_end])
                    ax.set_ylim(ylim_temp)
                    ax.set_ylabel("ampl. [a.u.]")
                    ax.plot(times[idx_start:idx_end], temporal_envelope_mean[itemp][0][int(idx_sort[icomp]), idx_start:idx_end],
                            color='black', linewidth=3.0)


        # ----------------------------------------------
        # finally plot a color bar
        # ----------------------------------------------
        if not bar_plot:
            # first normalize the color table
            norm = Normalize(vmin=np.round(vmin_cur, 2), vmax=np.round(vmax_cur, 2))
            sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
            sm.set_array(np.linspace(vmin_cur, 1.0))

            # estimate position of the color bar
            xpos = 0.405 + 0.5/(ntemp + 1.0)
            if n_keys > 1:
                cbaxes = fig.add_axes([xpos, 0.135, 0.2, 0.006])
            else:
                cbaxes = fig.add_axes([xpos, 0.03, 0.2, 0.006])

            ticks_fac = (vmax_cur - vmin_cur) * 0.3333
            ticks = np.round([vmin_cur, vmin_cur + ticks_fac, vmax_cur - ticks_fac, vmax_cur], 2)
            # ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]

            # now plot color bar
            cb = plt.colorbar(sm, ax=p2, cax=cbaxes, use_gridspec=False,
                              orientation='horizontal', ticks=ticks,
                              format='%1.2g')
            cb.ax.tick_params(labelsize=18)


        # ----------------------------------------------
        # save image
        # ----------------------------------------------
        if fnout:
            fnout_complete = '%s_%02d.png' % (fnout, iimg + 1)
            plt.savefig(fnout_complete, format='png', dpi=300)




    # close plot and set plotting back to screen
    plt.close('FourierICA plots')
    plt.ion()

    # remove temporary directory for
    # spatial profile plots
    if exists(temp_plot_dir):
        rmdir(temp_plot_dir)


    return mni_coords, classification, labels



