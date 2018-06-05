import numpy as np
from mne.io.pick import pick_types, pick_channels, pick_info
from mne.transforms import apply_trans
from mne.forward import _map_meg_channels
from mne.channels.interpolation import _do_interp_dots


def interpolate_bads(inst, reset_bads=True, mode='accurate', origin=None, verbose=None):
    """Interpolate bad MEG and EEG channels.

    Operates in place.

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    reset_bads : bool
        If True, remove the bads from info.
    mode : str
        Either ``'accurate'`` or ``'fast'``, determines the quality of the
        Legendre polynomial expansion used for interpolation of MEG
        channels.
    origin : None | list
        If None, origin is set to sensor center of mass, otherwise use the
        coordinates provided as origin.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see
        :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
        for more).

    Returns
    -------
    inst : instance of Raw, Epochs, or Evoked
        The modified instance.

    """

    from mne.channels.interpolation import _interpolate_bads_eeg

    if getattr(inst, 'preload', None) is False:
        raise ValueError('Data must be preloaded.')

    _interpolate_bads_eeg(inst)
    _interpolate_bads_meg(inst, origin=origin, mode=mode)

    if reset_bads is True:
        inst.info['bads'] = []

    return inst


def _interpolate_bads_meg(inst, mode='accurate', origin=None, verbose=None):
    """Interpolate bad channels from data in good channels.

    Parameters
    ----------
    inst : mne.io.Raw, mne.Epochs or mne.Evoked
        The data to interpolate. Must be preloaded.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used for interpolation. `'fast'` should
        be sufficient for most applications.
    origin : None | list
        If None, origin is set to sensor center of mass, otherwise use the
        coordinates provided as origin.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).
    """
    picks_meg = pick_types(inst.info, meg=True, eeg=False, exclude=[])
    picks_good = pick_types(inst.info, meg=True, eeg=False, exclude='bads')
    meg_ch_names = [inst.info['ch_names'][p] for p in picks_meg]
    bads_meg = [ch for ch in inst.info['bads'] if ch in meg_ch_names]

    # select the bad meg channel to be interpolated
    if len(bads_meg) == 0:
        picks_bad = []
    else:
        picks_bad = pick_channels(inst.info['ch_names'], bads_meg,
                                  exclude=[])

    # return without doing anything if there are no meg channels
    if len(picks_meg) == 0 or len(picks_bad) == 0:
        return
    info_from = pick_info(inst.info, picks_good)
    info_to = pick_info(inst.info, picks_bad)

    if origin is None:

        posvec = np.array([inst.info['chs'][p]['loc'][0:3] for p in picks_meg])
        cogpos = np.mean(posvec, axis=0)
        print ">_interpolate_bads_meg\\DBG> cog(sens) = [%8.5f  %8.5f  %8.5f]" % \
              (cogpos[0], cogpos[1], cogpos[2])
        cogposhd = apply_trans(inst.info['dev_head_t']['trans'], cogpos, move=True)
        print ">_interpolate_bads_meg\\DBG> cog(hdcs) = [%8.5f  %8.5f  %8.5f]" % \
              (cogposhd[0], cogposhd[1], cogposhd[2])
        print ">_interpolate_bads_meg\\DBG> calling _map_meg_channels(..., origin=(%8.5f  %8.5f  %8.5f))" % \
              (cogposhd[0], cogposhd[1], cogposhd[2])

        origin = (cogposhd[0], cogposhd[1], cogposhd[2])

    else:
        origin = origin

    mapping = _map_meg_channels(info_from, info_to, mode=mode, origin=origin)
    _do_interp_dots(inst, mapping, picks_good, picks_bad)
