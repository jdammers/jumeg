# Authors: Lukas Breuer <l.breuer@fz-juelich.de>
#
# License: Simplified BSD

"""
----------------------------------------------------------------------
--- jumeg.decompose --------------------------------------------------
----------------------------------------------------------------------
 autor      : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 12.11.2015
 version    : 1.1 (NOTE: Current version is only able to handle data
                         recorded with the magnesWH3600 system)

----------------------------------------------------------------------
 Based on following publications:
----------------------------------------------------------------------

COMPLEX_ICA:

A. Hyvaerinen, P. Ramkumar, L. Parkkonen and R. Hari, 'Independent
component analysis of short-time Fourier transforms for spontaneous
EEG/MEG analysis', NeuroImage 49(1):257-271, 2010.

**********************************************************************

FOURIER ICA:
A. Hyvaerinen, P. Ramkumar, L. Pakkonen, and R. Hari, 'Independent
component analysis of short-time Fourier transforms for spontaneous
EEG/MEG analysis', NeuroImage, 49(1):257-271, 2010.

P. Ramkumar, L. Pakkonen, and A. Hyvaerinen, 'Group-level
independent component analysis of Fourier envelopes of resting-state
MEG data', NeuroImage, 86(1): 480-491, 2014.

**********************************************************************

ICA:
(Infomax)
A.J. Bell and T.J. Sejnowski, 'An information_maximization
approach to blind separation and blind deconvolution',
Neural Comput, 7(6): 1129-1159, Nov. 1995.

(Extended Infomax)
T.-W.W. Lee, M. Girolami, and T.J. Sejnowski, 'Independent
Component Analysis Using an Extended Infomax Algorithm for
Mixed Subgaussian and Supergaussian Sources',
Neural Comput., 11(2): 417-441, Feb. 1999.

(FastICA)
A. Hyvaerinen, 'Survey on independent component analysis',
Neural Comput. Surv., 10(3): 626-634, Jan. 1999.

**********************************************************************

ICASSO:
J. Himberg, A. Hyvaerinen, and F. Esposito. 'Validating the
independent components of neuroimaging time-series via
clustering and visualization', Neuroimage, 22:3(1214-1222), 2004.

**********************************************************************

OCARTA:

L. Breuer, J. Dammers, T.P.L. Roberts, and N.J. Shah, 'Ocular and
Cardiac Artifact Rejection for Real-Time Analysis in MEG',
Journal of Neuroscience Methods, Jun. 2014
(doi:10.1016/j.jneumeth.2014.06.016).

L. Breuer, J. Dammers, T.P.L. Roberts, and N.J. Shah, 'A Constrained
ICA Approach for Real-Time Cardiac Artifact Rejection in
Magnetoencephalography', IEEE Transactions on Biomedical Engineering,
Feb. 2014 (doi:10.1109/TBME.2013.2280143).

----------------------------------------------------------------------
"""

from .complex_ica import complex_ica
# from . import dimension_selection
from .fourier_ica import (apply_ICASSO_fourierICA, apply_stft,
                          JuMEG_fourier_ica, stft_source_localization, fourier_ica)
# from . import fourier_ica_plot
from .group_ica import (group_fourierICA_src_space,
                        group_fourierICA_src_space_resting_state, plot_group_fourierICA)
# from . import ica
from .icasso import JuMEG_icasso
from .ocarta import JuMEG_ocarta


