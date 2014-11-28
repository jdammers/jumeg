# Authors: Lukas Breuer <l.breuer@fz-juelich.de>
#
# License: Simplified BSD

"""
----------------------------------------------------------------------
--- jumeg.decompose --------------------------------------------------
----------------------------------------------------------------------
 autor      : Lukas Breuer
 email      : l.breuer@fz-juelich.de
 last update: 27.11.2014
 version    : 1.0 (NOTE: Current version is only able to handle data
                         recorded with the magnesWH3600 system)

----------------------------------------------------------------------
 Based on following publications:
----------------------------------------------------------------------

L. Breuer, J. Dammers, T.P.L. Roberts, and N.J. Shah, 'Ocular and
Cardiac Artifact Rejection for Real-Time Analysis in MEG',
Journal of Neuroscience Methods, Jun. 2014
(doi:10.1016/j.jneumeth.2014.06.016)

L. Breuer, J. Dammers, T.P.L. Roberts, and N.J. Shah, 'A Constrained
ICA Approach for Real-Time Cardiac Artifact Rejection in
Magnetoencephalography', IEEE Transactions on Biomedical Engineering,
Feb. 2014 (doi:10.1109/TBME.2013.2280143).

----------------------------------------------------------------------
"""

from . import ica
from . import ocarta


