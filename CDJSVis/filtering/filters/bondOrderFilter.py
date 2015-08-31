
"""
Bond order filter

"""
from . import base


################################################################################

class BondOrderFilterSettings(base.BaseSettings):
    """
    Settings for the bond order filter:
    
        Maximum bond distance
            Used for spatially decomposing the system to speed up the algorithm.
            Should be set large enough that the required neighbours are included
            for the given system.
    
        Filter Q4
            Enable filtering by the Q4 value.
    
        Minimum Q4
            The minimum visible Q4 value.
    
        Maximum Q4
            The maximum visible Q4 value.
        
        Filter Q6
            Enable filtering by the Q6 value.
    
        Minimum Q6
            The minimum visible Q6 value.
    
        Maximum Q6
            The maximum visible Q6 value.
    
    """
    def __init__(self):
        super(BondOrderFilterSettings, self).__init__()
        
        self.registerSetting("filterQ4Enabled", default=False)
        self.registerSetting("filterQ6Enabled", default=False)
        self.registerSetting("maxBondDistance", default=4.0)
        self.registerSetting("minQ4", default=0.0)
        self.registerSetting("maxQ4", default=99.0)
        self.registerSetting("minQ6", default=0.0)
        self.registerSetting("maxQ6", default=99.0)

################################################################################

class BondOrderFilter(base.BaseFilter):
    """
    The bond order calculator/filter calculates the *Steinhardt order parameters* as described in [1]_.  
    Currently the Q\ :sub:`4` and Q\ :sub:`6` parameters are calculated (Equation 3 in the referenced paper) and made available as scalars 
    for colouring/plotting and will be displayed when clicking on an atom.  Filtering is not currently implemented for this
    property type.

    On the settings form you must set the parameter *Max bond distance* to be something sensible for your system.  
    For example, for FCC somewhere between 1NN and 2NN; for BCC somewhere between 2NN and 3NN.

    For a perfect lattice you would obtain the following values [2]_:

        * Q\ :sub:`4`\ :sup:`fcc`\ = 0.191; Q\ :sub:`6`\ :sup:`fcc`\ = 0.575
        * Q\ :sub:`4`\ :sup:`bcc`\ = 0.036; Q\ :sub:`6`\ :sup:`bcc`\ = 0.511
        * Q\ :sub:`4`\ :sup:`hcp`\ = 0.097; Q\ :sub:`6`\ :sup:`hcp`\ = 0.485

    .. [1] W. Lechner and C. Dellago. *J. Chem. Phys.* **129** (2008) 114707; `doi: 10.1063/1.2977970 <http://dx.doi.org/10.1063/1.2977970>`_.
    .. [2] A. Stukowski. *Modelling Simul. Mater. Sci. Eng.* **20** (2012) 045021; `doi: 10.1088/0965-0393/20/4/045021 <http://dx.doi.org/10.1088/0965-0393/20/4/045021>`_.
    
    """
