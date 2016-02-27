
"""
Bond order
==========

The bond order calculator/filter calculates the *Steinhardt order parameters* as described in [1]_.  
Currently the Q\ :sub:`4` and Q\ :sub:`6` parameters are calculated (Equation 3 in the referenced paper) and made available as scalars 
for colouring/plotting and will be displayed when clicking on an atom. You can also filter by these values.

For a perfect lattice you would obtain the following values [2]_:

    * Q\ :sub:`4`\ :sup:`fcc`\ = 0.191; Q\ :sub:`6`\ :sup:`fcc`\ = 0.575
    * Q\ :sub:`4`\ :sup:`bcc`\ = 0.036; Q\ :sub:`6`\ :sup:`bcc`\ = 0.511
    * Q\ :sub:`4`\ :sup:`hcp`\ = 0.097; Q\ :sub:`6`\ :sup:`hcp`\ = 0.485

The bond order calculator has the following parameters:

.. glossary::

    Maximum bond distance
        Used for spatially decomposing the system to speed up the algorithm.
        Should be set large enough that the required neighbours are included
        for the given system. For example for an FCC lattice set this to be
        somewhere between 1NN and 2NN, or for BCC somewhere between 2NN and
        3NN.
    
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

.. [1] W. Lechner and C. Dellago. *J. Chem. Phys.* **129** (2008) 114707; `doi: 10.1063/1.2977970 <http://dx.doi.org/10.1063/1.2977970>`_.
.. [2] A. Stukowski. *Modelling Simul. Mater. Sci. Eng.* **20** (2012) 045021; `doi: 10.1088/0965-0393/20/4/045021 <http://dx.doi.org/10.1088/0965-0393/20/4/045021>`_.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from . import base
from . import _bond_order

import numpy as np


################################################################################

class BondOrderFilterSettings(base.BaseSettings):
    """
    Settings for the bond order filter.

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
    The Bond Order filter.
    
    """
    def apply(self, filterInput, settings):
        """Run the bond order filter."""
        # check the inputs are correct
        if not isinstance(filterInput, base.FilterInput):
            raise TypeError("First argument of BondOrderFilter.apply must be of type FilterInput")
        if not isinstance(settings, BondOrderFilterSettings):
            raise TypeError("Second argument of BondOrderFilter.apply must be of type BondOrderFilterSettings")
        
        # unpack inputs
        inputState = filterInput.inputState
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        
        # settings
        maxBondDistance = settings.getSetting("maxBondDistance")
        filterQ4Enabled = int(settings.getSetting("filterQ4Enabled"))
        filterQ6Enabled = int(settings.getSetting("filterQ6Enabled"))
        minQ4 = settings.getSetting("minQ4")
        maxQ4 = settings.getSetting("maxQ4")
        minQ6 = settings.getSetting("minQ6")
        maxQ6 = settings.getSetting("maxQ6")
        
        # new scalars array
        scalarsQ4 = np.zeros(len(visibleAtoms), dtype=np.float64)
        scalarsQ6 = np.zeros(len(visibleAtoms), dtype=np.float64)
        
        # call C lib
        NVisible = _bond_order.bondOrderFilter(visibleAtoms, inputState.pos, maxBondDistance, scalarsQ4, scalarsQ6,
                                               inputState.cellDims, inputState.PBC, NScalars, fullScalars, filterQ4Enabled,
                                               minQ4, maxQ4, filterQ6Enabled, minQ6, maxQ6, NVectors, fullVectors)
        
        # resize visible atoms and scalars
        visibleAtoms.resize(NVisible, refcheck=False)
        scalarsQ4.resize(NVisible, refcheck=False)
        scalarsQ6.resize(NVisible, refcheck=False)
        
        # create result and add scalars
        result = base.FilterResult()
        result.addScalars("Q4", scalarsQ4)
        result.addScalars("Q6", scalarsQ6)
        
        return result
