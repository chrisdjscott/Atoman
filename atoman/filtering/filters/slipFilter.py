
"""
Slip
====

Calculate slip within the lattice, on an atom by atom basis. This filter works by
comparing the displacement of an atom from its reference position, to the
equivalent displacements of neighbouring atoms from the reference lattice.
If an atom as moved in a different direction to one of its neighbours in the
reference then it has "slipped". Once slip is calculated you can filter atoms by
their slip value.

Parameters affecting this filter are:

.. glossary::

    Neighbour cut-off
        Atoms are said to have been neighbours in the reference lattice if their
        separation was less than this value.
    
    Slip tolerance
        If the magnitude of the slip contribution between an atom and one of its
        neighbours is less than this value we ignore it.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np

from . import base
from . import _filtering


class SlipFilterSettings(base.BaseSettings):
    """
    Settings for the slip filter
    
    """
    def __init__(self):
        super(SlipFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("minSlip", default=0.0)
        self.registerSetting("maxSlip", default=9999.0)
        self.registerSetting("neighbourCutOff", 3.0)
        self.registerSetting("slipTolerance", default=0.3)


class SlipFilter(base.BaseFilter):
    """
    Slip filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the filter."""
        # unpack inputs
        inputState = filterInput.inputState
        refState = filterInput.refState
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        driftCompensation = filterInput.driftCompensation
        driftVector = filterInput.driftVector
        
        # settings
        minSlip = settings.getSetting("minSlip")
        maxSlip = settings.getSetting("maxSlip")
        cutoff = settings.getSetting("neighbourCutOff")
        tol = settings.getSetting("slipTolerance")
        filteringEnabled = settings.getSetting("filteringEnabled")
        
        # new scalars array
        scalars = np.zeros(len(visibleAtoms), dtype=np.float64)
        
        # call C library
        NVisible = _filtering.slipFilter(visibleAtoms, scalars, inputState.pos, refState.pos, inputState.cellDims,
                                         inputState.PBC, minSlip, maxSlip, NScalars, fullScalars, filteringEnabled,
                                         driftCompensation, driftVector, NVectors, fullVectors, cutoff, tol)
        
        # resize visible atoms and scalars
        visibleAtoms.resize(NVisible, refcheck=False)
        scalars.resize(NVisible, refcheck=False)
        
        # make result and add scalars
        result = base.FilterResult()
        result.addScalars("Slip", scalars)
        
        return result
