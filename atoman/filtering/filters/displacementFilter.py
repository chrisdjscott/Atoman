
"""
Displacement
============

The calculator computes the displacement of the atoms from a reference state.
Note it can only be used when the reference and input lattices have the
same numbers of atoms. Optionally the user can filter by displacement.

Parameters for this filter are:

.. glossary::

    Filtering enabled
        Enable filtering of atoms by displacement.
    
    Minimum displacement
        The minimum displacement for an atom to be visible.
    
    Maximum displacement
        The maximum displacement for an atom to be visible.
    
    Draw displacement vectors
        Draw displacement vectors showing the movement of the atoms.
    
    Bond thickness (VTK)
        The thickness of the bonds showing the movement of the atoms (VTK).
    
    Bond thickness (POV)
        The thickness of the bonds showing the movement of the atoms (POV-Ray).
    
    Bond number of sides
        The number of sides the bonds have. More looks better but is slower.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np

from . import base
from . import _filtering


class DisplacementFilterSettings(base.BaseSettings):
    """
    Settings for the displacement filter
    
    """
    def __init__(self):
        super(DisplacementFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("drawDisplacementVectors", default=False)
        self.registerSetting("minDisplacement", default=1.2)
        self.registerSetting("maxDisplacement", default=1000.0)
        self.registerSetting("bondThicknessVTK", default=0.4)
        self.registerSetting("bondThicknessPOV", default=0.4)
        self.registerSetting("bondNumSides", default=5)


class DisplacementFilter(base.BaseFilter):
    """
    The displacement filter.
    
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
        minDisplacement = settings.getSetting("minDisplacement")
        maxDisplacement = settings.getSetting("maxDisplacement")
        filteringEnabled = int(settings.getSetting("filteringEnabled"))
        
        # only run displacement filter if input and reference NAtoms are the same
        if inputState.NAtoms != refState.NAtoms:
            raise RuntimeError("Cannot run displacement filter with different numbers of input and reference atoms")
        
        # new scalars array
        scalars = np.zeros(len(visibleAtoms), dtype=np.float64)
        
        # call C library
        NVisible = _filtering.displacementFilter(visibleAtoms, scalars, inputState.pos, refState.pos, refState.cellDims, 
                                                 inputState.PBC, minDisplacement, maxDisplacement, NScalars, fullScalars,
                                                 filteringEnabled, driftCompensation, driftVector, NVectors, fullVectors)
        
        # resize visible atoms and scalars
        visibleAtoms.resize(NVisible, refcheck=False)
        scalars.resize(NVisible, refcheck=False)
        
        # make result and add scalars
        result = base.FilterResult()
        result.addScalars("Displacement", scalars)
        
        return result
