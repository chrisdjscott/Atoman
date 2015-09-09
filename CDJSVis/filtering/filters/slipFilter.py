
"""
Slip
====

The slip filter...

"""
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
        filteringEnabled = settings.getSetting("filteringEnabled")
        
        # new scalars array
        scalars = np.zeros(len(visibleAtoms), dtype=np.float64)
        
        # call C library
        NVisible = _filtering.slipFilter(visibleAtoms, scalars, inputState.pos, refState.pos, inputState.cellDims,
                                         inputState.PBC, minSlip, maxSlip, NScalars, fullScalars,
                                         filteringEnabled, driftCompensation, driftVector, NVectors, fullVectors)
        
        # resize visible atoms and scalars
        visibleAtoms.resize(NVisible, refcheck=False)
        scalars.resize(NVisible, refcheck=False)
        
        # make result and add scalars
        result = base.FilterResult()
        result.addScalars("Slip", scalars)
        
        return result
