
"""
This filter can be used to filter a generic scalar array stored on
a Lattice object.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from . import base
from . import _filtering


class GenericScalarFilterSettings(base.BaseSettings):
    """
    Settings for the generic scalar filter
    
    """
    def __init__(self):
        super(GenericScalarFilterSettings, self).__init__()
        
        self.registerSetting("minVal", default=-999.0)
        self.registerSetting("maxVal", default=999.0)
        self.registerSetting("scalarsName")


class GenericScalarFilter(base.BaseFilter):
    """
    Generic scalar filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the filter."""
        # unpack inputs
        inputState = filterInput.inputState
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        
        # settings
        scalarsName = settings.getSetting("scalarsName")
        minVal = settings.getSetting("minVal")
        maxVal = settings.getSetting("maxVal")
        self.logger.debug("Generic scalar filter: '%s'", scalarsName)
        
        # scalars array (the full, unmodified one stored on the Lattice)
        scalarsArray = inputState.scalarsDict[scalarsName]
        
        # call C library
        NVisible = _filtering.genericScalarFilter(visibleAtoms, scalarsArray, minVal, maxVal, NScalars, fullScalars,
                                                  NVectors, fullVectors)
        
        # resize visible atoms
        visibleAtoms.resize(NVisible, refcheck=False)
        
        # result
        result = base.FilterResult()
        
        return result
