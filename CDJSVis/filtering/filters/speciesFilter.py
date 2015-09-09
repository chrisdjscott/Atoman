
"""
Species
=======

This filter allows you to filter atoms by their species. Only atoms of the
selected species will be visible.

"""
import numpy as np

from . import base
from . import _filtering


class SpeciesFilterSettings(base.BaseSettings):
    """
    Settings for the species filter
    
    """
    def __init__(self):
        super(SpeciesFilterSettings, self).__init__()
        
        self.registerSetting("visibleSpeciesList", default=[])


class SpeciesFilter(base.BaseFilter):
    """
    The species filter.
    
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
        specieList = inputState.specieList
        
        # settings
        visibleSpecieList = settings.getSetting("visibleSpeciesList")
        
        # make visible specie array
        visSpecArray = []
        for i, sym in enumerate(specieList):
            if sym in visibleSpecieList:
                visSpecArray.append(i)
        visSpecArray = np.asarray(visSpecArray, dtype=np.int32)
        
        # call C library
        NVisible = _filtering.specieFilter(visibleAtoms, visSpecArray, inputState.specie, NScalars, fullScalars, 
                                           NVectors, fullVectors)

        # resize visible atoms
        visibleAtoms.resize(NVisible, refcheck=False)
        
        # result
        result = base.FilterResult()
        
        return result
