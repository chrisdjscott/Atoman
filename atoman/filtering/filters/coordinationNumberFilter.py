
"""
Coordination number
===================

This filter calculates the coordination number of the atoms in the system.
It uses the values specified in the bonding table to determine whether two
atoms are bonded. If no minimum/maximum bond lengths are specified for a
given pair of elements then bonds between them will not be counted.

"""
import numpy as np

from . import base
from . import _filtering
from ...system.atoms import elements


class CoordinationNumberFilterSettings(base.BaseSettings):
    """
    Settings for the coordination number filter
    
    """
    def __init__(self):
        super(CoordinationNumberFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("minCoordNum", default=0)
        self.registerSetting("maxCoordNum", default=100)


class CoordinationNumberFilter(base.BaseFilter):
    """
    The coordination number filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the coordination number filter."""
        # unpack inputs
        inputState = filterInput.inputState
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        specieList = inputState.specieList
        NSpecies = len(specieList)
        bondDict = elements.bondDict
        
        # settings
        filteringEnabled = int(settings.getSetting("filteringEnabled"))
        minCoordNum = settings.getSetting("minCoordNum")
        maxCoordNum = settings.getSetting("maxCoordNum")
        
        # arrays to store min/max bond lengths
        bondMinArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        bondMaxArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        
        # construct bonds array (bond distances squared)
        calcBonds = False
        maxBond = -1
        for i in xrange(NSpecies):
            symi = specieList[i]
            
            if symi in bondDict:
                d = bondDict[symi]
                
                for j in xrange(NSpecies):
                    symj = specieList[j]
                    
                    if symj in d:
                        bondMin, bondMax = d[symj]
                        
                        bondMinArray[i][j] = bondMin * bondMin
                        bondMinArray[j][i] = bondMinArray[i][j]
                        
                        bondMaxArray[i][j] = bondMax * bondMax
                        bondMaxArray[j][i] = bondMaxArray[i][j]
                        
                        if bondMax > maxBond:
                            maxBond = bondMax
                        
                        if bondMax > 0:
                            calcBonds = True
                        
                        self.logger.info("  %s - %s; bond range: %f -> %f", symi, symj, bondMin, bondMax)
        
        if not calcBonds:
            self.logger.warning("No bonds defined: all coordination numbers will be zero")
        
        # new scalars array
        scalars = np.zeros(len(visibleAtoms), dtype=np.float64)
        
        # run filter
        NVisible = _filtering.coordNumFilter(visibleAtoms, inputState.pos, inputState.specie, NSpecies, bondMinArray, bondMaxArray,
                                             maxBond, inputState.cellDims, inputState.PBC, scalars, minCoordNum, maxCoordNum,
                                             NScalars, fullScalars, filteringEnabled, NVectors, fullVectors)
        
        # resize visible atoms and scalars
        visibleAtoms.resize(NVisible, refcheck=False)
        scalars.resize(NVisible, refcheck=False)
        
        # create result and add scalars
        result = base.FilterResult()
        result.addScalars("Coordination number", scalars)
        
        return result
