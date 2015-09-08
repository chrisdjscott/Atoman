
"""
Adaptive common neighbour analysis
==================================

The ACNA calculator/filter performs the adaptive common neighbour analysis of Stutowski [1]_.

This classifies an atom as either:

0. Disordered
1. FCC
2. HCP
3. BCC
4. Icosahedral 

On the settings form you must set the parameter *Max bond distance* to be something sensible for your system.  
This parameter is used to spatially decompose the system in order to speed up the algorithm and should be chosen
so that the required number of neighbours (14 for BCC, 12 for the others) will be found within this distance of
a given atom. If in doubt, set it to something large (the code will just run slower).

More information to follow...

.. [1] A. Stukowski. *Modelling Simul. Mater. Sci. Eng.* **20** (2012) 045021; `doi: 10.1088/0965-0393/20/4/045021 <http://dx.doi.org/10.1088/0965-0393/20/4/045021>`_.

"""
import numpy as np

from ..filterer import Filterer
from . import base
from . import acna


################################################################################

class AcnaFilterSettings(base.BaseSettings):
    """
    Settings for the ACNA filter
    
    """
    def __init__(self):
        super(AcnaFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("maxBondDistance", default=5.0)
        self.registerSetting("structureVisibility", default=np.ones(len(Filterer.knownStructures), dtype=np.int32))

################################################################################

class AcnaFilter(base.BaseFilter):
    """
    ACNA filter...
    
    """
    def apply(self, filterInput, settings):
        """Run the filter."""
        # unpack inputs
        inputState = filterInput.inputState
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        ompNumThreads = filterInput.ompNumThreads
        visibleAtoms = filterInput.visibleAtoms
        pbc = inputState.PBC
        
        # settings
        maxBondDistance = settings.getSetting("maxBondDistance")
        filteringEnabled = int(settings.getSetting("filteringEnabled"))
        structureVisibility = settings.getSetting("structureVisibility")
        
        # acna scalars array
        scalars = np.zeros(len(visibleAtoms), dtype=np.float64)
        
        # counter array
        counters = np.zeros(len(Filterer.knownStructures), np.int32)
        
        # call C library
        NVisible = acna.adaptiveCommonNeighbourAnalysis(visibleAtoms, inputState.pos, scalars, inputState.cellDims, pbc,
                                                        NScalars, fullScalars, maxBondDistance, counters, filteringEnabled,
                                                        structureVisibility, ompNumThreads, NVectors, fullVectors)
        
        # result
        result = base.FilterResult()
        
        # resize visible atoms
        visibleAtoms.resize(NVisible, refcheck=False)
        
        # resize scalars and add to result
        scalars.resize(NVisible, refcheck=False)
        result.addScalars("ACNA", scalars)
        
        # structure counters dict
        result.setStructureCounterName("ACNA structure type")
        for i, structure in enumerate(Filterer.knownStructures):
            if counters[i] > 0:
                result.addStructureCount(structure, counters[i])
        
        return result
