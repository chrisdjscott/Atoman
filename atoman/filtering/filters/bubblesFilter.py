# -*- coding: utf-8 -*-

"""
Bubbles
=======

Locate bubbles in the system by the following steps:

* Remove bubble species from the lattice
* Identify vacancy clusters within the lattice
* Associate bubble species with vacancies using *Vacancy bubble radius*
* If an unassociated vacancy is within the *Vacancy-interstitial association radius* of
  an interstitial then it is not a vacancy
* Identify vacancy clusters using modified list
* Add bubble species and (as before) associate with a vacancy

The following parameters apply to this filter:

.. glossary::
    
    Bubble species
        The species of the bubbles.
    
    Vacancy radius
        The vacancy radius to use when identifying vacancy clusters.
    
    Vacancy neighbour radius
        The cut-off to use when identifying vacancy clusters. If two vacancies
        are within this distance of each other they are part of the same
        vacancy cluster.
    
    Vacancy bubble radius
        A bubble atom is associated with a vacancy if it is within this distance
        of the vacancy. Bubble atoms will only be associated to the vacancy they
        are closest to.
    
    Vacancy-interstitial association radius
        A vacancy is ignored if no bubble atom is within `vacancy bubble radius`
        of it and there is an interstitial within this distance.

"""
import numpy as np

from . import base
from . import _bubbles
from . import acnaFilter
from .. import voronoi


class BubblesFilterSettings(base.BaseSettings):
    """
    Setting for the bubbles filter.
    
    """
    def __init__(self):
        super(BubblesFilterSettings, self).__init__()
        
        self.registerSetting("bubbleSpecies", [])
        self.registerSetting("vacancyRadius", 1.3)
        self.registerSetting("vacNebRad", 4.0)
        self.registerSetting("vacancyBubbleRadius", 3.0)
        self.registerSetting("vacIntRad", 2.6)
        self.registerSetting("useAcna", default=False)
        self.registerSetting("acnaMaxBondDistance", default=5.0)
        self.registerSetting("acnaStructureType", default=1)
        
        # these settings are for compatibility with defects rendering
        self.registerSetting("vacScaleSize", default=0.75)
        self.registerSetting("vacOpacity", default=0.8)
        self.registerSetting("vacSpecular", default=0.4)
        self.registerSetting("vacSpecularPower", default=10)


class BubblesFilter(base.BaseFilter):
    """
    The bubbles filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the bubbles filter."""
        if not isinstance(settings, BubblesFilterSettings):
            raise TypeError("BubbleFilter requires a settings object of type BubbleFilterSettings.")
        
        if not isinstance(filterInput, base.FilterInput):
            raise TypeError("BubbleFilter requires an input object of type FilterInput")
        
        # unpack input object
        inputState = filterInput.inputState
        refState = filterInput.refState
        
        # settings
        bubbleSpecies = settings.getSetting("bubbleSpecies")
        if not len(bubbleSpecies):
            self.logger.warning("No bubble species have been specified therefore no bubbles can be detected!")
            result = base.FilterResult()
            return result
        self.logger.debug("Bubble species: %r", bubbleSpecies)
        
        # how many bubbles species atoms
        numBubbleAtoms = 0
        for sym in bubbleSpecies:
            index = inputState.getSpecieIndex(sym)
            numBubbleAtoms += inputState.specieCount[index]
            self.logger.debug("%d %s atoms in the lattice", inputState.specieCount[index], sym)
        self.logger.debug("Total number of bubble atoms in the lattice: %d", numBubbleAtoms)
        
        # get the indices of the bubble atoms
        bubbleAtomIndexes = []
        for i in xrange(inputState.NAtoms):
            if inputState.atomSym(i) in bubbleSpecies:
                bubbleAtomIndexes.append(i)
        bubbleAtomIndexes = np.asarray(bubbleAtomIndexes, dtype=np.int32)
        assert len(bubbleAtomIndexes) == numBubbleAtoms
        
        # compute ACNA if required
        acnaArray = None
        if settings.getSetting("useAcna"):
            self._logger.debug("Computing ACNA from point defects filter...")
            
            # acna settings
            acnaSettings = acnaFilter.AcnaFilterSettings()
            acnaSettings.updateSetting("maxBondDistance", settings.getSetting("acnaMaxBondDistance"))
            
            # acna input
            acnaInput = base.FilterInput()
            acnaInput.inputState = inputState
            acnaInput.NScalars = 0
            acnaInput.fullScalars = np.empty(acnaInput.NScalars, np.float64)
            acnaInput.NVectors = 0
            acnaInput.fullVectors = np.empty(acnaInput.NVectors, np.float64)
            acnaInput.ompNumThreads = ompNumThreads
            acnaInput.visibleAtoms = np.arange(inputState.NAtoms, dtype=np.int32)
            
            # acna filter
            acna = acnaFilter.AcnaFilter("ACNA - Defects")
            
            # run filter
            acnaResult = acna.apply(acnaInput, acnaSettings)
            
            # get scalars array from result
            acnaArray = acnaResult.getScalars()["ACNA"]
            
            # structure counters
            sd = acnaResult.getStructureCounterDict()
            self._logger.debug("  %r", sd)
        
        # check ACNA array is valid
        if acnaArray is None or len(acnaArray) != inputLattice.NAtoms:
            acnaArray = np.empty(0, np.float64)
        
        # call C library
        vacancyRadius = settings.getSetting("vacancyRadius")
        vacBubbleRad = settings.getSetting("vacancyBubbleRadius")
        vacNebRad = settings.getSetting("vacNebRad")
        vacIntRad = settings.getSetting("vacIntRad")
        acnaStructureType = settings.getSetting("acnaStructureType")
        result = _bubbles.identifyBubbles(inputState.NAtoms, inputState.pos, refState.NAtoms, refState.pos,
                                          filterInput.driftCompensation, filterInput.driftVector, inputState.cellDims,
                                          inputState.PBC, numBubbleAtoms, bubbleAtomIndexes, vacBubbleRad, acnaArray,
                                          acnaStructureType, vacancyRadius, vacNebRad, vacIntRad)
        
        # unpack
        bubbleVacList = result[0]
        bubbleAtomList = result[1]
        bubbleVacAsIndexList = result[2]
        vacancies = result[3]
        numBubbles = len(bubbleVacList)
        
        # compute voronoi volumes of atoms and vacancies
        voronoiOptions = filterInput.voronoiOptions
        voroCalc = voronoi.VoronoiDefectsCalculator(voronoiOptions)
        vor = voroCalc.getVoronoi(inputState, refState, vacancies)
        
        # create list of bubbles
        bubbleList = []
        for bubbleIndex in xrange(numBubbles):
            volume = 0.0
            
            # add volumes of bubble atoms
            bubbleAtoms = bubbleAtomList[bubbleIndex]
            for index in bubbleAtoms:
                volume += vor.atomVolume(index)
            
            # add volumes of vacancies
            bubbleVacs = bubbleVacList[bubbleIndex]
            bubbleVacsAsIndexes = bubbleVacAsIndexList[bubbleIndex]
            for i in xrange(len(bubbleVacs)):
                vacind = bubbleVacsAsIndexes[i]
                index = inputState.NAtoms + vacind
                volume += vor.atomVolume(index)
        
            # create bubble object
            bubble = Bubble()
            bubble.setRefState(refState)
            bubble.setInputState(inputState)
            bubble.setVacancies(bubbleVacList[bubbleIndex])
            bubble.setBubbleAtoms(bubbleAtomList[bubbleIndex])
            bubble.setVolume(volume)
            self.logger.debug("Adding bubble %d: %d vacancies, %d atoms (ratio: %.2f); volume is %f", bubbleIndex,
                              bubble.getNVacancies(), bubble.getNAtoms(), bubble.getRatio(), bubble.getVolume())
            bubbleList.append(bubble)
        
        # optionally show H that do not belong to a bubble as atoms too!?
        # for now we just set visible atoms to zero
        filterInput.visibleAtoms.resize(0, refcheck=False)
        
        # TODO: optionally show all defects!? (differentiate from bubbles somehow...)
        
        # result
        result = base.FilterResult()
        result.setBubbleList(bubbleList)
        
        return result


class Bubble(object):
    """
    Information about a bubble.
    
    """
    def __init__(self):
        self._vacancies = np.asarray([], dtype=np.int32)
        self._bubbleAtoms = np.asarray([], dtype=np.int32)
        self._refState = None
        self._inputState = None
        self._volume = None
        self._ratio = None
    
    def getNVacancies(self):
        """Return the number of vacancies."""
        return len(self._vacancies)
    
    def getNAtoms(self):
        """Return the number of bubble atoms."""
        return len(self._bubbleAtoms)
    
    def setVacancies(self, vacancies):
        """Set the vacancies."""
        self._vacancies = np.asarray(vacancies, dtype=np.int32)
    
    def getVacancy(self, index):
        """Return the given vacancy."""
        return self._vacancies[index]
    
    def getBubbleAtom(self, index):
        """Return the given bubble atom."""
        return self._bubbleAtoms[index]
    
    def setBubbleAtoms(self, bubbleAtoms):
        """Set the bubble atoms."""
        self._bubbleAtoms = np.asarray(bubbleAtoms, dtype=np.int32)
    
    def vacancies(self):
        """Iterator over bubble vacancies."""
        for index in self._vacancies:
            yield index
    
    def atoms(self):
        """Iterator over bubble atoms."""
        for index in self._bubbleAtoms:
            yield index
    
    def setRefState(self, refState):
        """Set the reference state."""
        self._refState = refState
    
    def setInputState(self, inputState):
        """Set the input state."""
        self._inputState = inputState
    
    def setVolume(self, volume):
        """Set the volume."""
        self._volume = volume
    
    def getVolume(self):
        """Get the volume."""
        return self._volume
    
    def computeRatio(self):
        """Compute the ratio."""
        self._ratio = float(len(self._bubbleAtoms)) / float(len(self._vacancies))
    
    def getRatio(self):
        """Return the ratio."""
        if self._ratio is None:
            self.computeRatio()
        return self._ratio
    
    def calculateCOM(self, includeAtoms=True, includeVacs=True):
        """
        Calculate the COM of the bubble.
        
        """
