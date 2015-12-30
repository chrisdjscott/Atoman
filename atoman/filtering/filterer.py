
"""
The filterer object.

@author: Chris Scott

"""
import copy
import time
import logging
import itertools

import numpy as np

from .filters import _filtering as filtering_c
from ..system.atoms import elements
from . import voronoi
from .filters import base
from . import filters
from . import atomStructure
from ..rendering import _rendering


class Filterer(object):
    """
    Filterer class.
    
    Applies the selected filters in order.
    
    """
    # known atom structure types
    knownStructures = atomStructure.knownStructures
    
    # all available filters
    defaultFilters = [
        "Species",
        "Point defects",
        "Crop box",
        "Cluster",
        "Displacement",
        "Charge",
        "Crop sphere",
        "Slice",
        "Coordination number",
        "Voronoi neighbours",
        "Voronoi volume",
        "Bond order",
        "Atom ID",
        "ACNA",
        "Slip",
        "Bubbles",
    ]
    defaultFilters.sort()
    
    # filters that are compatible with the 'Point defects' filter
    defectCompatibleFilters = [
        "Crop box",
        "Slice",
    ]
    
    def __init__(self, voronoiOptions):
        self.logger = logging.getLogger(__name__)
        self.voronoiOptions = voronoiOptions
        self._driftCompensation = False
        self.reset()
    
    def toggleDriftCompensation(self, driftCompensation):
        """Toggle the drift setting."""
        self._driftCompensation = driftCompensation
    
    def reset(self):
        """
        Reset to initial state.
        
        """
        self.inputState = None
        self.refState = None
        self.currentFilters = []
        self.currentSettings = []
        self.visibleAtoms = np.asarray([], dtype=np.int32)
        self.interstitials = np.asarray([], dtype=np.int32)
        self.vacancies = np.asarray([], dtype=np.int32)
        self.antisites = np.asarray([], dtype=np.int32)
        self.onAntisites = np.asarray([], dtype=np.int32)
        self.splitInterstitials = np.asarray([], dtype=np.int32)
        self.visibleSpecieCount = np.asarray([], dtype=np.int32)
        self.vacancySpecieCount = np.asarray([], dtype=np.int32)
        self.interstitialSpecieCount = np.asarray([], dtype=np.int32)
        self.antisiteSpecieCount = np.asarray([], dtype=np.int32)
        self.splitIntSpecieCount = np.asarray([], dtype=np.int32)
        self.driftVector = np.zeros(3, np.float64)
        self.clusterList = []
        self.bubbleList = []
        self.structureCounterDicts = {}
        self.voronoiAtoms = voronoi.VoronoiAtomsCalculator(self.voronoiOptions)
        self.voronoiDefects = voronoi.VoronoiDefectsCalculator(self.voronoiOptions)
        self.scalarsDict = {}
        self.latticeScalarsDict = {}
        self.vectorsDict = {}
        self.defectFilterSelected = False
        self.bubblesFilterSelected = False
    
    def runFilters(self, currentFilters, currentSettings, inputState, refState, sequencer=False):
        """
        Run the filters.
        
        """
        # time
        runFiltersTime = time.time()
        
        # reset the filterer
        self.reset()
        
        # validate the list of filters
        defectFilterSelected = False
        bubblesFilterSelected = False
        for filterName in currentFilters:
            if filterName not in self.defaultFilters and not filterName.startswith("Scalar:"):
                # TODO: check the scalar exists too
                raise ValueError("Unrecognised filter passed to Filterer: '%s'" % filterName)
            
            # check if the defect filter in the list
            if filterName == "Point defects":
                defectFilterSelected = True
            elif filterName == "Bubbles":
                bubblesFilterSelected = True
        self.logger.debug("Defect filter selected: %s", defectFilterSelected)
        self.defectFilterSelected = defectFilterSelected
        self.bubblesFilterSelected = bubblesFilterSelected
        
        # store refs to inputs
        self.inputState = inputState
        self.refState = refState
        self.currentFilters = currentFilters
        self.currentSettings = currentSettings
        
        # set up visible atoms or defect arrays
        if not defectFilterSelected:
            self.logger.debug("Setting all atoms visible initially")
            self.visibleAtoms = np.arange(inputState.NAtoms, dtype=np.int32)
            self.logger.info("%d visible atoms", len(self.visibleAtoms))
            
            # set Lattice scalars
            self.logger.debug("Adding initial scalars from inputState")
            for scalarsName, scalars in inputState.scalarsDict.iteritems():
                self.logger.debug("  Adding '%s' scalars", scalarsName)
                self.latticeScalarsDict[scalarsName] = copy.deepcopy(scalars)
            
            # set initial vectors
            self.logger.debug("Adding initial vectors from inputState")
            for vectorsName, vectors in inputState.vectorsDict.iteritems():
                self.logger.debug("  Adding '%s' vectors", vectorsName)
                self.vectorsDict[vectorsName] = vectors
        
        else:
            # initialise defect arrays
            self.interstitials = np.empty(inputState.NAtoms, dtype=np.int32)
            self.vacancies = np.empty(refState.NAtoms, dtype=np.int32)
            self.antisites = np.empty(refState.NAtoms, dtype=np.int32)
            self.onAntisites = np.empty(refState.NAtoms, dtype=np.int32)
            self.splitInterstitials = np.empty(3 * refState.NAtoms, dtype=np.int32)
        
        # drift compensation
        if self._driftCompensation:
            filtering_c.calculate_drift_vector(inputState.NAtoms, inputState.pos, refState.pos,
                                               refState.cellDims, inputState.PBC, self.driftVector)
            self.logger.info("Calculated drift vector: (%f, %f, %f)" % tuple(self.driftVector))
        
        # run filters
        applyFiltersTime = time.time()
        for filterName, filterSettings in zip(currentFilters, currentSettings):
            # determine the name of filter module to be loaded
            if filterName.startswith("Scalar: "):
                moduleName = "genericScalarFilter"
                filterObjectName = "GenericScalarFilter"
            else:
                words = str(filterName).title().split()
                filterObjectName = "%sFilter" % "".join(words)
                moduleName = filterObjectName[:1].lower() + filterObjectName[1:]
            self.logger.debug("Loading filter module: '%s'", moduleName)
            self.logger.debug("Creating filter object: '%s'", filterObjectName)
            
            # get module
            filterModule = getattr(filters, moduleName)
            
            # load dialog
            filterObject = getattr(filterModule, filterObjectName, None)
            if filterObject is None:
                self.logger.error("Could not locate filter object for: '%s'", filterName)
            
            else:
                self.logger.info("Running filter: '%s'", filterName)
                
                # filter
                filterObject = filterObject(filterName)
                
                # construct filter input object
                filterInput = base.FilterInput()
                filterInput.visibleAtoms = self.visibleAtoms
                filterInput.inputState = inputState
                filterInput.refState = refState
                filterInput.voronoiOptions = self.voronoiOptions
                filterInput.bondDict = elements.bondDict
                filterInput.NScalars, filterInput.fullScalars = self.makeFullScalarsArray()
                filterInput.NVectors, filterInput.fullVectors = self.makeFullVectorsArray()
                filterInput.voronoiAtoms = self.voronoiAtoms
                filterInput.voronoiDefects = self.voronoiDefects
                filterInput.driftCompensation = self._driftCompensation
                filterInput.driftVector = self.driftVector
                filterInput.vacancies = self.vacancies
                filterInput.interstitials = self.interstitials
                filterInput.splitInterstitials = self.splitInterstitials
                filterInput.antisites = self.antisites
                filterInput.onAntisites = self.onAntisites
                filterInput.defectFilterSelected = defectFilterSelected
                
                # run the filter
                result = filterObject.apply(filterInput, filterSettings)
                
                # cluster list
                if result.hasClusterList():
                    self.clusterList = result.getClusterList()
                
                # bubble list
                if result.hasBubbleList():
                    self.bubbleList = result.getBubbleList()
                
                # structure counters
                if result.hasStructureCounterDict():
                    self.structureCounterDicts[result.getStructureCounterName()] = result.getStructureCounterDict()
                
                # full vectors/scalars
                self.storeFullScalarsArray(len(self.visibleAtoms), filterInput.NScalars, filterInput.fullScalars)
                self.storeFullVectorsArray(len(self.visibleAtoms), filterInput.NVectors, filterInput.fullVectors)
                
                # new scalars
                self.scalarsDict.update(result.getScalars())
            
            if defectFilterSelected:
                nint = len(self.interstitials)
                nvac = len(self.vacancies)
                nant = len(self.antisites)
                nsplit = len(self.splitInterstitials) / 3
                num = nint + nvac + nant + nsplit
                self.logger.info("%d visible defects", num)
            else:
                self.logger.info("%d visible atoms", len(self.visibleAtoms))
        
        # species counts here
        if len(self.visibleAtoms):
            self.visibleSpecieCount = _rendering.countVisibleBySpecie(self.visibleAtoms, len(inputState.specieList),
                                                                      inputState.specie)
        if len(self.interstitials) + len(self.vacancies) + len(self.antisites) + len(self.splitInterstitials) > 0:
            self.vacancySpecieCount = _rendering.countVisibleBySpecie(self.vacancies, len(refState.specieList),
                                                                      refState.specie)
            self.interstitialSpecieCount = _rendering.countVisibleBySpecie(self.interstitials,
                                                                           len(inputState.specieList),
                                                                           inputState.specie)
            self.antisiteSpecieCount = _rendering.countAntisitesBySpecie(self.antisites, len(refState.specieList),
                                                                         refState.specie, self.onAntisites,
                                                                         len(inputState.specieList), inputState.specie)
            self.splitIntSpecieCount = _rendering.countSplitIntsBySpecie(self.splitInterstitials,
                                                                         len(inputState.specieList), inputState.specie)
        
        # TODO: dictionary of calculated properties... ?? sum of voro vols etc...
        
        # time to apply filters
        applyFiltersTime = time.time() - applyFiltersTime
        self.logger.debug("Apply filter(s) time: %f s", applyFiltersTime)
        
        # refresh available scalars in extra options dialog
        # self.parent.colouringOptions.refreshScalarColourOption()
        
        # time
        runFiltersTime = time.time() - runFiltersTime
        self.logger.debug("Apply list total time: %f s", runFiltersTime)
    
    def getBubblesIndices(self):
        """Return arrays for bubble vacancy and atom indices."""
        bubbleVacs = []
        bubbleAtoms = []
        for bubble in self.bubbleList:
            for index in bubble.vacancies():
                bubbleVacs.append(index)
            for index in bubble.atoms():
                bubbleAtoms.append(index)
        bubbleVacs = np.asarray(bubbleVacs, dtype=np.int32)
        bubbleAtoms = np.asarray(bubbleAtoms, dtype=np.int32)
        
        return bubbleVacs, bubbleAtoms
    
    def povrayAtomsWrittenSlot(self, status, povtime, uniqueID):
        """
        POV-Ray atoms have been written
        
        """
        if not status:
            self.povrayAtomsWritten = True
        
        self.logger.debug("Povray atoms written in %f s (%s)", povtime, uniqueID)
    
    def makeFullScalarsArray(self):
        """
        Combine scalars array into one big array for passing to C
        
        """
        self.logger.debug("Making full scalars array (N=%d)", len(self.scalarsDict) + len(self.latticeScalarsDict))
        
        scalarsList = []
        for name, scalars in self.scalarsDict.iteritems():
            self.logger.debug("  Adding '%s' scalars", name)
            scalarsList.append(scalars)
            if len(scalars) != len(self.visibleAtoms):
                raise RuntimeError("Wrong length for scalars: '{0}'".format(name))
        
        for name, scalars in self.latticeScalarsDict.iteritems():
            self.logger.debug("  Adding '%s' scalars (Lattice)", name)
            scalarsList.append(scalars)
            if len(scalars) != len(self.visibleAtoms):
                raise RuntimeError("Wrong length for scalars: '{0}' (Lattice)".format(name))
        
        if len(scalarsList):
            scalarsFull = np.concatenate(scalarsList)
        else:
            scalarsFull = np.array([], dtype=np.float64)
        
        return len(scalarsList), scalarsFull
    
    def makeFullVectorsArray(self):
        """
        Combine vectors array into one big array for passing to C
        
        """
        self.logger.debug("Making full vectors array (N=%d)", len(self.vectorsDict))
        
        vectorsList = []
        for name, vectors in self.vectorsDict.iteritems():
            self.logger.debug("Adding '%s' vectors", name)
            vectorsList.append(vectors)
            if vectors.shape != (len(self.visibleAtoms), 3):
                raise RuntimeError("Shape wrong for vectors array '%s': %r != %r" % (name, vectors.shape,
                                                                                     (len(self.visibleAtoms), 3)))
        
        if len(vectorsList):
            vectorsFull = np.concatenate(vectorsList)
        else:
            vectorsFull = np.array([], dtype=np.float64)
        
        return len(vectorsList), vectorsFull
    
    def storeFullScalarsArray(self, NVisible, NScalars, scalarsFull):
        """
        Split and resize full scalars array; store in dict
        
        Assumes scalarsDict was not modified since we called
        makeFullScalarsArray.
        
        """
        if NScalars > 0:
            self.logger.debug("Storing full scalars array")
            scalarsList = np.split(scalarsFull, NScalars)
            
            # Filterer.scalarsDict
            keys = self.scalarsDict.keys()
            for i, key in enumerate(keys):
                self.logger.debug("Storing '%s' scalars", key)
                scalars = scalarsList[i]
                assert len(scalars) >= NVisible, "ERROR: scalars (%s) smaller than expected (%d < %d)" % (key,
                                                                                                          len(scalars),
                                                                                                          NVisible)
                scalars_cp = copy.copy(scalars)
                scalars_cp.resize(NVisible, refcheck=False)
                self.scalarsDict[key] = scalars_cp
            
            # Lattice.scalarsDict
            offset = len(keys)
            keys = self.latticeScalarsDict.keys()
            for j, key in enumerate(keys):
                self.logger.debug("  Storing '%s' scalars (Lattice)", key)
                i = j + offset
                scalars = scalarsList[i]
                assert len(scalars) >= NVisible, "ERROR: scalars (%s) smaller than expected (%d < %d)" % (key,
                                                                                                          len(scalars),
                                                                                                          NVisible)
                scalars_cp = copy.copy(scalars)
                scalars_cp.resize(NVisible, refcheck=False)
                self.latticeScalarsDict[key] = scalars_cp
    
    def storeFullVectorsArray(self, NVisible, NVectors, vectorsFull):
        """
        Split and resize full vectors array; store in dict
        
        Assumes vectorsDict was not modified since we called
        makeFullVectorsArray.
        
        """
        if NVectors > 0:
            self.logger.debug("Storing full vectors array in dict")
            vectorsList = np.split(vectorsFull, NVectors)
            keys = self.vectorsDict.keys()
            
            for key, vectors in itertools.izip(keys, vectorsList):
                self.logger.debug("  Storing '%s' vectors", key)
                assert len(vectors) >= NVisible, "ERROR: vectors (%s) smaller than expected (%d < %d)" % (key,
                                                                                                          len(vectors),
                                                                                                          NVisible)
                vectors_cp = copy.copy(vectors)
                vectors_cp.resize((NVisible, 3), refcheck=False)
                self.vectorsDict[key] = vectors_cp
