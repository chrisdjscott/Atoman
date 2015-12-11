
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
from . import _clusters as clusters_c
from . import clusters
from ..system.atoms import elements
from . import voronoi
from .filters import base
from . import filters
from . import atomStructure
from ..rendering import _rendering


class Filterer(object):
    """
    Filter class.
    
    Contains list of subfilters to be performed in order.
    
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
        
        # self.colouringOptions = self.parent.colouringOptions
        # self.bondsOptions = self.parent.bondsOptions
        # self.displayOptions = self.parent.displayOptions
        # self.traceOptions = self.parent.traceOptions
        # self.actorsOptions = self.parent.actorsOptions
        # self.scalarBarAdded = False
        # self.scalarBar_white_bg = None
        # self.scalarBar_black_bg = None
        # self.povrayAtomsWritten = False
        
        # self._persistentList = False
        
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
        self.voronoi = None
        self.scalarsDict = {}
        self.latticeScalarsDict = {}
        self.vectorsDict = {}
        self.defectFilterSelected = False
    
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
        self.defectFilterSelected = defectFilterSelected
        for filterName in currentFilters:
            if filterName not in self.defaultFilters and not filterName.startswith("Scalar:"): #TODO: check the scalar exists too
                raise ValueError("Unrecognised filter passed to Filterer: '%s'" % filterName)
            
            # check if the defect filter in the list
            if filterName == "Point defects":
                defectFilterSelected = True
        self.logger.debug("Defect filter selected: %s", defectFilterSelected)
        
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
        
        # pov-ray hull file
        # hullFile = os.path.join(self.mainWindow.tmpDirectory, "pipeline%d_hulls%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID)))
        
        # drift compensation
        if self._driftCompensation:
            filtering_c.calculate_drift_vector(inputState.NAtoms, inputState.pos, refState.pos,
                                               refState.cellDims, inputState.PBC, self.driftVector)
            self.logger.info("Calculated drift vector: (%f, %f, %f)" % tuple(self.driftVector))
        
        # run filters
        applyFiltersTime = time.time()
        drawDisplacementVectors = False
        displacementSettings = None
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
                self.logger.error("Could not locate filter object for: '%s'", filterNameString)
            
            else:
                self.logger.info("Running filter: '%s'", filterName)
                
                # filter
                filterObject = filterObject(filterName)
                
                # check if we need to compute the Voronoi tessellation
                if filterObject.requiresVoronoi:
                    self.calculateVoronoi(inputState)
                
                # construct filter input object
                filterInput = base.FilterInput()
                filterInput.visibleAtoms = self.visibleAtoms
                filterInput.inputState = inputState
                filterInput.refState = refState
                filterInput.voronoiOptions = self.voronoiOptions
                filterInput.bondDict = elements.bondDict
                filterInput.NScalars, filterInput.fullScalars = self.makeFullScalarsArray()
                filterInput.NVectors, filterInput.fullVectors = self.makeFullVectorsArray()
                filterInput.voronoi = self.voronoi
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
                    #TODO: calculate volumes should be here
                    if filterSettings.getSetting("calculateVolumes"):
                        pass
                
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
                
                # custom (ideally generalise this too...)
                # if (filterName == "Displacement" or filterName == "Point defects") and inputState.NAtoms == refState.NAtoms:
                #     drawDisplacementVectors = filterSettings.getSetting("drawDisplacementVectors")
                #     displacementSettings = filterSettings
                
                # elif filterName == "Cluster":
                    # if filterSettings.getSetting("drawConvexHulls"):
                    #     self.clusterFilterDrawHulls(filterSettings, hullFile)
                     
                    # if filterSettings.getSetting("calculateVolumes"):
                    #     self.clusterFilterCalculateVolumes(filterSettings)
            
            # calculate numbers of atoms/defects
            # if defectFilterSelected:
            #     self.NVis = len(self.interstitials) + len(self.vacancies) + len(self.antisites) + len(self.splitInterstitials)
            #     self.NVac = len(self.vacancies)
            #     self.NInt = len(self.interstitials) + len(self.splitInterstitials) / 3
            #     self.NAnt = len(self.antisites)
            # else:
            #     self.NVis = len(self.visibleAtoms)
            # 
            # self.logger.info("  %d visible atoms", self.NVis)
            
            if defectFilterSelected:
                ndef = len(self.interstitials) + len(self.vacancies) + len(self.antisites) + len(self.splitInterstitials)
                self.logger.info("  %d visible defects", ndef)
            else:
                self.logger.info("  %d visible atoms", len(self.visibleAtoms))
        
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
        
        # TODO: dictionary of calculated properties... ??
        
        
        # report total Voro volume if filter selected
        # if "Voronoi volume" in self.scalarsDict:
        #     sumVoroVol = np.sum(self.scalarsDict["Voronoi volume"])
        #     self.logger.info("Sum of visible Voronoi volumes = %f units^3", sumVoroVol)
        
        # time to apply filters
        applyFiltersTime = time.time() - applyFiltersTime
        self.logger.debug("Apply filter(s) time: %f s", applyFiltersTime)
        
        # refresh available scalars in extra options dialog
        # self.parent.colouringOptions.refreshScalarColourOption()
        
        # render
#         renderTime = time.time()
#         povfile = "pipeline%d_atoms%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
#         if self.parent.defectFilterSelected:
#             # defects filter always first (for now...)
#             filterSettingsGui = currentSettings[0]
#             filterSettings = filterSettingsGui.getSettings()
#             if filterSettings is None:
#                 filterSettings = filterSettingsGui
#             
#             # render convex hulls
#             if filterSettings.getSetting("findClusters") and filterSettings.getSetting("drawConvexHulls"):
#                 self.pointDefectFilterDrawHulls(filterSettings, hullFile)
#             
#             # cluster volume
#             if filterSettings.getSetting("findClusters") and filterSettings.getSetting("calculateVolumes"):
#                 self.pointDefectFilterCalculateClusterVolumes(filterSettings)
#             
#             # render defects
#             if filterSettings.getSetting("findClusters") and filterSettings.getSetting("drawConvexHulls") and filterSettings.getSetting("hideDefects"):
#                 pass
#             
#             else:
#                 counters = renderer.getActorsForFilteredDefects(self.interstitials, self.vacancies, self.antisites, self.onAntisites,
#                                                                 self.splitInterstitials, self.actorsDict, self.colouringOptions,
#                                                                 filterSettings, self.displayOptions, self.pipelinePage)
#                 
#                 self.vacancySpecieCount = counters[0]
#                 self.interstitialSpecieCount = counters[1]
#                 self.antisiteSpecieCount = counters[2]
#                 self.splitIntSpecieCount = counters[3]
#                 self.scalarBar_white_bg = counters[4]
#                 self.scalarBar_black_bg = counters[5]
#                 
#                 # write pov-ray file too
#                 povfile = "pipeline%d_defects%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
#                 renderer.writePovrayDefects(povfile, self.vacancies, self.interstitials, self.antisites, self.onAntisites,
#                                             filterSettings, self.mainWindow, self.displayOptions, self.splitInterstitials, self.pipelinePage)
#                 self.povrayAtomsWritten = True
#             
#             # draw displacement vectors on interstitials atoms
#             if drawDisplacementVectors:
#                 self.renderInterstitialDisplacementVectors(displacementSettings)
#         
#         elif filterName == "Bubbles":
#             self.logger.debug("Rendering bubbles...")
#             
#             # full list of vacancies and bubble atoms
#             bubbleAtoms = []
#             bubbleVacancies = []
#             for bubble in self.bubbleList:
#                 for i in xrange(bubble.getNAtoms()):
#                     bubbleAtoms.append(bubble.getBubbleAtom(i))
#                 for i in xrange(bubble.getNVacancies()):
#                     bubbleVacancies.append(bubble.getVacancy(i))
#             bubbleAtoms = np.asarray(bubbleAtoms, dtype=np.int32)
#             bubbleVacancies = np.asarray(bubbleVacancies, dtype=np.int32)
#             self.NVac = len(bubbleVacancies)
#             
#             # render atoms
#             self.scalarBar_white_bg, self.scalarBar_black_bg, visSpecCount = renderer.getActorsForFilteredSystem(bubbleAtoms, self.mainWindow, 
#                                                                                                                  self.actorsDict, self.colouringOptions, 
#                                                                                                                  povfile, self.scalarsDict, self.latticeScalarsDict,
#                                                                                                                  self.displayOptions, self.pipelinePage,
#                                                                                                                  self.povrayAtomsWrittenSlot, self.vectorsDict,
#                                                                                                                  self.vectorsOptions, NVisibleForRes=None,
#                                                                                                                  sequencer=sequencer)
#             self.visibleSpecieCount = visSpecCount
#             self.NVis = len(bubbleAtoms)
#             
#             # render defects
#             counters = renderer.getActorsForFilteredDefects(self.interstitials, self.vacancies, bubbleVacancies, self.onAntisites,
#                                                             self.splitInterstitials, self.actorsDict, self.colouringOptions,
#                                                             filterSettings, self.displayOptions, self.pipelinePage)
#             
#             self.vacancySpecieCount = counters[0]
#             self.interstitialSpecieCount = counters[1]
#             self.antisiteSpecieCount = counters[2]
#             self.splitIntSpecieCount = counters[3]
# #             self.scalarBar_white_bg = counters[4]
# #             self.scalarBar_black_bg = counters[5]
#             
#             for bubble in self.bubbleList:
#                 for i in xrange(bubble.getNVacancies()):
#                     index = bubble.getVacancy(i)
#                     self.vacancySpecieCount[self.pipelinePage.refState.specie[index]] += 1
#             
#             # write pov-ray file too
#             povfile = "pipeline%d_defects%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
#             renderer.writePovrayDefects(povfile, bubbleVacancies, self.interstitials, self.antisites, self.onAntisites,
#                                         filterSettings, self.mainWindow, self.displayOptions, self.splitInterstitials, self.pipelinePage)
#             self.povrayAtomsWritten = True
#             
#             # store for picker
#             self.vacancies = bubbleVacancies
#             self.visibleAtoms = bubbleAtoms
#         
#         else:
#             if filterName == "Cluster" and filterSettings.getSetting("drawConvexHulls") and filterSettings.getSetting("hideAtoms"):
#                 pass
#             
#             else:
#                 # this is a hack!! not ideal
#                 if self.parent.isPersistentList():
#                     NVisibleForRes = 800
#                 else:
#                     NVisibleForRes = None
#                 
#                 self.scalarBar_white_bg, self.scalarBar_black_bg, visSpecCount = renderer.getActorsForFilteredSystem(self.visibleAtoms, self.mainWindow, 
#                                                                                                                      self.actorsDict, self.colouringOptions, 
#                                                                                                                      povfile, self.scalarsDict, self.latticeScalarsDict,
#                                                                                                                      self.displayOptions, self.pipelinePage,
#                                                                                                                      self.povrayAtomsWrittenSlot, self.vectorsDict,
#                                                                                                                      self.vectorsOptions, NVisibleForRes=NVisibleForRes,
#                                                                                                                      sequencer=sequencer)
#                 
#                 self.visibleSpecieCount = visSpecCount
#             
#             # do displacement vectors on visible atoms
#             if drawDisplacementVectors:
#                 self.renderDisplacementVectors(displacementSettings)
#             
#             # render trace
#             if self.traceOptions.drawTraceVectors:
#                 self.renderTrace()
#             
#             if self.bondsOptions.drawBonds:
#                 # find bonds
#                 self.calculateBonds()
#             
#             # voronoi
#             if self.voronoiOptions.displayVoronoi:
#                 self.renderVoronoi()
#         
#         # time to render
#         renderTime = time.time() - renderTime
#         self.logger.debug("Create actors time: %f s", renderTime)
#         
#         if self.parent.visible:
#             addActorsTime = time.time()
#             
#             self.addActors()
#             
#             addActorsTime = time.time() - addActorsTime
#             self.logger.debug("Add actors time: %f s" % addActorsTime)
#         
# #         for name, scalars in self.scalarsDict.iteritems():
# #             assert len(scalars) == len(self.visibleAtoms)
# #             f = open("%s_after.dat" % name.replace(" ", "_"), "w")
# #             for tup in itertools.izip(self.visibleAtoms, scalars):
# #                 f.write("%d %f\n" % tup)
# #             f.close()
#         
#         self.actorsOptions.refresh(self.actorsDict)
        
        # time
        runFiltersTime = time.time() - runFiltersTime
        self.logger.debug("Apply list total time: %f s", runFiltersTime)
    
    def povrayAtomsWrittenSlot(self, status, povtime, uniqueID):
        """
        POV-Ray atoms have been written
        
        """
        if not status:
            self.povrayAtomsWritten = True
        
        self.logger.debug("Povray atoms written in %f s (%s)", povtime, uniqueID)
    
    def calculateVoronoi(self, inputState):
        """
        Calc voronoi tesselation
        
        """
        if self.voronoi is None:
            self.voronoi = voronoi.computeVoronoi(inputState, self.voronoiOptions, inputState.PBC)
    
    # def renderVoronoi(self):
    #     """
    #     Render Voronoi cells
    #     
    #     """
    #     inputState = self.pipelinePage.inputState
    #     
    #     status = self.calculateVoronoi()
    #     if status:
    #         return status
    #     
    #     if not len(self.visibleAtoms):
    #         return 2
    #     
    #     if len(self.visibleAtoms) > 2000:
    #         # warn that this will be slow
    #         msg = """<p>You are about to render a large number of Voronoi cells (%d).</p>
    #                  <p>This will probably be very slow!</p>
    #                  <p>Do you wish to continue?</p>""" % len(self.visibleAtoms)
    #         
    #         reply = QtGui.QMessageBox.question(self.mainWindow, "Message", msg, QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.Yes)
    #         
    #         if reply == QtGui.QMessageBox.No:
    #             return
    #     
    #     # POV-RAY file
    #     voroFile = os.path.join(self.mainWindow.tmpDirectory, "pipeline%d_voro%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID)))
    #     
    #     # get actors for vis atoms only!
    #     renderVoronoi.getActorsForVoronoiCells(self.visibleAtoms, inputState, self.voronoi, 
    #                                            self.colouringOptions, self.voronoiOptions, self.actorsDict, 
    #                                            voroFile, self.scalarsDict, log=self.log)
        
    # def addScalarBar(self):
    #     """
    #     Add scalar bar.
    #     
    #     """
    #     if self.scalarBar_white_bg is not None and self.parent.scalarBarButton.isChecked() and not self.parent.filterTab.scalarBarAdded:
    #         for rw in self.rendererWindows:
    #             if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
    #                 # which scalar bar to add
    #                 if rw.blackBackground:
    #                     scalarBar = self.scalarBar_black_bg
    #                 else:
    #                     scalarBar = self.scalarBar_white_bg
    #                 
    #                 rw.vtkRen.AddActor2D(scalarBar)
    #                 rw.vtkRenWinInteract.ReInitialize()
    #         
    #         self.parent.filterTab.scalarBarAdded = True
    #         self.scalarBarAdded = True
    #     
    #     return self.scalarBarAdded
    
    # def hideScalarBar(self):
    #     """
    #     Remove scalar bar.
    #     
    #     """
    #     if self.scalarBarAdded:
    #         for rw in self.rendererWindows:
    #             if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
    #                 # which scalar bar was added
    #                 if rw.blackBackground:
    #                     scalarBar = self.scalarBar_black_bg
    #                 else:
    #                     scalarBar = self.scalarBar_white_bg
    #                 
    #                 rw.vtkRen.RemoveActor2D(scalarBar)
    #                 rw.vtkRenWinInteract.ReInitialize()
    #         
    #         self.parent.filterTab.scalarBarAdded = False
    #         self.scalarBarAdded = False
    
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
            self.logger.debug("  Adding '%s' vectors", name)
            vectorsList.append(vectors)
            if vectors.shape != (len(self.visibleAtoms), 3):
                raise RuntimeError("Shape wrong for vectors array '%s': %r != %r" % (name, vectors.shape, (len(self.visibleAtoms), 3)))
        
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
                self.logger.debug("  Storing '%s' scalars", key)
                scalars = scalarsList[i]
                assert len(scalars) >= NVisible, "ERROR: scalars (%s) smaller than expected (%d < %d)" % (key, len(scalars), NVisible)
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
                assert len(scalars) >= NVisible, "ERROR: scalars (%s) smaller than expected (%d < %d)" % (key, len(scalars), NVisible)
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
                assert len(vectors) >= NVisible, "ERROR: vectors (%s) smaller than expected (%d < %d)" % (key, len(vectors), NVisible)
                vectors_cp = copy.copy(vectors)
                vectors_cp.resize((NVisible, 3), refcheck=False)
                self.vectorsDict[key] = vectors_cp
    
    def pointDefectFilterCalculateClusterVolumes(self, settings):
        """
        Calculate volumes of clusters
        
        """
        self.logger.debug("Calculating volumes of defect clusters")
        self.logger.warning("If your clusters cross PBCs this may or may not give correct volumes; please test and let me know")
        
        inputLattice = self.pipelinePage.inputState
        refLattice = self.pipelinePage.refState
        clusterList = self.clusterList
        
        if settings.getSetting("calculateVolumesHull"):
            count = 0
            for cluster in clusterList:
                # first make pos array for this cluster
                clusterPos = cluster.makeClusterPos(inputLattice, refLattice)
                NDefects = len(clusterPos) / 3
                
                # now get convex hull
                if NDefects < 4:
                    pass
                
                else:
                    appliedPBCs = np.zeros(7, np.int32)
                    clusters_c.prepareClusterToDrawHulls(NDefects, clusterPos, inputLattice.cellDims, 
                                                         self.pipelinePage.PBC, appliedPBCs, settings.getSetting("neighbourRadius"))
                    
                    cluster.volume, cluster.facetArea = clusters.findConvexHullVolume(NDefects, clusterPos)
                
                self.logger.info("  Cluster %d (%d defects)", count, cluster.getNDefects())
                if cluster.facetArea is not None:
                    self.logger.info("    volume is %f; facet area is %f", cluster.volume, cluster.facetArea)
                
                count += 1
        
        elif settings.getSetting("calculateVolumesVoro"):
            # compute Voronoi
            vor = voronoi.computeVoronoiDefects(inputLattice, refLattice, self.vacancies, self.voronoiOptions, self.pipelinePage.PBC)
            
            count = 0
            for cluster in clusterList:
                volume = 0.0
                
                # add volumes of interstitials
                for i in xrange(cluster.getNInterstitials()):
                    index = cluster.interstitials[i]
                    volume += vor.atomVolume(index)
                
                # add volumes of split interstitial atoms
                for i in xrange(cluster.getNSplitInterstitials()):
                    index = cluster.splitInterstitials[3*i+1]
                    volume += vor.atomVolume(index)
                    index = cluster.splitInterstitials[3*i+2]
                    volume += vor.atomVolume(index)
                
                # add volumes of on antisite atoms
                for i in xrange(cluster.getNAntisites()):
                    index = cluster.onAntisites[i]
                    volume += vor.atomVolume(index)
                
                # add volumes of vacancies
                for i in xrange(cluster.getNVacancies()):
                    vacind = cluster.vacAsIndex[i]
                    index = inputLattice.NAtoms + vacind
                    volume += vor.atomVolume(index)
                
                cluster.volume = volume
                
                self.logger.info("  Cluster %d (%d defects)", count, cluster.getNDefects())
                self.logger.info("    volume is %f", volume)
                
                count += 1
        
        else:
            self.logger.error("Method to calculate defect cluster volumes not specified")
    
    def clusterFilterCalculateVolumes(self, filterSettings):
        """
        Calculate volumes of clusters.
        
        """
        # this will not work properly over PBCs at the moment
        lattice = self.pipelinePage.inputState
        
        # calculate volumes
        if filterSettings.getSetting("calculateVolumesHull"):
            count = 0
            for cluster in self.clusterList:
                # first make pos array for this cluster
                clusterPos = np.empty(3 * len(cluster), np.float64)
                for i in xrange(len(cluster)):
                    index = cluster[i]
                    
                    clusterPos[3*i] = lattice.pos[3*index]
                    clusterPos[3*i+1] = lattice.pos[3*index+1]
                    clusterPos[3*i+2] = lattice.pos[3*index+2]
                
                # now get convex hull
                if len(cluster) < 4:
                    pass
                
                else:
                    PBC = self.pipelinePage.PBC
                    if PBC[0] or PBC[1] or PBC[2]:
                        appliedPBCs = np.zeros(7, np.int32)
                        neighbourRadius = filterSettings.getSetting("neighbourRadius")
                        clusters_c.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, 
                                                             PBC, appliedPBCs, neighbourRadius)
                    
                    volume, area = clusters.findConvexHullVolume(len(cluster), clusterPos)
                
                self.logger.info("  Cluster %d (%d atoms)", count, len(cluster))
                if area is not None:
                    self.logger.info("    volume is %f; facet area is %f", volume, area)
                
                # store volume/facet area
                cluster.volume = volume
                cluster.facetArea = area
                
                count += 1
        
        elif filterSettings.getSetting("calculateVolumesVoro"):
            # compute Voronoi
            self.calculateVoronoi()
            vor = self.voronoi
            
            count = 0
            for cluster in self.clusterList:
                volume = 0.0
                for index in cluster:
                    volume += vor.atomVolume(index)
                
                self.logger.info("  Cluster %d (%d atoms)", count, len(cluster))
                self.logger.info("    volume is %f", volume)
                
                # store volume
                cluster.volume = volume
                
                count += 1
        
        else:
            self.logger.error("Method to calculate cluster volumes not specified")
