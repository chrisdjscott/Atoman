
"""
The filterer object.

@author: Chris Scott

"""
import os
import copy
import time
import logging
import itertools

import numpy as np
from PySide import QtGui, QtCore

from . import _filtering as filtering_c
from . import _defects as defects_c
from . import _clusters as clusters_c
from . import bonds as bonds_c
from . import bond_order as bond_order
from . import acna
from ..rendering import renderer
from ..rendering import renderBonds
from ..algebra import vectors
from . import clusters
from ..state.atoms import elements
from . import voronoi
from ..rendering import renderVoronoi


################################################################################
class Filterer(object):
    """
    Filter class.
    
    Contains list of subfilters to be performed in order.
    
    """
    # this must correspond to "atom_structures.h"
    knownStructures = [
        "disordered",
        "FCC",
        "HCP",
        "BCC",
        "icosahedral",
        "sigma11_tilt1",
        "sigma11_tilt2",
    ]
    
    def __init__(self, parent):
        self.parent = parent
        self.filterTab = parent.filterTab
        self.mainWindow = self.parent.mainWindow
        self.rendererWindows = self.mainWindow.rendererWindows
        self.mainToolbar = self.parent.mainToolbar
        self.pipelineIndex = self.filterTab.pipelineIndex
        self.pipelinePage = self.filterTab
        
        self.log = self.mainWindow.console.write
        self.logger = logging.getLogger(__name__)
        
        self.NVis = 0
        self.NVac = 0
        self.NInt = 0
        self.NAnt = 0
        self.visibleAtoms = np.empty(0, np.int32)
        self.visibleSpecieCount = np.asarray([], dtype=np.int32)
        self.vacancySpecieCount = np.asarray([], dtype=np.int32)
        self.interstitialSpecieCount = np.asarray([], dtype=np.int32)
        self.antisiteSpecieCount = np.asarray([], dtype=np.int32)
        self.splitIntSpecieCount = np.asarray([], dtype=np.int32)
        self.vacancies = np.empty(0, np.int32)
        self.interstitials = np.empty(0, np.int32)
        self.antisites = np.empty(0, np.int32)
        self.onAntisites = np.empty(0, np.int32)
        self.splitInterstitials = np.empty(0, np.int32)
        
        self.driftVector = np.zeros(3, np.float64)
        
        self.actorsDict = {}
        
        self.traceDict = {}
        self.previousPosForTrace = None
        
        self.colouringOptions = self.parent.colouringOptions
        self.bondsOptions = self.parent.bondsOptions
        self.displayOptions = self.parent.displayOptions
        self.voronoiOptions = self.parent.voronoiOptions
        self.traceOptions = self.parent.traceOptions
        self.vectorsOptions = self.parent.vectorsOptions
        self.actorsOptions = self.parent.actorsOptions
        self.scalarBarAdded = False
        self.scalarsDict = {}
        self.latticeScalarsDict = {}
        self.vectorsDict = {}
        self.scalarBar_white_bg = None
        self.scalarBar_black_bg = None
        self.povrayAtomsWritten = False
        self.clusterList = []
        self.voronoi = None
        
        self.structureCounterDicts = {}
    
    def removeActors(self, sequencer=False):
        """
        Remove actors.
        
        """
        self.hideActors()
        
        self.actorsDict = {}
        if not sequencer:
            self.traceDict = {}
            self.previousPosForTrace = None
        
        self.scalarsDict = {}
        self.latticeScalarsDict = {}
        self.vectorsDict = {}
        self.scalarBar_white_bg = None
        self.scalarBar_black_bg = None
        
        self.NVis = 0
        self.NVac = 0
        self.NInt = 0
        self.NAnt = 0
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
        
        self.povrayAtomsWritten = False
        self.clusterList = []
        self.structureCounterDicts = {}
        self.voronoi = None
    
    def hideActors(self):
        """
        Hide all actors
        
        """
        for actorName, val in self.actorsDict.iteritems():
            if isinstance(val, dict):
                self.logger.debug("Removing actors for: '%s'", actorName)
                for actorName2, actorObj in val.iteritems():
                    if actorObj.visible:
                        self.logger.debug("  Removing actor: '%s'", actorName2)
                        for rw in self.rendererWindows:
                            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                                rw.vtkRen.RemoveActor(actorObj.actor)
                        
                        actorObj.visible = False
            
            else:
                actorObj = val
                if actorObj.visible:
                    self.logger.debug("Removing actor: '%s'", actorName)
                    for rw in self.rendererWindows:
                        if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                            rw.vtkRen.RemoveActor(actorObj.actor)
                    
                    actorObj.visible = False
        
        for rw in self.rendererWindows:
            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                rw.vtkRenWinInteract.ReInitialize()
        
        self.hideScalarBar()
    
    def setActorAmbient(self, actorName, parentName, ambient, reinit=True):
        """
        Set ambient property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        actorObj.actor.GetProperty().SetAmbient(ambient)
        
        if reinit:
            self.reinitialiseRendererWindows()
    
    def setActorSpecular(self, actorName, parentName, specular, reinit=True):
        """
        Set specular property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        actorObj.actor.GetProperty().SetSpecular(specular)
        
        if reinit:
            self.reinitialiseRendererWindows()
    
    def setActorSpecularPower(self, actorName, parentName, specularPower, reinit=True):
        """
        Set specular power property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        actorObj.actor.GetProperty().SetSpecularPower(specularPower)
        
        if reinit:
            self.reinitialiseRendererWindows()
    
    def getActorAmbient(self, actorName, parentName):
        """
        Get ambient property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        ambient = actorObj.actor.GetProperty().GetAmbient()
        
        return ambient
    
    def getActorSpecular(self, actorName, parentName):
        """
        Get specular property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        specular = actorObj.actor.GetProperty().GetSpecular()
        
        return specular
    
    def getActorSpecularPower(self, actorName, parentName):
        """
        Get specular power property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        specularPower = actorObj.actor.GetProperty().GetSpecularPower()
        
        return specularPower
    
    def addActor(self, actorName, parentName=None, reinit=True):
        """
        Add individual actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        changes = False
        if not actorObj.visible:
            self.logger.debug("Adding actor: '%s'", actorName)
            for rw in self.rendererWindows:
                if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                    rw.vtkRen.AddActor(actorObj.actor)
                    changes = True
            
            actorObj.visible = True
        
        if changes and reinit:
            self.reinitialiseRendererWindows()
        
        return changes
    
    def hideActor(self, actorName, parentName=None, reinit=True):
        """
        Remove individual actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        changes = False
        if actorObj.visible:
            self.logger.debug("Removing actor: '%s'", actorName)
            for rw in self.rendererWindows:
                if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                    rw.vtkRen.RemoveActor(actorObj.actor)
                    changes = True
            
            actorObj.visible = False
        
        if changes and reinit:
            self.reinitialiseRendererWindows()
        
        return changes
    
    def reinitialiseRendererWindows(self):
        """
        Reinit renderer windows
        
        """
        for rw in self.rendererWindows:
            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                rw.vtkRenWinInteract.ReInitialize()
    
    def addActors(self):
        """
        Add all actors
        
        """
        for actorName, val in self.actorsDict.iteritems():
            if isinstance(val, dict):
                self.logger.debug("Adding actors for: '%s'", actorName)
                for actorName2, actorObj in val.iteritems():
                    if not actorObj.visible:
                        self.logger.debug("  Adding actor: '%s'", actorName2)
                        for rw in self.rendererWindows:
                            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                                rw.vtkRen.AddActor(actorObj.actor)
                        
                        actorObj.visible = True
            
            else:
                actorObj = val
                if not actorObj.visible:
                    self.logger.debug("Adding actor: '%s'", actorName)
                    for rw in self.rendererWindows:
                        if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                            rw.vtkRen.AddActor(actorObj.actor)
                    
                    actorObj.visible = True
        
        for rw in self.rendererWindows:
            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                rw.vtkRenWinInteract.ReInitialize()
        
        self.addScalarBar()
    
    def runFilters(self, sequencer=False):
        """
        Run the filters.
        
        """
        # time
        runFiltersTime = time.time()
        
        # remove actors
        if not self.parent.isPersistentList():
            self.removeActors(sequencer=sequencer)
        
        # first set up visible atoms arrays
        inputState = self.pipelinePage.inputState
        NAtoms = inputState.NAtoms
        
        if not self.parent.defectFilterSelected:
            self.visibleAtoms = np.arange(NAtoms, dtype=np.int32)
            self.NVis = NAtoms
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
        
        # pov-ray hull file
        hullFile = os.path.join(self.mainWindow.tmpDirectory, "pipeline%d_hulls%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID)))
        
        # drift compensation
        if self.parent.driftCompensation:
            filtering_c.calculate_drift_vector(NAtoms, self.pipelinePage.inputState.pos, self.pipelinePage.refState.pos, self.pipelinePage.refState.cellDims, 
                                               self.pipelinePage.PBC, self.driftVector)
            self.logger.info("Calculated drift vector: (%f, %f, %f)" % tuple(self.driftVector))
        
        # run filters
        applyFiltersTime = time.time()
        drawDisplacementVectors = False
        displacementSettings = None
        filterName = ""
        currentFilters = self.parent.getCurrentFilterNames()
        currentSettings = self.parent.getCurrentFilterSettings()
        for i in xrange(len(currentFilters)):
            # filter name
            filterNameString = currentFilters[i]
            array = filterNameString.split("[")
            filterName = array[0].strip()
            
            # filter settings
            filterSettingsGui = currentSettings[i]
            filterSettings = filterSettingsGui.getSettings()
            if filterSettings is None:
                filterSettings = filterSettingsGui
            
            self.logger.info("Running filter: '%s'", filterName)
            
            if filterName == "Species":
                self.filterSpecie(filterSettings)
            
            elif filterName == "Crop box":
                self.cropFilter(filterSettings)
            
            elif filterName == "Displacement":
                self.displacementFilter(filterSettings)
                drawDisplacementVectors = filterSettings.getSetting("drawDisplacementVectors")
                displacementSettings = filterSettings
            
            elif filterName == "Point defects":
                interstitials, vacancies, antisites, onAntisites, splitInterstitials = self.pointDefectFilter(filterSettings)
                
                self.interstitials = interstitials
                self.vacancies = vacancies
                self.antisites = antisites
                self.onAntisites = onAntisites
                self.splitInterstitials = splitInterstitials
            
            elif filterName == "Charge":
                self.chargeFilter(filterSettings)
            
            elif filterName == "Cluster":
                self.clusterFilter(filterSettings)
                
                if filterSettings.getSetting("drawConvexHulls"):
                    self.clusterFilterDrawHulls(filterSettings, hullFile)
                
                if filterSettings.getSetting("calculateVolumes"):
                    self.clusterFilterCalculateVolumes(filterSettings)
            
            elif filterName == "Crop sphere":
                self.cropSphereFilter(filterSettings)
            
            elif filterName == "Slice":
                self.sliceFilter(filterSettings)
            
            elif filterName == "Coordination number":
                self.coordinationNumberFilter(filterSettings)
            
            elif filterName == "Voronoi volume":
                self.voronoiVolumeFilter(filterSettings)
            
            elif filterName == "Voronoi neighbours":
                self.voronoiNeighboursFilter(filterSettings)
            
            elif filterName == "Bond order":
                self.bondOrderFilter(filterSettings)
            
            elif filterName == "Atom ID":
                self.atomIndexFilter(filterSettings)
            
            elif filterName == "ACNA":
                self.acnaFilter(filterSettings)
            
            elif filterName.startswith("Scalar: "):
                self.genericScalarFilter(filterName, filterSettings)
            
            elif filterName.startswith("Slip"):
                self.slipFilter(filterSettings)
            
            else:
                self.logger.warning("Unrecognised filter: '%s'; skipping", filterName)
            
            # write to log
            if self.parent.defectFilterSelected:
                self.NVis = len(self.interstitials) + len(self.vacancies) + len(self.antisites) + len(self.splitInterstitials)
                self.NVac = len(self.vacancies)
                self.NInt = len(self.interstitials) + len(self.splitInterstitials) / 3
                self.NAnt = len(self.antisites)
            else:
                self.NVis = len(self.visibleAtoms)
            
            self.logger.info("  %d visible atoms", self.NVis)
        
        # report total Voro volume if filter selected
        if "Voronoi volume" in self.scalarsDict:
            sumVoroVol = np.sum(self.scalarsDict["Voronoi volume"])
            self.logger.info("Sum of visible Voronoi volumes = %f units^3", sumVoroVol)
        
        # time to apply filters
        applyFiltersTime = time.time() - applyFiltersTime
        self.logger.debug("Apply filter(s) time: %f s", applyFiltersTime)
        
        # refresh available scalars in extra options dialog
        self.parent.colouringOptions.refreshScalarColourOption()
        
        # render
        renderTime = time.time()
        povfile = "pipeline%d_atoms%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
        if self.parent.defectFilterSelected:
            # defects filter always first (for now...)
            filterSettings = currentSettings[0]
            
            # render convex hulls
            if filterSettings.findClusters and filterSettings.drawConvexHulls:
                self.pointDefectFilterDrawHulls(filterSettings, hullFile)
            
            # cluster volume
            if filterSettings.findClusters and filterSettings.calculateVolumes:
                self.pointDefectFilterCalculateClusterVolumes(filterSettings)
            
            # render defects
            if filterSettings.findClusters and filterSettings.drawConvexHulls and filterSettings.hideDefects:
                pass
            
            else:
                counters = renderer.getActorsForFilteredDefects(interstitials, vacancies, antisites, onAntisites, splitInterstitials, self.actorsDict, 
                                                                self.colouringOptions, filterSettings, self.displayOptions, self.pipelinePage)
                
                self.vacancySpecieCount = counters[0]
                self.interstitialSpecieCount = counters[1]
                self.antisiteSpecieCount = counters[2]
                self.splitIntSpecieCount = counters[3]
                self.scalarBar_white_bg = counters[4]
                self.scalarBar_black_bg = counters[5]
                
                # write pov-ray file too
                povfile = "pipeline%d_defects%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
                renderer.writePovrayDefects(povfile, vacancies, interstitials, antisites, onAntisites, filterSettings, self.mainWindow, 
                                            self.displayOptions, splitInterstitials, self.pipelinePage)
                self.povrayAtomsWritten = True
        
        else:
            if filterName == "Cluster" and filterSettings.getSetting("drawConvexHulls") and filterSettings.getSetting("hideAtoms"):
                pass
            
            else:
                # this is a hack!! not ideal
                if self.parent.isPersistentList():
                    NVisibleForRes = 800
                else:
                    NVisibleForRes = None
                
                self.scalarBar_white_bg, self.scalarBar_black_bg, visSpecCount = renderer.getActorsForFilteredSystem(self.visibleAtoms, self.mainWindow, 
                                                                                                                     self.actorsDict, self.colouringOptions, 
                                                                                                                     povfile, self.scalarsDict, self.latticeScalarsDict,
                                                                                                                     self.displayOptions, self.pipelinePage,
                                                                                                                     self.povrayAtomsWrittenSlot, self.vectorsDict,
                                                                                                                     self.vectorsOptions, NVisibleForRes=NVisibleForRes,
                                                                                                                     sequencer=sequencer)
                
                self.visibleSpecieCount = visSpecCount
            
            # do displacement vectors on visible atoms
            if drawDisplacementVectors:
                self.renderDisplacementVectors(displacementSettings)
            
            # render trace
            if self.traceOptions.drawTraceVectors:
                self.renderTrace()
            
            if self.bondsOptions.drawBonds:
                # find bonds
                self.calculateBonds()
            
            # voronoi
            if self.voronoiOptions.displayVoronoi:
                self.renderVoronoi()
        
        # time to render
        renderTime = time.time() - renderTime
        self.logger.debug("Create actors time: %f s", renderTime)
        
        if self.parent.visible:
            addActorsTime = time.time()
            
            self.addActors()
            
            addActorsTime = time.time() - addActorsTime
            self.logger.debug("Add actors time: %f s" % addActorsTime)
        
#         for name, scalars in self.scalarsDict.iteritems():
#             assert len(scalars) == len(self.visibleAtoms)
#             f = open("%s_after.dat" % name.replace(" ", "_"), "w")
#             for tup in itertools.izip(self.visibleAtoms, scalars):
#                 f.write("%d %f\n" % tup)
#             f.close()
        
        self.actorsOptions.refresh(self.actorsDict)
        
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
    
    def voronoiNeighboursFilter(self, settings):
        """
        Voronoi neighbours filter
        
        """
        # calculate Voronoi tessellation
        status = self.calculateVoronoi()
        
        if status:
            self.logger.error("Calculate Voronoi volume failed")
            self.visibleAtoms.resize(0, refcheck=False)
            return status
        
        vor = self.voronoi
        
        # new scalars array
        scalars = np.zeros(len(self.visibleAtoms), dtype=np.float64)
        
        # full scalars array
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        # make array of neighbours
        num_nebs_array = vor.atomNumNebsArray()
        
        NVisible = filtering_c.voronoiNeighboursFilter(self.visibleAtoms, num_nebs_array, settings.minVoroNebs, settings.maxVoroNebs, 
                                                       scalars, NScalars, fullScalars, settings.filteringEnabled, NVectors, fullVectors)
        
        # update scalars/vectors dicts
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)

        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)

        # store scalars
        scalars.resize(NVisible, refcheck=False)
        self.scalarsDict["Voronoi neighbours"] = scalars
    
    def voronoiVolumeFilter(self, settings):
        """
        Voronoi volume filter
        
        """
        # calculate Voronoi tessellation
        status = self.calculateVoronoi()
        
        if status:
            self.logger.error("Calculate Voronoi volume failed")
            self.visibleAtoms.resize(0, refcheck=False)
            return status
        
        vor = self.voronoi
        
        # new scalars array
        scalars = np.zeros(len(self.visibleAtoms), dtype=np.float64)
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        # make array of volumes
        atom_volumes = vor.atomVolumesArray()
        
        NVisible = filtering_c.voronoiVolumeFilter(self.visibleAtoms, atom_volumes, settings.minVoroVol, settings.maxVoroVol, 
                                                   scalars, NScalars, fullScalars, settings.filteringEnabled, NVectors, fullVectors)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)

        # store scalars
        scalars.resize(NVisible, refcheck=False)
        self.scalarsDict["Voronoi volume"] = scalars
    
    def slipFilter(self, settings):
        """
        Slip filter
        
        """
        # input and ref
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        
        # new scalars array
        scalars = np.zeros(len(self.visibleAtoms), dtype=np.float64)
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        # call C library
        NVisible = filtering_c.slipFilter(self.visibleAtoms, scalars, inputState.pos, refState.pos, inputState.cellDims,
                                          inputState.PBC, settings.minSlip, settings.maxSlip, NScalars, fullScalars,
                                          settings.filteringEnabled, self.parent.driftCompensation, self.driftVector,
                                          NVectors, fullVectors)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)

        # store scalars
        scalars.resize(NVisible, refcheck=False)
        self.scalarsDict["Slip"] = scalars
    
    def bondOrderFilter(self, settings):
        """
        Bond order filter
        
        """
        inputState = self.pipelinePage.inputState
        
        # new scalars array
        scalarsQ4 = np.zeros(len(self.visibleAtoms), dtype=np.float64)
        scalarsQ6 = np.zeros(len(self.visibleAtoms), dtype=np.float64)
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        # num threads
        ompNumThreads = self.mainWindow.preferences.generalForm.openmpNumThreads
        
        maxBondDistance = settings.getSetting("maxBondDistance")
        filterQ4Enabled = int(settings.getSetting("filterQ4Enabled"))
        filterQ6Enabled = int(settings.getSetting("filterQ6Enabled"))
        minQ4 = int(settings.getSetting("minQ4"))
        maxQ4 = int(settings.getSetting("maxQ4"))
        minQ6 = int(settings.getSetting("minQ6"))
        maxQ6 = int(settings.getSetting("maxQ6"))
        NVisible = bond_order.bondOrderFilter(self.visibleAtoms, inputState.pos, maxBondDistance, scalarsQ4, scalarsQ6,
                                                inputState.cellDims, self.pipelinePage.PBC, NScalars, fullScalars, filterQ4Enabled,
                                                minQ4, maxQ4, filterQ6Enabled, minQ6, maxQ6, ompNumThreads, NVectors, fullVectors)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)

        # store scalars
        scalarsQ4.resize(NVisible, refcheck=False)
        scalarsQ6.resize(NVisible, refcheck=False)
        self.scalarsDict["Q4"] = scalarsQ4
        self.scalarsDict["Q6"] = scalarsQ6
    
    def acnaFilter(self, settings):
        """
        Adaptive common neighbour analysis
        
        """
        inputState = self.pipelinePage.inputState
        
        # new scalars array
        scalars = np.zeros(len(self.visibleAtoms), dtype=np.float64)
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        # number of openmp threads
        ompNumThreads = self.mainWindow.preferences.generalForm.openmpNumThreads
        
        # counter array
        counters = np.zeros(7, np.int32)
        
        maxBondDistance = settings.getSetting("maxBondDistance")
        filteringEnabled = int(settings.getSetting("filteringEnabled"))
        structureVisibility = settings.getSetting("structureVisibility")
        
        NVisible = acna.adaptiveCommonNeighbourAnalysis(self.visibleAtoms, inputState.pos, scalars, inputState.cellDims, self.pipelinePage.PBC,
                                                        NScalars, fullScalars, maxBondDistance, counters, filteringEnabled,
                                                        structureVisibility, ompNumThreads, NVectors, fullVectors)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)

        # store scalars
        scalars.resize(NVisible, refcheck=False)
        self.scalarsDict["ACNA"] = scalars
        
        # store counters
        d = {}
        for i, structure in enumerate(self.knownStructures):
            if counters[i] > 0:
                d[structure] = counters[i]
        self.structureCounterDicts["ACNA structure count"] = d
    
    def calculateVoronoi(self):
        """
        Calc voronoi tesselation
        
        """
        PBC = self.pipelinePage.PBC
        inputState = self.pipelinePage.inputState
        if self.voronoi is None:
            self.voronoi = voronoi.computeVoronoi(inputState, self.voronoiOptions, PBC)
        
        return 0
    
    def renderVoronoi(self):
        """
        Render Voronoi cells
        
        """
        inputState = self.pipelinePage.inputState
        
        status = self.calculateVoronoi()
        if status:
            return status
        
        if not len(self.visibleAtoms):
            return 2
        
        if len(self.visibleAtoms) > 2000:
            # warn that this will be slow
            msg = """<p>You are about to render a large number of Voronoi cells (%d).</p>
                     <p>This will probably be very slow!</p>
                     <p>Do you want to continue?</p>""" % len(self.visibleAtoms)
            
            reply = QtGui.QMessageBox.question(self.mainWindow, "Message", msg, QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.Yes)
            
            if reply == QtGui.QMessageBox.No:
                return
        
        # POV-RAY file
        voroFile = os.path.join(self.mainWindow.tmpDirectory, "pipeline%d_voro%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID)))
        
        # get actors for vis atoms only!
        renderVoronoi.getActorsForVoronoiCells(self.visibleAtoms, inputState, self.voronoi, 
                                               self.colouringOptions, self.voronoiOptions, self.actorsDict, 
                                               voroFile, self.scalarsDict, log=self.log)
    
    def calculateBonds(self):
        """
        Calculate and render bonds.
        
        """
        if not len(self.visibleAtoms):
            return 1
        
        self.logger.info("  Calculating bonds")
                
        inputState = self.pipelinePage.inputState
        specieList = inputState.specieList
        NSpecies = len(specieList)
        
        bondMinArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        bondMaxArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        bondSpecieCounter = np.zeros((NSpecies, NSpecies), dtype=np.int32)
        
        # construct bonds array
        calcBonds = False
        maxBond = -1
        drawList = []
        for i in xrange(self.bondsOptions.bondsList.count()):
            item = self.bondsOptions.bondsList.item(i)
            
            if item.checkState() == QtCore.Qt.Checked:
                syma = item.syma
                symb = item.symb
                
                # check if in current specie list and if so what indexes
                if syma in specieList:
                    indexa = inputState.getSpecieIndex(syma)
                else:
                    continue
                
                if symb in specieList:
                    indexb = inputState.getSpecieIndex(symb)
                else:
                    continue
                
                bondDict = elements.bondDict
                
                if syma in bondDict:
                    d = bondDict[syma]
                    
                    if symb in d:
                        bondMin, bondMax = d[symb]
                        
                        bondMinArray[indexa][indexb] = bondMin
                        bondMinArray[indexb][indexa] = bondMin
                        
                        bondMaxArray[indexa][indexb] = bondMax
                        bondMaxArray[indexb][indexa] = bondMax
                        
                        if bondMax > maxBond:
                            maxBond = bondMax
                        
                        if bondMax > 0:
                            calcBonds = True
                        
                        drawList.append("%s-%s" % (syma, symb))
                        
                        self.logger.info("    Pair: %s - %s; bond range: %f -> %f", syma, symb, bondMin, bondMax)
        
        if not calcBonds:
            self.logger.info("    No bonds to calculate")
            return 1
        
        # arrays for results
        maxBondsPerAtom = 50
        size = int(self.NVis * maxBondsPerAtom / 2)
        bondArray = np.empty(size, np.int32)
        NBondsArray = np.zeros(self.NVis, np.int32)
        bondVectorArray = np.empty(3 * size, np.float64)
        
        status = bonds_c.calculateBonds(self.visibleAtoms, inputState.pos, inputState.specie, len(specieList), bondMinArray, bondMaxArray,
                                        maxBond, maxBondsPerAtom, inputState.cellDims, self.pipelinePage.PBC, bondArray, NBondsArray,
                                        bondVectorArray, bondSpecieCounter)
        
        if status:
            self.logger.error("    Error in bonds clib (%d)", status)
            
            if status == 1:
                self.mainWindow.displayError("Max bonds per atom exceeded!\n\nThis would suggest you bond range is too big!")
            
            return 1
        
        # total number of bonds
        NBondsTotal = np.sum(NBondsArray)
        self.logger.info("    Total number of bonds: %d (x2 for actors)", NBondsTotal)
        
        # resize bond array
        bondArray.resize(NBondsTotal)
        bondVectorArray.resize(NBondsTotal * 3)
        
        # specie counters
        for i in xrange(NSpecies):
            syma = specieList[i]
            
            for j in xrange(i, NSpecies):
                symb = specieList[j]
                
                # check if selected
                pairStr = "%s-%s" % (syma, symb)
                pairStr2 = "%s-%s" % (symb, syma)
                
                if pairStr in drawList or pairStr2 in drawList:
                    NBondsPair = bondSpecieCounter[i][j]
                    if i != j:
                        NBondsPair += bondSpecieCounter[j][i]
                    
                    self.logger.info("      %d %s - %s bonds", NBondsPair, syma, symb)
        
        # draw bonds
        if NBondsTotal > 0:
            # pov file for bonds
            povfile = "pipeline%d_bonds%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
            
            renderBonds.renderBonds(self.visibleAtoms, self.mainWindow, self.pipelinePage, self.actorsDict, self.colouringOptions, 
                                    povfile, self.scalarsDict, bondArray, NBondsArray, bondVectorArray, self.bondsOptions)
        
        return 0
    
    def addScalarBar(self):
        """
        Add scalar bar.
        
        """
        if self.scalarBar_white_bg is not None and self.parent.scalarBarButton.isChecked() and not self.parent.filterTab.scalarBarAdded:
            for rw in self.rendererWindows:
                if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                    # which scalar bar to add
                    if rw.blackBackground:
                        scalarBar = self.scalarBar_black_bg
                    else:
                        scalarBar = self.scalarBar_white_bg
                    
                    rw.vtkRen.AddActor2D(scalarBar)
                    rw.vtkRenWinInteract.ReInitialize()
            
            self.parent.filterTab.scalarBarAdded = True
            self.scalarBarAdded = True
        
        return self.scalarBarAdded
    
    def hideScalarBar(self):
        """
        Remove scalar bar.
        
        """
        if self.scalarBarAdded:
            for rw in self.rendererWindows:
                if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                    # which scalar bar was added
                    if rw.blackBackground:
                        scalarBar = self.scalarBar_black_bg
                    else:
                        scalarBar = self.scalarBar_white_bg
                    
                    rw.vtkRen.RemoveActor2D(scalarBar)
                    rw.vtkRenWinInteract.ReInitialize()
            
            self.parent.filterTab.scalarBarAdded = False
            self.scalarBarAdded = False
    
    def makeFullScalarsArray(self):
        """
        Combine scalars array into one big array for passing to C
        
        """
        self.logger.debug("Making full scalars array (N=%d)", len(self.scalarsDict) + len(self.latticeScalarsDict))
        
        scalarsList = []
        for name, scalars in self.scalarsDict.iteritems():
            self.logger.debug("  Adding '%s' scalars", name)
            scalarsList.append(scalars)
            assert len(scalars) == len(self.visibleAtoms), "Wrong length for scalars: '%s'" % name
            
#             f = open("%s_before.dat" % name.replace(" ", "_"), "w")
#             for tup in itertools.izip(self.visibleAtoms, scalars):
#                 f.write("%d %f\n" % tup)
#             f.close()
        
        for name, scalars in self.latticeScalarsDict.iteritems():
            self.logger.debug("  Adding '%s' scalars (Lattice)", name)
            scalarsList.append(scalars)
            assert len(scalars) == len(self.visibleAtoms), "Wrong length for scalars: '%s' (Lattice)" % name
        
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
            assert vectors.shape == (len(self.visibleAtoms), 3), "Shape wrong for vectors array '%s': %r != %r" % (name, vectors.shape, (len(self.visibleAtoms), 3))
        
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
    
    def filterSpecie(self, settings):
        """
        Filter by specie
        
        """
        visibleSpecieList = settings.getVisibleSpecieList()
        specieList = self.pipelinePage.inputState.specieList
        
        # make visible specie array
        visSpecArray = []
        for i, sym in enumerate(specieList):
            if sym in visibleSpecieList:
                visSpecArray.append(i)
        visSpecArray = np.asarray(visSpecArray, dtype=np.int32)
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        NVisible = filtering_c.specieFilter(self.visibleAtoms, visSpecArray, self.pipelinePage.inputState.specie, NScalars, fullScalars, 
                                            NVectors, fullVectors)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)

        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def renderDisplacementVectors(self, settings):
        """
        Compute and render displacement vectors for visible atoms
        
        """
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        
        if inputState.NAtoms != refState.NAtoms:
            self.logger.warning("Cannot render displacement vectors with different numbers of input and reference atoms: skipping")
            return
        
        # number of visible atoms
        NVisible = len(self.visibleAtoms)
        
        # draw displacement vectors
        self.logger.debug("Drawing displacement vectors")
        
        # calculate vectors
        bondVectorArray = np.empty(3 * NVisible, np.float64)
        drawBondVector = np.empty(NVisible, np.int32)
        numBonds = bonds_c.calculateDisplacementVectors(self.visibleAtoms, inputState.pos, refState.pos, 
                                                        refState.cellDims, self.pipelinePage.PBC, bondVectorArray,
                                                        drawBondVector)
        
        self.logger.debug("  Number of displacement vectors to draw = %d (/ %d)", numBonds, len(self.visibleAtoms))
        
        # pov file for bonds
        povfile = "pipeline%d_dispvects%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
        
        # draw displacement vectors as bonds
        renderBonds.renderDisplacementVectors(self.visibleAtoms, self.mainWindow, self.pipelinePage, self.actorsDict, 
                                              self.colouringOptions, povfile, self.scalarsDict, numBonds, bondVectorArray, 
                                              drawBondVector, settings)
    
    def displacementFilter(self, settings):
        """
        Displacement filter
        
        """
        # only run displacement filter if input and reference NAtoms are the same
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        
        if inputState.NAtoms != refState.NAtoms:
            self.logger.warning("Cannot run displacement filter with different numbers of input and reference atoms: skipping this filter list")
            #TODO: display warning too
            self.visibleAtoms.resize(0, refcheck=False)
        
        else:
            # new scalars array
            scalars = np.zeros(len(self.visibleAtoms), dtype=np.float64)
            
            # old scalars arrays (resize as appropriate)
            NScalars, fullScalars = self.makeFullScalarsArray()
            
            # full vectors array
            NVectors, fullVectors = self.makeFullVectorsArray()
            
            # run displacement filter
            minDisplacement = settings.getSetting("minDisplacement")
            maxDisplacement = settings.getSetting("maxDisplacement")
            filteringEnabled = int(settings.getSetting("filteringEnabled"))
            NVisible = filtering_c.displacementFilter(self.visibleAtoms, scalars, inputState.pos, refState.pos, refState.cellDims, 
                                                      self.pipelinePage.PBC, minDisplacement, maxDisplacement, NScalars, fullScalars,
                                                      filteringEnabled, self.parent.driftCompensation, self.driftVector, NVectors,
                                                      fullVectors)
            
            # update scalars dict
            self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
            self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
            
            # resize visible atoms
            self.visibleAtoms.resize(NVisible, refcheck=False)
            
            # store scalars
            scalars.resize(NVisible, refcheck=False)
            self.scalarsDict["Displacement"] = scalars
    
    def renderTrace(self):
        """
        Render trace vectors
        
        """
        self.logger.debug("Computing trace...")
        
        inputState = self.pipelinePage.inputState
        refState = self.pipelinePage.refState
        settings = self.traceOptions
        NVisible = len(self.visibleAtoms)
        
        # trace
        if self.previousPosForTrace is None:
            self.previousPosForTrace = refState.pos
        
        if len(self.previousPosForTrace) == len(inputState.pos):
            # trace atoms that have moved...
            # first we calculate displacement vectors between current and last position
            bondVectorArray = np.empty(3 * NVisible, np.float64)
            drawTrace = np.empty(NVisible, np.int32)
            numBonds = bonds_c.calculateDisplacementVectors(self.visibleAtoms, inputState.pos, self.previousPosForTrace, 
                                                            refState.cellDims, self.pipelinePage.PBC, bondVectorArray,
                                                            drawTrace)
            
            # pov file for trace
            povfile = "pipeline%d_trace%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
            
            # render trace vectors
            self.traceDict = renderBonds.renderTraceVectors(self.visibleAtoms, self.mainWindow, self.pipelinePage, 
                                                            self.actorsDict, self.colouringOptions, povfile, 
                                                            self.scalarsDict, numBonds, bondVectorArray, drawTrace, 
                                                            settings, self.traceDict)
            
#             self.traceDict = renderBonds.renderTraceVectors2(self.visibleAtoms, self.mainWindow, self.pipelinePage, self.actorsCollection, 
#                                                              self.colouringOptions, povfile, self.scalarsDict, numBonds, 
#                                                              self.previousPosForTrace, drawTrace, settings, self.traceDict)
        
        else:
            self.logger.warning("Cannot compute trace with differing number of atoms between steps (undefined behaviour!)")
        
        # store pos for next step
        self.previousPosForTrace = copy.deepcopy(inputState.pos)
    
    def atomIndexFilter(self, settings):
        """
        Atom index filter
        
        """
        # input string
        text = settings.getSetting("filterString")
        self.logger.debug("Atom ID raw text: '%s'", text)
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
    
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        if not text:
            # return no visible atoms if input string was empty
            self.logger.warning("No visible atoms specified in AtomID filter")
            NVisible = 0
        
        else:
            # parse text
            array = [val for val in text.split(",") if val]
            num = len(array)
            rangeArray = np.empty((num, 2), np.int32)
            for i, item in enumerate(array):
                if "-" in item:
                    values = [val for val in item.split("-") if val]
                    minval = int(values[0])
                    if len(values) == 1:
                        maxval = minval
                    else:
                        maxval = int(values[1])
                else:
                    minval = maxval = int(item)
            
                self.logger.debug("  %d: %d -> %d", i, minval, maxval)
                rangeArray[i][0] = minval
                rangeArray[i][1] = maxval
        
            # input state
            lattice = self.pipelinePage.inputState
        
            # run displacement filter
            NVisible = filtering_c.atomIndexFilter(self.visibleAtoms, lattice.atomID, rangeArray, 
                                                   NScalars, fullScalars, NVectors, fullVectors)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def cropFilter(self, settings):
        """
        Crop lattice
        
        """
        # settings
        xmin = settings.getSetting("xmin")
        xmax = settings.getSetting("xmax")
        ymin = settings.getSetting("ymin")
        ymax = settings.getSetting("ymax")
        zmin = settings.getSetting("zmin")
        zmax = settings.getSetting("zmax")
        xEnabled = int(settings.getSetting("xEnabled"))
        yEnabled = int(settings.getSetting("yEnabled"))
        zEnabled = int(settings.getSetting("zEnabled"))
        invertSelection = int(settings.getSetting("invertSelection"))
        
        if self.parent.defectFilterSelected:
            inp = self.pipelinePage.inputState
            ref = self.pipelinePage.refState
            
            result = filtering_c.cropDefectsFilter(self.interstitials, self.vacancies, self.antisites, self.onAntisites, self.splitInterstitials,
                                                   inp.pos, ref.pos, xmin, xmax, ymin, ymax, zmin, zmax, xEnabled, yEnabled, zEnabled,
                                                   invertSelection)
            
            # unpack
            NInt, NVac, NAnt, NSplit = result
            self.vacancies.resize(NVac, refcheck=False)
            self.interstitials.resize(NInt, refcheck=False)
            self.antisites.resize(NAnt, refcheck=False)
            self.onAntisites.resize(NAnt, refcheck=False)
            self.splitInterstitials.resize(NSplit*3, refcheck=False)
        
        else:
            lattice = self.pipelinePage.inputState
            
            # old scalars arrays (resize as appropriate)
            NScalars, fullScalars = self.makeFullScalarsArray()
            
            # full vectors array
            NVectors, fullVectors = self.makeFullVectorsArray()
            
            NVisible = filtering_c.cropFilter(self.visibleAtoms, lattice.pos, xmin, xmax, ymin, ymax, zmin, zmax, xEnabled,
                                              yEnabled, zEnabled, invertSelection, NScalars, fullScalars, NVectors, fullVectors)
            
            # update scalars dict
            self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
            self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
    
            # resize visible atoms
            self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def cropSphereFilter(self, settings):
        """
        Crop sphere filter.
        
        """
        lattice = self.pipelinePage.inputState
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        xCentre = settings.getSetting("xCentre")
        yCentre = settings.getSetting("yCentre")
        zCentre = settings.getSetting("zCentre")
        radius = settings.getSetting("radius")
        invertSelection = int(settings.getSetting("invertSelection"))
        NVisible = filtering_c.cropSphereFilter(self.visibleAtoms, lattice.pos, xCentre, yCentre, zCentre, 
                                                radius, lattice.cellDims, self.pipelinePage.PBC, invertSelection, 
                                                NScalars, fullScalars, NVectors, fullVectors)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)

        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def sliceFilter(self, settings):
        """
        Slice filter.
        
        """
        if self.parent.defectFilterSelected:
            inp = self.pipelinePage.inputState
            ref = self.pipelinePage.refState
            
            self.logger.debug("Calling sliceDefectsFilter C function")
            result = filtering_c.sliceDefectsFilter(self.interstitials, self.vacancies, self.antisites, self.onAntisites, self.splitInterstitials,
                                                    inp.pos, ref.pos, settings.x0, settings.y0, settings.z0, settings.xn, settings.yn, settings.zn,
                                                    settings.invert)
            
            # unpack
            NInt, NVac, NAnt, NSplit = result
            self.vacancies.resize(NVac, refcheck=False)
            self.interstitials.resize(NInt, refcheck=False)
            self.antisites.resize(NAnt, refcheck=False)
            self.onAntisites.resize(NAnt, refcheck=False)
            self.splitInterstitials.resize(NSplit*3, refcheck=False)
        
        else:
            lattice = self.pipelinePage.inputState
            
            # old scalars arrays (resize as appropriate)
            NScalars, fullScalars = self.makeFullScalarsArray()
            
            # full vectors array
            NVectors, fullVectors = self.makeFullVectorsArray()
            
            self.logger.debug("Calling sliceFilter C function")
            NVisible = filtering_c.sliceFilter(self.visibleAtoms, lattice.pos, settings.x0, settings.y0, settings.z0, 
                                               settings.xn, settings.yn, settings.zn, settings.invert, NScalars, fullScalars, 
                                               NVectors, fullVectors)
            
            # update scalars dict
            self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
            self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
    
            # resize visible atoms
            self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def chargeFilter(self, settings):
        """
        Charge filter.
        
        """
        lattice = self.pipelinePage.inputState
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        self.logger.debug("Calling chargeFilter C function")
        minCharge = settings.getSetting("minCharge")
        maxCharge = settings.getSetting("maxCharge")
        NVisible = filtering_c.chargeFilter(self.visibleAtoms, lattice.charge, minCharge, maxCharge, 
                                            NScalars, fullScalars, NVectors, fullVectors)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)

        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def pointDefectFilter(self, settings, acnaArray=None):
        """
        Point defects filter
        
        """
        inputLattice = self.pipelinePage.inputState
        refLattice = self.pipelinePage.refState
        
        if settings.useAcna:
            self.logger.debug("Computing ACNA from point defects filter...")
            
            # dummy visible atoms and scalars arrays
            visAtoms = np.arange(inputLattice.NAtoms, dtype=np.int32)
            acnaArray = np.empty(inputLattice.NAtoms, np.float64)
            NScalars = 0
            fullScalars = np.empty(NScalars, np.float64)
            NVectors = 0
            fullVectors = np.empty(NVectors, np.float64)
            structVis = np.ones(len(self.knownStructures), np.int32)
            
            # number of threads
            ompNumThreads = self.mainWindow.preferences.generalForm.openmpNumThreads
            
            # counter array
            counters = np.zeros(7, np.int32)
            
            acna.adaptiveCommonNeighbourAnalysis(visAtoms, inputLattice.pos, acnaArray, inputLattice.cellDims,
                                                 self.pipelinePage.PBC, NScalars, fullScalars, 
                                                 settings.acnaMaxBondDistance, counters, 0, structVis, ompNumThreads, 
                                                 NVectors, fullVectors) 
            
            # store counters
            d = {}
            for i, structure in enumerate(self.knownStructures):
                if counters[i] > 0:
                    d[structure] = counters[i]
            
            self.logger.debug("  %r", d)
        
        elif acnaArray is None or len(acnaArray) != inputLattice.NAtoms:
            acnaArray = np.empty(0, np.float64)
        
        # set up arrays
        interstitials = np.empty(inputLattice.NAtoms, np.int32)
        
        if settings.identifySplitInts:
            splitInterstitials = np.empty(inputLattice.NAtoms, np.int32)
        
        else:
            splitInterstitials = np.empty(0, np.int32)
                
        vacancies = np.empty(refLattice.NAtoms, np.int32)
        
        antisites = np.empty(refLattice.NAtoms, np.int32)
        onAntisites = np.empty(refLattice.NAtoms, np.int32)
        
        # set up excluded specie arrays
        visibleSpecieList = settings.getVisibleSpecieList()
        
        exclSpecsInput = []
        for i, spec in enumerate(inputLattice.specieList):
            if spec not in visibleSpecieList:
                exclSpecsInput.append(i)
        exclSpecsInput = np.asarray(exclSpecsInput, dtype=np.int32)
        
        exclSpecsRef = []
        for i, spec in enumerate(refLattice.specieList):
            if spec not in visibleSpecieList:
                exclSpecsRef.append(i)
        exclSpecsRef = np.asarray(exclSpecsRef, dtype=np.int32)
        
        # specie counter arrays
        vacSpecCount = np.zeros( len(refLattice.specieList), np.int32 )
        intSpecCount = np.zeros( len(inputLattice.specieList), np.int32 )
        antSpecCount = np.zeros( len(refLattice.specieList), np.int32 )
        onAntSpecCount = np.zeros( (len(refLattice.specieList), len(inputLattice.specieList)), np.int32 )
        splitIntSpecCount = np.zeros( (len(inputLattice.specieList), len(inputLattice.specieList)), np.int32 )
        
        NDefectsByType = np.zeros(6, np.int32)
        
        if settings.findClusters:
            defectCluster = np.empty(inputLattice.NAtoms + refLattice.NAtoms, np.int32)
        
        else:
            defectCluster = np.empty(0, np.int32)
        
        # call C library
        status = defects_c.findDefects(settings.showVacancies, settings.showInterstitials, settings.showAntisites, NDefectsByType, vacancies, 
                                       interstitials, antisites, onAntisites, exclSpecsInput, exclSpecsRef, inputLattice.NAtoms, inputLattice.specieList,
                                       inputLattice.specie, inputLattice.pos, refLattice.NAtoms, refLattice.specieList, refLattice.specie, 
                                       refLattice.pos, refLattice.cellDims, self.pipelinePage.PBC, settings.vacancyRadius,
                                       settings.findClusters, settings.neighbourRadius, defectCluster, vacSpecCount, intSpecCount, antSpecCount,
                                       onAntSpecCount, splitIntSpecCount, settings.minClusterSize, settings.maxClusterSize, splitInterstitials, 
                                       settings.identifySplitInts, self.parent.driftCompensation, self.driftVector, acnaArray, settings.acnaStructureType)
        
        # summarise
        NDef = NDefectsByType[0]
        NVac = NDefectsByType[1]
        NInt = NDefectsByType[2]
        NAnt = NDefectsByType[3]
        NSplit = NDefectsByType[5]
        vacancies.resize(NVac)
        interstitials.resize(NInt)
        antisites.resize(NAnt)
        onAntisites.resize(NAnt)
        splitInterstitials.resize(NSplit*3)
        
        # report counters
        self.logger.info("Found %d defects", NDef)
        
        if settings.showVacancies:
            self.logger.info("  %d vacancies", NVac)
            for i in xrange(len(refLattice.specieList)):
                self.logger.info("    %d %s vacancies", vacSpecCount[i], refLattice.specieList[i])
        
        if settings.showInterstitials:
            self.logger.info("  %d interstitials", NInt + NSplit)
            for i in xrange(len(inputLattice.specieList)):
                self.logger.info("    %d %s interstitials", intSpecCount[i], inputLattice.specieList[i])
        
            if settings.identifySplitInts:
                self.logger.info("    %d split interstitials", NSplit)
                for i in xrange(len(inputLattice.specieList)):
                    for j in xrange(i, len(inputLattice.specieList)):
                        if j == i:
                            N = splitIntSpecCount[i][j]
                        else:
                            N = splitIntSpecCount[i][j] + splitIntSpecCount[j][i]
                        self.logger.info("      %d %s - %s split interstitials", N, inputLattice.specieList[i], inputLattice.specieList[j])
        
        if settings.showAntisites:
            self.logger.info("  %d antisites", NAnt)
            for i in xrange(len(refLattice.specieList)):
                for j in xrange(len(inputLattice.specieList)):
                    if inputLattice.specieList[j] == refLattice.specieList[i]:
                        continue
                    
                    self.logger.info("    %d %s on %s antisites", onAntSpecCount[i][j], inputLattice.specieList[j], refLattice.specieList[i])
        
        if settings.identifySplitInts:
            self.logger.info("Splint interstitial analysis")
            
            PBC = self.pipelinePage.PBC
            cellDims = inputLattice.cellDims
            
            for i in xrange(NSplit):
                ind1 = splitInterstitials[3*i+1]
                ind2 = splitInterstitials[3*i+2]
                
                pos1 = inputLattice.pos[3*ind1:3*ind1+3]
                pos2 = inputLattice.pos[3*ind2:3*ind2+3]
                
                sepVec = vectors.separationVector(pos1, pos2, cellDims, PBC)
                norm = vectors.normalise(sepVec)
                
                self.logger.info("  Orientation of split int %d: (%.3f %.3f %.3f)", i, norm[0], norm[1], norm[2])
        
        # sort clusters here
        self.clusterList = []
        clusterList = self.clusterList
        if settings.findClusters:
            NClusters = NDefectsByType[4]
            
            defectCluster.resize(NDef)
            
            # build cluster lists
            for i in xrange(NClusters):
                clusterList.append(clusters.DefectCluster())
            
            # add atoms to cluster lists
            clusterIndexMapper = {}
            count = 0
            for i in xrange(NVac):
                atomIndex = vacancies[i]
                clusterIndex = defectCluster[i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                
                clusterList[clusterListIndex].vacancies.append(atomIndex)
                clusterList[clusterListIndex].vacAsIndex.append(i)
            
            for i in xrange(NInt):
                atomIndex = interstitials[i]
                clusterIndex = defectCluster[NVac + i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                
                clusterList[clusterListIndex].interstitials.append(atomIndex)
            
            for i in xrange(NAnt):
                atomIndex = antisites[i]
                atomIndex2 = onAntisites[i]
                clusterIndex = defectCluster[NVac + NInt + i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                
                clusterList[clusterListIndex].antisites.append(atomIndex)
                clusterList[clusterListIndex].onAntisites.append(atomIndex2)
            
            for i in xrange(NSplit):
                clusterIndex = defectCluster[NVac + NInt + NAnt + i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                
                atomIndex = splitInterstitials[3*i]
                clusterList[clusterListIndex].splitInterstitials.append(atomIndex)
                
                atomIndex = splitInterstitials[3*i+1]
                clusterList[clusterListIndex].splitInterstitials.append(atomIndex)
                
                atomIndex = splitInterstitials[3*i+2]
                clusterList[clusterListIndex].splitInterstitials.append(atomIndex)
        
        # draw displacement vectors
        if settings.drawVectorsCheck.isEnabled() and settings.drawDisplacementVectors:
            self.logger.debug("Drawing displacement vectors for interstitials (%d)", NInt)
            
            # need to make a unique list for interstitials and split intersitials and antisites
            
            
            # calculate vectors
            bondVectorArray = np.empty(3 * NInt, np.float64)
            drawTrace = np.empty(NInt, np.int32)
            numBonds = bonds_c.calculateDisplacementVectors(interstitials, inputLattice.pos, refLattice.pos, 
                                                          refLattice.cellDims, self.pipelinePage.PBC, bondVectorArray,
                                                          drawTrace)
            
            self.logger.debug("  Number of interstitial displacement vectors to draw = %d (/ %d)", numBonds, NInt)
            
            # pov file for bonds
            povfile = "pipeline%d_intdispvects%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
            
            # draw displacement vectors as bonds
            renderBonds.renderDisplacementVectors(interstitials, self.mainWindow, self.pipelinePage, self.actorsDict, 
                                                  self.colouringOptions, povfile, self.scalarsDict, numBonds, bondVectorArray, 
                                                  drawTrace, settings)
        
        return interstitials, vacancies, antisites, onAntisites, splitInterstitials
    
    def pointDefectFilterCalculateClusterVolumes(self, settings):
        """
        Calculate volumes of clusters
        
        """
        self.logger.debug("Calculating volumes of defect clusters")
        self.logger.warning("If your clusters cross PBCs this may or may not give correct volumes; please test and let me know")
        
        inputLattice = self.pipelinePage.inputState
        refLattice = self.pipelinePage.refState
        clusterList = self.clusterList
        
        if settings.calculateVolumesHull:
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
                                                         self.pipelinePage.PBC, appliedPBCs, settings.neighbourRadius)
                    
                    cluster.volume, cluster.facetArea = clusters.findConvexHullVolume(NDefects, clusterPos)
                
                self.logger.info("  Cluster %d (%d defects)", count, cluster.getNDefects())
                if cluster.facetArea is not None:
                    self.logger.info("    volume is %f; facet area is %f", cluster.volume, cluster.facetArea)
                
                count += 1
        
        elif settings.calculateVolumesVoro:
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
    
    def pointDefectFilterDrawHulls(self, settings, hullPovFile):
        """
        Draw convex hulls around defect volumes
        
        """
        # PBCs
        PBC = self.pipelinePage.PBC
        if PBC[0] or PBC[1] or PBC[2]:
            PBCFlag = True
        
        else:
            PBCFlag = False
        
        inputLattice = self.pipelinePage.inputState
        refLattice = self.pipelinePage.refState
        clusterList = self.clusterList
        
        actorsDictLocal = {}
        
        # loop over clusters
        for clusterIndex, cluster in enumerate(clusterList):
            appliedPBCs = np.zeros(7, np.int32)
            clusterPos = cluster.makeClusterPos(inputLattice, refLattice)
            NDefects = len(clusterPos) / 3
            
            # determine if the cluster crosses PBCs
            clusters_c.prepareClusterToDrawHulls(NDefects, clusterPos, inputLattice.cellDims, 
                                                 self.pipelinePage.PBC, appliedPBCs, settings.neighbourRadius)
            
            facets = None
            if NDefects > 3:
                facets = clusters.findConvexHullFacets(NDefects, clusterPos)
            
            elif NDefects == 3:
                facets = []
                facets.append([0, 1, 2])
            
            # now render
            if facets is not None:
                # make sure not facets more than neighbour rad from cell
                facets = clusters.checkFacetsPBCs(facets, clusterPos, 2.0 * settings.neighbourRadius, self.pipelinePage.PBC, 
                                                  inputLattice.cellDims)
                
                renderer.getActorsForHullFacets(facets, clusterPos, self.mainWindow, actorsDictLocal, 
                                                settings, "Defects {0}".format(clusterIndex))
                
                # write povray file too
                renderer.writePovrayHull(facets, clusterPos, self.mainWindow, hullPovFile, settings)
            
            # handle PBCs
            if NDefects > 1 and PBCFlag:
                count = 0
                while max(appliedPBCs) > 0:
                    tmpClusterPos = copy.deepcopy(clusterPos)
                    clusters.applyPBCsToCluster(tmpClusterPos, inputLattice.cellDims, appliedPBCs)
                    
                    # get facets
                    facets = None
                    if NDefects > 3:
                        facets = clusters.findConvexHullFacets(NDefects, tmpClusterPos)
                    
                    elif NDefects == 3:
                        facets = []
                        facets.append([0, 1, 2])
                    
                    # render
                    if facets is not None:
                        #TODO: make sure not facets more than neighbour rad from cell
                        facets = clusters.checkFacetsPBCs(facets, tmpClusterPos, 2.0 * settings.neighbourRadius, 
                                                          self.pipelinePage.PBC, inputLattice.cellDims)
                        
                        renderer.getActorsForHullFacets(facets, tmpClusterPos, self.mainWindow, actorsDictLocal, 
                                                        settings, "Defects {0} (PBC {1})".format(clusterIndex, count))
                        
                        # write povray file too
                        renderer.writePovrayHull(facets, tmpClusterPos, self.mainWindow, hullPovFile, settings)
                        
                        count += 1
        
        self.actorsDict["Defect clusters"] = actorsDictLocal
        
    def clusterFilter(self, settings, PBC=None, minSize=None, maxSize=None, nebRad=None):
        """
        Run the cluster filter
        
        """
        lattice = self.pipelinePage.inputState
        
        atomCluster = np.empty(len(self.visibleAtoms), np.int32)
        result = np.empty(2, np.int32)
        
        if PBC is not None and len(PBC) == 3:
            pass
        else:
            PBC = self.pipelinePage.PBC
        
        if minSize is None:
            minSize = settings.getSetting("minClusterSize")
        
        if maxSize is None:
            maxSize = settings.getSetting("maxClusterSize")
        
        if nebRad is None:
            nebRad = settings.getSetting("neighbourRadius")
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        clusters_c.findClusters(self.visibleAtoms, lattice.pos, atomCluster, nebRad, lattice.cellDims, PBC, 
                                minSize, maxSize, result, NScalars, fullScalars, NVectors, fullVectors)
        
        NVisible = result[0]
        NClusters = result[1]
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
        
        self.visibleAtoms.resize(NVisible, refcheck=False)
        atomCluster.resize(NVisible, refcheck=False)
        
        # store cluster indexes as scalars
#         self.scalarsDict["Cluster"] = np.asarray(atomCluster, dtype=np.float64)
        
        # build cluster lists
        self.clusterList = []
        for i in xrange(NClusters):
            self.clusterList.append(clusters.AtomCluster())
        
        # add atoms to cluster lists
        clusterIndexMapper = {}
        count = 0
        for i in xrange(NVisible):
            atomIndex = self.visibleAtoms[i]
            clusterIndex = atomCluster[i]
            
            if clusterIndex not in clusterIndexMapper:
                clusterIndexMapper[clusterIndex] = count
                count += 1
            
            clusterListIndex = clusterIndexMapper[clusterIndex]
            
            self.clusterList[clusterListIndex].indexes.append(atomIndex)
    
    def clusterFilterDrawHulls(self, settings, hullPovFile):
        """
        Draw hulls around filters.
        
        If the clusterList was created using PBCs we need to recalculate each 
        cluster without PBCs.
        
        """
        PBC = self.pipelinePage.PBC
        if PBC[0] or PBC[1] or PBC[2]:
            self.clusterFilterDrawHullsWithPBCs(settings, hullPovFile)
        
        else:
            self.clusterFilterDrawHullsNoPBCs(settings, hullPovFile)
    
    def clusterFilterDrawHullsNoPBCs(self, settings, hullPovFile):
        """
        SHOULD BE ABLE TO GET RID OF THIS AND JUST USE PBCs ONE
        
        """
        lattice = self.pipelinePage.inputState
        clusterList = self.clusterList
        
        # draw them as they are
        actorsDictLocal = {}
        for clusterIndex, cluster in enumerate(clusterList):
            # first make pos array for this cluster
            clusterPos = np.empty(3 * len(cluster), np.float64)
            for i in xrange(len(cluster)):
                index = cluster[i]
                
                clusterPos[3*i] = lattice.pos[3*index]
                clusterPos[3*i+1] = lattice.pos[3*index+1]
                clusterPos[3*i+2] = lattice.pos[3*index+2]
            
            # now get convex hull
            if len(cluster) == 3:
                facets = []
                facets.append([0, 1, 2])
            
            elif len(cluster) == 2:
                # draw bond
                continue
            
            elif len(cluster) < 2:
                continue
            
            else:
                facets = clusters.findConvexHullFacets(len(cluster), clusterPos)
            
            # now render
            if facets is not None:
                renderer.getActorsForHullFacets(facets, clusterPos, self.mainWindow, actorsDictLocal, 
                                                settings, "Clusters {0}".format(clusterIndex))
                
                # write povray file too
                renderer.writePovrayHull(facets, clusterPos, self.mainWindow, hullPovFile, settings)
        
        self.actorsDict["Clusters"] = actorsDictLocal
    
    def clusterFilterDrawHullsWithPBCs(self, settings, hullPovFile):
        """
        
        
        """
        lattice = self.pipelinePage.inputState
        clusterList = self.clusterList
        
        actorsDictLocal = {}
        for clusterIndex, cluster in enumerate(clusterList):
            appliedPBCs = np.zeros(7, np.int32)
            clusterPos = np.empty(3 * len(cluster), np.float64)
            for i in xrange(len(cluster)):
                index = cluster[i]
                
                clusterPos[3*i] = lattice.pos[3*index]
                clusterPos[3*i+1] = lattice.pos[3*index+1]
                clusterPos[3*i+2] = lattice.pos[3*index+2]
            
            neighbourRadius = settings.getSetting("neighbourRadius")
            
            clusters_c.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, 
                                                 self.pipelinePage.PBC, appliedPBCs, neighbourRadius)
            
            facets = None
            if len(cluster) > 3:
                facets = clusters.findConvexHullFacets(len(cluster), clusterPos)
            
            elif len(cluster) == 3:
                facets = []
                facets.append([0, 1, 2])
            
            # now render
            if facets is not None:
                #TODO: make sure not facets more than neighbour rad from cell
                facets = clusters.checkFacetsPBCs(facets, clusterPos, 2.0 * neighbourRadius, self.pipelinePage.PBC, lattice.cellDims)
                
                renderer.getActorsForHullFacets(facets, clusterPos, self.mainWindow, actorsDictLocal, 
                                                settings, "Cluster {0}".format(clusterIndex))
                
                # write povray file too
                renderer.writePovrayHull(facets, clusterPos, self.mainWindow, hullPovFile, settings)
            
            # handle PBCs
            if len(cluster) > 1:
                count = 0
                while max(appliedPBCs) > 0:
                    tmpClusterPos = copy.deepcopy(clusterPos)
                    clusters.applyPBCsToCluster(tmpClusterPos, lattice.cellDims, appliedPBCs)
                    
                    # get facets
                    facets = None
                    if len(cluster) > 3:
                        facets = clusters.findConvexHullFacets(len(cluster), tmpClusterPos)
                    
                    elif len(cluster) == 3:
                        facets = []
                        facets.append([0, 1, 2])
                    
                    # render
                    if facets is not None:
                        #TODO: make sure not facets more than neighbour rad from cell
                        facets = clusters.checkFacetsPBCs(facets, tmpClusterPos, 2.0 * neighbourRadius, self.pipelinePage.PBC, lattice.cellDims)
                        
                        renderer.getActorsForHullFacets(facets, tmpClusterPos, self.mainWindow, actorsDictLocal, settings,
                                                        "Cluster {0} (PBC {1})".format(clusterIndex, count))
                         
                        count += 1
                        
                        # write povray file too
                        renderer.writePovrayHull(facets, tmpClusterPos, self.mainWindow, hullPovFile, settings)
        
        self.actorsDict["Clusters"] = actorsDictLocal
    
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
    
    def coordinationNumberFilter(self, filterSettings):
        """
        Coordination number filter.
        
        """
        inputState = self.pipelinePage.inputState
        specieList = inputState.specieList
        NSpecies = len(specieList)
        bondDict = elements.bondDict
        
        bondMinArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        bondMaxArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        
        # construct bonds array
        calcBonds = False
        maxBond = -1
        
        # populate bond arrays
        for i in xrange(NSpecies):
            symi = specieList[i]
            
            if symi in bondDict:
                d = bondDict[symi]
                
                for j in xrange(NSpecies):
                    symj = specieList[j]
                    
                    if symj in d:
                        bondMin, bondMax = d[symj]
                        
                        bondMinArray[i][j] = bondMin
                        bondMinArray[j][i] = bondMin
                        
                        bondMaxArray[i][j] = bondMax
                        bondMaxArray[j][i] = bondMax
                        
                        if bondMax > maxBond:
                            maxBond = bondMax
                        
                        if bondMax > 0:
                            calcBonds = True
                        
                        self.logger.info("  %s - %s; bond range: %f -> %f", symi, symj, bondMin, bondMax)
        
        if not calcBonds:
            self.logger.warning("No bonds defined: all coordination numbers will be zero")
        
        # new scalars array
        scalars = np.zeros(len(self.visibleAtoms), dtype=np.float64)
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # full vectors array
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        # run filter
        filteringEnabled = int(filterSettings.getSetting("filteringEnabled"))
        minCoordNum = filterSettings.getSetting("minCoordNum")
        maxCoordNum = filterSettings.getSetting("maxCoordNum")
        NVisible = filtering_c.coordNumFilter(self.visibleAtoms, inputState.pos, inputState.specie, NSpecies, bondMinArray, bondMaxArray,
                                              maxBond, inputState.cellDims, self.pipelinePage.PBC, scalars, minCoordNum, maxCoordNum,
                                              NScalars, fullScalars, filteringEnabled, NVectors, fullVectors)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
        
        # store scalars
        scalars.resize(NVisible, refcheck=False)
        self.scalarsDict["Coordination number"] = scalars
    
    def genericScalarFilter(self, filterName, settings):
        """
        Generic scalar filter
        
        """
        self.logger.debug("Generic scalar filter: '%s'", filterName)
        
        # full scalars/vectors
        NScalars, fullScalars = self.makeFullScalarsArray()
        NVectors, fullVectors = self.makeFullVectorsArray()
        
        # scalars array (the full, unmodified one stored on the Lattice)
        scalarsName = filterName[8:]
        scalarsArray = self.pipelinePage.inputState.scalarsDict[scalarsName]
        
        # run filter
        minVal = settings.getSetting("minVal")
        maxVal = settings.getSetting("maxVal")
        NVisible = filtering_c.genericScalarFilter(self.visibleAtoms, scalarsArray, minVal, maxVal, NScalars, fullScalars,
                                                   NVectors, fullVectors)
        
        # update scalars/vectors
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        self.storeFullVectorsArray(NVisible, NVectors, fullVectors)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
