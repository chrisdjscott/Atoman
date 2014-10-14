
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
import vtk
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
        self.visibleSpecieCount = []
        self.vacancySpecieCount = []
        self.interstitialSpecieCount = []
        self.antisiteSpecieCount = []
        self.splitIntSpecieCount = []
        self.vacancies = np.empty(0, np.int32)
        self.interstitials = np.empty(0, np.int32)
        self.antisites = np.empty(0, np.int32)
        self.onAntisites = np.empty(0, np.int32)
        self.splitInterstitials = np.empty(0, np.int32)
        
        self.driftVector = np.zeros(3, np.float64)
        
        self.actorsCollection = vtk.vtkActorCollection()
        
        self.traceDict = {}
        self.previousPosForTrace = None
        
        self.colouringOptions = self.parent.colouringOptions
        self.bondsOptions = self.parent.bondsOptions
        self.displayOptions = self.parent.displayOptions
        self.voronoiOptions = self.parent.voronoiOptions
        self.traceOptions = self.parent.traceOptions
        self.scalarBarAdded = False
        self.scalarsDict = {}
        self.scalarBar_white_bg = None
        self.scalarBar_black_bg = None
        self.povrayAtomsWritten = False
        self.clusterList = []
        self.voronoi = None
        
        self.structureCounterDicts = {}
#         self.knownStructures = [
#             "disordered",
#             "FCC",
#             "HCP",
#             "BCC",
#             "icosahedral",
#             "sigma11_tilt1",
#             "sigma11_tilt2",
#         ]
    
    def removeActors(self, sequencer=False):
        """
        Remove actors.
        
        """
        self.hideActors()
        
        self.actorsCollection = vtk.vtkActorCollection()
        if not sequencer:
            self.traceDict = {}
            self.previousPosForTrace = None
        
        self.scalarsDict = {}
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
        self.actorsCollection.InitTraversal()
        actor = self.actorsCollection.GetNextItem()
        while actor is not None:
            for rw in self.rendererWindows:
                if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                    try:
                        rw.vtkRen.RemoveActor(actor)
                    except:
                        pass
            
            actor = self.actorsCollection.GetNextItem()
        
        for rw in self.rendererWindows:
            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                rw.vtkRenWinInteract.ReInitialize()
        
        self.hideScalarBar()
    
    def addActors(self):
        """
        Add all actors
        
        """
        self.actorsCollection.InitTraversal()
        actor = self.actorsCollection.GetNextItem()
        while actor is not None:
            for rw in self.rendererWindows:
                if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                    try:
                        rw.vtkRen.AddActor(actor)
                    except:
                        pass
            
            actor = self.actorsCollection.GetNextItem()
        
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
        NAtoms = self.pipelinePage.inputState.NAtoms
        
        if not self.parent.defectFilterSelected:
            self.visibleAtoms = np.arange(NAtoms, dtype=np.int32)
            self.NVis = NAtoms
            self.logger.info("%d visible atoms", len(self.visibleAtoms))
        
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
            filterSettings = currentSettings[i]
            
            self.logger.info("Running filter: '%s'", filterName)
            
            if filterName == "Specie":
                self.filterSpecie(filterSettings)
            
            elif filterName == "Crop box":
                self.cropFilter(filterSettings)
            
            elif filterName == "Displacement":
                self.displacementFilter(filterSettings)
                drawDisplacementVectors = filterSettings.drawDisplacementVectors
                displacementSettings = filterSettings
            
            elif filterName == "Point defects":
                interstitials, vacancies, antisites, onAntisites, splitInterstitials = self.pointDefectFilter(filterSettings)
                
                self.interstitials = interstitials
                self.vacancies = vacancies
                self.antisites = antisites
                self.onAntisites = onAntisites
                self.splitInterstitials = splitInterstitials
            
            elif filterName == "Kinetic energy":
                self.KEFilter(filterSettings)
            
            elif filterName == "Potential energy":
                self.PEFilter(filterSettings)
            
            elif filterName == "Charge":
                self.chargeFilter(filterSettings)
            
            elif filterName == "Cluster":
                self.clusterFilter(filterSettings)
                
                if filterSettings.drawConvexHulls:
                    self.clusterFilterDrawHulls(filterSettings, hullFile)
                
                if filterSettings.calculateVolumes:
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
            
            elif filterName == "Atom index":
                self.atomIndexFilter(filterSettings)
            
            elif filterName == "ACNA":
                self.acnaFilter(filterSettings)
            
            # write to log
            if self.parent.defectFilterSelected:
                self.NVis = len(interstitials) + len(vacancies) + len(antisites) + len(splitInterstitials)
                self.NVac = len(vacancies)
                self.NInt = len(interstitials) + len(splitInterstitials) / 3
                self.NAnt = len(antisites)
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
                counters = renderer.getActorsForFilteredDefects(interstitials, vacancies, antisites, onAntisites, splitInterstitials, self.actorsCollection, 
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
            if filterName == "Cluster" and filterSettings.drawConvexHulls and filterSettings.hideAtoms:
                pass
            
            else:
                # this is a hack!! not ideal
                if self.parent.isPersistentList():
                    NVisibleForRes = 800
                else:
                    NVisibleForRes = None
                
                self.scalarBar_white_bg, self.scalarBar_black_bg, visSpecCount = renderer.getActorsForFilteredSystem(self.visibleAtoms, self.mainWindow, 
                                                                                                                     self.actorsCollection, self.colouringOptions, 
                                                                                                                     povfile, self.scalarsDict, self.displayOptions, 
                                                                                                                     self.pipelinePage, self.povrayAtomsWrittenSlot,
                                                                                                                     NVisibleForRes=NVisibleForRes,
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
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # make array of neighbours
        num_nebs_array = vor.atomNumNebsArray()
        
        NVisible = filtering_c.voronoiNeighboursFilter(self.visibleAtoms, num_nebs_array, settings.minVoroNebs, settings.maxVoroNebs, 
                                                       scalars, NScalars, fullScalars, settings.filteringEnabled)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)

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
        
        # make array of volumes
        atom_volumes = vor.atomVolumesArray()
        
        NVisible = filtering_c.voronoiVolumeFilter(self.visibleAtoms, atom_volumes, settings.minVoroVol, settings.maxVoroVol, 
                                                   scalars, NScalars, fullScalars, settings.filteringEnabled)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)

        # store scalars
        scalars.resize(NVisible, refcheck=False)
        self.scalarsDict["Voronoi volume"] = scalars
    
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
        
        # num threads
        ompNumThreads = self.mainWindow.preferences.generalForm.openmpNumThreads
        
        NVisible = bond_order.bondOrderFilter(self.visibleAtoms, inputState.pos, settings.maxBondDistance, scalarsQ4, scalarsQ6, inputState.minPos, 
                                                inputState.maxPos, inputState.cellDims, self.pipelinePage.PBC, NScalars, fullScalars, settings.filterQ4Enabled, 
                                                settings.minQ4, settings.maxQ4, settings.filterQ6Enabled, settings.minQ6, settings.maxQ6, ompNumThreads)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        
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
        
        # number of openmp threads
        ompNumThreads = self.mainWindow.preferences.generalForm.openmpNumThreads
        
        # counter array
        counters = np.zeros(7, np.int32)
        
        NVisible = acna.adaptiveCommonNeighbourAnalysis(self.visibleAtoms, inputState.pos, scalars, inputState.minPos, inputState.maxPos, 
                                                        inputState.cellDims, self.pipelinePage.PBC, NScalars, fullScalars, settings.maxBondDistance,
                                                        counters, settings.filteringEnabled, settings.structureVisibility, ompNumThreads)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        
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
                                               self.colouringOptions, self.voronoiOptions, self.actorsCollection, 
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
                                        maxBond, maxBondsPerAtom, inputState.cellDims, self.pipelinePage.PBC, inputState.minPos, inputState.maxPos, 
                                        bondArray, NBondsArray, bondVectorArray, bondSpecieCounter)
        
        if status:
            self.logger.error("    Error in bonds clib (%d)", status)
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
            
            renderBonds.renderBonds(self.visibleAtoms, self.mainWindow, self.pipelinePage, self.actorsCollection, self.colouringOptions, 
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
        self.logger.debug("Making full scalars array (N=%d)", len(self.scalarsDict))
        
        scalarsList = []
        for name, scalars in self.scalarsDict.iteritems():
            self.logger.debug("  Adding '%s' scalars", name)
            scalarsList.append(scalars)
            
#             assert len(scalars) == len(self.visibleAtoms)
#             f = open("%s_before.dat" % name.replace(" ", "_"), "w")
#             for tup in itertools.izip(self.visibleAtoms, scalars):
#                 f.write("%d %f\n" % tup)
#             f.close()
        
        if len(scalarsList):
            scalarsFull = np.concatenate(scalarsList)
        else:
            scalarsFull = np.array([], dtype=np.float64)
        
        return len(scalarsList), scalarsFull
    
    def storeFullScalarsArray(self, NVisible, NScalars, scalarsFull):
        """
        Split and resize full scalars array; store in dict
        
        Assumes scalarsDict was not modified since we called
        makeFullScalarsArray.
        
        """
        if NScalars > 0:
            self.logger.debug("Storing full scalars array in dict")
            scalarsList = np.split(scalarsFull, NScalars)
            keys = self.scalarsDict.keys()
            
            for key, scalars in itertools.izip(keys, scalarsList):
                self.logger.debug("  Storing '%s' scalars", key)
                assert len(scalars) >= NVisible, "ERROR: scalars (%s) smaller than expected (%d < %d)" % (key, len(scalars), NVisible)
                scalars_cp = copy.copy(scalars)
                scalars_cp.resize(NVisible, refcheck=False)
                self.scalarsDict[key] = scalars_cp
    
    def filterSpecie(self, settings):
        """
        Filter by specie
        
        """
        if settings.allSpeciesSelected:
            visSpecArray = np.arange(len(self.pipelinePage.inputState.specieList), dtype=np.int32)
        
        else:
            visSpecArray = np.empty(len(settings.visibleSpecieList), np.int32)
            count = 0
            for i in xrange(len(self.pipelinePage.inputState.specieList)):
                if self.pipelinePage.inputState.specieList[i] in settings.visibleSpecieList:
                    visSpecArray[count] = i
                    count += 1
        
            if count != len(visSpecArray):
                visSpecArray.resize(count)
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        NVisible = filtering_c.specieFilter(self.visibleAtoms, visSpecArray, self.pipelinePage.inputState.specie, NScalars, fullScalars)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)

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
        renderBonds.renderDisplacementVectors(self.visibleAtoms, self.mainWindow, self.pipelinePage, self.actorsCollection, 
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
            
            # run displacement filter
            NVisible = filtering_c.displacementFilter(self.visibleAtoms, scalars, inputState.pos, refState.pos, refState.cellDims, 
                                                      self.pipelinePage.PBC, settings.minDisplacement, settings.maxDisplacement, 
                                                      NScalars, fullScalars, settings.filteringEnabled, self.parent.driftCompensation, 
                                                      self.driftVector)
            
            # update scalars dict
            self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
            
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
                                                            self.actorsCollection, self.colouringOptions, povfile, 
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
        lattice = self.pipelinePage.inputState
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        # run displacement filter
        NVisible = filtering_c.atomIndexFilter(self.visibleAtoms, lattice.atomID, settings.filteringEnabled, settings.minVal, settings.maxVal, 
                                               NScalars, fullScalars)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def cropFilter(self, settings):
        """
        Crop lattice
        
        """
        lattice = self.pipelinePage.inputState
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        NVisible = filtering_c.cropFilter(self.visibleAtoms, lattice.pos, settings.xmin, settings.xmax, settings.ymin, 
                                          settings.ymax, settings.zmin, settings.zmax, settings.xEnabled, 
                                          settings.yEnabled, settings.zEnabled, settings.invertSelection, NScalars, fullScalars)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)

        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def cropSphereFilter(self, settings):
        """
        Crop sphere filter.
        
        """
        lattice = self.pipelinePage.inputState
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        NVisible = filtering_c.cropSphereFilter(self.visibleAtoms, lattice.pos, settings.xCentre, settings.yCentre, settings.zCentre, 
                                                settings.radius, lattice.cellDims, self.pipelinePage.PBC, settings.invertSelection, 
                                                NScalars, fullScalars)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)

        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def sliceFilter(self, settings):
        """
        Slice filter.
        
        """
        lattice = self.pipelinePage.inputState
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        NVisible = filtering_c.sliceFilter(self.visibleAtoms, lattice.pos, settings.x0, settings.y0, settings.z0, 
                                           settings.xn, settings.yn, settings.zn, settings.invert, NScalars, fullScalars)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)

        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def chargeFilter(self, settings):
        """
        Charge filter.
        
        """
        lattice = self.pipelinePage.inputState
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        NVisible = filtering_c.chargeFilter(self.visibleAtoms, lattice.charge, settings.minCharge, settings.maxCharge, NScalars, fullScalars)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)

        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def KEFilter(self, settings):
        """
        Filter kinetic energy.
        
        """
        lattice = self.pipelinePage.inputState
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        NVisible = filtering_c.KEFilter(self.visibleAtoms, lattice.KE, settings.minKE, settings.maxKE, NScalars, fullScalars)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)

        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def PEFilter(self, settings):
        """
        Filter potential energy.
        
        """
        lattice = self.pipelinePage.inputState
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        NVisible = filtering_c.PEFilter(self.visibleAtoms, lattice.PE, settings.minPE, settings.maxPE, NScalars, fullScalars)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)

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
            structVis = np.ones(len(self.knownStructures), np.int32)
            
            # number of threads
            ompNumThreads = self.mainWindow.preferences.generalForm.openmpNumThreads
            
            # counter array
            counters = np.zeros(7, np.int32)
            
            acna.adaptiveCommonNeighbourAnalysis(visAtoms, inputLattice.pos, acnaArray, inputLattice.minPos, inputLattice.maxPos, 
                                                 inputLattice.cellDims, self.pipelinePage.PBC, NScalars, fullScalars, 
                                                 settings.acnaMaxBondDistance, counters, 0, structVis, ompNumThreads) 
            
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
        
        # set min/max pos to lattice (for boxing)
        minPos = refLattice.minPos
        maxPos = refLattice.maxPos
        
        if settings.findClusters:
            defectCluster = np.empty(inputLattice.NAtoms + refLattice.NAtoms, np.int32)
        
        else:
            defectCluster = np.empty(0, np.int32)
        
        # call C library
        status = defects_c.findDefects(settings.showVacancies, settings.showInterstitials, settings.showAntisites, NDefectsByType, vacancies, 
                                       interstitials, antisites, onAntisites, exclSpecsInput, exclSpecsRef, inputLattice.NAtoms, inputLattice.specieList,
                                       inputLattice.specie, inputLattice.pos, refLattice.NAtoms, refLattice.specieList, refLattice.specie, 
                                       refLattice.pos, refLattice.cellDims, self.pipelinePage.PBC, settings.vacancyRadius, minPos, maxPos, 
                                       settings.findClusters, settings.neighbourRadius, defectCluster, vacSpecCount, intSpecCount, antSpecCount,
                                       onAntSpecCount, splitIntSpecCount, settings.minClusterSize, settings.maxClusterSize, splitInterstitials, 
                                       settings.identifySplitInts, self.parent.driftCompensation, self.driftVector, acnaArray)
        
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
        if settings.drawVectorsGroup.isEnabled() and settings.drawDisplacementVectors:
            self.logger.debug("Drawing displacement vectors for interstitials (%d)", NInt)
            
            # need to make a unique list for interstitials and split intersitials and antisites
            
            
            # calculate vectors
            bondVectorArray = np.empty(3 * NInt, np.float64)
            status = bonds_c.calculateDisplacementVectors(interstitials, inputLattice.pos, refLattice.pos, 
                                                          refLattice.cellDims, self.pipelinePage.PBC, bondVectorArray)
            
            # pov file for bonds
            povfile = "pipeline%d_intdispvects%d_%s.pov" % (self.pipelineIndex, self.parent.tab, str(self.filterTab.currentRunID))
            
            # draw displacement vectors as bonds
            renderBonds.renderDisplacementVectors(interstitials, self.mainWindow, self.pipelinePage, self.actorsCollection, 
                                                  self.colouringOptions, povfile, self.scalarsDict, bondVectorArray, settings)
        
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
        
        # loop over clusters
        for cluster in clusterList:
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
                
                renderer.getActorsForHullFacets(facets, clusterPos, self.mainWindow, self.actorsCollection, settings)
                
                # write povray file too
                renderer.writePovrayHull(facets, clusterPos, self.mainWindow, hullPovFile, settings)
            
            # handle PBCs
            if NDefects > 1 and PBCFlag:
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
                        
                        renderer.getActorsForHullFacets(facets, tmpClusterPos, self.mainWindow, self.actorsCollection, settings)
                        
                        # write povray file too
                        renderer.writePovrayHull(facets, tmpClusterPos, self.mainWindow, hullPovFile, settings)
        
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
            minSize = settings.minClusterSize
        
        if maxSize is None:
            maxSize = settings.maxClusterSize
        
        if nebRad is None:
            nebRad = settings.neighbourRadius
        
        # set min/max pos to lattice (for boxing)
        minPos = np.zeros(3, np.float64)
        maxPos = copy.deepcopy(lattice.cellDims)
        
        # old scalars arrays (resize as appropriate)
        NScalars, fullScalars = self.makeFullScalarsArray()
        
        clusters_c.findClusters(self.visibleAtoms, lattice.pos, atomCluster, nebRad, lattice.cellDims, PBC, 
                                minPos, maxPos, minSize, maxSize, result, NScalars, fullScalars)
        
        NVisible = result[0]
        NClusters = result[1]
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        
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
        for cluster in clusterList:
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
                renderer.getActorsForHullFacets(facets, clusterPos, self.mainWindow, self.actorsCollection, settings)
                
                # write povray file too
                renderer.writePovrayHull(facets, clusterPos, self.mainWindow, hullPovFile, settings)
    
    def clusterFilterDrawHullsWithPBCs(self, settings, hullPovFile):
        """
        
        
        """
        lattice = self.pipelinePage.inputState
        clusterList = self.clusterList
        
        for cluster in clusterList:
            appliedPBCs = np.zeros(7, np.int32)
            clusterPos = np.empty(3 * len(cluster), np.float64)
            for i in xrange(len(cluster)):
                index = cluster[i]
                
                clusterPos[3*i] = lattice.pos[3*index]
                clusterPos[3*i+1] = lattice.pos[3*index+1]
                clusterPos[3*i+2] = lattice.pos[3*index+2]
            
            clusters_c.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, 
                                                 self.pipelinePage.PBC, appliedPBCs, settings.neighbourRadius)
            
            facets = None
            if len(cluster) > 3:
                facets = clusters.findConvexHullFacets(len(cluster), clusterPos)
            
            elif len(cluster) == 3:
                facets = []
                facets.append([0, 1, 2])
            
            # now render
            if facets is not None:
                #TODO: make sure not facets more than neighbour rad from cell
                facets = clusters.checkFacetsPBCs(facets, clusterPos, 2.0 * settings.neighbourRadius, self.pipelinePage.PBC, lattice.cellDims)
                
                renderer.getActorsForHullFacets(facets, clusterPos, self.mainWindow, self.actorsCollection, settings)
                
                # write povray file too
                renderer.writePovrayHull(facets, clusterPos, self.mainWindow, hullPovFile, settings)
            
            # handle PBCs
            if len(cluster) > 1:
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
                        facets = clusters.checkFacetsPBCs(facets, tmpClusterPos, 2.0 * settings.neighbourRadius, self.pipelinePage.PBC, lattice.cellDims)
                        
                        renderer.getActorsForHullFacets(facets, tmpClusterPos, self.mainWindow, self.actorsCollection, settings)
                        
                        # write povray file too
                        renderer.writePovrayHull(facets, tmpClusterPos, self.mainWindow, hullPovFile, settings)
    
    def clusterFilterCalculateVolumes(self, filterSettings):
        """
        Calculate volumes of clusters.
        
        """    
        # this will not work properly over PBCs at the moment
        lattice = self.pipelinePage.inputState
        
        # draw them as they are
        if filterSettings.calculateVolumesHull:
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
                        clusters_c.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, 
                                                             PBC, appliedPBCs, filterSettings.neighbourRadius)
                    
                    volume, area = clusters.findConvexHullVolume(len(cluster), clusterPos)
                
                self.logger.info("  Cluster %d (%d atoms)", count, len(cluster))
                if area is not None:
                    self.logger.info("    volume is %f; facet area is %f", volume, area)
                
                # store volume/facet area
                cluster.volume = volume
                cluster.facetArea = area
                
                count += 1
        
        elif filterSettings.calculateVolumesVoro:
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
            self.log("ERROR: method to calculate cluster volumes not specified")
    
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
        
        # run displacement filter
        NVisible = filtering_c.coordNumFilter(self.visibleAtoms, inputState.pos, inputState.specie, NSpecies, bondMinArray, bondMaxArray, 
                                              maxBond, inputState.cellDims, self.pipelinePage.PBC, inputState.minPos, inputState.maxPos, 
                                              scalars, filterSettings.minCoordNum, filterSettings.maxCoordNum, NScalars, fullScalars, 
                                              filterSettings.filteringEnabled)
        
        # update scalars dict
        self.storeFullScalarsArray(NVisible, NScalars, fullScalars)
        
        # resize visible atoms
        self.visibleAtoms.resize(NVisible, refcheck=False)
        
        # store scalars
        scalars.resize(NVisible, refcheck=False)
        self.scalarsDict["Coordination number"] = scalars
