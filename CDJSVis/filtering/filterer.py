
"""
The filter tab for the main toolbar

@author: Chris Scott

"""
import os
import copy

import numpy as np
import vtk

from ..visclibs import filtering_c
from ..visclibs import defects_c
from ..visclibs import clusters_c
from ..rendering import renderer
from ..rendering import renderBonds
from ..visutils import vectors
from . import clusters
from ..atoms import elements
from ..visclibs_ctypes import bonds as bonds_c
from ..visclibs_ctypes import numpy_utils as nputils


################################################################################
class Filterer(object):
    """
    Filter class.
    
    Contains list of subfilters to be performed in order.
    
    """
    def __init__(self, parent):
        
        self.parent = parent
        self.filterTab = parent.filterTab
        self.mainWindow = self.parent.mainWindow
        
        self.log = self.mainWindow.console.write
        
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
        
        self.actorsCollection = vtk.vtkActorCollection()
        
        self.availableScreenInfo = {}
        
        self.colouringOptions = self.parent.colouringOptions
        self.bondsOptions = self.parent.bondsOptions
        self.scalarBarAdded = False
        self.scalarBar = None
        
        self.scalars = np.asarray([], dtype=np.float64)
        self.scalarsType = ""
    
    def removeActors(self):
        """
        Remove actors.
        
        """
        self.hideActors()
        
        self.actorsCollection = vtk.vtkActorCollection()
        
        self.scalarBar = None
    
    def hideActors(self):
        """
        Hide all actors
        
        """
        self.actorsCollection.InitTraversal()
        actor = self.actorsCollection.GetNextItem()
        while actor is not None:
            try:
                self.mainWindow.VTKRen.RemoveActor(actor)
            except:
                pass
            
            actor = self.actorsCollection.GetNextItem()
        
        self.mainWindow.VTKWidget.ReInitialize()
        
        self.hideScalarBar()
    
    def addActors(self):
        """
        Add all actors
        
        """
        self.actorsCollection.InitTraversal()
        actor = self.actorsCollection.GetNextItem()
        while actor is not None:
            try:
                self.mainWindow.VTKRen.AddActor(actor)
            except:
                pass
            
            actor = self.actorsCollection.GetNextItem()
        
        self.mainWindow.VTKWidget.ReInitialize()
        
        self.addScalarBar()
    
    def runFilters(self):
        """
        Run the filters.
        
        """
        if not self.parent.isPersistentList():
            self.removeActors()
        
        # reset
        self.visibleAtoms = np.asarray([], dtype=np.int32)
        self.interstitials = np.asarray([], dtype=np.int32)
        self.vacancies = np.asarray([], dtype=np.int32)
        self.antisites = np.asarray([], dtype=np.int32)
        self.onAntisites = np.asarray([], dtype=np.int32)
        self.splitInterstitials = np.asarray([], dtype=np.int32)
        self.scalars = np.asarray([], dtype=np.float64)
        
        # first set up visible atoms arrays
        NAtoms = self.parent.mainWindow.inputState.NAtoms
        
        if not self.parent.defectFilterSelected:
            self.visibleAtoms = np.arange(NAtoms, dtype=np.int32)
            NVis = NAtoms
            self.NVis = NAtoms
#            self.scalars = np.empty(NAtoms, dtype=np.float64)
            self.log("%d visible atoms" % (len(self.visibleAtoms),), 0, 2)
        
        self.availableScreenInfo = {}
        
        hullFile = os.path.join(self.mainWindow.tmpDirectory, "hulls%d.pov" % self.parent.tab)
        if os.path.exists(hullFile):
            os.unlink(hullFile)
        
        # run filters
        filterName = ""
        self.scalarsType = ""
        currentFilters = self.parent.currentFilters
        currentSettings = self.parent.currentSettings
        for i in xrange(len(currentFilters)):
            # filter name
            filterNameString = currentFilters[i]
            array = filterNameString.split("[")
            filterName = array[0].strip()
            
            # filter settings
            filterSettings = currentSettings[i]
            
            self.log("Running filter: %s" % (filterName,), 0, 2)
            
            if filterName == "Specie":
                self.filterSpecie(filterSettings)
            
            elif filterName == "Crop":
                self.cropFilter(filterSettings)
            
            elif filterName == "Displacement":
                self.displacementFilter(filterSettings)
                self.scalarsType = filterName
            
            elif filterName == "Point defects":
                interstitials, vacancies, antisites, onAntisites, splitInterstitials, clusterList = self.pointDefectFilter(filterSettings)
                
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
                clusterList = self.clusterFilter(filterSettings)
                
                if filterSettings.drawConvexHulls:
                    self.clusterFilterDrawHulls(clusterList, filterSettings, hullFile)
                
                if filterSettings.calculateVolumes:
                    self.clusterFilterCalculateVolumes(clusterList, filterSettings)
            
            elif filterName == "Crop sphere":
                self.cropSphereFilter(filterSettings)
            
            elif filterName == "Slice":
                self.sliceFilter(filterSettings)
            
            # write to log
            if self.parent.defectFilterSelected:
                NVis = len(interstitials) + len(vacancies) + len(antisites) + len(splitInterstitials)
                self.NVac = len(vacancies)
                self.NInt = len(interstitials) + len(splitInterstitials) / 3
                self.NAnt = len(antisites)
                
            else:
                NVis = len(self.visibleAtoms)
            
            self.NVis = NVis
            
            self.log("%d visible atoms" % (NVis,), 0, 3)
            self.availableScreenInfo["visible"] = NVis
        
        # refresh available scalars in extra options dialog
        self.parent.colouringOptions.refreshScalarColourOption(self.scalarsType)
        
        # render
        povfile = "atoms%d.pov" % (self.parent.tab,)
        if self.parent.defectFilterSelected:
            colourBy = self.colouringOptions.colourBy
            self.colouringOptions.colourBy = "Specie"
            
            # vtk render
            if filterSettings.findClusters and filterSettings.drawConvexHulls:
                self.pointDefectFilterDrawHulls(clusterList, filterSettings, hullFile)
            
            if filterSettings.findClusters and filterSettings.drawConvexHulls and filterSettings.hideDefects:
                pass
            
            else:
                counters = renderer.getActorsForFilteredDefects(interstitials, vacancies, antisites, onAntisites, splitInterstitials, self.mainWindow, self.actorsCollection, self.colouringOptions, filterSettings)
                
                self.vacancySpecieCount = counters[0]
                self.interstitialSpecieCount = counters[1]
                self.antisiteSpecieCount = counters[2]
                self.splitIntSpecieCount = counters[3]
                
                # write pov-ray file too
                povfile = "defects%d.pov" % self.parent.tab
                renderer.writePovrayDefects(povfile, vacancies, interstitials, antisites, onAntisites, filterSettings, self.mainWindow)
            
            self.colouringOptions.colourBy = colourBy
            
            # add defect info to text screen?
            
        
        else:
            if filterName == "Cluster" and filterSettings.drawConvexHulls and filterSettings.hideAtoms:
                pass
            
            else:
                # this is a hack!! not ideal
                if self.parent.isPersistentList():
                    NVisibleForRes = 800
                
                else:
                    NVisibleForRes = None
                
                self.scalarBar, visSpecCount = renderer.getActorsForFilteredSystem(self.visibleAtoms, self.mainWindow, self.actorsCollection, 
                                                                                    self.colouringOptions, povfile, self.scalars, NVisibleForRes=NVisibleForRes)
                
                self.visibleSpecieCount = visSpecCount
                
                # write pov-ray file too (only if pov-ray located??)
#                renderer.writePovrayAtoms(povfile, self.visibleAtoms, self.mainWindow)
            
            if self.bondsOptions.drawBonds:
                # find bonds
                status = self.calculateBonds()
                
                if not status:
                    povfile = "bonds%d.pov" % (self.parent.tab,)
                    
                    # draw bonds
                    renderBonds.renderBonds(self.visibleAtoms, self.mainWindow, self.actorsCollection, self.colouringOptions, povfile, 
                                            self.scalars)
        
        if self.parent.visible:
            self.addActors()
    
    def calculateBonds(self):
        """
        Calculate bonds.
        
        """
        print "CALCULATING BONDS"
                
        inputState = self.mainWindow.inputState
        specieList = inputState.specieList
        NSpecies = len(specieList)
        
        bondMinArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        bondMaxArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        
        # construct bonds array
        calcBonds = False
        maxBond = -1
        for i in xrange(self.bondsOptions.NBondPairs):
            pair = self.bondsOptions.bondPairsList[i]
            drawPair = self.bondsOptions.bondPairDrawStatus[i]
            
            if drawPair:
                syma, symb = pair
                
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
                        
                        calcBonds = True
                
                        print "PAIR: %s - %s; bond range: %f -> %f" % (pair[0], pair[1], bondMin, bondMax)
        
        if not calcBonds:
            print "NO BONDS TO CALC"
            return 1
        
        # arrays for results
        maxBondsPerAtom = 50
        size = self.NVis * maxBondsPerAtom
        print "SIZE", size
        print "MAX BOND", maxBond
        bondArray = np.empty(size, np.int32)
        NBondsArray = np.zeros(self.NVis, np.int32)
        bondVectorArray = np.empty(3 * size, np.float64)
        
        status = bonds_c.calculateBonds(self.NVis, self.visibleAtoms, inputState.pos, inputState.specie, len(specieList), bondMinArray, bondMaxArray, 
                                        maxBond, maxBondsPerAtom, inputState.cellDims, self.mainWindow.PBC, inputState.minPos, inputState.maxPos, 
                                        bondArray, NBondsArray, bondVectorArray)
        
        print "BACK IN PY"
        
        if status:
            print "ERROR IN BONDS LIB (%d)" % status
            return 1
        
        # total number of bonds
        NBondsTotal = np.sum(NBondsArray)
        print "NBONDSTOT", NBondsTotal
        
        # resize bond array
        bondArray.resize(NBondsTotal)
        bondVectorArray.resize(NBondsTotal * 3)
        
        
        
        
        
        return 0
    
    def addScalarBar(self):
        """
        Add scalar bar.
        
        """
        if self.scalarBar is not None and self.parent.scalarBarButton.isChecked() and not self.parent.filterTab.scalarBarAdded:
            self.mainWindow.VTKRen.AddActor2D(self.scalarBar)
            self.mainWindow.VTKWidget.ReInitialize()
            
            self.parent.filterTab.scalarBarAdded = True
            self.scalarBarAdded = True
        
        return self.scalarBarAdded
    
    def hideScalarBar(self):
        """
        Remove scalar bar.
        
        """
        if self.scalarBarAdded:
            self.mainWindow.VTKRen.RemoveActor2D(self.scalarBar)
            self.mainWindow.VTKWidget.ReInitialize()
            
            self.parent.filterTab.scalarBarAdded = False
            self.scalarBarAdded = False
    
    def getActorsForFilteredSystem(self):
        """
        Render systems after applying filters.
        
        """
        pass
            
    def filterSpecie(self, settings):
        """
        Filter by specie
        
        """
        if settings.allSpeciesSelected:
            visSpecArray = np.arange(len(self.mainWindow.inputState.specieList), dtype=np.int32)
        
        else:
            visSpecArray = np.empty(len(settings.visibleSpecieList), np.int32)
            count = 0
            for i in xrange(len(self.mainWindow.inputState.specieList)):
                if self.mainWindow.inputState.specieList[i] in settings.visibleSpecieList:
                    visSpecArray[count] = i
                    count += 1
        
            if count != len(visSpecArray):
                visSpecArray.resize(count)
        
        NVisible = filtering_c.specieFilter(self.visibleAtoms, visSpecArray, self.mainWindow.inputState.specie)
        
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def displacementFilter(self, settings):
        """
        Displacement filter
        
        """
        # only run displacement filter if input and reference NAtoms are the same
        inputState = self.mainWindow.inputState
        refState = self.mainWindow.refState
        
        if inputState.NAtoms != refState.NAtoms:
            self.log("WARNING: cannot run displacement filter with different numbers of input and reference atoms: skipping this filter list")
            self.visibleAtoms.resize(0, refcheck=False)
        
        else:
            # scalars array
            if len(self.scalars) != len(self.visibleAtoms):
                self.scalars = np.empty(len(self.visibleAtoms), dtype=np.float64)
            
            # run displacement filter
            NVisible = filtering_c.displacementFilter(self.visibleAtoms, self.scalars, inputState.pos, refState.pos, refState.cellDims, 
                                                      self.mainWindow.PBC, settings.minDisplacement, settings.maxDisplacement)
            
            self.visibleAtoms.resize(NVisible, refcheck=False)
            self.scalars.resize(NVisible, refcheck=False)
    
    def cropFilter(self, settings):
        """
        Crop lattice
        
        """
        lattice = self.mainWindow.inputState
        
        NVisible = filtering_c.cropFilter(self.visibleAtoms, lattice.pos, settings.xmin, settings.xmax, settings.ymin, 
                                          settings.ymax, settings.zmin, settings.zmax, settings.xEnabled, 
                                          settings.yEnabled, settings.zEnabled)
        
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def cropSphereFilter(self, settings):
        """
        Crop sphere filter.
        
        """
        lattice = self.mainWindow.inputState
        
        NVisible = filtering_c.cropSphereFilter(self.visibleAtoms, lattice.pos, settings.xCentre, settings.yCentre, settings.zCentre, 
                                                settings.radius, lattice.cellDims, self.mainWindow.PBC, settings.invertSelection)
        
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def sliceFilter(self, settings):
        """
        Slice filter.
        
        """
        lattice = self.mainWindow.inputState
        
        NVisible = filtering_c.sliceFilter(self.visibleAtoms, lattice.pos, settings.x0, settings.y0, settings.z0, 
                                           settings.xn, settings.yn, settings.zn, settings.invert)
        
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def chargeFilter(self, settings):
        """
        Charge filter.
        
        """
        lattice = self.mainWindow.inputState
        
        NVisible = filtering_c.chargeFilter(self.visibleAtoms, lattice.charge, settings.minCharge, settings.maxCharge)
        
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def KEFilter(self, settings):
        """
        Filter kinetic energy.
        
        """
        lattice = self.mainWindow.inputState
        
        NVisible = filtering_c.KEFilter(self.visibleAtoms, lattice.KE, settings.minKE, settings.maxKE)
        
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def PEFilter(self, settings):
        """
        Filter potential energy.
        
        """
        lattice = self.mainWindow.inputState
        
        NVisible = filtering_c.PEFilter(self.visibleAtoms, lattice.PE, settings.minPE, settings.maxPE)
        
        self.visibleAtoms.resize(NVisible, refcheck=False)
    
    def pointDefectFilter(self, settings):
        """
        Point defects filter
        
        """
        inputLattice = self.mainWindow.inputState
        refLattice = self.mainWindow.refState
        
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
        if settings.allSpeciesSelected:
            exclSpecsInput = np.zeros(0, np.int32)
            exclSpecsRef = np.zeros(0, np.int32)
        
        else:
            exclSpecs = []
            for i in xrange(len(inputLattice.specieList)):
                spec = inputLattice.specieList[i]
                if spec not in settings.visibleSpecieList:
                    exclSpecs.append(i)
            exclSpecsInput = np.empty(len(exclSpecs), np.int32)
            for i in xrange(len(exclSpecs)):
                exclSpecsInput[i] = exclSpecs[i]
            
            exclSpecs = []
            for i in xrange(len(refLattice.specieList)):
                spec = refLattice.specieList[i]
                if spec not in settings.visibleSpecieList:
                    exclSpecs.append(i)
            exclSpecsRef = np.empty(len(exclSpecs), np.int32)
            for i in xrange(len(exclSpecs)):
                exclSpecsRef[i] = exclSpecs[i]
        
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
                                       refLattice.pos, refLattice.cellDims, self.mainWindow.PBC, settings.vacancyRadius, minPos, maxPos, 
                                       settings.findClusters, settings.neighbourRadius, defectCluster, vacSpecCount, intSpecCount, antSpecCount,
                                       onAntSpecCount, splitIntSpecCount, settings.minClusterSize, settings.maxClusterSize, splitInterstitials, 
                                       settings.identifySplitInts)
        
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
        self.log("Found %d defects" % (NDef,), 0, 3)
        
        if settings.showVacancies:
            self.log("%d vacancies" % (NVac,), 0, 4)
            for i in xrange(len(refLattice.specieList)):
                self.log("%d %s vacancies" % (vacSpecCount[i], refLattice.specieList[i]), 0, 5)
        
        if settings.showInterstitials:
            self.log("%d interstitials" % (NInt + NSplit,), 0, 4)
            for i in xrange(len(inputLattice.specieList)):
                self.log("%d %s interstitials" % (intSpecCount[i], inputLattice.specieList[i]), 0, 5)
        
            if settings.identifySplitInts:
                self.log("%d split interstitials" % (NSplit,), 0, 5)
                for i in xrange(len(inputLattice.specieList)):
                    for j in xrange(i, len(inputLattice.specieList)):
                        if j == i:
                            N = splitIntSpecCount[i][j]
                        else:
                            N = splitIntSpecCount[i][j] + splitIntSpecCount[j][i]
                        self.log("%d %s - %s split interstitials" % (N, inputLattice.specieList[i], inputLattice.specieList[j]), 0, 6)
        
        if settings.showAntisites:
            self.log("%d antisites" % (NAnt,), 0, 4)
            for i in xrange(len(refLattice.specieList)):
                for j in xrange(len(inputLattice.specieList)):
                    if inputLattice.specieList[j] == refLattice.specieList[i]:
                        continue
                    
                    self.log("%d %s on %s antisites" % (onAntSpecCount[i][j], inputLattice.specieList[j], refLattice.specieList[i]), 0, 6)
        
        if settings.identifySplitInts:
            self.log("Split int analysis")
            
            PBC = self.mainWindow.PBC
            cellDims = inputLattice.cellDims
            
            for i in xrange(NSplit):
                ind1 = splitInterstitials[3*i+1]
                ind2 = splitInterstitials[3*i+2]
                
                pos1 = inputLattice.pos[3*ind1:3*ind1+3]
                pos2 = inputLattice.pos[3*ind2:3*ind2+3]
                
                sepVec = vectors.separationVector(pos1, pos2, cellDims, PBC)
                norm = vectors.normalise(sepVec)
                
                self.log("Orientation of split int %d: (%.3f %.3f %.3f)" % (i, norm[0], norm[1], norm[2]), 0, 1)
        
        # sort clusters here
        clusterList = []
        defectType = []
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
        
        return (interstitials, vacancies, antisites, onAntisites, splitInterstitials, clusterList)
    
    def pointDefectFilterDrawHulls(self, clusterList, settings, hullPovFile):
        """
        Draw convex hulls around defect volumes
        
        """
#        PBC = self.mainWindow.PBC
#        if PBC[0] or PBC[1] or PBC[2]:
#            self.pointDefectFilterDrawHullsWithPBCs(clusterList, settings)
#        
#        else:
#            self.pointDefectFilterDrawHullsWithPBCs(clusterList, settings)
#    
#    def pointDefectFilterDrawHullsWithPBCs(self, clusterList, settings, hullPovFile):
#        """
#        Draw hulls around defect volumes (PBCs)
#        
#        """
        PBC = self.mainWindow.PBC
        if PBC[0] or PBC[1] or PBC[2]:
            PBCFlag = True
        
        else:
            PBCFlag = False
        
        inputLattice = self.mainWindow.inputState
        refLattice = self.mainWindow.refState
        
        for cluster in clusterList:
            
            NDefects = cluster.getNDefects()
            
            appliedPBCs = np.zeros(7, np.int32)
            clusterPos = np.empty(3 * NDefects, np.float64)
            
            # vacancy positions
            count = 0
            for i in xrange(cluster.getNVacancies()):
                index = cluster.vacancies[i]
                
                clusterPos[3*count] = refLattice.pos[3*index]
                clusterPos[3*count+1] = refLattice.pos[3*index+1]
                clusterPos[3*count+2] = refLattice.pos[3*index+2]
                
                count += 1
            
            # antisite positions
            for i in xrange(cluster.getNAntisites()):
                index = cluster.antisites[i]
                
                clusterPos[3*count] = refLattice.pos[3*index]
                clusterPos[3*count+1] = refLattice.pos[3*index+1]
                clusterPos[3*count+2] = refLattice.pos[3*index+2]
                
                count += 1
            
            # interstitial positions
            for i in xrange(cluster.getNInterstitials()):
                index = cluster.interstitials[i]
                
                clusterPos[3*count] = inputLattice.pos[3*index]
                clusterPos[3*count+1] = inputLattice.pos[3*index+1]
                clusterPos[3*count+2] = inputLattice.pos[3*index+2]
                
                count += 1
            
            clusters_c.prepareClusterToDrawHulls(NDefects, clusterPos, inputLattice.cellDims, 
                                                 self.mainWindow.PBC, appliedPBCs, settings.neighbourRadius)
            
            facets = None
            if NDefects > 3:
                facets = clusters.findConvexHullFacets(NDefects, clusterPos)
            
            elif NDefects == 3:
                facets = []
                facets.append([0, 1, 2])
            
            # now render
            if facets is not None:
                #TODO: make sure not facets more than neighbour rad from cell
                facets = clusters.checkFacetsPBCs(facets, clusterPos, 2.0 * settings.neighbourRadius, self.mainWindow.PBC, 
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
                                                          self.mainWindow.PBC, inputLattice.cellDims)
                        
                        renderer.getActorsForHullFacets(facets, tmpClusterPos, self.mainWindow, self.actorsCollection, settings)
                        
                        # write povray file too
                        renderer.writePovrayHull(facets, tmpClusterPos, self.mainWindow, hullPovFile, settings)
        
    def clusterFilter(self, settings, PBC=None, minSize=None, maxSize=None, nebRad=None):
        """
        Run the cluster filter
        
        """
        lattice = self.mainWindow.inputState
        
        atomCluster = np.empty(len(self.visibleAtoms), np.int32)
        result = np.empty(2, np.int32)
        
        if PBC is not None and len(PBC) == 3:
            pass
        else:
            PBC = self.mainWindow.PBC
        
        if minSize is None:
            minSize = settings.minClusterSize
        
        if maxSize is None:
            maxSize = settings.maxClusterSize
        
        if nebRad is None:
            nebRad = settings.neighbourRadius
        
        # set min/max pos to lattice (for boxing)
        minPos = np.zeros(3, np.float64)
        maxPos = copy.deepcopy(lattice.cellDims)
        
        clusters_c.findClusters(self.visibleAtoms, lattice.pos, atomCluster, nebRad, lattice.cellDims, PBC, 
                                minPos, maxPos, minSize, maxSize, result)
        
        NVisible = result[0]
        NClusters = result[1]
        
        self.visibleAtoms.resize(NVisible, refcheck=False)
        atomCluster.resize(NVisible, refcheck=False)
        
        # build cluster lists
        clusterList = []
        for i in xrange(NClusters):
            clusterList.append([])
        
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
            
            clusterList[clusterListIndex].append(atomIndex)
        
        #TODO: rebuild scalars array of atom cluster (so can colour by cluster maybe?)
        
        
        return clusterList
    
    def clusterFilterDrawHulls(self, clusterList, settings, hullPovFile):
        """
        Draw hulls around filters.
        
        If the clusterList was created using PBCs we need to recalculate each 
        cluster without PBCs.
        
        """
        PBC = self.mainWindow.PBC
        if PBC[0] or PBC[1] or PBC[2]:
            self.clusterFilterDrawHullsWithPBCs(clusterList, settings, hullPovFile)
        
        else:
            self.clusterFilterDrawHullsNoPBCs(clusterList, settings, hullPovFile)
    
    def clusterFilterDrawHullsNoPBCs(self, clusterList, settings, hullPovFile):
        """
        SHOULD BE ABLE TO GET RID OF THIS AND JUST USE PBCs ONE
        
        """
        lattice = self.mainWindow.inputState
        
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
    
    def clusterFilterDrawHullsWithPBCs(self, clusterList, settings, hullPovFile):
        """
        
        
        """
        lattice = self.mainWindow.inputState
        
        for cluster in clusterList:
            
            appliedPBCs = np.zeros(7, np.int32)
            clusterPos = np.empty(3 * len(cluster), np.float64)
            for i in xrange(len(cluster)):
                index = cluster[i]
                
                clusterPos[3*i] = lattice.pos[3*index]
                clusterPos[3*i+1] = lattice.pos[3*index+1]
                clusterPos[3*i+2] = lattice.pos[3*index+2]
            
            clusters_c.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, 
                                                 self.mainWindow.PBC, appliedPBCs, settings.neighbourRadius)
            
            facets = None
            if len(cluster) > 3:
                facets = clusters.findConvexHullFacets(len(cluster), clusterPos)
            
            elif len(cluster) == 3:
                facets = []
                facets.append([0, 1, 2])
            
            # now render
            if facets is not None:
                #TODO: make sure not facets more than neighbour rad from cell
                facets = clusters.checkFacetsPBCs(facets, clusterPos, 2.0 * settings.neighbourRadius, self.mainWindow.PBC, lattice.cellDims)
                
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
                        facets = clusters.checkFacetsPBCs(facets, tmpClusterPos, 2.0 * settings.neighbourRadius, self.mainWindow.PBC, lattice.cellDims)
                        
                        renderer.getActorsForHullFacets(facets, tmpClusterPos, self.mainWindow, self.actorsCollection, settings)
                        
                        # write povray file too
                        renderer.writePovrayHull(facets, tmpClusterPos, self.mainWindow, hullPovFile, settings)
                
    
    def clusterFilterCalculateVolumes(self, clusterList, filterSettings):
        """
        Calculate volumes of clusters.
        
        """    
        # this will not work properly over PBCs at the moment
        lattice = self.mainWindow.inputState
        
        # draw them as they are
        count = 0
        for cluster in clusterList:
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
                PBC = self.mainWindow.PBC
                if PBC[0] or PBC[1] or PBC[2]:
                    appliedPBCs = np.zeros(7, np.int32)
                    clusters_c.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, 
                                                         PBC, appliedPBCs, filterSettings.neighbourRadius)
                
                volume, area = clusters.findConvexHullVolume(len(cluster), clusterPos)
            
            self.log("Cluster %d (%d atoms)" % (count, len(cluster)), 0, 4)
            self.log("volume is %f; facet area is %f" % (volume, area), 0, 5)
            
            count += 1


