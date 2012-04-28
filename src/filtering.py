
"""
The filter tab for the main toolbar

@author: Chris Scott

"""
import os
import sys
import tempfile
import subprocess
import copy

from PyQt4 import QtGui, QtCore
import numpy as np
import vtk

from visclibs import filtering_c
from visclibs import defects_c
from visclibs import clusters_c
import rendering


################################################################################
class Filterer:
    """
    Filter class.
    
    Contains list of subfilters to be performed in order.
    
    """
    def __init__(self, parent):
        
        self.parent = parent
        self.mainWindow = self.parent.mainWindow
        
        self.log = self.mainWindow.console.write
        
        self.actorsCollection = vtk.vtkActorCollection()
    
    def removeActors(self):
        """
        Remove actors.
        
        """
        self.hideActors()
        
        self.actorsCollection = vtk.vtkActorCollection()
    
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
    
    def runFilters(self):
        """
        Run the filters.
        
        """
        self.removeActors()
        
        # first set up visible atoms arrays
        NAtoms = self.parent.mainWindow.inputState.NAtoms
        
        if not self.parent.defectFilterSelected:
            visibleAtoms = np.arange(NAtoms, dtype=np.int32)
            self.log("%d visible atoms" % (len(visibleAtoms),), 0, 2)
        
        # run filters
        currentFilters = self.parent.currentFilters
        currentSettings = self.parent.currentSettings
        for i in xrange(len(currentFilters)):
            filterName = currentFilters[i]
            filterSettings = currentSettings[i]
            
            self.log("Running filter: %s" % (filterName,), 0, 2)
            
            if filterName == "Specie":
                self.filterSpecie(visibleAtoms, filterSettings)
            
            elif filterName == "Crop":
                self.cropFilter(visibleAtoms, filterSettings)
            
            elif filterName == "Point defects":
                interstitials, vacancies, antisites, onAntisites = self.pointDefectFilter(filterSettings)
            
            elif filterName == "Clusters":
                clusterList = self.clusterFilter(visibleAtoms, filterSettings)
                
                if filterSettings.drawConvexHulls:
                    self.clusterFilterDrawHulls(clusterList, filterSettings)
            
            if self.parent.defectFilterSelected:
                NVis = len(interstitials) + len(vacancies) + len(antisites)
                self.log("%d visible atoms" % (NVis,), 0, 3)
            
            else:
                self.log("%d visible atoms" % (len(visibleAtoms),), 0, 3)
        
        # render
        if self.parent.defectFilterSelected:
            rendering.getActorsForFilteredDefects(interstitials, vacancies, antisites, onAntisites, self.mainWindow, self.actorsCollection)
        
        else:
            rendering.getActorsForFilteredSystem(visibleAtoms, self.mainWindow, self.actorsCollection)
                
        if self.parent.visible:
            self.addActors()
    
    def getActorsForFilteredSystem(self, visibleAtoms):
        """
        Render systems after applying filters.
        
        """
        pass
            
    def filterSpecie(self, visibleAtoms, settings):
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
        
        NVisible = filtering_c.specieFilter(visibleAtoms, visSpecArray, self.mainWindow.inputState.specie)
        
        visibleAtoms.resize(NVisible, refcheck=False)

    def cropFilter(self, visibleAtoms, settings):
        """
        Crop lattice
        
        """
        lattice = self.mainWindow.inputState
        
        NVisible = filtering_c.cropFilter(visibleAtoms, lattice.pos, settings.xmin, settings.xmax, settings.ymin, 
                                          settings.ymax, settings.zmin, settings.zmax, settings.xEnabled, 
                                          settings.yEnabled, settings.zEnabled)
        
        visibleAtoms.resize(NVisible, refcheck=False)
    
    def pointDefectFilter(self, settings):
        """
        Point defects filter
        
        """
        inputLattice = self.mainWindow.inputState
        refLattice = self.mainWindow.refState
        
        # set up arrays
        if settings.showInterstitials:
            interstitials = np.empty(inputLattice.NAtoms, np.int32)
        else:
            interstitials = np.empty(0, np.int32)
        if settings.showVacancies:
            vacancies = np.empty(refLattice.NAtoms, np.int32)
        else:
            vacancies = np.empty(0, np.int32)
        if settings.showAntisites:
            antisites = np.empty(refLattice.NAtoms, np.int32)
            onAntisites = np.empty(refLattice.NAtoms, np.int32)
        else:
            antisites = np.empty(0, np.int32)
            onAntisites = np.empty(0, np.int32)
        
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
        
        NDefectsByType = np.zeros(4, np.int32)
        
        # set min/max pos to lattice (for boxing)
        minPos = refLattice.minPos
        maxPos = refLattice.maxPos
        
        # call C library
        status = defects_c.findDefects(settings.showVacancies, settings.showInterstitials, settings.showAntisites, NDefectsByType, vacancies, 
                                       interstitials, antisites, onAntisites, exclSpecsInput, exclSpecsRef, inputLattice.NAtoms, inputLattice.specieList,
                                       inputLattice.specie, inputLattice.pos, refLattice.NAtoms, refLattice.specieList, refLattice.specie, 
                                       refLattice.pos, refLattice.cellDims, self.mainWindow.PBC, settings.vacancyRadius, minPos, maxPos)
        
        # summarise
        NDef = NDefectsByType[0]
        NVac = NDefectsByType[1]
        NInt = NDefectsByType[2]
        NAnt = NDefectsByType[3]
        vacancies.resize(NVac)
        interstitials.resize(NInt)
        antisites.resize(NAnt)
        onAntisites.resize(NAnt)
        
        self.log("Found %d defects" % (NDef,), 0, 3)
        self.log("%d vacancies" % (NVac,), 0, 4)
        self.log("%d interstitials" % (NInt,), 0, 4)
        self.log("%d antisites" % (NAnt,), 0, 4)
        
        return (interstitials, vacancies, antisites, onAntisites)
    
    def clusterFilter(self, visibleAtoms, settings, PBC=None, minSize=None, nebRad=None):
        """
        Run the cluster filter
        
        """
        lattice = self.mainWindow.inputState
        
        atomCluster = np.empty(len(visibleAtoms), np.int32)
        result = np.empty(2, np.int32)
        
        if PBC is not None and len(PBC) == 3:
            pass
        else:
            PBC = self.mainWindow.PBC
        
        if minSize is None:
            minSize = settings.minClusterSize
        
        if nebRad is None:
            nebRad = settings.neighbourRadius
        
        # set min/max pos to lattice (for boxing)
        minPos = np.zeros(3, np.float64)
        maxPos = copy.deepcopy(lattice.cellDims)
        
        clusters_c.findClusters(visibleAtoms, lattice.pos, atomCluster, nebRad, lattice.cellDims, PBC, 
                                minPos, maxPos, minSize, result)
        
        NVisible = result[0]
        NClusters = result[1]
        
        visibleAtoms.resize(NVisible, refcheck=False)
        atomCluster.resize(NVisible, refcheck=False)
        
        # build cluster lists
        clusterList = []
        for i in xrange(NClusters):
            clusterList.append([])
        
        # add atoms to cluster lists
        clusterIndexMapper = {}
        count = 0
        for i in xrange(NVisible):
            atomIndex = visibleAtoms[i]
            clusterIndex = atomCluster[i]
            
            if clusterIndex not in clusterIndexMapper:
                clusterIndexMapper[clusterIndex] = count
                count += 1
            
            clusterListIndex = clusterIndexMapper[clusterIndex]
            
            clusterList[clusterListIndex].append(atomIndex)
        
        #TODO: rebuild scalars array of atom cluster (so can colour by cluster maybe?)
        
        
        return clusterList
    
    def clusterFilterDrawHulls(self, clusterList, settings):
        """
        Draw hulls around filters.
        
        If the clusterList was created using PBCs we need to recalculate each 
        cluster without PBCs.
        
        """
        PBC = self.mainWindow.PBC
        if PBC[0] or PBC[1] or PBC[2]:
            self.clusterFilterDrawHullsWithPBCs(clusterList, settings)
        
        else:
            self.clusterFilterDrawHullsNoPBCs(clusterList, settings)
    
    def clusterFilterDrawHullsNoPBCs(self, clusterList, settings):
        """
        
        
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
                facets = findConvexHull(cluster, clusterPos, qconvex=settings.qconvex)
            
            # now render
            if facets is not None:
                rendering.getActorsForHullFacets(facets, clusterPos, self.mainWindow, self.actorsCollection)
    
    def clusterFilterDrawHullsWithPBCs(self, clusterList, settings):
        """
        
        
        """
        # recalc each volume with PBCs off
        for cluster in clusterList:
            clusterAtoms = np.empty(len(cluster), np.int32)
            for i in xrange(len(cluster)):
                clusterAtoms[i] = cluster[i]
            
            subClusterList = self.clusterFilter(clusterAtoms, settings, PBC=np.zeros(3, np.int32), minSize=1, nebRad=1.5*settings.neighbourRadius)
            
            self.clusterFilterDrawHullsNoPBCs(subClusterList, settings)
    





def findConvexHull(cluster, pos, qconvex="qconvex"):
    """
    Find convex hull of given points
    
    """
    # write to file
    fh, fn = tempfile.mkstemp(dir="/tmp")
    f = open(fn, "w")
    f.write("3\n")
    f.write("%d\n" % (len(cluster),))
    for i in xrange(len(cluster)):
        string = "%f %f %f\n" % (pos[3*i], pos[3*i+1], pos[3*i+2])
        f.write(string)
    f.close()
    
    command = "%s Qt i < %s" % (qconvex, fn,)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, stderr = proc.communicate()
    status = proc.poll()
    if status:
        print "qconvex failed"
        print stderr
        os.unlink(fn)
        sys.exit(35)
    
    output = output.strip()
    lines = output.split("\n")
    
    lines.pop(0)
    
    facets = []
    for line in lines:
        array = line.split()
        facets.append([np.int32(array[0]), np.int32(array[1]), np.int32(array[2])])
    
    # unlink temp file
    os.unlink(fn)
    
    return facets
