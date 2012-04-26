
"""
The filter tab for the main toolbar

@author: Chris Scott

"""

import os
import sys

from PyQt4 import QtGui, QtCore
import numpy as np
import vtk

from utilities import iconPath
from genericForm import GenericForm
from visclibs import filtering_c
from visclibs import defects_c
import rendering


################################################################################
class VisibleObjects:
    def __init__(self, visibleAtoms, useRefPos=0):
        
        self.useRefPos = useRefPos
        
        self.visibleDict = {}
        
        self.visibleDict["ATOMS"] = visibleAtoms
    
        


################################################################################
class Filterer:
    """
    Filter class.
    
    Contains list of subfilters to be performed in order.
    
    """
    def __init__(self, parent):
        
        self.parent = parent
        self.mainWindow = self.parent.mainWindow
        
        self.actorsCollection = vtk.vtkActorCollection()
    
    def removeActors(self):
        """
        Remove all actors
        
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
                print "ADDING ACTOR"
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
        
        self.actorsCollection = vtk.vtkActorCollection()
        
        # first set up visible atoms arrays
        NAtoms = self.parent.mainWindow.inputState.NAtoms
        
        if not self.parent.defectFilterSelected:
            visibleAtoms = np.arange(NAtoms, dtype=np.int32)
        
        # run filters
        currentFilters = self.parent.currentFilters
        currentSettings = self.parent.currentSettings
        for i in xrange(len(currentFilters)):
            filterName = currentFilters[i]
            filterSettings = currentSettings[i]
            
            if filterName == "Specie":
                self.filterSpecie(visibleAtoms, filterSettings)
            
            elif filterName == "Crop":
                self.cropFilter(visibleAtoms, filterSettings)
            
            elif filterName == "Point defects":
                self.pointDefectFilter(filterSettings)
        
        # render
        if self.parent.defectFilterSelected:
            print "NOT ADDED DEFECT RENDERING YET"
        
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
        print "RUNNING SPECIE FILTER"
        if settings.allSpeciesSelected:
            visSpecArray = np.arange(len(self.mainWindow.inputState.specieList), dtype=np.int32)
        
        else:
            visSpecArray = np.empty(len(settings.visibleSpecieList), np.int32)
            count = 0
            for i in xrange(len(self.mainWindow.inputState.specieList)):
                if self.mainWindow.inputState.specieList[i] in settings.visibleSpecieList:
                    visSpecArray[count] = i
                    count += 1
        
        NVisible = filtering_c.specieFilter(visibleAtoms, visSpecArray, self.mainWindow.inputState.specie)
        
        visibleAtoms.resize(NVisible, refcheck=False)
        
        print "NVISIBLE", NVisible

    def cropFilter(self, visibleAtoms, settings):
        """
        Crop lattice
        
        """
        lattice = self.mainWindow.inputState
        
        print "X", settings.xEnabled, settings.xmin, settings.xmax
        print "Y", settings.yEnabled, settings.ymin, settings.ymax
        print "Z", settings.zEnabled, settings.zmin, settings.zmax
        
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
        
        print "EXCLUDE SPECS INPUT"
        print inputLattice.specieList
        print exclSpecsInput
        print "EXCLUDE SPECS REF"
        print refLattice.specieList
        print exclSpecsRef
        
        NDefectsByType = np.zeros(4, np.int32)
        
        # call C library
        status = defects_c.findDefects(settings.showVacancies, settings.showInterstitials, settings.showAntisites, NDefectsByType, vacancies, 
                                       interstitials, antisites, onAntisites, exclSpecsInput, exclSpecsRef, inputLattice.NAtoms, inputLattice.specieList,
                                       inputLattice.specie, inputLattice.pos, refLattice.NAtoms, refLattice.specieList, refLattice.specie, 
                                       refLattice.pos, refLattice.cellDims[0], refLattice.cellDims[1], refLattice.cellDims[2], int(self.mainWindow.PBC[0]),
                                       int(self.mainWindow.PBC[1]), int(self.mainWindow.PBC[2]), settings.vacancyRadius, refLattice.minPos[0], 
                                       refLattice.minPos[1], refLattice.minPos[2], refLattice.maxPos[0], refLattice.maxPos[1], refLattice.maxPos[2])
        
        





