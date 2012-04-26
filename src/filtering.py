
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
        print "RUN FILTER NATOMS", NAtoms
        
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
        
        # render
        actors = []
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






