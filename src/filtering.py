
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
import renderer


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
        
        self.mainWindow.renWinInteract.ReInitialize()
    
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
        
        self.mainWindow.renWinInteract.ReInitialize()
    
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
        
        # render
        actors = []
        if self.parent.defectFilter:
            print "NOT ADDED DEFECT RENDERING YET"
        
        else:
            actors = renderer.getActorsForFilteredSystem(visibleAtoms, self.mainWindow)
        
        for actor in actors:
            self.actorsCollection.AddItem(actor)
        
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
        visSpecArray = np.empty(len(settings.visibleSpecieList), np.int32)
        
        count = 0
        for i in xrange(len(self.mainWindow.inputState.specieList)):
            if self.mainWindow.inputState.specieList[i] in settings.visibleSpecieList:
                visSpecArray[count] = i
                count += 1
        
        NVisible = filtering_c.specieFilter(visibleAtoms, visSpecArray, self.mainWindow.inputState.specie)
        
        visibleAtoms.resize(NVisible, refcheck=False)
        
        print "NVISIBLE", NVisible










