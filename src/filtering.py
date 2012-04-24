
"""
The filter tab for the main toolbar

@author: Chris Scott

"""

import os
import sys

from PyQt4 import QtGui, QtCore
import numpy as np

from utilities import iconPath
from genericForm import GenericForm



################################################################################
class Filter:
    """
    Filter class.
    
    Contains list of subfilters to be performed in order.
    
    """
    def __init__(self, parent):
        
        self.parent = parent
        self.mainWindow = self.parent.mainWindow
        
        self.filterList = []
        
        self.actorList = []
        
    def addFilter(self, name, index=None):
        """
        Add given filter to the list.
        
        """
        pass
    
    def removeFilter(self, index):
        """
        Remove filter from list.
        
        """
        # remove actors too?
        pass
    
    def runFilter(self):
        """
        Run the filters.
        
        """
        # first set up visible atoms arrays
        NAtomsInput = self.parent.mainWindow.inputState.NAtoms
        print "RUN FILTER NATOMSINPUT", NAtomsInput
        
        NAtomsRef = self.parent.mainWindow.refState.NAtoms
        print "RUN FILTER NATOMSREF", NAtomsRef
        
        # run filters
        for filter in self.filterList:
            filter.runFilter()
        
        
        




################################################################################
class SpecieFilter:
    def __init__(self):
        
        self.name = "SPECIE FILTER"
        
        self.visibleSpecies = []
    
    def addVisibleSpecie(self, specie):
        
        if specie not in self.visibleSpecies:
            self.visibleSpecies.append(specie)
    
    def remmoeVisibleSpecie(self, specie):
        
        if specie in self.visibleSpecies:
            index = self.visibleSpecies.index(specie)
            self.visibleSpecies.pop(index)
    
    def runFilter(self, mainWindow):
        pass









