
"""
The filter tab for the main toolbar

author: Chris Scott
last edited: February 2012
"""

import os
import sys

from PyQt4 import QtGui, QtCore

from utilities import iconPath
from genericForm import GenericForm








################################################################################
class SpecieFilter:
    def __init__(self):
        
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









