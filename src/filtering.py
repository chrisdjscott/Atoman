
"""
The filter tab for the main toolbar

author: Chris Scott
last edited: February 2012
"""

import os
import sys

try:
    from PyQt4 import QtGui, QtCore
except:
    print __name__+ ": ERROR: could not import PyQt4"

try:
    from utilities import iconPath
except:
    print __name__+ ": ERROR: could not import utilities"
try:
    from genericForm import GenericForm
except:
    print __name__+ ": ERROR: could not import genericForm"








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









