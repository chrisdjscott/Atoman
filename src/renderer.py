
"""
Module for rendering

author: Chris Scott
last edited: February 2012
"""

import os
import sys

#try:
#    from PyQt4 import QtGui, QtCore
#except:
#    print __name__+ ": ERROR: could not import PyQt4"
try:
    import vtk
except:
    print __name__+ ": ERROR: could not import vtk"





################################################################################
class Renderer:
    def __init__(self, mainWindow):
        
        self.mainWindow = mainWindow
        
        
    
    
    def setCameraToCell(self):
        """
        Point the camera at the centre of the cell.
        
        """
        pass
    
    def setCameraToCOM(self):
        """
        Point the camera at the centre of mass.
        
        """
        pass
    
    def writeCameraSettings(self):
        """
        Write the camera settings to file.
        So can be loaded back in future
        OPTION TO WRITE TO TMPDIR IF WANT!!!
        
        """
        pass
    
    def addCellOutline(self):
        """
        Add the cell outline.
        
        """
        cellDims = self.mainWindow.refState.cellDims
        
    def removeCellOutline(self):
        """
        Remove the cell outline.
        
        """
        pass
    
    def addAxes(self):
        """
        Add the axis label
        
        """
        pass
    
    def removeAxes(self):
        """
        Remove the axis label
        
        """
        pass
