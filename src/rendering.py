
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
    sys.exit(__name__+ ": ERROR: could not import PyQt4")
try:
    import vtk
except:
    sys.exit(__name__+ ": ERROR: could not import vtk")

try:
    from utilities import iconPath
except:
    sys.exit(__name__+ ": ERROR: could not import utilities")




################################################################################
class Renderer():
    def __init__(self, mainWindow):
        
        self.mainWindow = mainWindow
        
        
        
