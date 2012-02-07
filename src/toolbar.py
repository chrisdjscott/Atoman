
"""
The main toolbar

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
try:
    from inputTab import InputTab
except:
    print __name__+ ": ERROR: could not import inputTab"
try:
    from filterTab import FilterTab
except:
    print __name__+ ": ERROR: could not import inputTab"
try:
    import resources
except:
    print __name__+ ": ERROR: could not import resources"




################################################################################
class MainToolbar(QtGui.QDockWidget):
    def __init__(self, parent, width, height):
        super(MainToolbar, self).__init__()
        
        self.mainWindow = parent
        
        self.setWindowTitle("Main Toolbar")
        
        self.setFeatures(self.DockWidgetMovable | self.DockWidgetFloatable)
        
        # set size
        self.toolbarWidth = width
        self.toolbarHeight = height
        self.setFixedWidth(self.toolbarWidth)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        
        # create container for widgets
        self.container = QtGui.QWidget(self)
        containerLayout = QtGui.QVBoxLayout(self.container)
        containerLayout.setSpacing(0)
        containerLayout.setContentsMargins(0,0,0,0)
        containerLayout.setAlignment(QtCore.Qt.AlignTop)
        
        
        # display current file info
        self.currentFileBox = GenericForm(self, self.toolbarWidth, "Current file")
        self.currentFileBox.show()
        
        row = self.currentFileBox.newRow()
        self.currentRefLabel = QtGui.QLabel("Reference: " + str(self.mainWindow.refFile))
        row.addWidget(self.currentRefLabel)
        
        row = self.currentFileBox.newRow()
        self.currentInputLabel = QtGui.QLabel("Input: " + str(self.mainWindow.inputFile))
        row.addWidget(self.currentInputLabel)
        
        containerLayout.addWidget(self.currentFileBox)
                
        # create the tab bar
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        self.tabBar = QtGui.QTabWidget(row)
        self.tabBar.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum)
        
        # add tabs
        self.inputTab = InputTab(self, self.mainWindow, self.toolbarWidth)
        self.tabBar.addTab(self.inputTab, "Input")
        
        self.filterPage = FilterTab(self, self.mainWindow, self.toolbarWidth)
        self.tabBar.addTab(self.filterPage, "Filter")
        
        self.outputPage = QtGui.QWidget(self)
        self.tabBar.addTab(self.outputPage, "Output")
        
        rowLayout.addWidget(self.tabBar)
        
        containerLayout.addWidget(row)
        
        # set the main widget
        self.setWidget(self.container)
    
