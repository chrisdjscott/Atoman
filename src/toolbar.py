
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
    sys.exit(__name__, "ERROR: PyQt4 not found")

try:
    from utilities import iconPath
except:
    sys.exit(__name__, "ERROR: utilities not found")


################################################################################
class InputTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(InputTab, self).__init__()
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        
        # layout
        inputTabLayout = QtGui.QVBoxLayout(self)
        
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        
        self.tabBar = QtGui.QTabWidget(row)
        
        # add LBOMD page
        self.LBOMDTab = QtGui.QWidget(self)
        self.tabBar.addTab(self.LBOMDTab, "LBOMD")
        
        # add generic page
        self.genericTab = QtGui.QWidget(self)
        self.tabBar.addTab(self.genericTab, "Generic")
        
        rowLayout.addWidget(self.tabBar)
        
        inputTabLayout.addWidget(row)


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
#        containerLayout.setSpacing(0)
#        containerLayout.setContentsMargins(0,0,0,0)
        containerLayout.setAlignment(QtCore.Qt.AlignTop)
                
        # create info widget
        group = QtGui.QGroupBox("Current file")
        groupLayout = QtGui.QVBoxLayout()
        groupLayout.setSpacing(0)
        groupLayout.setContentsMargins(0,0,0,0)
        groupLayout.setAlignment(QtCore.Qt.AlignTop)
        
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setContentsMargins(0,0,0,0)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        self.currentRefLabel = QtGui.QLabel("Reference: " + str(self.mainWindow.refFile))
        self.currentRefLabel.setWordWrap(1)
        rowLayout.addWidget(self.currentRefLabel)
        groupLayout.addWidget(row)
        
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setContentsMargins(0,0,0,0)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        self.currentRefLabel = QtGui.QLabel("Input: " + str(self.mainWindow.inputFile))
        self.currentRefLabel.setWordWrap(1)
        rowLayout.addWidget(self.currentRefLabel)
        groupLayout.addWidget(row)
        
        group.setLayout(groupLayout)
        
        containerLayout.addWidget(group)
        
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
        
        self.filterPage = QtGui.QWidget(self)
        self.tabBar.addTab(self.filterPage, "Filter")
        
        self.outputPage = QtGui.QWidget(self)
        self.tabBar.addTab(self.outputPage, "Output")
        
        rowLayout.addWidget(self.tabBar)
        
        containerLayout.addWidget(row)
        
        # set the main widget
        self.setWidget(self.container)
    
