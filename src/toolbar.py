
"""
The main toolbar

author: Chris Scott
last edited: February 2012
"""

import os
import sys

try:
    from PyQt4 import QtGui, QtCore, Qt
except:
    sys.exit(__name__, "ERROR: PyQt4 not found")

try:
    from utilities import iconPath
except:
    sys.exit(__name__, "ERROR: utilities not found")


################################################################################
class MainToolbar(QtGui.QDockWidget):
    def __init__(self, parent, width, height):
        super(MainToolbar, self).__init__()
        
#        self.parent = parent
        self.mainWindow = parent
        
        self.setWindowTitle("Main Toolbar")
        
        # set size
        self.toolbarWidth = width
        self.toolbarHeight = height
        self.setFixedWidth(self.toolbarWidth)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        
        # create main widget
#        self.mainWidget = GenericForm(self, self.toolbarWidth, "Toolbar Main")
        
        # create the tab bar
        self.tabContainer = QtGui.QWidget(self)
        self.tabBar = QtGui.QTabWidget(self.tabContainer)
        self.tabBar.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Minimum)
        self.setTitleBarWidget(self.tabBar)
        
        # add tabs
        self.inputPage = QtGui.QWidget(self)
        self.tabBar.addTab(self.inputPage, "Input")
        
        self.filterPage = QtGui.QWidget(self)
        self.tabBar.addTab(self.filterPage, "Filter")
        
        self.outputPage = QtGui.QWidget(self)
        self.tabBar.addTab(self.outputPage, "Output")
        
        # set the main widget
        self.setWidget(self.tabContainer)
