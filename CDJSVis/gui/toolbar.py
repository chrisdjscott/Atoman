
"""
The main toolbar

@author: Chris Scott

"""
import sys

from PyQt4 import QtGui, QtCore

from .genericForm import GenericForm
from .inputTab import InputTab, InputTabNew
from .filterTab import FilterTab
from .outputTab import OutputTab

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)



################################################################################
class MainToolbar(QtGui.QDockWidget):
    def __init__(self, parent, width, height):
        super(MainToolbar, self).__init__(parent)
        
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
        self.inputTab = InputTabNew(self, self.mainWindow, self.toolbarWidth)
        self.tabBar.addTab(self.inputTab, "Input")
        
        self.filterPage = FilterTab(self, self.mainWindow, self.toolbarWidth)
        self.tabBar.addTab(self.filterPage, "Filter")
        self.tabBar.setTabEnabled(1, False)
        
        self.outputPage = OutputTab(self, self.mainWindow, self.toolbarWidth)
        self.tabBar.addTab(self.outputPage, "Output")
        self.tabBar.setTabEnabled(2, False)
        
        rowLayout.addWidget(self.tabBar)
        
        containerLayout.addWidget(row)
        
        # set the main widget
        self.setWidget(self.container)
    
