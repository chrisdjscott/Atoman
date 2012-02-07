
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
    print __name__+ "ERROR: could not import PyQt4"

try:
    from utilities import iconPath
except:
    print __name__+ "ERROR: could not import utilities"
try:
    from genericForm import GenericForm
except:
    print __name__+ "ERROR: could not import genericForm"
try:
    import resources
except:
    print __name__+ ": ERROR: could not import resources"
    

################################################################################
class FilterList(QtGui.QWidget):
    def __init__(self, parent, mainToolbar, mainWindow, tab, width, height=120):
        super(FilterList, self).__init__()
        
        self.filterTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.tab = tab
        self.tabWidth = width
        self.tabHieght = height
        
        


################################################################################
class FilterTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(FilterTab, self).__init__()
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        # layout
        filterTabLayout = QtGui.QVBoxLayout(self)
        filterTabLayout.setContentsMargins(0, 0, 0, 0)
        filterTabLayout.setSpacing(0)
        filterTabLayout.setAlignment(QtCore.Qt.AlignTop)
        
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)
        
        #----- buttons for new/trash filter list
        runAll = QtGui.QPushButton(QtGui.QIcon(iconPath('user-trash.svg')),'Apply lists')
        runAll.setStatusTip("Run all filter lists")
        self.connect(runAll, QtCore.SIGNAL('clicked()'), self.runAllFilterLists)
        add = QtGui.QPushButton(QtGui.QIcon(iconPath('tab-new.svg')),'New list')
        add.setStatusTip("New filter list")
        self.connect(add, QtCore.SIGNAL('clicked()'), self.addFilterList)
        clear = QtGui.QPushButton(QtGui.QIcon(iconPath('edit-delete.svg')),'Clear lists')
        clear.setStatusTip("Clear all filter lists")
        self.connect(clear, QtCore.SIGNAL('clicked()'), self.clearAllFilterLists)
        
        rowLayout.addWidget(add)
        rowLayout.addWidget(clear)
        rowLayout.addWidget(runAll)
        
        filterTabLayout.addWidget(row)
        
        #----- add tab bar for filter lists
        
        
        
        
        
        
    def runAllFilterLists(self):
        pass

    def addFilterList(self):
        pass
    
    def clearAllFilterLists(self):
        pass






