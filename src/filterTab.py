
"""
The filter tab for the main toolbar

@author: Chris Scott

"""

import os
import sys

from PyQt4 import QtGui, QtCore, Qt

from utilities import iconPath
from genericForm import GenericForm
import resources
import filtering
import filterList


################################################################################
class FilterTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(FilterTab, self).__init__(parent)
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        self.log = self.mainWindow.console.write
        
        self.filterListCount = 1
        self.filterLists = []
        
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
        runAll = QtGui.QPushButton(QtGui.QIcon(iconPath('view-refresh-all.svg')),'Apply lists')
        runAll.setStatusTip("Apply all filter lists")
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
        self.filterTabBar = QtGui.QTabWidget(self)
        self.filterTabBar.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.connect(self.filterTabBar, QtCore.SIGNAL('currentChanged(int)'), self.filterTabBarChanged)
        filterTabLayout.addWidget(self.filterTabBar)
        
        # widget to hold filter list
        self.filterListWidget = QtGui.QWidget()
        self.filterListLayout = QtGui.QVBoxLayout(self.filterListWidget)
        self.filterListLayout.setContentsMargins(0, 0, 0, 0)
        
        # add list
        list1 = filterList.FilterList(self, self.mainToolbar, self.mainWindow, self.filterListCount, self.toolbarWidth)
        self.filterListLayout.addWidget(list1)
        self.filterLists.append(list1)
        
        # add to tab bar
        self.filterTabBar.addTab(self.filterListWidget, str(self.filterListCount))
        
    def runAllFilterLists(self):
        """
        Run all the filter lists.
        
        """
        self.log("Running all filter lists")
        count = 0
        for filterList in self.filterLists:
            self.log("Running filter list %d" % (count,), 0, 1)
            filterList.filterer.runFilters()
            
            count += 1

    def addFilterList(self):
        pass
    
    def clearAllFilterLists(self):
        pass

    def filterTabBarChanged(self, val):
        # guess need to handle addition and removal of tabs here
        pass
    
    def removeFilterList(self):
        pass
    
    def refreshAllFilters(self):
        """
        Refresh filter settings
        
        """
        self.log("Refreshing filters", 3)
        for filterList in self.filterLists:
            for filterSettings in filterList.currentSettings:
                filterSettings.refresh()



