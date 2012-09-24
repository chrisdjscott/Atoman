
"""
The filter tab for the main toolbar

@author: Chris Scott

"""
import os
import sys

from PyQt4 import QtGui, QtCore
import vtk

from utilities import iconPath
import filterList
import rendering

try:
    import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################
class FilterTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(FilterTab, self).__init__(parent)
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        
        self.log = self.mainWindow.console.write
        
        self.filterListCount = 0
        self.filterLists = []
        self.onScreenInfo = {}
        self.onScreenInfoActors = vtk.vtkActor2DCollection()
        
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
        
        # add a filter list
        self.addFilterList()
    
    def showFilterSummary(self):
        """
        Show filtering summary.
        
        """
        pass
    
    def removeOnScreenInfo(self):
        """
        Remove on screen info.
        
        """
        self.onScreenInfoActors.InitTraversal()
        actor = self.onScreenInfoActors.GetNextItem()
        while actor is not None:
            try:
                self.mainWindow.VTKRen.RemoveActor(actor)
            except:
                pass
            
            actor = self.onScreenInfoActors.GetNextItem()
        
        self.mainWindow.VTKWidget.ReInitialize()
        
        self.onScreenInfoActors = vtk.vtkActor2DCollection()
    
    def runAllFilterLists(self):
        """
        Run all the filter lists.
        
        """
        self.log("Running all filter lists")
        
        # first remove all old povray files
        os.system("rm -f %s/*.pov" % self.mainWindow.tmpDirectory)
        
        self.scalarBarAdded = False
        self.onScreenInfo = {}
        self.removeOnScreenInfo()
        
        count = 0
        visCount = 0
        for filterList in self.filterLists:
            self.log("Running filter list %d" % (count,), 0, 1)
            
            if filterList.isStaticList():
                self.log("Static filter list: skipping", 0, 2)
            
            else:
                filterList.filterer.runFilters()
            
            count += 1
            
            if filterList.visible:
                visCount += filterList.filterer.NVis
        
        self.onScreenInfo["Visible count"] = "%d visible" % visCount
        
        self.refreshOnScreenInfo()
        
        self.mainWindow.setStatus("Ready")
    
    def refreshOnScreenInfo(self):
        """
        Refresh the on-screen information.
        
        """
        textSel = self.mainWindow.textSelector
        selectedText = textSel.selectedText
        
        print "COUNT", selectedText.count()
        if not selectedText.count():
            return
        
        # add stuff that doesn't change
        if "Atom count" not in self.onScreenInfo:
            self.onScreenInfo["Atom count"] = "%d atoms" % self.mainWindow.inputState.NAtoms
        
        if "Simulation time" not in self.onScreenInfo:
            self.onScreenInfo["Simulation time"] = "%.3f fs" % self.mainWindow.inputState.simTime
        
        # alignment/position stuff
        topy = self.mainWindow.VTKWidget.height() - 5
        topx = 5
#        topyRight = self.mainWindow.VTKWidget.height() - 5
#        topxRight = self.mainWindow.VTKWidget.width() - 220
        
        # loop over selected text
        for i in xrange(selectedText.count()):
            item = selectedText.item(i)
            item = str(item.text())
            print "  ITEM", item
            
            try:
                line = self.onScreenInfo[item]
                print "    LINE", line
                
                # add actor
                actor = rendering.vtkRenderWindowText(line, 20, topx, topy, 0, 0, 0)
                
                topy -= 20
                
                self.onScreenInfoActors.AddItem(actor)
            
            except KeyError:
                print "WARNING: %s not in onScreenInfo dict" % item
        
        # add to render window
        self.onScreenInfoActors.InitTraversal()
        actor = self.onScreenInfoActors.GetNextItem()
        while actor is not None:
            try:
                self.mainWindow.VTKRen.AddActor(actor)
            except:
                pass
            
            actor = self.onScreenInfoActors.GetNextItem()
        
        self.mainWindow.VTKWidget.ReInitialize()
    
    def addFilterList(self):
        """
        Add a new filter list
        
        """
        # widget to hold filter list
        filterListWidget = QtGui.QWidget()
        filterListLayout = QtGui.QVBoxLayout(filterListWidget)
        filterListLayout.setContentsMargins(0, 0, 0, 0)
        
        # add list
        list1 = filterList.FilterList(self, self.mainToolbar, self.mainWindow, self.filterListCount, self.toolbarWidth)
        filterListLayout.addWidget(list1)
        self.filterLists.append(list1)
        
        # add to tab bar
        self.filterTabBar.addTab(filterListWidget, str(self.filterListCount))
        
        # select new tab
        self.filterTabBar.setCurrentIndex(self.filterListCount)
        
        self.filterListCount += 1
    
    def clearAllFilterLists(self):
        """
        Clear all the filter lists
        
        """
        self.log("Clearing all filter lists")
        for filterList in self.filterLists:
            filterList.clearList()
            self.removeFilterList()
        
    def filterTabBarChanged(self, val):
        # guess need to handle addition and removal of tabs here
        pass
    
    def removeFilterList(self):
        """
        Remove a filter list
        
        """
        if self.filterListCount <= 1:
            return
        
        currentList = self.filterTabBar.currentIndex()
        
        self.filterLists[currentList].clearList()
        
        for i in xrange(self.filterListCount):
            if i > currentList:
                self.filterTabBar.setTabText(i, str(i - 1))
        
        self.filterTabBar.removeTab(currentList)
        
        self.filterLists.pop(currentList)
        
        self.filterListCount -= 1
    
    def refreshAllFilters(self):
        """
        Refresh filter settings
        
        """
        self.log("Refreshing filters", 3)
        for filterList in self.filterLists:
            for filterSettings in filterList.currentSettings:
                filterSettings.refresh()



