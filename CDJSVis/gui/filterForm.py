
"""
The filter tab for the main toolbar

@author: Chris Scott

"""
import os
import sys
import glob

from PySide import QtGui, QtCore
import vtk
import numpy as np

from ..visutils.utilities import iconPath
from . import filterList
from ..rendering.text import vtkRenderWindowText
from ..visutils import utilities

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################
class FilterForm(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width, pipelineIndex, pipelineString):
        super(FilterForm, self).__init__(parent)
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        self.pipelineIndex = pipelineIndex
        self.pipelineString = pipelineString
        
        self.log = self.mainWindow.console.write
        
        self.rendererWindows = self.mainWindow.rendererWindows
        
        self.filterListCount = 0
        self.filterLists = []
        self.onScreenInfo = {}
        self.onScreenInfoActors = vtk.vtkActor2DCollection()
        self.visAtomsList = []
        
        self.refState = None
        self.inputState = None
        self.extension = None
        
        # layout
        filterTabLayout = QtGui.QVBoxLayout(self)
        filterTabLayout.setContentsMargins(0, 0, 0, 0)
        filterTabLayout.setSpacing(0)
        filterTabLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # row 
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)
        
        # reference selector
        self.refCombo = QtGui.QComboBox()
        self.refCombo.currentIndexChanged.connect(self.refChanged)        
        
        # add to row
        rowLayout.addWidget(QtGui.QLabel("Reference:"))
        rowLayout.addWidget(self.refCombo)
        filterTabLayout.addWidget(row)
        
        # row 
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)
        
        # reference selector
        self.inputCombo = QtGui.QComboBox()
        self.inputCombo.currentIndexChanged.connect(self.inputChanged)
        
        # add to row
        rowLayout.addWidget(QtGui.QLabel("Input:"))
        rowLayout.addWidget(self.inputCombo)
        filterTabLayout.addWidget(row)
        
        # row
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
        
        # refresh if ref already loaded
        if self.mainWindow.refLoaded:
            self.refreshAllFilters()
    
    def addStateOption(self, filename):
        """
        Add state option to combo boxes
        
        """
        print "ADD STATE OPTION", filename
        
        self.refCombo.addItem(filename)
        self.inputCombo.addItem(filename)
    
    def checkStateChangeOk(self):
        """
        Check it was ok to change the state.
        
        """
        if self.inputState is None:
            return
        
        ref = self.refState
        inp = self.inputState
        
        diff = False
        for i in xrange(3):
            if inp.cellDims[i] != ref.cellDims[i]:
                diff = True
                break
        
        if diff:
            self.mainWindow.console.write("WARNING: cell dims differ")
            
        return diff
    
    def postRefLoaded(self, oldRef):
        """
        Do stuff after the ref has been loaded.
        
        """
        if oldRef is not None:
            self.clearAllActors()
            self.refreshAllFilters()
                
            for rw in self.rendererWindows:
                if rw.currentPipelineIndex == self.pipelineIndex:
                    rw.textSelector.refresh()
                    
                    rw.outputDialog.rdfTab.refresh()
        
        self.mainWindow.readLBOMDIN()
        
        for rw in self.rendererWindows:
            if rw.currentPipelineIndex == self.pipelineIndex:
                rw.renderer.postRefRender()
                rw.textSelector.refresh()
    
    def postInputLoaded(self):
        """
        Do stuff after the input has been loaded
        
        """
#         self.setCurrentInputFile(filename)
#         self.inputLoaded = 1
        
#         self.mainToolbar.loadInputForm.hide()
        self.mainToolbar.analysisPipelinesForm.show()
        
        self.refreshAllFilters()
        
        for rw in self.rendererWindows:
            if rw.currentPipelineIndex == self.pipelineIndex:
                rw.textSelector.refresh()
                rw.outputDialog.rdfTab.refresh()
    
    def refChanged(self, index):
        """
        Ref changed
        
        """
        old_ref = self.refState
        
        self.refState = self.mainWindow.systemsDialog.lattice_list[index]
        self.extension = self.mainWindow.systemsDialog.extensions_list[index]
        
        # read lbomd in?
        
        
        # post ref loaded
        self.postRefLoaded(old_ref)
        
        # check ok
        status = self.checkStateChangeOk()
        
        if status:
            # must change input too
            self.inputChanged(index)
    
    def inputChanged(self, index):
        """
        Input changed
        
        """
        self.inputState = self.mainWindow.systemsDialog.lattice_list[index]
        self.extension = self.mainWindow.systemsDialog.extensions_list[index]
        
        # check ok
        status = self.checkStateChangeOk()
        
        if status:
            # must change ref too
            self.refChanged(index)
        
        # post input loaded
        self.postInputLoaded()
    
    def showFilterSummary(self):
        """
        Show filtering summary.
        
        """
        pass
    
    def removeOnScreenInfo(self):
        """
        Remove on screen info.
        
        """
        for rw in self.mainWindow.rendererWindows:
            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                rw.removeOnScreenInfo()
    
    def clearAllActors(self):
        """
        Clear all actors.
        
        """
        self.log("Clearing all actors")
        
        for filterList in self.filterLists:
            filterList.filterer.removeActors()
    
    def runAllFilterLists(self):
        """
        Run all the filter lists.
        
        """
        self.log("Running all filter lists")
        
        # first remove all old povray files
        oldpovfiles = glob.glob(os.path.join(self.mainWindow.tmpDirectory, "pipeline%d_*.pov" % self.pipelineIndex))
        for fn in oldpovfiles:
            os.unlink(fn)
        
        self.scalarBarAdded = False
        
        count = 0
        for filterList in self.filterLists:
            self.log("Running filter list %d" % (count,), 0, 1)
            
            if filterList.isStaticList():
                self.log("Static filter list: skipping", 0, 2)
            
            else:
                filterList.filterer.runFilters()
            
            count += 1
        
        self.refreshOnScreenInfo()
        
        self.mainWindow.setStatus("Ready")
    
    def refreshOnScreenInfo(self):
        """
        Refresh the on-screen information.
        
        """
        for rw in self.mainWindow.rendererWindows:
            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                rw.refreshOnScreenInfo()
    
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
        self.visAtomsList.append([])
        
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
        
        self.visAtomsList.pop(currentList)
        
        self.filterListCount -= 1
    
    def refreshAllFilters(self):
        """
        Refresh filter settings
        
        """
        self.log("Refreshing filters", 3)
        for filterList in self.filterLists:
            for filterSettings in filterList.currentSettings:
                filterSettings.refresh()
            
            filterList.bondsOptions.refresh()
    
    def gatherVisibleAtoms(self):
        """
        Builds an array containing all (unique) visible atoms.
        
        """
        visibleAtomsFull = None
        for filterList in self.filterLists:
            visibleAtoms = filterList.filterer.visibleAtoms
            
            if visibleAtomsFull is None:
                visibleAtomsFull = visibleAtoms
            else:
                visibleAtomsFull = np.append(visibleAtomsFull, visibleAtoms)
        
        visibleAtomsFull = np.unique(visibleAtomsFull)
        
        return visibleAtomsFull
    
    def broadcastToRenderers(self, method, args=(), kwargs={}, globalBcast=False):
        """
        Broadcast command to associated renderers.
        
        """
        rwList = []
        for rw in self.mainWindow.rendererWindows:
            if globalBcast:
                rwList.append(rw)
            
            elif rw.currentPipelineString == self.pipelineString:
                rwList.append(rw)
        
        for rw in rwList:
            if hasattr(rw, method):
                call = getattr(rw, method)
                
                call(*args, **kwargs)



