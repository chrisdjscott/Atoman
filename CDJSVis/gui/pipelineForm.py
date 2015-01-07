# -*- coding: utf-8 -*-

"""
Analysis is performed using an *Analysis pipeline*, found on the *Analysis toolbar* on the left of the application (see image).  
Multiple pipelines can be configured at once; a pipeline is viewed in a renderer window. 

An individual pipeline takes a reference and an input system as its input and contains one or more filter/calculator lists.  These
lists operate independently of one another and calculate properties or filter the input system. Available filters/calculators are 
shown below:

"""
import os
import sys
import glob
import math
import logging
import functools
import uuid

from PySide import QtGui, QtCore
import vtk
import numpy as np

from ..visutils.utilities import iconPath
from . import filterList
from ..rendering.text import vtkRenderWindowText
from ..visutils import utilities
from . import picker as picker_c
from . import infoDialogs
from . import utils
from ..rendering import highlight
from . import dialogs

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################
class PipelineForm(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width, pipelineIndex, pipelineString):
        super(PipelineForm, self).__init__(parent)
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        self.pipelineIndex = pipelineIndex
        self.pipelineString = pipelineString
        self.systemsDialog = mainWindow.systemsDialog
        
        self.log = self.mainWindow.console.write
        self.logger = logging.getLogger(__name__)
        
        self.rendererWindows = self.mainWindow.rendererWindows
        
        self.pickerContextMenuID = uuid.uuid4()
        self.pickerContextMenu = QtGui.QMenu(self)
        self.pickerContextMenu.aboutToHide.connect(self.hidePickerMenuHighlight)
        
        self.filterListCount = 0
        self.filterLists = []
        self.onScreenInfo = {}
        self.onScreenInfoActors = vtk.vtkActor2DCollection()
        self.visAtomsList = []
        
        self.refState = None
        self.inputState = None
        self.extension = None
        self.inputStackIndex = None
        self.filename = None
        self.currentRunID = None
        self.abspath = None
        self.PBC = None
        
        self.analysisPipelineFormHidden = True
        
        # layout
        filterTabLayout = QtGui.QVBoxLayout(self)
        filterTabLayout.setContentsMargins(0, 0, 0, 0)
        filterTabLayout.setSpacing(0)
        filterTabLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # row 
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)
        label = QtGui.QLabel("<b>Pipeline %d settings</b>" % pipelineIndex)
        rowLayout.addWidget(label)
        filterTabLayout.addWidget(row)
        
        # row 
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)
        
        # reference selector
        self.refCombo = QtGui.QComboBox()
        self.refCombo.currentIndexChanged.connect(self.refChanged)
        for fn in self.systemsDialog.getDisplayNames():
            self.refCombo.addItem(fn)    
        
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
        for fn in self.systemsDialog.getDisplayNames():
            self.inputCombo.addItem(fn)
        
        # add to row
        rowLayout.addWidget(QtGui.QLabel("Input:"))
        rowLayout.addWidget(self.inputCombo)
        filterTabLayout.addWidget(row)
        
        row = QtGui.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.addWidget(QtGui.QLabel("<b>Property/filter lists:</b>"))
        filterTabLayout.addLayout(row)
        
        # row
        row = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)
        
        #----- buttons for new/trash filter list
        runAll = QtGui.QPushButton(QtGui.QIcon(iconPath('view-refresh-all.svg')),'Apply lists')
        runAll.setStatusTip("Apply all property/filter lists")
        runAll.setToolTip("Apply all property/filter lists")
        runAll.clicked.connect(self.runAllFilterLists)
        add = QtGui.QPushButton(QtGui.QIcon(iconPath('tab-new.svg')),'New list')
        add.setToolTip("New property/filter list")
        add.setStatusTip("New property/filter list")
        add.clicked.connect(self.addFilterList)
        clear = QtGui.QPushButton(QtGui.QIcon(iconPath('edit-delete.svg')),'Clear lists')
        clear.setStatusTip("Clear all property/filter lists")
        clear.setToolTip("Clear all property/filter lists")
        clear.clicked.connect(self.clearAllFilterLists)
        
        rowLayout.addWidget(add)
        rowLayout.addWidget(clear)
        rowLayout.addWidget(runAll)
        
        filterTabLayout.addWidget(row)
        
        #----- add tab bar for filter lists
        self.filterTabBar = QtGui.QTabWidget(self)
        self.filterTabBar.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.filterTabBar.currentChanged[int].connect(self.filterTabBarChanged)
        self.filterTabBar.setTabsClosable(True)
        self.filterTabBar.tabCloseRequested.connect(self.tabCloseRequested)
        filterTabLayout.addWidget(self.filterTabBar)
        
        # add a filter list
        self.addFilterList()
        
        # add pbc options
        group = QtGui.QGroupBox("Periodic boundaries")
        group.setAlignment(QtCore.Qt.AlignHCenter)
        
        groupLayout = QtGui.QVBoxLayout(group)
        
        self.replicateCellButton = QtGui.QPushButton("Replicate cell")
        self.replicateCellButton.clicked.connect(self.replicateCell)
        self.replicateCellButton.setToolTip("Replicate in periodic directions")
        groupLayout.addWidget(self.replicateCellButton)
        
        self.PBCXCheckBox = QtGui.QCheckBox("x")
        self.PBCXCheckBox.setChecked(1)
        self.PBCYCheckBox = QtGui.QCheckBox("y")
        self.PBCYCheckBox.setChecked(1)
        self.PBCZCheckBox = QtGui.QCheckBox("z")
        self.PBCZCheckBox.setChecked(1)
        
        self.PBCXCheckBox.stateChanged[int].connect(self.PBCXChanged)
        self.PBCYCheckBox.stateChanged[int].connect(self.PBCYChanged)
        self.PBCZCheckBox.stateChanged[int].connect(self.PBCZChanged)
        
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.addWidget(self.PBCXCheckBox)
        rowLayout.addWidget(self.PBCYCheckBox)
        rowLayout.addWidget(self.PBCZCheckBox)
        
        groupLayout.addWidget(row)
        
        filterTabLayout.addWidget(group)
        
        # refresh if ref already loaded
        if self.mainWindow.refLoaded:
            self.refreshAllFilters()
    
    def replicateCell(self):
        """
        Replicate cell?
        
        """
        self.logger.warning("'Replicate cell' is an experimental feature!")
        
        dlg = dialogs.ReplicateCellDialog(self.PBC, parent=self)
        status = dlg.exec_()
        
        if status == QtGui.QDialog.Accepted:
            print "Replicating cell"
            
            repDirs = np.zeros(3, np.int32)
            replicate = False
            
            if dlg.replicateInXCheck.isChecked():
                repDirs[0] = 1
                replicate = True
                self.PBCXCheckBox.setCheckState(QtCore.Qt.Unchecked)
            
            if dlg.replicateInYCheck.isChecked():
                repDirs[1] = 1
                replicate = True
                self.PBCYCheckBox.setCheckState(QtCore.Qt.Unchecked)
            
            if dlg.replicateInZCheck.isChecked():
                repDirs[2] = 1
                replicate = True
                self.PBCZCheckBox.setCheckState(QtCore.Qt.Unchecked)
            
            if replicate:
                self.logger.warning("Replicating cell: this will modify the current input state everywhere")
                self.logger.debug("Replicating cell: %r", repDirs)
                
                #TODO: write in C
                lattice = self.inputState
                numatoms = lattice.NAtoms
                halfCell = lattice.cellDims / 2.0
                newpos = np.empty(3, np.float64)
                cellDims = lattice.cellDims
                for i in xrange(numatoms):
                    sym = lattice.atomSym(i)
                    pos = lattice.atomPos(i)
                    q = lattice.charge[i]
                    ke = lattice.KE[i]
                    pe = lattice.PE[i]
                    for j in xrange(3):
                        if repDirs[j]:
                            if pos[j] < halfCell[j]:
                                newpos[:] = pos[:]
                                newpos[j] += cellDims[j]
                                lattice.addAtom(sym, newpos, q, KE=ke, PE=pe)
                            
                            else:
                                newpos[:] = pos[:]
                                newpos[j] -= cellDims[j]
                                lattice.addAtom(sym, newpos, q, KE=ke, PE=pe)
                
                # run all filter lists
                self.runAllFilterLists()
    
    def PBCXChanged(self, val):
        """
        PBC changed.
        
        """
        if self.PBCXCheckBox.isChecked():
            self.PBC[0] = 1
        
        else:
            self.PBC[0] = 0
    
    def PBCYChanged(self, val):
        """
        PBC changed.
        
        """
        if self.PBCYCheckBox.isChecked():
            self.PBC[1] = 1
        
        else:
            self.PBC[1] = 0
    
    def PBCZChanged(self, val):
        """
        PBC changed.
        
        """
        if self.PBCZCheckBox.isChecked():
            self.PBC[2] = 1
        
        else:
            self.PBC[2] = 0
    
    def getCurrentStateIndexes(self):
        """
        Return indexes of states that are currently selected
        
        """
        refIndex = self.refCombo.currentIndex()
        inputIndex = self.inputCombo.currentIndex()
        
        return refIndex, inputIndex
    
    def changeStateDisplayName(self, index, displayName):
        """
        Change display name of state
        
        """
        self.refCombo.setItemText(index, displayName)
        self.inputCombo.setItemText(index, displayName)
    
    def addStateOption(self, filename):
        """
        Add state option to combo boxes
        
        """
        self.refCombo.addItem(filename)
        self.inputCombo.addItem(filename)
    
    def removeStateOption(self, index):
        """
        Remove state option from combo boxes
        
        """
        self.refCombo.removeItem(index)
        self.inputCombo.removeItem(index)
    
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
#             if inp.cellDims[i] != ref.cellDims[i]:
            if math.fabs(inp.cellDims[i] - ref.cellDims[i]) > 1e-4:
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
            self.removeInfoWindows()
            self.refreshAllFilters()
                
            for rw in self.rendererWindows:
                if rw.currentPipelineIndex == self.pipelineIndex:
                    rw.textSelector.refresh()
                    rw.outputDialog.plotTab.rdfForm.refresh()
        
        self.mainWindow.readLBOMDIN()
        
        for rw in self.rendererWindows:
            if rw.currentPipelineIndex == self.pipelineIndex:
                rw.renderer.postRefRender()
                rw.textSelector.refresh()
    
    def postInputLoaded(self):
        """
        Do stuff after the input has been loaded
        
        """
        self.clearAllActors()
        self.removeInfoWindows()
        self.refreshAllFilters()
        
        if self.analysisPipelineFormHidden:
            self.mainToolbar.analysisPipelinesForm.show()
            self.analysisPipelineFormHidden = False
        
        else:
            # auto run filters... ?
            pass
        
        for rw in self.rendererWindows:
            if rw.currentPipelineIndex == self.pipelineIndex:
                rw.textSelector.refresh()
                rw.outputDialog.plotTab.rdfForm.refresh()
                rw.outputDialog.imageTab.imageSequenceTab.resetPrefix()
        
        settings = self.mainWindow.preferences.renderingForm
        
        if self.inputState.NAtoms <= settings.maxAtomsAutoRun:
            self.runAllFilterLists()
    
    def refChanged(self, index):
        """
        Ref changed
        
        """
        # item
        item = self.mainWindow.systemsDialog.systems_list_widget.item(index)
        
        # lattice
        state = item.lattice
        
        # check if has really changed
        if self.refState is state:
            return
        
        old_ref = self.refState
        
        self.refState = state
        self.extension = item.extension
        
        # read lbomd in?
        
        
        # post ref loaded
        self.postRefLoaded(old_ref)
        
        # check ok
        status = self.checkStateChangeOk()
        
        if status:
            # must change input too
            self.inputCombo.setCurrentIndex(index)
#             self.inputChanged(index)
    
    def inputChanged(self, index):
        """
        Input changed
        
        """
        # item
        item = self.mainWindow.systemsDialog.systems_list_widget.item(index)
        
        # lattice
        state = item.lattice
        
        # check if has really changed
        if self.inputState is state:
            return
        
        self.inputState = state
        self.inputStackIndex = item.stackIndex
        self.filename = item.displayName
        self.extension = item.extension
        self.abspath = item.abspath
        self.PBC = state.PBC
        self.setPBCChecks()
        
        # check ok
        status = self.checkStateChangeOk()
        
        if status:
            # must change ref too
            self.refCombo.setCurrentIndex(index)
#             self.refChanged(index)
        
        # post input loaded
        self.postInputLoaded()
    
    def setPBCChecks(self):
        """
        Set the PBC checks
        
        """
        self.PBCXCheckBox.setChecked(self.PBC[0])
        self.PBCYCheckBox.setChecked(self.PBC[1])
        self.PBCZCheckBox.setChecked(self.PBC[2])
    
    def removeInfoWindows(self):
        """
        Remove all info windows and associated highlighters
        
        """
        self.logger.debug("Clearing all info windows/highlighters")
        
        for filterList in self.filterLists:
            filterList.removeInfoWindows()
    
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
    
    def runAllFilterLists(self, sequencer=False):
        """
        Run all the filter lists.
        
        """
        self.logger.info("Running all filter lists")
        
        # unique id (used for POV-Ray file naming)
        self.currentRunID = uuid.uuid4()
        
        # first remove all old povray files
        oldpovfiles = glob.glob(os.path.join(self.mainWindow.tmpDirectory, "pipeline%d_*.pov" % self.pipelineIndex))
        for fn in oldpovfiles:
            os.unlink(fn)
        
        # remove old info windows
        self.removeInfoWindows()
        
        # set scalar bar false
        self.scalarBarAdded = False
        
        # progress dialog
        sequencer = True
        if not sequencer:
            progDiag = utils.showProgressDialog("Applying lists", "Applying lists...", self)
        
        try:
            count = 0
            for filterList in self.filterLists:
                self.logger.info("  Running filter list %d", count)
                
                if filterList.isStaticList():
                    self.logger.info("    Static filter list: skipping")
                
                else:
                    filterList.filterer.runFilters(sequencer=sequencer)
                
                count += 1
            
            self.refreshOnScreenInfo()
            
            # refresh plot options
            for rw in self.rendererWindows:
                if rw.currentPipelineIndex == self.pipelineIndex:
                    rw.outputDialog.plotTab.scalarsForm.refreshScalarPlotOptions()
        
        finally:
            if not sequencer:
                utils.cancelProgressDialog(progDiag)
        
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
    
    def tabCloseRequested(self, index):
        """
        Tab close requested
        
        """
        self.removeFilterList(index=index)
    
    def removeFilterList(self, index=None):
        """
        Remove a filter list
        
        """
        if self.filterListCount <= 1:
            return
        
        if index is not None:
            currentList = index
        
        else:
            currentList = self.filterTabBar.currentIndex()
        
        self.filterLists[currentList].clearList()
        
        for i in xrange(self.filterListCount):
            if i > currentList:
                self.filterTabBar.setTabText(i, str(i - 1))
                self.filterLists[i].tab -= 1
        
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
            currentSettings = filterList.getCurrentFilterSettings()
            for filterSettings in currentSettings:
                filterSettings.refresh()
            
            filterList.bondsOptions.refresh()
            filterList.vectorsOptions.refresh()
            filterList.colouringOptions.refreshScalarColourOption()
            filterList.refreshAvailableFilters()
    
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
        if globalBcast:
            rwList = self.mainWindow.rendererWindows
        
        else:
            rwList = [rw for rw in self.mainWindow.rendererWindows if rw.currentPipelineString == self.pipelineString]
        
        self.logger.debug("Broadcasting to renderers (%d/%d): %s", len(rwList), len(self.mainWindow.rendererWindows), method)
        
        for rw in rwList:
            if hasattr(rw, method):
                call = getattr(rw, method)
                
                call(*args, **kwargs)
    
    def pickObject(self, pickPos, clickType):
        """
        Pick object
        
        """
        logger = self.logger
        
        # loop over filter lists
        filterLists = self.filterLists
        
        # states
        refState = self.refState
        inputState = self.inputState
        
        # we don't want PBCs when picking
        pickPBC = np.zeros(3, np.int32)
        
        # min/max pos for boxing
        # we need the min/max of ref/input/pickPos
        minPos = np.zeros(3, np.float64)
        maxPos = np.zeros(3, np.float64)
        for i in xrange(3):
            # set to min ref pos
            minPos[i] = min(refState.pos[i::3])
            maxPos[i] = max(refState.pos[i::3])
            
            # see if min input pos is less
            minPos[i] = min(minPos[i], min(inputState.pos[i::3]))
            maxPos[i] = max(maxPos[i], max(inputState.pos[i::3]))
            
            # see if picked pos is less
            minPos[i] = min(minPos[i], pickPos[i])
            maxPos[i] = max(maxPos[i], pickPos[i])
        
        logger.debug("Min pos for picker: %r", minPos)
        logger.debug("Max pos for picker: %r", maxPos)
        
        # loop over filter lists, looking for closest object to pick pos
        minSepIndex = -1
        minSep = 9999999.0
        minSepType = None
        minSepFilterList = None
        for filterList in filterLists:
            if not filterList.visible:
                continue
            
            filterer = filterList.filterer
            
            visibleAtoms = filterer.visibleAtoms
            interstitials = filterer.interstitials
            vacancies = filterer.vacancies
            antisites = filterer.antisites
            onAntisites = filterer.onAntisites
            splitInts = filterer.splitInterstitials
            scalarsDict = filterer.scalarsDict
            latticeScalarsDict = filterer.latticeScalarsDict
            vectorsDict = filterer.vectorsDict
            
            result = np.empty(3, np.float64)
            
            status = picker_c.pickObject(visibleAtoms, vacancies, interstitials, onAntisites, splitInts, pickPos, 
                                         inputState.pos, refState.pos, pickPBC, inputState.cellDims,
                                         minPos, maxPos, inputState.specie, 
                                         refState.specie, inputState.specieCovalentRadius, 
                                         refState.specieCovalentRadius, result)
            
            tmp_type, tmp_index, tmp_sep = result
            
            if tmp_index >= 0 and tmp_sep < minSep:
                minSep = tmp_sep
                minSepType = int(tmp_type)
                minSepFilterList = filterList
                
                if minSepType == 0:
                    minSepIndex = visibleAtoms[int(tmp_index)]
                    defList = None
                
                else:
                    minSepIndex = int(tmp_index)
                    
                    if minSepType == 1:
                        defList = (vacancies,)
                    elif minSepType == 2:
                        defList = (interstitials,)
                    elif minSepType == 3:
                        defList = (antisites, onAntisites)
                    else:
                        defList = (splitInts,)
                
                minSepScalars = {}
                for scalarType, scalarArray in scalarsDict.iteritems():
                    minSepScalars[scalarType] = scalarArray[tmp_index]
                for scalarType, scalarArray in latticeScalarsDict.iteritems():
                    minSepScalars[scalarType] = scalarArray[tmp_index]
                
                minSepVectors = {}
                for vectorType, vectorArray in vectorsDict.iteritems():
                    minSepVectors[vectorType] = vectorArray[tmp_index]
        
        logger.debug("Closest object to pick: %f (threshold: %f)", minSep, 0.1)
        
        # check if close enough
        if minSep < 0.1:
            if clickType == "RightClick" and minSepType == 0:
                logger.debug("Picked object (right click)")
                
                viewAction = QtGui.QAction("View atom", self)
                viewAction.setToolTip("View atom info")
                viewAction.setStatusTip("View atom info")
                viewAction.triggered.connect(functools.partial(self.viewAtomClicked, minSepIndex, minSepType, minSepFilterList, minSepScalars, minSepVectors, defList))
                
                editAction = QtGui.QAction("Edit atom", self)
                editAction.setToolTip("Edit atom")
                editAction.setStatusTip("Edit atom")
                editAction.triggered.connect(functools.partial(self.editAtomClicked, minSepIndex))
                
                removeAction = QtGui.QAction("Remove atom", self)
                removeAction.setToolTip("Remove atom")
                removeAction.setStatusTip("Remove atom")
                removeAction.triggered.connect(functools.partial(self.removeAtomClicked, minSepIndex))
                
                menu = self.pickerContextMenu
                menu.clear()
                
                menu.addAction(viewAction)
                menu.addAction(editAction)
                menu.addAction(removeAction)
                
                # highlight atom
                lattice = self.inputState
                radius = lattice.specieCovalentRadius[lattice.specie[minSepIndex]] * minSepFilterList.displayOptions.atomScaleFactor
                highlighter = highlight.AtomHighlighter(lattice.atomPos(minSepIndex), radius * 1.1)
                self.broadcastToRenderers("addHighlighters", (self.pickerContextMenuID, [highlighter,]))
                
                cursor = QtGui.QCursor()
                
                menu.popup(cursor.pos())
            
            else:
                # show the info window
                self.showInfoWindow(minSepIndex, minSepType, minSepFilterList, minSepScalars, minSepVectors, defList)
    
    def checkIfAtomVisible(self, index):
        """
        Check if the selected atom is visible in one of the filter lists.
        
        """
        visible = False
        visibleFilterList = None
        for filterList in self.filterLists:
            if index in filterList.filterer.visibleAtoms:
                visible = True
                visibleFilterList = filterList
                break
        
        return visible, visibleFilterList
    
    def showInfoWindow(self, minSepIndex, minSepType, minSepFilterList, minSepScalars, minSepVectors, defList):
        """
        Show info window
        
        """
        logger = self.logger
        
        logger.debug("Showing info window for picked object")
        
        # key for storing the window
        windowKey = "%d_%d" % (minSepType, minSepIndex) 
        logger.debug("Picked object window key: '%s' (exists already: %s)", windowKey, windowKey in minSepFilterList.infoWindows)
        
        # check if key already exists, if so use stored window
        if windowKey in minSepFilterList.infoWindows:
            window = minSepFilterList.infoWindows[windowKey]
        
        # otherwise make new window
        else:
            if minSepType == 0:
                # atom info window
                window = infoDialogs.AtomInfoWindow(self, minSepIndex, minSepScalars, minSepVectors, minSepFilterList, parent=self)
            
            else:
                # defect info window
                window = infoDialogs.DefectInfoWindow(self, minSepIndex, minSepType, defList, minSepFilterList, parent=self)
            
            # store window for reuse
            minSepFilterList.infoWindows[windowKey] = window
        
        # position window
        utils.positionWindow(window, window.size(), self.mainWindow.desktop, self)
        
        # show window
        window.show()
    
    def viewAtomClicked(self, minSepIndex, minSepType, minSepFilterList, minSepScalars, minSepVectors, defList):
        """
        View atom
        
        """
        logger = self.logger
        logger.debug("View atom action; Index is %d", minSepIndex)
        
        self.showInfoWindow(minSepIndex, minSepType, minSepFilterList, minSepScalars, minSepVectors, defList)
    
    def editAtomClicked(self, index):
        """
        Edit atom
        
        """
        logger = self.logger
        logger.debug("Edit atom action; Index is %d", index)
        
        self.mainWindow.displayWarning("Edit atom not implemented yet.")
    
    def removeAtomClicked(self, index):
        """
        Remove atom
        
        """
        logger = self.logger
        logger.debug("Remove atom action; Index is %d", index)
        
        self.mainWindow.displayWarning("Remove atom not implemented yet.")
    
    def hidePickerMenuHighlight(self):
        """
        Hide picker menu highlighter
        
        """
        self.logger.debug("Hiding picker context menu highlighter")
        
        self.broadcastToRenderers("removeHighlighters", (self.pickerContextMenuID,))
