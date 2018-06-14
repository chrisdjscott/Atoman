# -*- coding: utf-8 -*-

"""
Analysis is performed using an *Analysis pipeline*, found on the *Analysis toolbar* on the left of the application (see
right). Multiple pipelines can be configured at once; a pipeline is viewed in a renderer window.

An individual pipeline takes a reference and an input system as its input and contains one or more filter/calculator
lists. These lists operate independently of one another and calculate properties or filter the input system. Available
filters/calculators are shown below:

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
import glob
import math
import logging
import functools
import uuid

from PySide2 import QtGui, QtCore, QtWidgets

import vtk
import numpy as np

from ..visutils.utilities import iconPath
from . import filterList
from . import picker
from .dialogs import infoDialogs
from . import utils
from ..rendering import highlight
from .dialogs import simpleDialogs
import six
from six.moves import range


################################################################################
class PipelineForm(QtWidgets.QWidget):
    def __init__(self, parent, mainWindow, width, pipelineIndex, pipelineString):
        super(PipelineForm, self).__init__(parent)

        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.toolbarWidth = width
        self.pipelineIndex = pipelineIndex
        self.pipelineString = pipelineString
        self.systemsDialog = mainWindow.systemsDialog

        self.logger = logging.getLogger(__name__)

        self.rendererWindows = self.mainWindow.rendererWindows

        self.pickerContextMenuID = uuid.uuid4()
        self.pickerContextMenu = QtWidgets.QMenu(self)
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
        self.linkedLattice = None
        self.fromSFTP = None
        self.scalarBarAdded = False

        # layout
        filterTabLayout = QtWidgets.QVBoxLayout(self)
        filterTabLayout.setContentsMargins(0, 0, 0, 0)
        filterTabLayout.setSpacing(0)
        filterTabLayout.setAlignment(QtCore.Qt.AlignTop)

        # row
        row = QtWidgets.QWidget()
        rowLayout = QtWidgets.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)
        label = QtWidgets.QLabel("<b>Pipeline %d settings</b>" % pipelineIndex)
        rowLayout.addWidget(label)
        filterTabLayout.addWidget(row)

        # row
        row = QtWidgets.QWidget()
        rowLayout = QtWidgets.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)

        # reference selector
        self.refCombo = QtWidgets.QComboBox()
        self.refCombo.setFixedWidth(220)
        self.refCombo.setToolTip("Select the reference system for this pipeline")
        self.refCombo.currentIndexChanged.connect(self.refChanged)

        # add to row
        rowLayout.addWidget(QtWidgets.QLabel("Reference:"))
        rowLayout.addWidget(self.refCombo)
        filterTabLayout.addWidget(row)

        # row
        row = QtWidgets.QWidget()
        rowLayout = QtWidgets.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)

        # reference selector
        self.inputCombo = QtWidgets.QComboBox()
        self.inputCombo.setFixedWidth(220)
        self.inputCombo.setToolTip("Select the input system for this pipeline")
        self.inputCombo.currentIndexChanged.connect(self.inputChanged)

        # add to row
        rowLayout.addWidget(QtWidgets.QLabel("Input:"))
        rowLayout.addWidget(self.inputCombo)
        filterTabLayout.addWidget(row)

        row = QtWidgets.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.addWidget(QtWidgets.QLabel("<b>Property/filter lists:</b>"))
        filterTabLayout.addLayout(row)

        # row
        row = QtWidgets.QWidget()
        rowLayout = QtWidgets.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setSpacing(0)

        # buttons for new/trash filter list
        runAll = QtWidgets.QPushButton(QtGui.QIcon(iconPath('oxygen/view-refresh.png')), 'Apply lists')
        runAll.setStatusTip("Apply all property/filter lists")
        runAll.setToolTip("Apply all property/filter lists")
        runAll.clicked.connect(self.runAllFilterLists)
        add = QtWidgets.QPushButton(QtGui.QIcon(iconPath('oxygen/tab-new-background.png')), 'New list')
        add.setToolTip("New property/filter list")
        add.setStatusTip("New property/filter list")
        add.clicked.connect(self.addFilterList)
        clear = QtWidgets.QPushButton(QtGui.QIcon(iconPath('oxygen/tab-close-other.png')), 'Clear lists')
        clear.setStatusTip("Clear all property/filter lists")
        clear.setToolTip("Clear all property/filter lists")
        clear.clicked.connect(self.clearAllFilterLists)

        rowLayout.addWidget(add)
        rowLayout.addWidget(clear)
        rowLayout.addWidget(runAll)

        filterTabLayout.addWidget(row)

        # add tab bar for filter lists
        self.filterTabBar = QtWidgets.QTabWidget(self)
        self.filterTabBar.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.filterTabBar.currentChanged[int].connect(self.filterTabBarChanged)
        self.filterTabBar.setTabsClosable(True)
        self.filterTabBar.tabCloseRequested.connect(self.tabCloseRequested)
        filterTabLayout.addWidget(self.filterTabBar)

        # add a filter list
        self.addFilterList()

        # add pbc options
        group = QtWidgets.QGroupBox("Periodic boundaries")
        group.setAlignment(QtCore.Qt.AlignHCenter)
        groupLayout = QtWidgets.QVBoxLayout(group)
        groupLayout.setSpacing(0)
        groupLayout.setContentsMargins(0, 0, 0, 0)

        # add PBC check boxes
        self.PBCXCheckBox = QtWidgets.QCheckBox("x")
        self.PBCXCheckBox.setChecked(QtCore.Qt.Checked)
        self.PBCYCheckBox = QtWidgets.QCheckBox("y")
        self.PBCYCheckBox.setChecked(QtCore.Qt.Checked)
        self.PBCZCheckBox = QtWidgets.QCheckBox("z")
        self.PBCZCheckBox.setChecked(QtCore.Qt.Checked)

        self.PBCXCheckBox.stateChanged[int].connect(self.PBCXChanged)
        self.PBCYCheckBox.stateChanged[int].connect(self.PBCYChanged)
        self.PBCZCheckBox.stateChanged[int].connect(self.PBCZChanged)

        row = QtWidgets.QWidget(self)
        rowLayout = QtWidgets.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.addWidget(self.PBCXCheckBox)
        rowLayout.addWidget(self.PBCYCheckBox)
        rowLayout.addWidget(self.PBCZCheckBox)

        groupLayout.addWidget(row)

        # add shift cell and replicate cell buttons
        self.replicateCellButton = QtWidgets.QPushButton("Replicate cell")
        self.replicateCellButton.clicked.connect(self.replicateCell)
        self.replicateCellButton.setToolTip("Replicate in periodic directions")
        self.shiftCellButton = QtWidgets.QPushButton("Shift cell")
        self.shiftCellButton.clicked.connect(self.shiftCell)
        self.shiftCellButton.setToolTip("Shift cell in periodic directions")
        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addStretch(1)
        hbox.addWidget(self.shiftCellButton)
        hbox.addWidget(self.replicateCellButton)
        hbox.addStretch(1)
        groupLayout.addLayout(hbox)

        # add shift atom button
        row = QtWidgets.QWidget(self)
        rowLayout = QtWidgets.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        self.shiftAtomButton = QtWidgets.QPushButton("Shift atoms")
        self.shiftAtomButton.clicked.connect(self.shiftAtom)
        self.shiftAtomButton.setToolTip("Shift an atom (or set of atoms) in periodic directions")
        rowLayout.addWidget(self.shiftAtomButton)
        groupLayout.addWidget(row)


        filterTabLayout.addWidget(group)

        # add systems to combos
        for fn in self.systemsDialog.getDisplayNames():
            self.refCombo.addItem(fn)

        for fn in self.systemsDialog.getDisplayNames():
            self.inputCombo.addItem(fn)

        # refresh if ref already loaded
        if self.mainWindow.refLoaded:
            self.refreshAllFilters()

    def shiftAtom(self):
        """
        Shift atom

        """
        # lattice
        lattice = self.inputState

        # show dialog
        dlg = simpleDialogs.ShiftAtomDialog(-1, self.PBC, lattice.cellDims, lattice.NAtoms, parent=self)
        status = dlg.exec_()

        if status == QtWidgets.QDialog.Accepted:
            # amount
            shift = np.empty(3, np.float64)
            shift[0] = dlg.shiftXSpin.value()
            shift[1] = dlg.shiftYSpin.value()
            shift[2] = dlg.shiftZSpin.value()

            # atomIDstring
            atomIDstring = dlg.lineEdit.text()

            # parse atomIDstring
            array = [val for val in atomIDstring.split(",") if val]
            num = len(array)
            rangeArray = np.empty((num, 2), np.int32)
            for i, item in enumerate(array):
                if "-" in item:
                    values = [val for val in item.split("-") if val]
                    minval = int(values[0])
                    if len(values) == 1:
                        maxval = minval
                    else:
                        maxval = int(values[1])
                else:
                    minval = maxval = int(item)

                rangeArray[i][0] = minval
                rangeArray[i][1] = maxval


            # loop over atoms
            if (shift[0] or shift[1] or shift[2]) and (num>0):
                self.logger.debug("Shifting atom: x = %f; y = %f; z = %f", shift[0], shift[1], shift[2])

                # set override cursor
                QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    # shift atoms
                    for i in range(num):
                        for k in range(rangeArray[i][1]-rangeArray[i][0]+1):
                            i3 = 3 * (rangeArray[i][0]+k-1)
                            for j in range(3):
                                lattice.pos[i3 + j] += shift[j]

                    # wrap atoms back into periodic cell
                    lattice.wrapAtoms()

                finally:
                    QtGui.QApplication.restoreOverrideCursor()

                # run post ref render of Renderer (redraws cell)
                for rw in self.rendererWindows:
                    if rw.currentPipelineIndex == self.pipelineIndex:
                        rw.renderer.postRefRender()
                        rw.textSelector.refresh()

                # run post input loaded method
                self.postInputLoaded()

    def shiftCell(self):
        """
        Shift cell

        """
        # lattice
        lattice = self.inputState

        # show dialog
        dlg = simpleDialogs.ShiftCellDialog(self.PBC, lattice.cellDims, parent=self)
        status = dlg.exec_()

        if status == QtWidgets.QDialog.Accepted:
            # amount
            shift = np.empty(3, np.float64)
            shift[0] = dlg.shiftXSpin.value()
            shift[1] = dlg.shiftYSpin.value()
            shift[2] = dlg.shiftZSpin.value()

            # loop over atoms
            if shift[0] or shift[1] or shift[2]:
                self.logger.debug("Shifting cell: x = %f; y = %f; z = %f", shift[0], shift[1], shift[2])

                # progress update interval
                progressInterval = int(lattice.NAtoms / 10)
                if progressInterval < 50:
                    progressInterval = 50
                elif progressInterval > 500:
                    progressInterval = 500

                # set override cursor
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    # add progress dialog
                    self.mainWindow.updateProgress(0, lattice.NAtoms, "Shifting cell")

                    # loop over atoms
                    for i in range(lattice.NAtoms):
                        i3 = 3 * i
                        for j in range(3):
                            lattice.pos[i3 + j] += shift[j]

                        # progress
                        if i % progressInterval == 0:
                            self.mainWindow.updateProgress(i, lattice.NAtoms, "Shifting cell")

                    # wrap atoms back into periodic cell
                    lattice.wrapAtoms()

                finally:
                    self.mainWindow.hideProgressBar()
                    QtWidgets.QApplication.restoreOverrideCursor()

                # run post ref render of Renderer (redraws cell)
                for rw in self.rendererWindows:
                    if rw.currentPipelineIndex == self.pipelineIndex:
                        rw.renderer.postRefRender()
                        rw.textSelector.refresh()

                # run post input loaded method
                self.postInputLoaded()

    def replicateCell(self):
        """
        Replicate cell

        """
        self.logger.warning("'Replicate cell' is an experimental feature!")

        dlg = simpleDialogs.ReplicateCellDialog(self.PBC, parent=self)
        status = dlg.exec_()

        if status == QtWidgets.QDialog.Accepted:
            repDirs = np.zeros(3, np.int32)
            replicate = False

            numx = dlg.replicateInXSpin.value()
            if numx:
                repDirs[0] = numx
                replicate = True

            numy = dlg.replicateInYSpin.value()
            if numy:
                repDirs[1] = numy
                replicate = True

            numz = dlg.replicateInZSpin.value()
            if numz:
                repDirs[2] = numz
                replicate = True

            if replicate:
                self.logger.warning("Replicating cell: this will modify the current input state everywhere")
                self.logger.debug("Replicating cell: %r", repDirs)

                # TODO: write in C
                lattice = self.inputState
                newpos = np.empty(3, np.float64)
                cellDims = lattice.cellDims

                # calculate final number of atoms
                numfin = lattice.NAtoms
                for i in range(3):
                    numfin += numfin * repDirs[i]
                numadd = numfin - lattice.NAtoms
                self.logger.debug("Replicating cell: adding %d atoms", numadd)

                # progress update interval
                progressInterval = int(numadd / 10)
                if progressInterval < 50:
                    progressInterval = 50
                elif progressInterval > 500:
                    progressInterval = 500

                # set override cursor
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    # add progress dialog
                    self.mainWindow.updateProgress(0, numadd, "Replicating cell")

                    # loop over directions
                    count = 0
                    for i in range(3):
                        self.logger.debug("Replicating along axis %d: %d times", i, repDirs[i])

                        # store num atoms at beginning of this direction
                        NAtoms = lattice.NAtoms

                        # loop over number of replications in this direction
                        for j in range(repDirs[i]):
                            # loop over atoms
                            for k in range(NAtoms):
                                # attributes to copy to new atom
                                sym = lattice.atomSym(k)
                                q = lattice.charge[k]
                                scalarVals = {}
                                for name, scalarsArray in six.iteritems(lattice.scalarsDict):
                                    scalarVals[name] = scalarsArray[k]
                                vectorVals = {}
                                for name, vectorsArray in six.iteritems(lattice.vectorsDict):
                                    vectorVals[name] = vectorsArray[k]

                                # new position
                                pos = lattice.atomPos(k)
                                newpos[:] = pos[:]
                                newpos[i] += (j + 1) * cellDims[i]

                                # add atom
                                lattice.addAtom(sym, newpos, q, scalarVals=scalarVals, vectorVals=vectorVals)

                                # progress
                                count += 1
                                if count % progressInterval == 0:
                                    self.mainWindow.updateProgress(count, numadd, "Replicating cell")

                        # change cell dimension
                        lattice.cellDims[i] += cellDims[i] * repDirs[i]
                        self.logger.debug("New cellDims along axis %d: %f", i, lattice.cellDims[i])

                finally:
                    self.mainWindow.hideProgressBar()
                    QtWidgets.QApplication.restoreOverrideCursor()

                # run post ref render of Renderer (redraws cell)
                for rw in self.rendererWindows:
                    if rw.currentPipelineIndex == self.pipelineIndex:
                        rw.renderer.postRefRender()
                        rw.textSelector.refresh()

                # run post input loaded method
                self.postInputLoaded()

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

        if not self.mainToolbar.tabWidget.isTabEnabled(1):
            # enable and switch to analysis tab after first file is loaded
            self.mainToolbar.tabWidget.setTabEnabled(1, True)
            self.mainToolbar.tabWidget.setCurrentIndex(1)

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
        for i in range(3):
            if math.fabs(inp.cellDims[i] - ref.cellDims[i]) > 1e-4:
                diff = True
                break

        if diff:
            self.logger.warning("Cell dims differ")

        return diff

    def postRefLoaded(self, oldRef):
        """
        Do stuff after the ref has been loaded.

        """
        self.logger.debug("Running postRefLoaded")

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

        self.logger.debug("Finished postRefLoaded")

    def postInputLoaded(self):
        """
        Do stuff after the input has been loaded

        """
        self.logger.debug("Running postInputLoaded")

        self.clearAllActors()
        self.removeInfoWindows()
        self.refreshAllFilters()

        for rw in self.rendererWindows:
            if rw.currentPipelineIndex == self.pipelineIndex:
                rw.postInputChanged()

        settings = self.mainWindow.preferences.renderingForm
        if self.inputState.NAtoms <= settings.maxAtomsAutoRun:
            self.logger.debug("Running all filter lists automatically")
            self.runAllFilterLists()

        self.logger.debug("Finished postInputLoaded")

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

        # post ref loaded
        self.postRefLoaded(old_ref)

        # check ok
        status = self.checkStateChangeOk()

        if status:
            # must change input too
            self.inputCombo.setCurrentIndex(index)

    def inputChanged(self, index):
        """
        Input changed

        """
        self.logger.debug("Running inputChanged")

        # item
        item = self.mainWindow.systemsDialog.systems_list_widget.item(index)

        # lattice
        state = item.lattice

        # check if has really changed
        if self.inputState is state:
            return

        self.inputState = state
        self.filename = item.displayName
        self.extension = item.extension
        self.abspath = item.abspath
        self.fileFormat = item.fileFormat
        self.linkedLattice = item.linkedLattice
        self.fromSFTP = item.fromSFTP
        self.PBC = state.PBC
        self.setPBCChecks()

        # check ok
        status = self.checkStateChangeOk()

        if status:
            # must change ref too
            self.refCombo.setCurrentIndex(index)

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

        for filterList_ in self.filterLists:
            filterList_.removeInfoWindows()

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
        self.logger.debug("Clearing all actors")

        for filterList_ in self.filterLists:
            filterList_.clearActors()

    def runAllFilterLists(self, sequencer=False):
        """
        Run all the filter lists.

        """
        self.logger.info("Running all filter lists")

        # unique id (used for POV-Ray file naming)
        self.currentRunID = uuid.uuid4()

        # first remove all old povray files
        oldpovfiles = glob.glob(os.path.join(self.mainWindow.tmpDirectory,
                                             "pipeline%d_*.pov" % self.pipelineIndex))
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
            status = 0
            for count, filterList_ in enumerate(self.filterLists):
                self.logger.info("Running filter list %d", count)
                filterList_.applyList(sequencer=sequencer)

        except:
            exctype, value = sys.exc_info()[:2]
            self.logger.exception("Run all filter lists failed!")
            self.mainWindow.displayError("Run all filter lists failed!\n\n%s: %s" % (exctype, value))
            status = 1

        finally:
            if not sequencer:
                utils.cancelProgressDialog(progDiag)

        self.mainWindow.setStatus("Ready")

        return status

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
        filterListWidget = QtWidgets.QWidget()
        filterListLayout = QtWidgets.QVBoxLayout(filterListWidget)
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
        self.logger.debug("Clearing all filter lists")
        for filterList_ in self.filterLists:
            filterList_.clearList()
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

        for i in range(self.filterListCount):
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
        self.logger.debug("Refreshing filters")
        for filterList_ in self.filterLists:
            currentSettings = filterList_.getCurrentFilterSettings()
            for filterSettings in currentSettings:
                filterSettings.refresh()

            filterList_.bondsOptions.refresh()
            filterList_.vectorsOptions.refresh()
            filterList_.colouringOptions.refreshScalarColourOption()
            filterList_.refreshAvailableFilters()

    def gatherVisibleAtoms(self):
        """
        Builds an array containing all (unique) visible atoms.

        """
        visibleAtomsFull = None
        for filterList_ in self.filterLists:
            visibleAtoms = filterList_.filterer.visibleAtoms

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

        self.logger.debug("Broadcasting to renderers (%d/%d): %s", len(rwList), len(self.mainWindow.rendererWindows),
                          method)

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
        for i in range(3):
            # set to min ref pos
            minPos[i] = min(refState.pos[i::3])
            maxPos[i] = max(refState.pos[i::3])

            # see if min input pos is less
            minPos[i] = min(minPos[i], min(inputState.pos[i::3]))
            maxPos[i] = max(maxPos[i], max(inputState.pos[i::3]))

            # see if picked pos is less
            minPos[i] = min(minPos[i], pickPos[i])
            maxPos[i] = max(maxPos[i], pickPos[i])

        logger.debug("Min pos for picker: %r", minPos)
        logger.debug("Max pos for picker: %r", maxPos)

        # loop over filter lists, looking for closest object to pick pos
        minSepIndex = -1
        minSep = 9999999.0
        minSepType = None
        minSepFilterList = None
        for filterList_ in filterLists:
            if not filterList_.visible:
                continue

            filterer = filterList_.filterer

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

            status = picker.pickObject(visibleAtoms, vacancies, interstitials, onAntisites, splitInts, pickPos,
                                       inputState.pos, refState.pos, pickPBC, inputState.cellDims,
                                       inputState.specie, refState.specie, inputState.specieCovalentRadius,
                                       refState.specieCovalentRadius, result)

            if status:
                raise RuntimeError("Picker exited with non zero status (%d)" % status)

            tmp_type, tmp_index, tmp_sep = result

            if tmp_index >= 0 and tmp_sep < minSep:
                minSep = tmp_sep
                minSepType = int(tmp_type)
                minSepFilterList = filterList_

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
                for scalarType, scalarArray in six.iteritems(scalarsDict):
                    minSepScalars[scalarType] = scalarArray[int(tmp_index)]
                for scalarType, scalarArray in six.iteritems(latticeScalarsDict):
                    minSepScalars[scalarType] = scalarArray[int(tmp_index)]

                minSepVectors = {}
                for vectorType, vectorArray in six.iteritems(vectorsDict):
                    minSepVectors[vectorType] = vectorArray[int(tmp_index)]

        logger.debug("Closest object to pick: %f (threshold: %f)", minSep, 0.1)

        # check if close enough
        if minSep < 0.1:
            if clickType == "RightClick" and minSepType == 0:
                logger.debug("Picked object (right click)")

                viewAction = QtWidgets.QAction("View atom", self)
                viewAction.setToolTip("View atom info")
                viewAction.setStatusTip("View atom info")
                viewAction.triggered.connect(functools.partial(self.viewAtomClicked, minSepIndex, minSepType,
                                             minSepFilterList, minSepScalars, minSepVectors, defList))

                editAction = QtWidgets.QAction("Edit atom", self)
                editAction.setToolTip("Edit atom")
                editAction.setStatusTip("Edit atom")
                editAction.triggered.connect(functools.partial(self.editAtomClicked, minSepIndex))

                removeAction = QtWidgets.QAction("Remove atom", self)
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
        for filterList_ in self.filterLists:
            if index in filterList_.filterer.visibleAtoms:
                visible = True
                visibleFilterList = filterList_
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
        logger.debug("Picked object window key: '%s' (exists already: %s)", windowKey,
                     windowKey in minSepFilterList.infoWindows)

        # check if key already exists, if so use stored window
        if windowKey in minSepFilterList.infoWindows:
            window = minSepFilterList.infoWindows[windowKey]

        # otherwise make new window
        else:
            if minSepType == 0:
                # atom info window
                window = infoDialogs.AtomInfoWindow(self, minSepIndex, minSepScalars, minSepVectors, minSepFilterList,
                                                    parent=self)

            else:
                # defect info window
                window = infoDialogs.DefectInfoWindow(self, minSepIndex, minSepType, defList, minSepScalars,
                                                      minSepVectors, minSepFilterList, parent=self)

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
