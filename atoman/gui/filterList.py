
"""
The filter tab for the main toolbar

@author: Chris Scott

"""
import logging
import functools
import copy
import traceback

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from ..filtering import filterer
from .filterListOptions import actorsOptions
from .filterListOptions import bondsOptions
from .filterListOptions import colouringOptions
from .filterListOptions import displayOptions
from .filterListOptions import traceOptions
from .filterListOptions import vectorsOptions
from .filterListOptions import voronoiOptions
from . import utils
from .dialogs import infoDialogs
from . import filterSettings
from ..rendering import filterListRenderer


class FilterListWidgetItem(QtGui.QListWidgetItem):
    """
    Item that goes in the filter list
    
    """
    def __init__(self, filterName, filterSettings):
        super(FilterListWidgetItem, self).__init__()
        
        self.filterName = filterName
        self.filterSettings = filterSettings
        self.setText(filterName)


class OptionsListItem(QtGui.QListWidgetItem):
    """
    Item that goes in the options list
    
    """
    def __init__(self, dialog):
        super(OptionsListItem, self).__init__()
        
        self.dialog = dialog


class FilterList(QtGui.QWidget):
    """
    Filter list widget
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, tab, width):
        super(FilterList, self).__init__(parent)
        
        self.filterTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.tab = tab
        self.tabWidth = width
        self.filterCounter = 0
        self.pipelinePage = self.filterTab
        
        self.logger = logging.getLogger(__name__)
        
        # have to treat defect filter differently
        self.defectFilterSelected = False
        
        # info windows stored here
        self.infoWindows = {}
        self.clusterInfoWindows = {}
        
        self.visible = True
        
        # layout
        self.filterListLayout = QtGui.QVBoxLayout(self)
        self.filterListLayout.setSpacing(0)
        
        # add the top set of buttons
        
        # visibility of filter list
        self.visibleButton = QtGui.QPushButton(QtGui.QIcon(iconPath("eye-ava.svg")), "")
        self.visibleButton.setFixedWidth(35)
        self.visibleButton.setStatusTip("Visible")
        self.visibleButton.setToolTip("Visible")
        self.visibleButton.setCheckable(1)
        self.visibleButton.setChecked(0)
        self.visibleButton.clicked.connect(self.visibilityChanged)
        
        # trash the list
        trashButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/edit-delete.png")), "")
        trashButton.setStatusTip("Delete property/filter list")
        trashButton.setToolTip("Delete property/filter list")
        trashButton.setFixedWidth(35)
        trashButton.clicked.connect(self.filterTab.removeFilterList)
        
        # drift compenstation
        self.driftCompButton = QtGui.QPushButton(QtGui.QIcon(iconPath("other/Drift.jpg")), "")
        self.driftCompButton.setStatusTip("Drift compensation")
        self.driftCompButton.setToolTip("Drift compensation")
        self.driftCompButton.setFixedWidth(35)
        self.driftCompButton.setCheckable(1)
        self.driftCompButton.setChecked(0)
        self.driftCompButton.clicked.connect(self.driftCompClicked)
        self.driftCompensation = False
        
        # static list button
        self.staticListButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/object-unlocked.png")), "")
        self.staticListButton.setFixedWidth(35)
        self.staticListButton.setStatusTip("Freeze property/filter list")
        self.staticListButton.setToolTip("Freeze property/filter list")
        self.staticListButton.setCheckable(1)
        self.staticListButton.setChecked(0)
        self.staticListButton.clicked.connect(self.staticListButtonClicked)
        
        # show scalar bar
        self.scalarBarButton = QtGui.QPushButton(QtGui.QIcon(iconPath("other/color-spectrum-hi.png")), "")
        self.scalarBarButton.setFixedWidth(35)
        self.scalarBarButton.setStatusTip("Show scalar bar")
        self.scalarBarButton.setToolTip("Show scalar bar")
        self.scalarBarButton.setCheckable(1)
        self.scalarBarButton.setChecked(0)
        self.scalarBarButton.clicked.connect(self.toggleScalarBar)
        
        # set up the row of buttons
        row1 = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row1)
        rowLayout.setAlignment(QtCore.Qt.AlignLeft)
        rowLayout.addWidget(self.visibleButton)
        rowLayout.addWidget(trashButton)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        
        row2 = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row2)
        rowLayout.setAlignment(QtCore.Qt.AlignRight)
        rowLayout.addWidget(self.driftCompButton)
        rowLayout.addWidget(self.staticListButton)
        rowLayout.addWidget(self.scalarBarButton)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        
        row3 = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row3)
        rowLayout.addWidget(row1)
        rowLayout.addWidget(row2)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        
        self.filterListLayout.addWidget(row3)
        
        # Now add the list widget
        self.listItems = QtGui.QListWidget(self)
        self.listItems.setFixedHeight(120)
        self.listItems.itemDoubleClicked.connect(self.openFilterSettings)
        self.listItems.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.listItems.customContextMenuRequested.connect(self.showListWidgetContextMenu)
        self.listItems.setDragEnabled(True)
        self.listItems.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.filterListLayout.addWidget(self.listItems)
        
        # quick add combo
        self.quickAddCombo = QtGui.QComboBox()
        self.quickAddCombo.addItem("Add property/filter ...")
        self.quickAddCombo.addItems(filterer.Filterer.defaultFilters)
        self.allFilters = copy.deepcopy(filterer.Filterer.defaultFilters)
        self.quickAddCombo.currentIndexChanged[str].connect(self.quickAddComboAction)
        
        # clear list button
        clearList = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/edit-clear.png")), "")
        clearList.setStatusTip("Clear current property/filter list")
        clearList.setToolTip("Clear current property/filter list")
        clearList.clicked.connect(self.clearList)
        
        # apply list button
        applyList = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/view-refresh.png")), "")
        applyList.setStatusTip("Apply current property/filter list")
        applyList.setToolTip("Apply current property/filter list")
        applyList.clicked.connect(self.applyList)
        
        row = QtGui.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.addWidget(self.quickAddCombo)
        row.addStretch()
        row.addWidget(clearList)
        row.addWidget(applyList)
        self.filterListLayout.addLayout(row)
        
        # add other option like colour by height etc
        extraOptionsGroupBox = QtGui.QGroupBox("Additional filter list options")
        extraOptionsGroupBox.setAlignment(QtCore.Qt.AlignHCenter)
        
        groupLayout = QtGui.QVBoxLayout(extraOptionsGroupBox)
        groupLayout.setAlignment(QtCore.Qt.AlignTop)
        groupLayout.setContentsMargins(0, 0, 0, 0)
        groupLayout.setSpacing(0)
        
        self.optionsList = QtGui.QListWidget()
        self.optionsList.itemClicked.connect(self.optionsListItemClicked)
        self.optionsList.setSelectionMode(self.optionsList.NoSelection)
        self.optionsList.setFixedHeight(120)
        groupLayout.addWidget(self.optionsList)
        
        # actor visibility
        self.actorsOptions = actorsOptions.ActorsOptionsWindow(self.mainWindow, parent=self)
        item = OptionsListItem(self.actorsOptions)
        item.setText("Actors options")
        self.optionsList.addItem(item)
        
        # bonding options
        self.bondsOptions = bondsOptions.BondsOptionsWindow(self.mainWindow, parent=self)
        item = OptionsListItem(self.bondsOptions)
        item.setText("Bonds options: Off")
        self.bondsOptions.modified.connect(item.setText)
        self.optionsList.addItem(item)
        
        # colouring options
        self.colouringOptions = colouringOptions.ColouringOptionsWindow(parent=self)
        item = OptionsListItem(self.colouringOptions)
        item.setText("Colouring: Species")
        self.colouringOptions.modified.connect(item.setText)
        self.optionsList.addItem(item)
        
        # display options
        self.displayOptions = displayOptions.DisplayOptionsWindow(self.mainWindow, parent=self)
        item = OptionsListItem(self.displayOptions)
        item.setText("Display options")
        self.optionsList.addItem(item)
        
        # trace options
        self.traceOptions = traceOptions.TraceOptionsWindow(self.mainWindow, parent=self)
        item = OptionsListItem(self.traceOptions)
        item.setText("Trace options: Off")
        self.traceOptions.modified.connect(item.setText)
        self.optionsList.addItem(item)
        
        # vectors options
        self.vectorsOptions = vectorsOptions.VectorsOptionsWindow(self.mainWindow, parent=self)
        item = OptionsListItem(self.vectorsOptions)
        item.setText("Vectors options: None")
        self.vectorsOptions.modified.connect(item.setText)
        self.optionsList.addItem(item)
        
        # Voronoi options
        self.voronoiOptions = voronoiOptions.VoronoiOptionsWindow(self.mainWindow, parent=self)
        item = OptionsListItem(self.voronoiOptions)
        item.setText("Voronoi options")
        self.optionsList.addItem(item)
        
        self.filterListLayout.addWidget(extraOptionsGroupBox)
        
        # the filterer (does the filtering)
        self.filterer = filterer.Filterer(self.voronoiOptions)
        
        # the renderer (does the rendering)
        self.renderer = filterListRenderer.FilterListRenderer(self)
    
    def optionsListItemClicked(self, item):
        """
        Item clicked
        
        """
        item.dialog.hide()
        item.dialog.show()
    
    def driftCompClicked(self):
        """
        Drift comp button clicked
        
        """
        if self.driftCompButton.isChecked():
            # check ok to have drift comp
            pp = self.pipelinePage
            if pp.inputState.NAtoms != pp.refState.NAtoms:
                message = "Drift compensation can only be used if the input and reference atoms match each other"
                self.mainWindow.displayWarning(message)
                self.driftCompButton.setChecked(QtCore.Qt.Unchecked)
            
            else:
                self.driftCompensation = True
        
        else:
            self.driftCompensation = False
    
    def showClusterInfoWindow(self, clusterIndex):
        """
        Show info window for cluster
        
        """
        self.logger.debug("Showing cluster info window for cluster %d", clusterIndex)
        
        # check if key already exists, if so use stored window
        if clusterIndex in self.clusterInfoWindows:
            window = self.clusterInfoWindows[clusterIndex]
        
        # otherwise make new window
        else:
            if self.defectFilterSelected:
                window = infoDialogs.DefectClusterInfoWindow(self.pipelinePage, self, clusterIndex, parent=self)
            else:
                window = infoDialogs.ClusterInfoWindow(self.pipelinePage, self, clusterIndex, parent=self)
            
            # store window
            self.clusterInfoWindows[clusterIndex] = window
        
        # position window
        utils.positionWindow(window, window.size(), self.mainWindow.desktop, self)
        
        # show the window
        window.show()
    
    def quickAddComboAction(self, text):
        """
        Quick add combo item selected.
        
        """
        text = str(text)
        if text in self.allFilters:
            self.logger.debug("Quick add: '%s'", text)
            self.addFilter(filterName=text)
            self.quickAddCombo.setCurrentIndex(0)
    
    def showListWidgetContextMenu(self, point):
        """
        Show context menu for listWidgetItem
        
        """
        logger = logging.getLogger(__name__)
        logger.debug("Show list widget context menu (property/filter list)")
        
        # position to open menu at
        globalPos = self.listItems.mapToGlobal(point)
        
        # index of item
        t = self.listItems.indexAt(point)
        index = t.row()
        
        # item
        item = self.listItems.item(index)
        
        if item is not None:
            logger.debug("  Showing context menu for item at row: %d", index)
            
            # context menu
            menu = QtGui.QMenu(self)
            
            # make actions
            # edit action
            editAction = QtGui.QAction("Edit settings", self)
            editAction.setToolTip("Edit settings")
            editAction.setStatusTip("Edit settings")
            editAction.triggered.connect(functools.partial(self.openFilterSettings, item))
            
            # removee action
            removeAction = QtGui.QAction("Remove from list", self)
            removeAction.setToolTip("Remove from list")
            removeAction.setStatusTip("Remove from list")
            removeAction.triggered.connect(functools.partial(self.removeFilter, index))
            
            # add action
            menu.addAction(editAction)
            menu.addAction(removeAction)
            
            # show menu
            menu.exec_(globalPos)
    
    def removeInfoWindows(self):
        """
        Remove info windows and highlighters
        
        """
        self.logger.debug("Removing info windows")
        
        # atom info windows
        keys = self.infoWindows.keys()
        for key in keys:
            win = self.infoWindows.pop(key)
            win.close()
        
        # cluster info windows
        keys = self.clusterInfoWindows.keys()
        for key in keys:
            win = self.clusterInfoWindows.pop(key)
            win.close()
    
    def toggleScalarBar(self):
        """
        Toggle scalar bar (if there is one).
        
        """
        if self.scalarBarButton.isChecked():
            added = self.renderer.addScalarBar()
            if not added:
                self.scalarBarButton.setChecked(0)
        else:
            self.renderer.hideScalarBar()
    
    def openFilterSettings(self, item=None):
        """
        Open filter settings window
        
        """
        self.logger.debug("Open filter settings dialog (%s)", item.filterName)
        
        if item is None:
            item = self.listItems.currentItem()
        
        item.filterSettings.hide()
        item.filterSettings.show()
    
    def clearActors(self, sequencer=False):
        """Remove all current actors."""
        self.renderer.removeActors(sequencer=sequencer)
        
        # TODO: remove scalar bar too
    
    def applyList(self, sequencer=False):
        """
        Run filters in this list.
        
        """
        # skip if static list
        if self.isStaticList():
            self.logger.info("Static filter list: skipping")
            return
        
        # otherwise process
        if not sequencer:
            # add a progress dialog
            progDiag = utils.showProgressDialog("Applying list", "Applying list...", self)
        
        try:
            # list of filters
            currentFilters = self.getCurrentFilterNames()
            currentSettingsGuis = self.getCurrentFilterSettings()
            currentSettings = [settingsGui.getSettings() for settingsGui in currentSettingsGuis]
            
            # current states
            inputState = self.pipelinePage.inputState
            refState = self.pipelinePage.refState
            
            # remove actors first
            self.clearActors(sequencer=sequencer)
            
            # apply filters
            self.filterer.runFilters(currentFilters, currentSettings, inputState, refState)
            
            # this is where the rendering should be done
            self.renderer.render(sequencer=sequencer)
            
            # update on screen text
            self.filterTab.refreshOnScreenInfo()
            
            # refresh available scalars in extra options dialog
            self.colouringOptions.refreshScalarColourOption()
            
            # add actors
            if self.visible:
                self.renderer.addActors()
            
            # refresh plot options and reinit
            for rw in self.pipelinePage.rendererWindows:
                if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                    rw.outputDialog.plotTab.scalarsForm.refreshScalarPlotOptions()
                    if self.visible:
                        rw.vtkRenWinInteract.ReInitialize()
        
        except:
            # print traceback
            errstring = traceback.format_exc()
            
            # show error
            # exctype, value = sys.exc_info()[:2]
            self.logger.error("Apply list failed!\n\n%s", errstring)
            self.mainWindow.displayError("Apply list failed!\n\n%s" % errstring)
        
        finally:
            if not sequencer:
                # always remove the progress dialog
                utils.cancelProgressDialog(progDiag)
    
    def getCurrentFilterSettings(self):
        """
        Return ordered list of current filter settings objects
        
        """
        currentSettings = []
        for i in xrange(self.listItems.count()):
            item = self.listItems.item(i)
            currentSettings.append(item.filterSettings)
        
        return currentSettings
    
    def getCurrentFilterNames(self):
        """
        Return ordered list of current filter names
        
        """
        currentNames = []
        for i in xrange(self.listItems.count()):
            item = self.listItems.item(i)
            currentNames.append(item.filterName.split("[")[0].strip())
        
        return currentNames
    
    def getCurrentFilterScalars(self):
        """
        Return a list of the scalars provided by the current filters
        
        """
        currentScalars = []
        for i in xrange(self.listItems.count()):
            item = self.listItems.item(i)
            currentScalars.extend(item.filterSettings.getProvidedScalars())
        
        return currentScalars
    
    def clearList(self):
        """
        Clear filters and actors from list.
        
        """
        self.renderer.removeActors()
        
        # close info windows
        self.removeInfoWindows()
        
        # close settings dialogs
        settingsDialogs = self.getCurrentFilterSettings()
        while len(settingsDialogs):
            dlg = settingsDialogs.pop()
            dlg.accept()
        
        # clear list items
        self.listItems.clear()
        
        self.staticListButton.setChecked(0)
        self.defectFilterSelected = False
        
        if self.filterer.scalarBarAdded:
            self.scalarBarButton.setChecked(0)
        
        # refresh available scalars
        self.colouringOptions.refreshScalarColourOption()
        
        self.filterTab.refreshOnScreenInfo()
    
    def staticListButtonClicked(self):
        """
        Static list button clicked
        
        """
        if self.staticListButton.isChecked():
            self.staticListButton.setIcon(QtGui.QIcon(iconPath("oxygen/object-locked.png")))
        
        else:
            self.staticListButton.setIcon(QtGui.QIcon(iconPath("oxygen/object-unlocked.png")))
    
    def isStaticList(self):
        """
        Check if the list is a static list.
        
        """
        return self.staticListButton.isChecked()
    
    def moveFilterDownInList(self):
        """
        Move filter down in list
        
        """
        if self.isStaticList():
            self.mainWindow.displayWarning("Cannot modify a frozen filter list")
            return
        
        # find which one is selected
        row = self.listItems.currentRow()
        
        if not self.listItems.count() or row < 0:
            return
        
        newRow = row + 1
        if newRow == self.listItems.count():
            return
        
        self.listItems.insertItem(newRow, self.listItems.takeItem(row))
    
    def moveFilterUpInList(self):
        """
        Move filter up in list
        
        """
        if self.isStaticList():
            self.mainWindow.displayWarning("Cannot modify a frozen filter list")
            return
        
        # find which one is selected
        row = self.listItems.currentRow()
        
        if not self.listItems.count() or row < 0:
            return
        
        newRow = row - 1
        if newRow < 0:
            return
        
        self.listItems.insertItem(newRow, self.listItems.takeItem(row))
    
    def warnDefectFilter(self, name=None):
        """
        Warn user that defect filter cannot
        be used with any other filter
        
        """
        if name is not None:
            message = "The '%s' filter cannot be used in conjuction with the 'Point defects' filter" % name
        else:
            message = "The 'Point defects' filter must be added to the filter list first"
        
        msgBox = QtGui.QMessageBox(self)
        msgBox.setText(message)
        msgBox.setWindowFlags(msgBox.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
        msgBox.setIcon(QtGui.QMessageBox.Warning)
        msgBox.exec_()
    
    def warnDisplacementFilter(self):
        """
        Displacement filter can only be used when number of atoms match
        
        """
        message = "The Displacement filter can only be used when the reference and input number of atoms match!"
        
        msgBox = QtGui.QMessageBox(self)
        msgBox.setText(message)
        msgBox.setWindowFlags(msgBox.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
        msgBox.setIcon(QtGui.QMessageBox.Warning)
        msgBox.exec_()
    
    def refreshAvailableFilters(self):
        """
        Refresh available filters
        
        """
        self.logger.debug("Refreshing available filters")
        
        inp = self.pipelinePage.inputState
        scalarsDict = inp.scalarsDict
        
        numDefault = len(filterer.Filterer.defaultFilters)
        scalarNames = scalarsDict.keys()
        additionalFilters = ["Scalar: {0}".format(s) for s in scalarNames]
        previousAdditionalFilters = self.allFilters[numDefault:]
        currentLen = len(self.allFilters)
        assert currentLen + 1 == self.quickAddCombo.count()
        
        self.logger.debug("Old additional filters: %r", previousAdditionalFilters)
        self.logger.debug("New additional filters: %r", additionalFilters)
        
        # remove filters that are no longer available
        for key in previousAdditionalFilters:
            if key not in additionalFilters:
                self.logger.debug("Removing filter: '%s'", key)
                
                for i in xrange(numDefault + 1, self.quickAddCombo.count()):
                    if str(self.quickAddCombo.itemText(i)) == key:
                        self.quickAddCombo.removeItem(i)
                        self.allFilters.pop(i - 1)
                        
                        # also delete settings and remove from list widget...
                        delinds = []
                        for j in xrange(self.listItems.count() - 1, -1, -1):
                            item = self.listItems.item(j)
                            if item.filterName.startswith(key):
                                delinds.append(j)
                        
                        for index in delinds:
                            item = self.listItems.takeItem(index)
                            dlg = item.filterSettings
                            dlg.close()
                            dlg.accept()
                            del item
        
        # add new filters
        for key in additionalFilters:
            if key in previousAdditionalFilters:
                self.logger.debug("Keeping filter: '%s'", key)
            
            else:
                self.logger.debug("Adding filter: '%s'", key)
                self.quickAddCombo.addItem(key)
                self.allFilters.append(key)
    
    def addFilter(self, filterName=None):
        """
        Add new filter
        
        """
        if self.isStaticList():
            self.mainWindow.displayWarning("Cannot modify a frozen filter list")
            return
        
        # first determine what filter is to be added
        pp = self.pipelinePage
        if filterName is not None and filterName in self.allFilters:
            if self.defectFilterSelected and filterName not in self.filterer.defectCompatibleFilters:
                self.warnDefectFilter(name=filterName)
            
            elif self.listItems.count() > 0 and str(filterName) == "Point defects":
                self.warnDefectFilter()
            
            elif str(filterName) == "Displacement" and pp.inputState.NAtoms != pp.refState.NAtoms:
                self.warnDisplacementFilter()
            
            else:
                # filter name
                filterNameString = "%s [%d]" % (filterName, self.filterCounter)
                self.logger.debug("Adding filter/property calculator: '%s'", filterNameString)
                
                # create options form
                form = self.createSettingsForm(filterName)
                
                # list widget item
                item = FilterListWidgetItem(str(filterNameString), form)
                
                # add
                self.listItems.addItem(item)
                
                # show options form
                form.show()
                
                if str(filterName) == "Point defects":
                    self.defectFilterSelected = True
                
                # refresh available scalars
                self.colouringOptions.refreshScalarColourOption()
    
    def removeFilter(self, row=None):
        """
        Remove new filter
        
        """
        if self.isStaticList():
            self.mainWindow.displayWarning("Cannot modify a frozen filter list")
            return
        
        # find which one is selected
        if row is None:
            row = self.listItems.currentRow()
        
        if not self.listItems.count() or row < 0:
            return
                
        # remove it from lists
        item = self.listItems.takeItem(row)
        filterName = item.filterName
        self.logger.debug("Removing filter/property calculation (%d): '%s'", row, filterName)
        dlg = item.filterSettings
        dlg.close()
        dlg.accept()
        del item
        
        if filterName.startswith("Point defects"):
            self.defectFilterSelected = False
        
        # refresh available scalars
        self.colouringOptions.refreshScalarColourOption()
    
    def createSettingsForm(self, filterName):
        """
        Create a settings form for the filter.
        
        """
        form = None
        
        if filterName.startswith("Scalar: "):
            self.logger.debug("Creating settings dialog for: '%s'", filterName)
            
            # title for form
            title = "%s settings (List %d - %d)" % (filterName, self.tab, self.filterCounter)
            
            # load module
            from .filterSettings import genericScalarSettingsDialog
            
            # load form
            form = genericScalarSettingsDialog.GenericScalarSettingsDialog(self.mainWindow, filterName, title,
                                                                           parent=self)
            self.filterCounter += 1
        
        else:
            words = str(filterName).title().split()
            
            dialogName = "%sSettingsDialog" % "".join(words)
            moduleName = dialogName[:1].lower() + dialogName[1:]
            self.logger.debug("Loading settings dialog module: '%s'", moduleName)
            self.logger.debug("Creating settings dialog: '%s'", dialogName)
            
            # get module
            formModule = getattr(filterSettings, moduleName)
            
            # load dialog
            formObject = getattr(formModule, dialogName, None)
            if formObject is not None:
                title = "%s settings (List %d - %d)" % (filterName, self.tab, self.filterCounter)
                form = formObject(self.mainWindow, title, parent=self)
                self.filterCounter += 1
            else:
                self.logger.error("Could not locate form '%s' in module '%s'", dialogName, moduleName)
        
        return form
    
    def visibilityChanged(self):
        """
        Update visibility of filter list
        
        """
        if self.visibleButton.isChecked():
            self.visibleButton.setIcon(QtGui.QIcon(iconPath("eye-close-ava.svg")))
            self.visible = False
            self.filterer.hideActors()
        
        else:
            self.visibleButton.setIcon(QtGui.QIcon(iconPath("eye-ava.svg")))
            self.visible = True
            self.actorsOptions.addCheckedActors()
        
        self.filterTab.refreshOnScreenInfo()
