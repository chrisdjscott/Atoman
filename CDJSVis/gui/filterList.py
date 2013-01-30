
"""
The filter tab for the main toolbar

@author: Chris Scott

"""
import sys

from PyQt4 import QtGui, QtCore

from ..visutils.utilities import iconPath
from ..filtering import filterer
from . import filterSettings
from . import filterListOptions

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################
class FilterList(QtGui.QWidget):
    def __init__(self, parent, mainToolbar, mainWindow, tab, width, height=150):
        super(FilterList, self).__init__(parent)
        
        self.filterTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.tab = tab
        self.tabWidth = width
        self.tabHeight = height
        
        self.defectFilterSelected = 0
        
        # all available filters
        self.allFilters = ["Specie", 
                           "Point defects", 
                           "Crop", 
                           "Cluster", 
                           "Displacement",
                           "Kinetic energy",
                           "Potential energy",
                           "Charge",
                           "Crop sphere",
                           "Slice"]
        self.allFilters.sort()
        
        # current selected filters
        self.currentFilters = []
        
        # settings (windows) for current filters
        self.currentSettings = []
        
        self.visible = 1
        
        # layout
        self.filterListLayout = QtGui.QVBoxLayout(self)
        
        # add the top set of buttons
        
        # visibility of filter list
        self.visibleButton = QtGui.QPushButton(QtGui.QIcon(iconPath("eye-ava.svg")), "")
        self.visibleButton.setFixedWidth(35)
        self.visibleButton.setStatusTip("Visible")
        self.visibleButton.setCheckable(1)
        self.visibleButton.setChecked(0)
        self.connect(self.visibleButton, QtCore.SIGNAL('clicked()'), self.visibilityChanged)
        
        # trash the list
        trashButton = QtGui.QPushButton(QtGui.QIcon(iconPath("edit-delete.svg")), "")
        trashButton.setStatusTip("Delete filter list")
        trashButton.setFixedWidth(35)
        self.connect(trashButton, QtCore.SIGNAL('clicked()'), self.filterTab.removeFilterList)
        
        # persistent list button
        self.persistButton = QtGui.QPushButton(QtGui.QIcon(iconPath("application-certificate.svg")), "")
        self.persistButton.setFixedWidth(35)
        self.persistButton.setStatusTip("Persistent filter list")
        self.persistButton.setCheckable(1)
        self.persistButton.setChecked(0)
        
        # static list button
        self.staticListButton = QtGui.QPushButton(QtGui.QIcon(iconPath("Stop_hand_nuvola_black.svg")), "")
        self.staticListButton.setFixedWidth(35)
        self.staticListButton.setStatusTip("Static filter list")
        self.staticListButton.setCheckable(1)
        self.staticListButton.setChecked(0)
        
        # show scalar bar
        self.scalarBarButton = QtGui.QPushButton(QtGui.QIcon(iconPath("preferences-desktop-locale.svg")), "")
        self.scalarBarButton.setFixedWidth(35)
        self.scalarBarButton.setStatusTip("Show scalar bar")
        self.scalarBarButton.setCheckable(1)
        self.scalarBarButton.setChecked(0)
        self.scalarBarButton.clicked.connect(self.toggleScalarBar)
        
        # set up the row of buttons
        row1 = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row1)
        rowLayout.setAlignment(QtCore.Qt.AlignLeft)
        rowLayout.addWidget(self.visibleButton)
        rowLayout.addWidget(trashButton)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        
        row2 = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row2)
        rowLayout.setAlignment(QtCore.Qt.AlignRight)
        rowLayout.addWidget(self.persistButton)
        rowLayout.addWidget(self.staticListButton)
        rowLayout.addWidget(self.scalarBarButton)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        
        row3 = QtGui.QWidget()
        rowLayout = QtGui.QHBoxLayout(row3)
        rowLayout.addWidget(row1)
        rowLayout.addWidget(row2)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        
        self.filterListLayout.addWidget(row3)
        
        # Now add the list widget
        self.listItems = QtGui.QListWidget(self)
        self.listItems.setFixedHeight(self.tabHeight)
        
        self.connect(self.listItems, QtCore.SIGNAL('itemDoubleClicked(QListWidgetItem*)'), self.openFilterSettings)
        
        self.filterListLayout.addWidget(self.listItems)
        
        # add more buttons
        addFilter = QtGui.QPushButton(QtGui.QIcon(iconPath("list-add.svg")), "")
        addFilter.setStatusTip("Add new filter")
        self.connect(addFilter, QtCore.SIGNAL('clicked()'), self.addFilter)
        
        removeFilter = QtGui.QPushButton(QtGui.QIcon(iconPath("list-remove.svg")), "")
        removeFilter.setStatusTip("Remove filter")
        self.connect(removeFilter, QtCore.SIGNAL('clicked()'), self.removeFilter)
        
        moveUp = QtGui.QPushButton(QtGui.QIcon(iconPath("go-up.svg")), "")
        moveUp.setStatusTip("Move up")
        self.connect(moveUp, QtCore.SIGNAL('clicked()'), self.moveFilterUpInList)
        
        moveDown = QtGui.QPushButton(QtGui.QIcon(iconPath("go-down.svg")), "")
        moveDown.setStatusTip("Move down")
        self.connect(moveDown, QtCore.SIGNAL('clicked()'), self.moveFilterDownInList)
        
        clearList = QtGui.QPushButton(QtGui.QIcon(iconPath("edit-clear.svg")), "")
        clearList.setStatusTip("Clear current filter list")
        self.connect(clearList, QtCore.SIGNAL('clicked()'), self.clearList)
        
        applyList = QtGui.QPushButton(QtGui.QIcon(iconPath("view-refresh.svg")), "")
        applyList.setStatusTip("Apply current filter list")
        self.connect(applyList, QtCore.SIGNAL('clicked()'), self.applyList)
        
        buttonWidget = QtGui.QWidget()
        buttonLayout = QtGui.QHBoxLayout(buttonWidget)
        buttonLayout.setSpacing(0)
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.setAlignment(QtCore.Qt.AlignTop)
        
        buttonLayout.addWidget(addFilter)
        buttonLayout.addWidget(removeFilter)
        buttonLayout.addWidget(moveUp)
        buttonLayout.addWidget(moveDown)
        buttonLayout.addWidget(clearList)
        buttonLayout.addWidget(applyList)
        
        self.filterListLayout.addWidget(buttonWidget)
        
        # add other option like colour by height etc
        extraOptionsGroupBox = QtGui.QGroupBox("Additional filter list options")
        extraOptionsGroupBox.setAlignment(QtCore.Qt.AlignHCenter)
        
        groupLayout = QtGui.QVBoxLayout(extraOptionsGroupBox)
        groupLayout.setAlignment(QtCore.Qt.AlignTop)
        groupLayout.setContentsMargins(0, 0, 0, 0)
        groupLayout.setSpacing(0)
        
        # colouring options
        self.colouringOptionsButton = QtGui.QPushButton("Colouring options: Specie")
        self.colouringOptionsButton.clicked.connect(self.showColouringOptions)
        
        self.colouringOptions = filterListOptions.ColouringOptionsWindow(parent=self)
        self.colouringOptionsOpen = False
        
        groupLayout.addWidget(self.colouringOptionsButton)
        
        self.filterListLayout.addWidget(extraOptionsGroupBox)
        
        # the filterer (does the filtering)
        self.filterer = filterer.Filterer(self)
    
    def toggleScalarBar(self):
        """
        Toggle scalar bar (if there is one).
        
        """
        if self.scalarBarButton.isChecked():
            added = self.filterer.addScalarBar()
            
            if not added:
                self.scalarBarButton.setChecked(0)
        
        else:
            self.filterer.hideScalarBar()
    
    def showColouringOptions(self):
        """
        Show the colouring options window.
        
        """
        if self.colouringOptionsOpen:
            self.colouringOptions.closeEvent(1)
        
        self.colouringOptions.show()
        self.colouringOptionsOpen = True
    
    def openFilterSettings(self):
        """
        Open filter settings window
        
        """
        row = self.listItems.currentRow()
        self.currentSettings[row].hide()
        self.currentSettings[row].show()
    
    def openOptionsWindow(self):
        """
        Open additional options window
        
        """
        print "NOT IMPLEMENTED YET"
    
    def applyList(self):
        """
        Run filters in this list.
        
        """
        # apply filters
        self.filterer.runFilters()
        
        self.filterTab.refreshOnScreenInfo()
            
    def clearList(self):
        """
        Clear filters and actors from list.
        
        """
        self.filterer.removeActors()
        
        self.listItems.clear()
        
        while len(self.currentFilters):
            self.currentFilters.pop()
        
        while len(self.currentSettings):
            dlg = self.currentSettings.pop()
            dlg.accept()
        
        self.staticListButton.setChecked(0)
        self.persistButton.setChecked(0)
        
        self.defectFilterSelected = 0
        
        if self.filterer.scalarBarAdded:
#            self.filterer.hideScalarBar()
            self.scalarBarButton.setChecked(0)
        
        self.filterTab.refreshOnScreenInfo()
    
    def isStaticList(self):
        """
        Check if the list is a static list.
        
        """
        return self.staticListButton.isChecked()
    
    def isPersistentList(self):
        """
        Check if the list is a persistent list.
        
        """
        return self.persistButton.isChecked()
    
    def moveFilterDownInList(self):
        """
        Move filter down in list
        
        """
        if self.isStaticList():
            self.mainWindow.displayWarning("Cannot modify a static filter list")
            return
        
        # find which one is selected
        row = self.listItems.currentRow()
        
        if not len(self.listItems) or row < 0:
            return
        
        newRow = row + 1
        if newRow == len(self.listItems):
            return
        
        self.listItems.insertItem(newRow, self.listItems.takeItem(row))
        self.currentFilters.insert(newRow, self.currentFilters.pop(row))
        self.currentSettings.insert(newRow, self.currentSettings.pop(row))
    
    def moveFilterUpInList(self):
        """
        Move filter up in list
        
        """
        if self.isStaticList():
            self.mainWindow.displayWarning("Cannot modify a static filter list")
            return
        
        # find which one is selected
        row = self.listItems.currentRow()
        
        if not len(self.listItems) or row < 0:
            return
        
        newRow = row - 1
        if newRow < 0:
            return
        
        self.listItems.insertItem(newRow, self.listItems.takeItem(row))
        self.currentFilters.insert(newRow, self.currentFilters.pop(row))
        self.currentSettings.insert(newRow, self.currentSettings.pop(row))
    
    def warnDefectFilter(self):
        """
        Warn user that defect filter cannot 
        be used with any other filter
        
        """
        QtGui.QMessageBox.warning(self, "Warning", "The point defects filter cannot be used in conjunction with any other filter!")
    
    def addFilter(self):
        """
        Add new filter
        
        """
        if self.isStaticList():
            self.mainWindow.displayWarning("Cannot modify a static filter list")
            return
        
        # first determine what filter is to be added
        filterName, ok = QtGui.QInputDialog.getItem(self, "Add filter", "Select filter:", self.allFilters, editable=False)
        
        if ok:
            if self.defectFilterSelected:
                self.warnDefectFilter()
            
            elif len(self.currentFilters) and str(filterName) == "Point defects":
                self.warnDefectFilter()
            
            else:
                if filterName not in self.currentFilters:
                    self.currentFilters.append(str(filterName))
                    self.listItems.addItem(filterName)
                    
                    # create option form
                    form = self.createSettingsForm(filterName)
                    form.show()
                    self.currentSettings.append(form)
                    
                    if str(filterName) == "Point defects":
                        self.defectFilterSelected = 1
    
    def removeFilter(self):
        """
        Remove new filter
        
        """
        if self.isStaticList():
            self.mainWindow.displayWarning("Cannot modify a static filter list")
            return
        
        # find which one is selected
        row = self.listItems.currentRow()
        
        if not len(self.listItems) or row < 0:
            return
                
        # remove it from lists
        self.listItems.takeItem(row)
        filterName = self.currentFilters.pop(row)
        dlg = self.currentSettings.pop(row)
        dlg.accept()
        
        if filterName == "Point defects":
            self.defectFilterSelected = 0
    
    def createSettingsForm(self, filterName):
        """
        Create a settings form for the filter.
        
        """
        form = None
        
        words = str(filterName).title().split()
        
        dialogName = "%sSettingsDialog" % "".join(words)
        
        formObject = getattr(filterSettings, dialogName, None)
        if formObject is not None:
            title = "%s filter settings" % filterName
            form = formObject(self.mainWindow, title, parent=self)
        
        return form
    
    def visibilityChanged(self):
        """
        Update visibility of filter list
        
        """
        if self.visibleButton.isChecked():
            self.visibleButton.setIcon(QtGui.QIcon(iconPath("eye-close-ava.svg")))
            self.visible = 0
            self.filterer.hideActors()
        
        else:
            self.visibleButton.setIcon(QtGui.QIcon(iconPath("eye-ava.svg")))
            self.visible = 1
            self.filterer.addActors()
        
        self.filterTab.refreshOnScreenInfo()

