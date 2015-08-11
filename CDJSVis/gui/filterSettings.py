
"""
Settings for filters.

Dialogs must be named like: FilterNameSettingsDialog
where FilterName is the (capitalised) name of the
filter with no spaces. Eg "Point defects" becomes
"PointDefectsSettingsDialog".

@author: Chris Scott

"""
import logging
import functools

import numpy as np
from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from . import genericForm
from ..rendering import slicePlane
from ..state.atoms import elements


################################################################################
class GenericSettingsDialog(QtGui.QDialog):
    def __init__(self, title, parent):
        super(GenericSettingsDialog, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.mainWindow = self.parent.mainWindow
        self.pipelinePage = self.parent.filterTab
        
        self.logger = logging.getLogger(__name__)
        
        # get tab and filter id's
        array = title.split("(")[1].split(")")[0].split()
        self.listID = int(array[1])
        self.filterID = int(array[3])
        
        self.setModal(0)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/configure.png")))
#        self.resize(500,300)
        
        dialogLayout = QtGui.QVBoxLayout()
        dialogLayout.setAlignment(QtCore.Qt.AlignTop)
        
        tabWidget = QtGui.QTabWidget()
        
        # layout/widget
        self.contentLayout = QtGui.QFormLayout()
        contentWidget = QtGui.QWidget()
        contentWidget.setLayout(self.contentLayout)
        
        tabWidget.addTab(contentWidget, "Calculate")
        
        # display settings
        self.displaySettingsLayout = QtGui.QFormLayout()
        displaySettingsWidget = QtGui.QWidget()
        displaySettingsWidget.setLayout(self.displaySettingsLayout)
        
        tabWidget.addTab(displaySettingsWidget, "Display")
        
        dialogLayout.addWidget(tabWidget)
        self.setLayout(dialogLayout)
        
        # button box
        self.buttonBox = QtGui.QDialogButtonBox()
        
        # add close button
        closeButton = self.buttonBox.addButton(self.buttonBox.Close)
        closeButton.setDefault(True)
        self.buttonBox.rejected.connect(self.close)
        
        dialogLayout.addWidget(self.buttonBox)
        
        # filtering enabled by default
        self.filteringEnabled = True
        
        # help page
        self.helpPage = None
        
        # does this filter provide scalars
        self.providedScalars = []
    
    def addProvidedScalar(self, name):
        """
        Add scalar option
        
        """
        self.providedScalars.append(name)
    
    def addHorizontalDivider(self, displaySettings=False):
        """
        Add horizontal divider (QFrame)
        
        """
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        if displaySettings:
            self.displaySettingsLayout.addRow(line)
        else:
            self.contentLayout.addRow(line)
    
    def addLinkToHelpPage(self, page):
        """
        Add button with link to help page
        
        """
        helpButton = self.buttonBox.addButton(self.buttonBox.Help)
        helpButton.setAutoDefault(False)
        helpButton.setToolTip("Show help page")
        self.buttonBox.helpRequested.connect(self.loadHelpPage)
        
        self.helpPage = page
    
    def loadHelpPage(self):
        """
        Load the help page
        
        """
        self.logger.debug("Help requested: '%s'", self.helpPage)
        
        if self.helpPage is None:
            return
        
        self.mainWindow.helpWindow.loadPage(self.helpPage)
        self.mainWindow.showHelp()
    
    def addFilteringGroupBox(self, title="Enable filtering", slot=None, checked=False):
        """
        Add a group box that contains filtering options
        
        """
        # widget
        grp = QtGui.QGroupBox(title)
        grp.setCheckable(True)
        
        # layout
        grpLayout = QtGui.QVBoxLayout()
        grpLayout.setAlignment(QtCore.Qt.AlignTop)
        grpLayout.setContentsMargins(0,0,0,0)
        grpLayout.setSpacing(0)
        grp.setLayout(grpLayout)
        
        # connect toggled signal
        if slot is not None:
            grp.toggled.connect(slot)
        
        # initial check status
        grp.setChecked(checked)
        
        # add to form layout
        row = self.newRow()
        row.addWidget(grp)
        
        return grpLayout
    
    def addEnableFilteringCheck(self):
        """
        Add the enable filtering check to the form
        
        """
        self.enableFilteringCheck = QtGui.QCheckBox("Enable filtering")
        self.enableFilteringCheck.stateChanged.connect(self.enableFilteringChanged)
        self.enableFilteringCheck.setCheckState(QtCore.Qt.Unchecked)
        self.filteringEnabled = False
        row = self.newRow()
        row.addWidget(self.enableFilteringCheck)
        row = self.newRow()
    
    def enableFilteringChanged(self, checkState):
        """
        Enable filtering option has changed
        
        """
        if checkState == QtCore.Qt.Unchecked:
            self.filteringEnabled = False
        else:
            self.filteringEnabled = True
        
        self.logger.debug("Enable filtering changed (%s): %r", self.filterType, self.filteringEnabled)
    
    def newRow(self, align=None):
        """
        New filter settings row.
        
        """
        row = genericForm.FormRow(align=align)
        self.contentLayout.addWidget(row)
        
        return row
    
    def removeRow(self,row):
        """
        Remove filter settings row
        
        """
        self.contentLayout.removeWidget(row)  
    
    def newDisplayRow(self, align=None):
        """
        New display settings row.
        
        """
        row = genericForm.FormRow(align=align)
        self.displaySettingsLayout.addWidget(row)
        
        return row
    
    def closeEvent(self, event):
        self.hide()
    
    def refresh(self):
        """
        Called whenever a new input is loaded.
        
        Should be overridden if required.
        
        """
        pass


################################################################################
class SpeciesSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(SpeciesSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Species"
        
        # specie list
        self.specieList = QtGui.QListWidget(self)
#         self.specieList.setFixedHeight(100)
        self.specieList.setFixedWidth(200)
        row = self.newRow()
        row.addWidget(self.specieList)
        
        self.refresh()
    
    def getVisibleSpecieList(self):
        """
        Return list of visible species
        
        """
        visibleSpecieList = []
        for i in xrange(self.specieList.count()):
            item = self.specieList.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                visibleSpecieList.append(item.symbol)
        
        return visibleSpecieList
    
    def refresh(self):
        """
        Refresh the specie list
        
        """
        self.logger.debug("Refreshing species filter options")
        
        inputSpecieList = self.pipelinePage.inputState.specieList
        refSpecieList = self.pipelinePage.refState.specieList
        
        # set of added species
        currentSpecies = set()
        
        # remove species that don't exist
        num = self.specieList.count()
        for i in xrange(num - 1, -1, -1):
            item = self.specieList.item(i)
            
            # remove if doesn't exist in both ref and input
            if item.symbol not in inputSpecieList and item.symbol not in refSpecieList:
                self.logger.debug("  Removing species option: %s", item.symbol)
                self.specieList.takeItem(i) # does this delete it?
            
            else:
                currentSpecies.add(item.symbol)
        
        # unique species from ref/input
        combinedSpecieList = list(inputSpecieList) + list(refSpecieList)
        uniqueCurrentSpecies = set(combinedSpecieList)
        
        # add species that aren't already added
        for sym in uniqueCurrentSpecies:
            if sym in currentSpecies:
                self.logger.debug("  Keeping species option: %s", sym)
            
            else:
                self.logger.debug("  Adding species option: %s", sym)
                name = elements.atomName(sym)
                item = SpecieListItem(sym, name=name)
                self.specieList.addItem(item)

################################################################################

class CropBoxSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(CropBoxSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Crop box"
        
        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0
        self.zmin = 0.0
        self.zmax = 0.0
        self.xEnabled = 0
        self.yEnabled = 0
        self.zEnabled = 0
        self.invertSelection = 0
        
        # x
        self.xCropCheckBox = QtGui.QCheckBox()
        self.xCropCheckBox.setChecked(0)
        self.xCropCheckBox.setToolTip("Enable cropping in the x direction")
        self.xCropCheckBox.stateChanged[int].connect(self.changedXEnabled)
        self.xMinRangeSpinBox = QtGui.QDoubleSpinBox()
        self.xMinRangeSpinBox.setSingleStep(1)
        self.xMinRangeSpinBox.setMinimum(-9999.0)
        self.xMinRangeSpinBox.setMaximum(9999.0)
        self.xMinRangeSpinBox.setValue(self.xmin)
        self.xMinRangeSpinBox.setToolTip("Minimum x value")
        self.xMinRangeSpinBox.valueChanged.connect(self.setXMin)
        self.xMaxRangeSpinBox = QtGui.QDoubleSpinBox()
        self.xMaxRangeSpinBox.setSingleStep(1)
        self.xMaxRangeSpinBox.setMinimum(-9999.0)
        self.xMaxRangeSpinBox.setMaximum(9999.0)
        self.xMaxRangeSpinBox.setValue(self.xmax)
        self.xMaxRangeSpinBox.setToolTip("Maximum x value")
        self.xMaxRangeSpinBox.valueChanged.connect(self.setXMax)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.xMinRangeSpinBox)
        hbox.addWidget(QtGui.QLabel("-"))
        hbox.addWidget(self.xMaxRangeSpinBox)
        
        self.contentLayout.addRow("X Crop Enabled", self.xCropCheckBox)
        self.contentLayout.addRow("Range", hbox)
        
        # divider
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        self.contentLayout.addRow(line)
        
        # y
        self.yCropCheckBox = QtGui.QCheckBox()
        self.yCropCheckBox.setChecked(0)
        self.yCropCheckBox.setToolTip("Enable cropping in the y direction")
        self.yCropCheckBox.stateChanged[int].connect(self.changedYEnabled)
        self.yMinRangeSpinBox = QtGui.QDoubleSpinBox()
        self.yMinRangeSpinBox.setSingleStep(1)
        self.yMinRangeSpinBox.setMinimum(-9999.0)
        self.yMinRangeSpinBox.setMaximum(9999.0)
        self.yMinRangeSpinBox.setValue(self.ymin)
        self.yMinRangeSpinBox.setToolTip("Minimum y value")
        self.yMinRangeSpinBox.valueChanged.connect(self.setYMin)
        self.yMaxRangeSpinBox = QtGui.QDoubleSpinBox()
        self.yMaxRangeSpinBox.setSingleStep(1)
        self.yMaxRangeSpinBox.setMinimum(-9999.0)
        self.yMaxRangeSpinBox.setMaximum(9999.0)
        self.yMaxRangeSpinBox.setValue(self.ymax)
        self.yMaxRangeSpinBox.setToolTip("Maximum y value")
        self.yMaxRangeSpinBox.valueChanged.connect(self.setYMax)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.yMinRangeSpinBox)
        hbox.addWidget(QtGui.QLabel("-"))
        hbox.addWidget(self.yMaxRangeSpinBox)
        
        self.contentLayout.addRow("Y Crop Enabled", self.yCropCheckBox)
        self.contentLayout.addRow("Range", hbox)
        
        # divider
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        self.contentLayout.addRow(line)
        
        # z
        self.zCropCheckBox = QtGui.QCheckBox()
        self.zCropCheckBox.setChecked(0)
        self.zCropCheckBox.setToolTip("Enable cropping in the z direction")
        self.zCropCheckBox.stateChanged[int].connect(self.changedZEnabled)
        self.zMinRangeSpinBox = QtGui.QDoubleSpinBox()
        self.zMinRangeSpinBox.setSingleStep(1)
        self.zMinRangeSpinBox.setMinimum(-9999.0)
        self.zMinRangeSpinBox.setMaximum(9999.0)
        self.zMinRangeSpinBox.setValue(self.zmin)
        self.zMinRangeSpinBox.setToolTip("Minimum z value")
        self.zMinRangeSpinBox.valueChanged.connect(self.setZMin)
        self.zMaxRangeSpinBox = QtGui.QDoubleSpinBox()
        self.zMaxRangeSpinBox.setSingleStep(1)
        self.zMaxRangeSpinBox.setMinimum(-9999.0)
        self.zMaxRangeSpinBox.setMaximum(9999.0)
        self.zMaxRangeSpinBox.setValue(self.zmax)
        self.zMaxRangeSpinBox.setToolTip("Maximum z value")
        self.zMaxRangeSpinBox.valueChanged.connect(self.setZMax)
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.zMinRangeSpinBox)
        hbox.addWidget(QtGui.QLabel("-"))
        hbox.addWidget(self.zMaxRangeSpinBox)
        
        self.contentLayout.addRow("Z Crop Enabled", self.zCropCheckBox)
        self.contentLayout.addRow("Range", hbox)
        
        # divider
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        self.contentLayout.addRow(line)
        
        # invert selection
        self.invertCheckBox = QtGui.QCheckBox()
        self.invertCheckBox.setChecked(0)
        self.invertCheckBox.stateChanged.connect(self.invertChanged)
        self.invertCheckBox.setToolTip("Invert selection")
        self.contentLayout.addRow("Invert selection", self.invertCheckBox)
        
        # reset
        self.setToLatticeButton = QtGui.QPushButton('Set to lattice')
        self.setToLatticeButton.setAutoDefault(0)
        self.setToLatticeButton.setToolTip('Set crop to lattice dimensions')
        self.setToLatticeButton.clicked.connect(self.setCropToLattice)
        self.contentLayout.addRow(self.setToLatticeButton)
    
    def invertChanged(self, index):
        """
        Invert setting changed.
        
        """
        if self.invertCheckBox.isChecked():
            self.invertSelection = 1
        
        else:
            self.invertSelection = 0
    
    def setCropToLattice(self):
        self.xMinRangeSpinBox.setValue(0.0)
        self.xMaxRangeSpinBox.setValue(self.pipelinePage.inputState.cellDims[0])
        self.yMinRangeSpinBox.setValue(0.0)
        self.yMaxRangeSpinBox.setValue(self.pipelinePage.inputState.cellDims[1])
        self.zMinRangeSpinBox.setValue(0.0)
        self.zMaxRangeSpinBox.setValue(self.pipelinePage.inputState.cellDims[2])
    
    def changedXEnabled(self):
        if self.xCropCheckBox.isChecked():
            self.xEnabled = 1
        else:
            self.xEnabled = 0
    
    def changedYEnabled(self):
        if self.yCropCheckBox.isChecked():
            self.yEnabled = 1
        else:
            self.yEnabled = 0
    
    def changedZEnabled(self):
        if self.zCropCheckBox.isChecked():
            self.zEnabled = 1
        else:
            self.zEnabled = 0
    
    def setXMin(self, val):
        self.xmin = val
    
    def setXMax(self, val):
        self.xmax = val
    
    def setYMin(self, val):
        self.ymin = val
    
    def setYMax(self, val):
        self.ymax = val
    
    def setZMin(self, val):
        self.zmin = val
    
    def setZMax(self, val):
        self.zmax = val
    
    def refresh(self):
        self.xMinRangeSpinBox.setValue(self.xmin)
        self.xMaxRangeSpinBox.setValue(self.xmax)
        self.yMinRangeSpinBox.setValue(self.ymin)
        self.yMaxRangeSpinBox.setValue(self.ymax)
        self.zMinRangeSpinBox.setValue(self.zmin)
        self.zMaxRangeSpinBox.setValue(self.zmax)
        self.xCropCheckBox.setChecked( self.xEnabled )
        self.yCropCheckBox.setChecked( self.yEnabled )
        self.zCropCheckBox.setChecked( self.zEnabled )
    
    def reset(self):
        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0
        self.zmin = 0.0
        self.zmax = 0.0
        self.xEnabled = 0
        self.yEnabled = 0
        self.zEnabled = 0
        self.refresh()


################################################################################
class CropSphereSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(CropSphereSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Crop sphere"
        
        self.xCentre = 0.0
        self.yCentre = 0.0
        self.zCentre = 0.0
        self.radius = 1.0
        self.invertSelection = 0
        
        self.xCentreSpinBox = QtGui.QDoubleSpinBox()
        self.xCentreSpinBox.setSingleStep(0.01)
        self.xCentreSpinBox.setMinimum(-9999.0)
        self.xCentreSpinBox.setMaximum( 9999.0)
        self.xCentreSpinBox.setValue(self.xCentre)
        self.xCentreSpinBox.setToolTip("Centre of crop region (x)")
        self.xCentreSpinBox.valueChanged.connect(self.xCentreChanged)
        
        self.yCentreSpinBox = QtGui.QDoubleSpinBox()
        self.yCentreSpinBox.setSingleStep(0.01)
        self.yCentreSpinBox.setMinimum(-9999.0)
        self.yCentreSpinBox.setMaximum( 9999.0)
        self.yCentreSpinBox.setValue(self.yCentre)
        self.yCentreSpinBox.setToolTip("Centre of crop region (y)")
        self.yCentreSpinBox.valueChanged.connect(self.yCentreChanged)
        
        self.zCentreSpinBox = QtGui.QDoubleSpinBox()
        self.zCentreSpinBox.setSingleStep(0.01)
        self.zCentreSpinBox.setMinimum(-9999.0)
        self.zCentreSpinBox.setMaximum( 9999.0)
        self.zCentreSpinBox.setValue(self.zCentre)
        self.zCentreSpinBox.setToolTip("Centre of crop region (z)")
        self.zCentreSpinBox.valueChanged.connect(self.zCentreChanged)
        
        self.contentLayout.addRow("Centre (x)", self.xCentreSpinBox)
        self.contentLayout.addRow("Centre (y)", self.yCentreSpinBox)
        self.contentLayout.addRow("Centre (z)", self.zCentreSpinBox)
        
        # radius
        self.radiusSpinBox = QtGui.QDoubleSpinBox()
        self.radiusSpinBox.setSingleStep(1)
        self.radiusSpinBox.setMinimum(0.0)
        self.radiusSpinBox.setMaximum(9999.0)
        self.radiusSpinBox.setValue(self.radius)
        self.radiusSpinBox.setToolTip("Radius of sphere")
        self.radiusSpinBox.valueChanged.connect(self.radiusChanged)
        self.contentLayout.addRow("Radius", self.radiusSpinBox)
        
        # invert selection
        self.invertCheckBox = QtGui.QCheckBox()
        self.invertCheckBox.setChecked(0)
        self.invertCheckBox.setToolTip("Invert selection")
        self.invertCheckBox.stateChanged.connect(self.invertChanged)
        self.contentLayout.addRow("Invert selection", self.invertCheckBox)
        
        # set to centre
        self.setToLatticeButton = QtGui.QPushButton('Set to lattice centre')
        self.setToLatticeButton.setAutoDefault(0)
        self.setToLatticeButton.setToolTip('Set to lattice centre')
        self.setToLatticeButton.clicked.connect(self.setToLattice)
        self.contentLayout.addRow(self.setToLatticeButton)
    
    def invertChanged(self, index):
        """
        Invert setting changed.
        
        """
        if self.invertCheckBox.isChecked():
            self.invertSelection = 1
        
        else:
            self.invertSelection = 0
    
    def setToLattice(self):
        """
        Set centre to lattice centre.
        
        """
        self.xCentreSpinBox.setValue(self.pipelinePage.inputState.cellDims[0] / 2.0)
        self.yCentreSpinBox.setValue(self.pipelinePage.inputState.cellDims[1] / 2.0)
        self.zCentreSpinBox.setValue(self.pipelinePage.inputState.cellDims[2] / 2.0)
    
    def radiusChanged(self, val):
        """
        Radius changed.
        
        """
        self.radius = val
    
    def xCentreChanged(self, val):
        """
        X centre changed.
        
        """
        self.xCentre = val
    
    def yCentreChanged(self, val):
        """
        Y centre changed.
        
        """
        self.yCentre = val
    
    def zCentreChanged(self, val):
        """
        Z centre changed.
        
        """
        self.zCentre = val
        
################################################################################

class SpecieListItem(QtGui.QListWidgetItem):
    """
    Item in a specie list widget.
    
    """
    def __init__(self, symbol, name=None):
        super(SpecieListItem, self).__init__()
        
        # add check box
        self.setFlags(self.flags() | QtCore.Qt.ItemIsUserCheckable)
        
        # don't allow it to be selected
        self.setFlags(self.flags() & ~QtCore.Qt.ItemIsSelectable)
        
        # set unchecked initially
        self.setCheckState(QtCore.Qt.Checked)
        
        # store bond pair
        self.symbol = symbol
        
        # set text
        if name is not None:
            self.setText("%s - %s" % (symbol, name))
        
        else:
            self.setText("%s" % symbol)

################################################################################

class PointDefectsSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(PointDefectsSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Point defects"
        
        # settings
        self.vacancyRadius = 1.3
        self.showInterstitials = 1
        self.showAntisites = 1
        self.showVacancies = 1
        self.findClusters = 0
        self.neighbourRadius = 3.5
        self.minClusterSize = 3
        self.maxClusterSize = -1
        self.hullCol = [0]*3
        self.hullCol[2] = 1
        self.hullOpacity = 0.5
        self.calculateVolumes = False
        self.calculateVolumesVoro = True
        self.calculateVolumesHull = False
        self.drawConvexHulls = 0
        self.hideDefects = 0
        self.identifySplitInts = 1
        self.vacScaleSize = 0.75
        self.vacOpacity = 0.8
        self.vacSpecular = 0.4
        self.vacSpecularPower = 10
        
        # vacancy radius option
        self.vacRadSpinBox = QtGui.QDoubleSpinBox()
        self.vacRadSpinBox.setSingleStep(0.1)
        self.vacRadSpinBox.setMinimum(0.01)
        self.vacRadSpinBox.setMaximum(10.0)
        self.vacRadSpinBox.setValue(self.vacancyRadius)
        self.vacRadSpinBox.setToolTip("The vacancy radius is used to determine if an input atom "
                                      "is associated with a reference site.")
        self.vacRadSpinBox.valueChanged[float].connect(self.vacRadChanged)
        self.contentLayout.addRow("Vacancy radius", self.vacRadSpinBox)
        
        self.addHorizontalDivider()
        
        # defect type options
        self.intTypeCheckBox = QtGui.QCheckBox("Interstitials")
        self.intTypeCheckBox.setChecked(1)
        self.intTypeCheckBox.stateChanged[int].connect(self.intVisChanged)
        self.intTypeCheckBox.setToolTip("Show interstitials")
        
        self.vacTypeCheckBox = QtGui.QCheckBox("Vacancies")
        self.vacTypeCheckBox.setChecked(1)
        self.vacTypeCheckBox.stateChanged[int].connect(self.vacVisChanged)
        self.vacTypeCheckBox.setToolTip("Show vacancies")
        
        self.antTypeCheckBox = QtGui.QCheckBox("Antisites")
        self.antTypeCheckBox.setChecked(1)
        self.antTypeCheckBox.stateChanged[int].connect(self.antVisChanged)
        self.antTypeCheckBox.setToolTip("Show antisites")
        
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.intTypeCheckBox)
        vbox.addWidget(self.vacTypeCheckBox)
        vbox.addWidget(self.antTypeCheckBox)
        self.contentLayout.addRow("Defect visibility", vbox)
        
        self.addHorizontalDivider()
        
        # identify split ints check box
        self.identifySplitsCheck = QtGui.QCheckBox()
        self.identifySplitsCheck.setChecked(1)
        self.identifySplitsCheck.stateChanged.connect(self.identifySplitsChanged)
        self.identifySplitsCheck.setToolTip("Attempt to identify split interstitials (2 interstitials and 1 vacancy). "
                                            "Note: a split interstitial is counted as 1 defect, not 3.")
        self.contentLayout.addRow("Identify split interstitials", self.identifySplitsCheck)
        
        self.addHorizontalDivider()
        
        # use acna options
        self.useAcna = False
        useAcnaCheck = QtGui.QCheckBox()
        useAcnaCheck.setChecked(self.useAcna)
        useAcnaCheck.stateChanged.connect(self.useAcnaToggled)
        useAcnaCheck.setToolTip("Use Adaptive Common Neighbour Analysis to complement the comparison "
                                "to a reference system. Interstitials with the selected structure are "
                                "not classified as defects if they have a neighbouring vacancy.")
        self.contentLayout.addRow("<b>Use ACNA</b>", useAcnaCheck)
        
        # acna max bond distance
        self.acnaMaxBondDistance = 5.0
        self.maxBondDistanceSpin = QtGui.QDoubleSpinBox()
        self.maxBondDistanceSpin.setSingleStep(0.1)
        self.maxBondDistanceSpin.setMinimum(2.0)
        self.maxBondDistanceSpin.setMaximum(9.99)
        self.maxBondDistanceSpin.setValue(self.acnaMaxBondDistance)
        self.maxBondDistanceSpin.valueChanged[float].connect(self.setAcnaMaxBondDistance)
        self.maxBondDistanceSpin.setToolTip("This value is used for spatially decomposing the system. It "
                                            "should be set large enough to include all required atoms. "
                                            "If unsure just set it to something big, eg. 10.0")
        self.contentLayout.addRow("Max bond distance", self.maxBondDistanceSpin)
        
        # acna ideal structure
        self.acnaStructureType = 1
        self.acnaStructureCombo = QtGui.QComboBox()
        filterer = self.parent.filterer
        self.acnaStructureCombo.addItems(filterer.knownStructures)
        self.acnaStructureCombo.setCurrentIndex(self.acnaStructureType)
        self.acnaStructureCombo.currentIndexChanged.connect(self.acnaStructureTypeChanged)
        self.contentLayout.addRow("Structure", self.acnaStructureCombo)
        
        # make sure set up properly
        self.useAcnaToggled(QtCore.Qt.Unchecked)
        
        self.addHorizontalDivider()
        
        # find clusters settings
        findClustersCheck = QtGui.QCheckBox()
        findClustersCheck.setToolTip("Identify clusters of defects in the system.")
        findClustersCheck.stateChanged.connect(self.findClustersChanged)
        findClustersCheck.setChecked(self.findClusters)
        self.contentLayout.addRow("<b>Identify clusters</b>", findClustersCheck)
        
        # neighbour rad spin box
        self.nebRadSpinBox = QtGui.QDoubleSpinBox()
        self.nebRadSpinBox.setSingleStep(0.1)
        self.nebRadSpinBox.setMinimum(0.01)
        self.nebRadSpinBox.setMaximum(100.0)
        self.nebRadSpinBox.setValue(self.neighbourRadius)
        self.nebRadSpinBox.valueChanged[float].connect(self.nebRadChanged)
        self.nebRadSpinBox.setToolTip("Clusters are constructed using a recursive algorithm where "
                                      "two defects are said to be neighbours if their separation "
                                      "is less than this value.")
        self.contentLayout.addRow("Neighbour radius", self.nebRadSpinBox)
        
        # minimum size spin box
        self.minNumSpinBox = QtGui.QSpinBox()
        self.minNumSpinBox.setMinimum(1)
        self.minNumSpinBox.setMaximum(1000)
        self.minNumSpinBox.setValue(self.minClusterSize)
        self.minNumSpinBox.valueChanged[int].connect(self.minNumChanged)
        self.minNumSpinBox.setToolTip("Only show clusters that contain more than this number of defects.")
        self.contentLayout.addRow("Minimum cluster size", self.minNumSpinBox)
        
        # maximum size spin box
        self.maxNumSpinBox = QtGui.QSpinBox()
        self.maxNumSpinBox.setMinimum(-1)
        self.maxNumSpinBox.setMaximum(9999)
        self.maxNumSpinBox.setValue(self.maxClusterSize)
        self.maxNumSpinBox.valueChanged[int].connect(self.maxNumChanged)
        self.maxNumSpinBox.setToolTip("Only show clusters that contain less than this number of defects. "
                                      "Set to '-1' to disable this condition.")
        self.contentLayout.addRow("Maximum cluster size", self.maxNumSpinBox)
        
        # calculate volumes options
        self.calcVolsCheck = QtGui.QCheckBox()
        self.calcVolsCheck.setToolTip("Calculate volumes of defect clusters.")
        self.calcVolsCheck.stateChanged.connect(self.calcVolsChanged)
        self.calcVolsCheck.setChecked(self.calculateVolumes)
        self.contentLayout.addRow("<b>Calculate volumes</b>", self.calcVolsCheck)
        
        # radio buttons
        self.convHullVolRadio = QtGui.QRadioButton(parent=self.calcVolsCheck)
        self.convHullVolRadio.toggled.connect(self.calcVolsMethodChanged)
        self.convHullVolRadio.setToolTip("Volume is determined from the convex hull of the defect positions.")
        self.voroVolRadio = QtGui.QRadioButton(parent=self.calcVolsCheck)
        self.voroVolRadio.setToolTip("Volume is determined by summing the Voronoi volumes of the defects in "
                                     "the cluster. Ghost atoms are added for vacancies when computing the "
                                     "individual Voronoi volumes.")
        self.voroVolRadio.setChecked(True)
        self.contentLayout.addRow("Convex hull volumes", self.convHullVolRadio)
        self.contentLayout.addRow("Sum Voronoi volumes", self.voroVolRadio)
        
        # make sure setup properly
        self.calcVolsChanged(QtCore.Qt.Unchecked)
        
        self.addHorizontalDivider()
        
        # filter species group
        self.filterSpecies = False
        self.filterSpeciesCheck = QtGui.QCheckBox()
        self.filterSpeciesCheck.setChecked(self.filterSpecies)
        self.filterSpeciesCheck.setToolTip("Filter visible defects by species")
        self.filterSpeciesCheck.stateChanged.connect(self.filterSpeciesToggled)
        
        self.specieList = QtGui.QListWidget(self)
        self.specieList.setFixedHeight(80)
        self.specieList.setFixedWidth(100)
        self.specieList.setEnabled(self.filterSpecies)
        
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.filterSpeciesCheck)
        vbox.addWidget(self.specieList)
        self.contentLayout.addRow("Filter species", vbox)
        
        # draw hulls options
        self.drawHullsCheck = QtGui.QCheckBox()
        self.drawHullsCheck.setChecked(False)
        self.drawHullsCheck.setToolTip("Draw convex hulls of defect clusters")
        self.drawHullsCheck.stateChanged.connect(self.drawHullsChanged)
        self.displaySettingsLayout.addRow("<b>Draw convex hulls</b>", self.drawHullsCheck)
        
        # hull colour
        col = QtGui.QColor(self.hullCol[0]*255.0, self.hullCol[1]*255.0, self.hullCol[2]*255.0)
        self.hullColourButton = QtGui.QPushButton("")
        self.hullColourButton.setFixedWidth(50)
        self.hullColourButton.setFixedHeight(30)
        self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
        self.hullColourButton.clicked.connect(self.showColourDialog)
        self.hullColourButton.setToolTip("The colour of the hull.")
        self.displaySettingsLayout.addRow("Hull colour", self.hullColourButton)
        
        # hull opacity
        self.hullOpacitySpinBox = QtGui.QDoubleSpinBox()
        self.hullOpacitySpinBox.setSingleStep(0.01)
        self.hullOpacitySpinBox.setMinimum(0.01)
        self.hullOpacitySpinBox.setMaximum(1.0)
        self.hullOpacitySpinBox.setValue(self.hullOpacity)
        self.hullOpacitySpinBox.setToolTip("The opacity of the convex hulls")
        self.hullOpacitySpinBox.valueChanged[float].connect(self.hullOpacityChanged)
        self.displaySettingsLayout.addRow("Hull opacity", self.hullOpacitySpinBox)
        
        # hide atoms
        self.hideAtomsCheckBox = QtGui.QCheckBox()
        self.hideAtomsCheckBox.stateChanged.connect(self.hideDefectsChanged)
        self.hideAtomsCheckBox.setToolTip("Don't show the defects when rendering the convex hulls")
        self.displaySettingsLayout.addRow("Hide defects", self.hideAtomsCheckBox)
        
        self.drawHullsChanged(QtCore.Qt.Unchecked)
        
        self.addHorizontalDivider(displaySettings=True)
        
        # vac display settings
        # scale size
        vacScaleSizeSpin = QtGui.QDoubleSpinBox()
        vacScaleSizeSpin.setMinimum(0.1)
        vacScaleSizeSpin.setMaximum(2.0)
        vacScaleSizeSpin.setSingleStep(0.1)
        vacScaleSizeSpin.setValue(self.vacScaleSize)
        vacScaleSizeSpin.valueChanged.connect(self.vacScaleSizeChanged)
        vacScaleSizeSpin.setToolTip("When rendering vacancies scale the atomic radius by this amount (usually < 1)")
        self.displaySettingsLayout.addRow("Vacancy scale size", vacScaleSizeSpin)
        
        # opacity
        vacOpacitySpin = QtGui.QDoubleSpinBox()
        vacOpacitySpin.setMinimum(0.01)
        vacOpacitySpin.setMaximum(1.0)
        vacOpacitySpin.setSingleStep(0.1)
        vacOpacitySpin.setValue(self.vacOpacity)
        vacOpacitySpin.setToolTip("The opacity value for vacancies.")
        vacOpacitySpin.valueChanged.connect(self.vacOpacityChanged)
        self.displaySettingsLayout.addRow("Vacancy opacity", vacOpacitySpin)
        
        # specular
        vacSpecularSpin = QtGui.QDoubleSpinBox()
        vacSpecularSpin.setMinimum(0.01)
        vacSpecularSpin.setMaximum(1.0)
        vacSpecularSpin.setSingleStep(0.01)
        vacSpecularSpin.setValue(self.vacSpecular)
        vacSpecularSpin.setToolTip("Vacancy specular value")
        vacSpecularSpin.valueChanged.connect(self.vacSpecularChanged)
        self.displaySettingsLayout.addRow("Vacancy specular", vacSpecularSpin)
        
        # specular power
        vacSpecularPowerSpin = QtGui.QDoubleSpinBox()
        vacSpecularPowerSpin.setMinimum(0)
        vacSpecularPowerSpin.setMaximum(100)
        vacSpecularPowerSpin.setSingleStep(0.1)
        vacSpecularPowerSpin.setValue(self.vacSpecularPower)
        vacSpecularPowerSpin.setToolTip("Vacancy specu;ar power value")
        vacSpecularPowerSpin.valueChanged.connect(self.vacSpecularPowerChanged)
        self.displaySettingsLayout.addRow("Vacancy specular power", vacSpecularPowerSpin)
        
        self.addHorizontalDivider(displaySettings=True)
        
        self.drawDisplacementVectors = False
        self.bondThicknessVTK = 0.4
        self.bondThicknessPOV = 0.4
        self.bondNumSides = 5
        
        # draw displacement vector settings
        self.drawVectorsCheck = QtGui.QCheckBox()
        self.drawVectorsCheck.stateChanged.connect(self.drawVectorsChanged)
        self.drawVectorsCheck.setCheckState(QtCore.Qt.Unchecked)
        self.drawVectorsCheck.setToolTip("Draw displacement vectors (movement) of defects")
        self.disableDrawVectorsCheck()
        
        self.displaySettingsLayout.addRow("<b>Draw displacement vectors</b>", self.drawVectorsCheck)
        
        # vtk thickness
        self.vtkThickSpin = QtGui.QDoubleSpinBox()
        self.vtkThickSpin.setMinimum(0.01)
        self.vtkThickSpin.setMaximum(10)
        self.vtkThickSpin.setSingleStep(0.1)
        self.vtkThickSpin.setValue(self.bondThicknessVTK)
        self.vtkThickSpin.valueChanged.connect(self.vtkThickChanged)
        self.vtkThickSpin.setToolTip("Thickness of lines showing defect movement (VTK)")
        self.displaySettingsLayout.addRow("Bond thickness (VTK)", self.vtkThickSpin)
        
        # pov thickness
        self.povThickSpin = QtGui.QDoubleSpinBox()
        self.povThickSpin.setMinimum(0.01)
        self.povThickSpin.setMaximum(10)
        self.povThickSpin.setSingleStep(0.01)
        self.povThickSpin.setValue(self.bondThicknessPOV)
        self.povThickSpin.valueChanged.connect(self.povThickChanged)
        self.povThickSpin.setToolTip("Thickness of lines showing defect movement (POV-Ray)")
        self.displaySettingsLayout.addRow("Bond thickness (POV)", self.povThickSpin)
        
        # num sides
        self.numSidesSpin = QtGui.QSpinBox()
        self.numSidesSpin.setMinimum(3)
        self.numSidesSpin.setMaximum(999)
        self.numSidesSpin.setSingleStep(1)
        self.numSidesSpin.setValue(self.bondNumSides)
        self.numSidesSpin.valueChanged.connect(self.numSidesChanged)
        self.numSidesSpin.setToolTip("Number of sides when rendering displacement vectors (more looks better but is slower)")
        self.displaySettingsLayout.addRow("Bond number of sides", self.numSidesSpin)
        
        self.drawVectorsChanged(QtCore.Qt.Unchecked)
        self.findClustersChanged(QtCore.Qt.Unchecked)
        
        self.refresh()
    
    def acnaStructureTypeChanged(self, index):
        """
        ACNA structure type has changed
        
        """
        self.acnaStructureType = index
    
    def setAcnaMaxBondDistance(self, val):
        """
        Set max bond distance for ACNA
        
        """
        self.acnaMaxBondDistance = val
    
    def useAcnaToggled(self, state):
        """
        Use ACNA toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.useAcna = False
            
            # disable associated
            self.maxBondDistanceSpin.setEnabled(False)
            self.acnaStructureCombo.setEnabled(False)
        
        else:
            self.useAcna = True
            
            # disable associated
            self.maxBondDistanceSpin.setEnabled(True)
            self.acnaStructureCombo.setEnabled(True)
    
    def numSidesChanged(self, val):
        """
        Number of sides changed.
        
        """
        self.bondNumSides = val
    
    def vtkThickChanged(self, val):
        """
        VTK thickness changed.
        
        """
        self.bondThicknessVTK = val
    
    def povThickChanged(self, val):
        """
        POV thickness changed.
        
        """
        self.bondThicknessPOV = val
    
    def enableDrawVectorsCheck(self):
        """
        Enable
        
        """
        self.drawVectorsCheck.setEnabled(True)
        if self.drawDisplacementVectors:
            self.vtkThickSpin.setEnabled(True)
            self.povThickSpin.setEnabled(True)
    
    def disableDrawVectorsCheck(self):
        """
        Disable
        
        """
        self.drawVectorsCheck.setEnabled(False)
        if self.drawDisplacementVectors:
            self.vtkThickSpin.setEnabled(False)
            self.povThickSpin.setEnabled(False)
        
    
    def drawVectorsChanged(self, state):
        """
        Draw displacement vectors toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.drawDisplacementVectors = False
            
            self.vtkThickSpin.setEnabled(False)
            self.povThickSpin.setEnabled(False)
            self.numSidesSpin.setEnabled(False)
        
        else:
            self.drawDisplacementVectors = True
            
            self.vtkThickSpin.setEnabled(True)
            self.povThickSpin.setEnabled(True)
            self.numSidesSpin.setEnabled(True)
        
        self.logger.debug("Draw defect displacement vectors: %r", self.drawDisplacementVectors)
    
    def vacSpecularPowerChanged(self, val):
        """
        Vac specular power changed.
        
        """
        self.vacSpecularPower = val
    
    def vacSpecularChanged(self, val):
        """
        Vac specular changed.
        
        """
        self.vacSpecular = val
    
    def vacOpacityChanged(self, val):
        """
        Vac opacity changed.
        
        """
        self.vacOpacity = val
    
    def vacScaleSizeChanged(self, val):
        """
        Vac scale size changed.
        
        """
        self.vacScaleSize = val
    
    def identifySplitsChanged(self, state):
        """
        Identity splits changed.
        
        """
        if self.identifySplitsCheck.isChecked():
            self.identifySplitInts = 1
        
        else:
            self.identifySplitInts = 0
    
    def hideDefectsChanged(self, val):
        """
        Hide atoms check changed.
        
        """
        if self.hideAtomsCheckBox.isChecked():
            self.hideDefects = 1
        
        else:
            self.hideDefects = 0
    
    def minNumChanged(self, val):
        """
        Change min cluster size.
        
        """
        self.minClusterSize = val
    
    def maxNumChanged(self, val):
        """
        Change max cluster size.
        
        """
        self.maxClusterSize = val
    
    def nebRadChanged(self, val):
        """
        Change neighbour radius.
        
        """
        self.neighbourRadius = val
    
    def disableCalcVolsCheck(self):
        """
        Disable calc vols check box
        
        """
        self.calcVolsCheck.setEnabled(False)
        if self.calculateVolumes:
            self.convHullVolRadio.setEnabled(False)
            self.voroVolRadio.setEnabled(False)
    
    def enableCalcVolsCheck(self):
        """
        Enable calc vols check box
        
        """
        self.calcVolsCheck.setEnabled(True)
        if self.calculateVolumes:
            self.convHullVolRadio.setEnabled(True)
            self.voroVolRadio.setEnabled(True)
    
    def findClustersChanged(self, state):
        """
        Change find volumes setting.
        
        """
        if state == QtCore.Qt.Unchecked:
            self.findClusters = 0
            
            # disable associated settings
            self.nebRadSpinBox.setEnabled(False)
            self.minNumSpinBox.setEnabled(False)
            self.maxNumSpinBox.setEnabled(False)
            self.disableCalcVolsCheck()
            self.disableDrawHullsCheck()
        
        else:
            self.findClusters = 1
            
            # enable associated settings
            self.nebRadSpinBox.setEnabled(True)
            self.minNumSpinBox.setEnabled(True)
            self.maxNumSpinBox.setEnabled(True)
            self.enableCalcVolsCheck()
            self.enableDrawHullsCheck()
    
    def vacRadChanged(self, val):
        """
        Update vacancy radius
        
        """
        self.vacancyRadius = val
    
    def intVisChanged(self, val):
        """
        Change visibility of interstitials
        
        """
        if self.intTypeCheckBox.isChecked():
            self.showInterstitials = 1
        else:
            self.showInterstitials = 0
    
    def vacVisChanged(self, val):
        """
        Change visibility of vacancies
        
        """
        if self.vacTypeCheckBox.isChecked():
            self.showVacancies = 1
        else:
            self.showVacancies = 0
    
    def antVisChanged(self, val):
        """
        Change visibility of antisites
        
        """
        if self.antTypeCheckBox.isChecked():
            self.showAntisites = 1
        else:
            self.showAntisites = 0
    
    def filterSpeciesToggled(self, state):
        """
        Filter species toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filterSpecies = False
            self.specieList.setEnabled(False)
        
        else:
            self.filterSpecies = True
            self.specieList.setEnabled(True)
    
    def getVisibleSpecieList(self):
        """
        Return list of visible species
        
        """
        visibleSpecieList = []
        if self.filterSpecies:
            for i in xrange(self.specieList.count()):
                item = self.specieList.item(i)
                if item.checkState() == QtCore.Qt.Checked:
                    visibleSpecieList.append(item.symbol)
        
        else:
            for i in xrange(self.specieList.count()):
                item = self.specieList.item(i)
                visibleSpecieList.append(item.symbol)
        
        return visibleSpecieList
    
    def refresh(self):
        """
        Refresh the specie list
        
        """
        self.logger.debug("Refreshing point defect filter options")
        
        refState = self.pipelinePage.refState
        inputState = self.pipelinePage.inputState
        refSpecieList = refState.specieList
        inputSpecieList = inputState.specieList
        
        # set of added species
        currentSpecies = set()
        
        # remove species that don't exist
        num = self.specieList.count()
        for i in xrange(num - 1, -1, -1):
            item = self.specieList.item(i)
            
            # remove if doesn't exist both ref and input
            if item.symbol not in inputSpecieList and item.symbol not in refSpecieList:
                self.logger.debug("  Removing specie option: %s", item.symbol)
                self.specieList.takeItem(i) # does this delete it?
            
            else:
                currentSpecies.add(item.symbol)
        
        # unique species from ref/input
        combinedSpecieList = list(inputSpecieList) + list(refSpecieList)
        uniqueCurrentSpecies = set(combinedSpecieList)
        
        # add species that aren't already added
        for sym in uniqueCurrentSpecies:
            if sym in currentSpecies:
                self.logger.debug("  Keeping specie option: %s", sym)
            
            else:
                self.logger.debug("  Adding specie option: %s", sym)
                item = SpecieListItem(sym)
                self.specieList.addItem(item)
        
        # enable/disable draw vectors
        if inputState.NAtoms == refState.NAtoms:
            self.enableDrawVectorsCheck()
        else:
            self.disableDrawVectorsCheck()
    
    def disableDrawHullsCheck(self):
        """
        Disable the check box and associated widgets
        
        """
        self.drawHullsCheck.setEnabled(False)
        if self.drawConvexHulls:
            self.hullColourButton.setEnabled(False)
            self.hullOpacitySpinBox.setEnabled(False)
            self.hideAtomsCheckBox.setEnabled(False)
    
    def enableDrawHullsCheck(self):
        """
        Enable the check box and associated widgets
        
        """
        self.drawHullsCheck.setEnabled(True)
        if self.drawConvexHulls:
            self.hullColourButton.setEnabled(True)
            self.hullOpacitySpinBox.setEnabled(True)
            self.hideAtomsCheckBox.setEnabled(True)
    
    def drawHullsChanged(self, state):
        """
        Change draw hulls setting.
        
        """
        if state == QtCore.Qt.Unchecked:
            self.drawConvexHulls = 0
            
            self.hullColourButton.setEnabled(False)
            self.hullOpacitySpinBox.setEnabled(False)
            self.hideAtomsCheckBox.setEnabled(False)
        
        else:
            self.drawConvexHulls = 1
            
            self.hullColourButton.setEnabled(True)
            self.hullOpacitySpinBox.setEnabled(True)
            self.hideAtomsCheckBox.setEnabled(True)
    
    def hullOpacityChanged(self, val):
        """
        Change hull opacity setting.
        
        """
        self.hullOpacity = val
    
    def showColourDialog(self):
        """
        Show hull colour dialog.
        
        """
        col = QtGui.QColorDialog.getColor()
        
        if col.isValid():
            self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
            
            self.hullCol[0] = float(col.red()) / 255.0
            self.hullCol[1] = float(col.green()) / 255.0
            self.hullCol[2] = float(col.blue()) / 255.0
    
    def calcVolsMethodChanged(self, val=None):
        """
        Calc vols method changed
        
        """
        if self.convHullVolRadio.isChecked():
            self.calculateVolumesHull = True
        else:
            self.calculateVolumesHull = False
        
        if self.voroVolRadio.isChecked():
            self.calculateVolumesVoro = True
        else:
            self.calculateVolumesVoro = False
    
    def calcVolsChanged(self, state):
        """
        Changed calc vols.
        
        """
        if state == QtCore.Qt.Unchecked:
            self.calculateVolumes = False
            
            # disable buttons
            self.convHullVolRadio.setEnabled(False)
            self.voroVolRadio.setEnabled(False)
        
        else:
            self.calculateVolumes = True
            
            # enable buttons
            self.convHullVolRadio.setEnabled(True)
            self.voroVolRadio.setEnabled(True)
        
        self.calcVolsMethodChanged()

################################################################################
class ClusterSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(ClusterSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Clusters"
        
        self.minClusterSize = 8
        self.maxClusterSize = -1
        self.drawConvexHulls = 0
        self.neighbourRadius = 5.0
        self.calculateVolumes = False
        self.calculateVolumesVoro = True
        self.calculateVolumesHull = False
        self.hullCol = [0]*3
        self.hullCol[2] = 1
        self.hullOpacity = 0.5
        self.hideAtoms = 0
        
        # neighbour rad spin box
        self.nebRadSpinBox = QtGui.QDoubleSpinBox()
        self.nebRadSpinBox.setSingleStep(0.01)
        self.nebRadSpinBox.setMinimum(0.01)
        self.nebRadSpinBox.setMaximum(100.0)
        self.nebRadSpinBox.setValue(self.neighbourRadius)
        self.nebRadSpinBox.valueChanged.connect(self.nebRadChanged)
        self.nebRadSpinBox.setToolTip("Clusters are constructed using a recursive algorithm where "
                                      "two atoms are said to be neighbours if their separation "
                                      "is less than this value.")
        self.contentLayout.addRow("Neighbour radius", self.nebRadSpinBox)
        
        # minimum size spin box
        self.minNumSpinBox = QtGui.QSpinBox()
        self.minNumSpinBox.setMinimum(1)
        self.minNumSpinBox.setMaximum(1000)
        self.minNumSpinBox.setValue(self.minClusterSize)
        self.minNumSpinBox.valueChanged.connect(self.minNumChanged)
        self.minNumSpinBox.setToolTip("Only show clusters that contain more than this number of atoms.")
        self.contentLayout.addRow("Minimum cluster size", self.minNumSpinBox)
        
        # maximum size spin box
        self.maxNumSpinBox = QtGui.QSpinBox()
        self.maxNumSpinBox.setMinimum(-1)
        self.maxNumSpinBox.setMaximum(999999)
        self.maxNumSpinBox.setValue(self.maxClusterSize)
        self.maxNumSpinBox.valueChanged.connect(self.maxNumChanged)
        self.maxNumSpinBox.setToolTip("Only show clusters that contain less than this number of atoms. Set to "
                                      "'-1' to disable this condition.")
        self.contentLayout.addRow("Maximum cluster size", self.maxNumSpinBox)
        
        self.addHorizontalDivider()
        
        # calculate volumes options
        self.calcVolsCheck = QtGui.QCheckBox()
        self.calcVolsCheck.setToolTip("Calculate volumes of clusters of atoms.")
        self.calcVolsCheck.stateChanged.connect(self.calcVolsChanged)
        self.calcVolsCheck.setChecked(self.calculateVolumes)
        self.contentLayout.addRow("<b>Calculate volumes</b>", self.calcVolsCheck)
        
        # radio buttons
        self.convHullVolRadio = QtGui.QRadioButton(parent=self.calcVolsCheck)
        self.convHullVolRadio.toggled.connect(self.calcVolsMethodChanged)
        self.convHullVolRadio.setToolTip("Volume is determined from the convex hull of the atom positions.")
        self.voroVolRadio = QtGui.QRadioButton(parent=self.calcVolsCheck)
        self.voroVolRadio.setToolTip("Volume is determined by summing the Voronoi volumes of the atoms in "
                                     "the cluster.")
        self.voroVolRadio.setChecked(True)
        self.contentLayout.addRow("Convex hull volumes", self.convHullVolRadio)
        self.contentLayout.addRow("Sum Voronoi volumes", self.voroVolRadio)
        
        # make sure setup properly
        self.calcVolsChanged(QtCore.Qt.Unchecked)
        
        # draw hulls options
        self.drawHullsCheck = QtGui.QCheckBox()
        self.drawHullsCheck.setChecked(False)
        self.drawHullsCheck.setToolTip("Draw convex hulls of atom clusters")
        self.drawHullsCheck.stateChanged.connect(self.drawHullsChanged)
        self.displaySettingsLayout.addRow("<b>Draw convex hulls</b>", self.drawHullsCheck)
        
        # hull colour
        col = QtGui.QColor(self.hullCol[0]*255.0, self.hullCol[1]*255.0, self.hullCol[2]*255.0)
        self.hullColourButton = QtGui.QPushButton("")
        self.hullColourButton.setFixedWidth(50)
        self.hullColourButton.setFixedHeight(30)
        self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
        self.hullColourButton.clicked.connect(self.showColourDialog)
        self.hullColourButton.setToolTip("The colour of the hull.")
        self.displaySettingsLayout.addRow("Hull colour", self.hullColourButton)
        
        # hull opacity
        self.hullOpacitySpinBox = QtGui.QDoubleSpinBox()
        self.hullOpacitySpinBox.setSingleStep(0.01)
        self.hullOpacitySpinBox.setMinimum(0.01)
        self.hullOpacitySpinBox.setMaximum(1.0)
        self.hullOpacitySpinBox.setValue(self.hullOpacity)
        self.hullOpacitySpinBox.setToolTip("The opacity of the convex hulls")
        self.hullOpacitySpinBox.valueChanged[float].connect(self.hullOpacityChanged)
        self.displaySettingsLayout.addRow("Hull opacity", self.hullOpacitySpinBox)
        
        # hide atoms
        self.hideAtomsCheckBox = QtGui.QCheckBox()
        self.hideAtomsCheckBox.stateChanged.connect(self.hideAtomsChanged)
        self.hideAtomsCheckBox.setToolTip("Don't show the atoms when rendering the convex hulls")
        self.displaySettingsLayout.addRow("Hide atoms", self.hideAtomsCheckBox)
        
        self.drawHullsChanged(QtCore.Qt.Unchecked)
    
    def hideAtomsChanged(self, val):
        """
        Hide atoms check changed.
        
        """
        if self.hideAtomsCheckBox.isChecked():
            self.hideAtoms = 1
        
        else:
            self.hideAtoms = 0
    
    def hullOpacityChanged(self, val):
        """
        Change hull opacity setting.
        
        """
        self.hullOpacity = val
    
    def showColourDialog(self):
        """
        Show hull colour dialog.
        
        """
        col = QtGui.QColorDialog.getColor()
        
        if col.isValid():
            self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
            
            self.hullCol[0] = float(col.red()) / 255.0
            self.hullCol[1] = float(col.green()) / 255.0
            self.hullCol[2] = float(col.blue()) / 255.0
    
    def calcVolsMethodChanged(self, val=None):
        """
        Calc vols method changed
        
        """
        if self.convHullVolRadio.isChecked():
            self.calculateVolumesHull = True
        else:
            self.calculateVolumesHull = False
        
        if self.voroVolRadio.isChecked():
            self.calculateVolumesVoro = True
        else:
            self.calculateVolumesVoro = False
    
    def calcVolsChanged(self, state):
        """
        Changed calc vols.
        
        """
        if state == QtCore.Qt.Unchecked:
            self.calculateVolumes = False
            
            # disable buttons
            self.convHullVolRadio.setEnabled(False)
            self.voroVolRadio.setEnabled(False)
        
        else:
            self.calculateVolumes = True
            
            # enable buttons
            self.convHullVolRadio.setEnabled(True)
            self.voroVolRadio.setEnabled(True)
        
        self.calcVolsMethodChanged()
    
    def minNumChanged(self, val):
        """
        Change min cluster size.
        
        """
        self.minClusterSize = val
    
    def maxNumChanged(self, val):
        """
        Change max cluster size.
        
        """
        self.maxClusterSize = val
    
    def nebRadChanged(self, val):
        """
        Change neighbour radius.
        
        """
        self.neighbourRadius = val
    
    def drawHullsChanged(self, state):
        """
        Change draw hulls setting.
        
        """
        if state == QtCore.Qt.Unchecked:
            self.drawConvexHulls = 0
            
            self.hullColourButton.setEnabled(False)
            self.hullOpacitySpinBox.setEnabled(False)
            self.hideAtomsCheckBox.setEnabled(False)
        
        else:
            self.drawConvexHulls = 1
            
            self.hullColourButton.setEnabled(True)
            self.hullOpacitySpinBox.setEnabled(True)
            self.hideAtomsCheckBox.setEnabled(True)

################################################################################

class SlipSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(SlipSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Slip"
        self.addProvidedScalar("Slip")
        
        self.minSlip = 0.0
        self.maxSlip = 9999.0
        self.filteringEnabled = False
        
        # filtering options
        filterCheck = QtGui.QCheckBox()
        filterCheck.setChecked(self.filteringEnabled)
        filterCheck.setToolTip("Filter atoms by slip")
        filterCheck.stateChanged.connect(self.filteringToggled)
        self.contentLayout.addRow("<b>Enable filtering</b>", filterCheck)
        
        self.minSlipSpin = QtGui.QDoubleSpinBox()
        self.minSlipSpin.setSingleStep(0.1)
        self.minSlipSpin.setMinimum(0.0)
        self.minSlipSpin.setMaximum(9999.0)
        self.minSlipSpin.setValue(self.minSlip)
        self.minSlipSpin.valueChanged.connect(self.setMinSlip)
        self.minSlipSpin.setEnabled(False)
        self.contentLayout.addRow("Min", self.minSlipSpin)
        
        self.maxSlipSpin = QtGui.QDoubleSpinBox()
        self.maxSlipSpin.setSingleStep(0.1)
        self.maxSlipSpin.setMinimum(0.0)
        self.maxSlipSpin.setMaximum(9999.0)
        self.maxSlipSpin.setValue(self.maxSlip)
        self.maxSlipSpin.valueChanged.connect(self.setMaxSlip)
        self.maxSlipSpin.setEnabled(False)
        self.contentLayout.addRow("Max", self.maxSlipSpin)
    
    def filteringToggled(self, state):
        """
        Filtering toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filteringEnabled = False
            
            self.minSlipSpin.setEnabled(False)
            self.maxSlipSpin.setEnabled(False)
        
        else:
            self.filteringEnabled = True
            
            self.minSlipSpin.setEnabled(True)
            self.maxSlipSpin.setEnabled(True)
    
    def setMinSlip(self, val):
        """
        Set the minimum slip.
        
        """
        self.minSlip = val

    def setMaxSlip(self, val):
        """
        Set the maximum slip.
        
        """
        self.maxSlip = val

################################################################################

class DisplacementSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(DisplacementSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Displacement"
        self.addProvidedScalar("Displacement")
        
        self.minDisplacement = 1.2
        self.maxDisplacement = 1000.0
        self.bondThicknessVTK = 0.4
        self.bondThicknessPOV = 0.4
        self.bondNumSides = 5
        self.drawDisplacementVectors = False
        
        # draw displacement vector settings
        self.drawVectorsCheck = QtGui.QCheckBox()
        self.drawVectorsCheck.stateChanged.connect(self.drawVectorsChanged)
        self.drawVectorsCheck.setCheckState(QtCore.Qt.Unchecked)
        self.drawVectorsCheck.setToolTip("Draw displacement vectors (movement) of defects")
        
        self.displaySettingsLayout.addRow("<b>Draw displacement vectors</b>", self.drawVectorsCheck)
        
        # vtk thickness
        self.vtkThickSpin = QtGui.QDoubleSpinBox()
        self.vtkThickSpin.setMinimum(0.01)
        self.vtkThickSpin.setMaximum(10)
        self.vtkThickSpin.setSingleStep(0.1)
        self.vtkThickSpin.setValue(self.bondThicknessVTK)
        self.vtkThickSpin.valueChanged.connect(self.vtkThickChanged)
        self.vtkThickSpin.setToolTip("Thickness of lines showing defect movement (VTK)")
        self.displaySettingsLayout.addRow("Bond thickness (VTK)", self.vtkThickSpin)
        
        # pov thickness
        self.povThickSpin = QtGui.QDoubleSpinBox()
        self.povThickSpin.setMinimum(0.01)
        self.povThickSpin.setMaximum(10)
        self.povThickSpin.setSingleStep(0.01)
        self.povThickSpin.setValue(self.bondThicknessPOV)
        self.povThickSpin.valueChanged.connect(self.povThickChanged)
        self.povThickSpin.setToolTip("Thickness of lines showing defect movement (POV-Ray)")
        self.displaySettingsLayout.addRow("Bond thickness (POV)", self.povThickSpin)
        
        # num sides
        self.numSidesSpin = QtGui.QSpinBox()
        self.numSidesSpin.setMinimum(3)
        self.numSidesSpin.setMaximum(999)
        self.numSidesSpin.setSingleStep(1)
        self.numSidesSpin.setValue(self.bondNumSides)
        self.numSidesSpin.valueChanged.connect(self.numSidesChanged)
        self.numSidesSpin.setToolTip("Number of sides when rendering displacement vectors (more looks better but is slower)")
        self.displaySettingsLayout.addRow("Bond number of sides", self.numSidesSpin)
        
        self.drawVectorsChanged(QtCore.Qt.Unchecked)
        
        # filtering options
        self.filteringEnabled = False
        filterCheck = QtGui.QCheckBox()
        filterCheck.setChecked(self.filteringEnabled)
        filterCheck.setToolTip("Filter atoms by displacement")
        filterCheck.stateChanged.connect(self.filteringToggled)
        self.contentLayout.addRow("<b>Enable filtering</b>", filterCheck)
        
        self.minDisplacementSpinBox = QtGui.QDoubleSpinBox()
        self.minDisplacementSpinBox.setSingleStep(0.1)
        self.minDisplacementSpinBox.setMinimum(0.0)
        self.minDisplacementSpinBox.setMaximum(9999.0)
        self.minDisplacementSpinBox.setValue(self.minDisplacement)
        self.minDisplacementSpinBox.valueChanged.connect(self.setMinDisplacement)
        self.minDisplacementSpinBox.setEnabled(False)
        self.contentLayout.addRow("Min", self.minDisplacementSpinBox)
        
        self.maxDisplacementSpinBox = QtGui.QDoubleSpinBox()
        self.maxDisplacementSpinBox.setSingleStep(0.1)
        self.maxDisplacementSpinBox.setMinimum(0.0)
        self.maxDisplacementSpinBox.setMaximum(9999.0)
        self.maxDisplacementSpinBox.setValue(self.maxDisplacement)
        self.maxDisplacementSpinBox.valueChanged.connect(self.setMaxDisplacement)
        self.maxDisplacementSpinBox.setEnabled(False)
        self.contentLayout.addRow("Max", self.maxDisplacementSpinBox)
    
    def numSidesChanged(self, val):
        """
        Number of sides changed.
        
        """
        self.bondNumSides = val
    
    def vtkThickChanged(self, val):
        """
        VTK thickness changed.
        
        """
        self.bondThicknessVTK = val
    
    def povThickChanged(self, val):
        """
        POV thickness changed.
        
        """
        self.bondThicknessPOV = val
    
    def drawVectorsChanged(self, state):
        """
        Draw displacement vectors toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.drawDisplacementVectors = False
            
            self.vtkThickSpin.setEnabled(False)
            self.povThickSpin.setEnabled(False)
            self.numSidesSpin.setEnabled(False)
        
        else:
            self.drawDisplacementVectors = True
            
            self.vtkThickSpin.setEnabled(True)
            self.povThickSpin.setEnabled(True)
            self.numSidesSpin.setEnabled(True)
        
        self.logger.debug("Draw displacement vectors: %r", self.drawDisplacementVectors)
    
    def filteringToggled(self, state):
        """
        Filtering toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filteringEnabled = False
            
            self.minDisplacementSpinBox.setEnabled(False)
            self.maxDisplacementSpinBox.setEnabled(False)
        
        else:
            self.filteringEnabled = True
            
            self.minDisplacementSpinBox.setEnabled(True)
            self.maxDisplacementSpinBox.setEnabled(True)
    
    def setMinDisplacement(self, val):
        """
        Set the minimum displacement.
        
        """
        self.minDisplacement = val

    def setMaxDisplacement(self, val):
        """
        Set the maximum displacement.
        
        """
        self.maxDisplacement = val


################################################################################
class ChargeSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(ChargeSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Charge"
        
        self.minCharge = -100.0
        self.maxCharge = 100.0
        
        self.minChargeSpinBox = QtGui.QDoubleSpinBox()
        self.minChargeSpinBox.setSingleStep(0.1)
        self.minChargeSpinBox.setMinimum(-999.0)
        self.minChargeSpinBox.setMaximum(999.0)
        self.minChargeSpinBox.setValue(self.minCharge)
        self.minChargeSpinBox.valueChanged.connect(self.setMinCharge)
        self.contentLayout.addRow("Min charge", self.minChargeSpinBox)
        
        self.maxChargeSpinBox = QtGui.QDoubleSpinBox()
        self.maxChargeSpinBox.setSingleStep(0.1)
        self.maxChargeSpinBox.setMinimum(-999.0)
        self.maxChargeSpinBox.setMaximum(999.0)
        self.maxChargeSpinBox.setValue(self.maxCharge)
        self.maxChargeSpinBox.valueChanged.connect(self.setMaxCharge)
        self.contentLayout.addRow("Max charge", self.maxChargeSpinBox)
    
    def setMinCharge(self, val):
        """
        Set the minimum charge.
        
        """
        self.minCharge = val

    def setMaxCharge(self, val):
        """
        Set the maximum charge.
        
        """
        self.maxCharge = val

################################################################################

class SliceSettingsDialog(GenericSettingsDialog):
    """
    Slice filter settings.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(SliceSettingsDialog, self).__init__(title, parent)
        
        # slice plane
        self.slicePlane = slicePlane.SlicePlane(self.pipelinePage)
        
        # defaults
        lattice = self.pipelinePage.inputState
        self.x0 = lattice.cellDims[0] / 2.0
        self.y0 = lattice.cellDims[1] / 2.0
        self.z0 = lattice.cellDims[2] / 2.0
        self.xn = 1.0
        self.yn = 0.0
        self.zn = 0.0
        self.invert = 0
        self.showSlicePlaneChecked = False
        
        # plane centre group box
        planeCentreGroup = QtGui.QGroupBox("Plane centre")
        planeCentreGroup.setAlignment(QtCore.Qt.AlignHCenter)
        
        planeCentreLayout = QtGui.QVBoxLayout(planeCentreGroup)
        planeCentreLayout.setAlignment(QtCore.Qt.AlignTop)
        planeCentreLayout.setContentsMargins(0, 0, 0, 0)
        planeCentreLayout.setSpacing(0)
        
        # spin boxes
        x0SpinBox = QtGui.QDoubleSpinBox()
        x0SpinBox.setSingleStep(1)
        x0SpinBox.setMinimum(-1000)
        x0SpinBox.setMaximum(1000)
        x0SpinBox.setValue(self.x0)
        x0SpinBox.valueChanged.connect(self.x0Changed)
        
        y0SpinBox = QtGui.QDoubleSpinBox()
        y0SpinBox.setSingleStep(1)
        y0SpinBox.setMinimum(-1000)
        y0SpinBox.setMaximum(1000)
        y0SpinBox.setValue(self.y0)
        y0SpinBox.valueChanged.connect(self.y0Changed)
        
        z0SpinBox = QtGui.QDoubleSpinBox()
        z0SpinBox.setSingleStep(1)
        z0SpinBox.setMinimum(-1000)
        z0SpinBox.setMaximum(1000)
        z0SpinBox.setValue(self.z0)
        z0SpinBox.valueChanged.connect(self.z0Changed)
        
        # row
        row = QtGui.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(x0SpinBox)
        row.addWidget(y0SpinBox)
        row.addWidget(z0SpinBox)
        
        planeCentreLayout.addLayout(row)
        
        row = self.newRow()
        row.addWidget(planeCentreGroup)
        
        # plane normal group box
        planeNormalGroup = QtGui.QGroupBox("Plane normal")
        planeNormalGroup.setAlignment(QtCore.Qt.AlignHCenter)
        
        planeNormalLayout = QtGui.QVBoxLayout(planeNormalGroup)
        planeNormalLayout.setAlignment(QtCore.Qt.AlignTop)
        planeNormalLayout.setContentsMargins(0, 0, 0, 0)
        planeNormalLayout.setSpacing(0)
        
        # spin boxes
        xnSpinBox = QtGui.QDoubleSpinBox()
        xnSpinBox.setSingleStep(0.1)
        xnSpinBox.setMinimum(-10)
        xnSpinBox.setMaximum(10)
        xnSpinBox.setValue(self.xn)
        xnSpinBox.valueChanged.connect(self.xnChanged)
        
        ynSpinBox = QtGui.QDoubleSpinBox()
        ynSpinBox.setSingleStep(0.1)
        ynSpinBox.setMinimum(-10)
        ynSpinBox.setMaximum(10)
        ynSpinBox.setValue(self.yn)
        ynSpinBox.valueChanged.connect(self.ynChanged)
        
        znSpinBox = QtGui.QDoubleSpinBox()
        znSpinBox.setSingleStep(0.1)
        znSpinBox.setMinimum(-10)
        znSpinBox.setMaximum(10)
        znSpinBox.setValue(self.zn)
        znSpinBox.valueChanged.connect(self.znChanged)
        
        # row
        row = QtGui.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(xnSpinBox)
        row.addWidget(ynSpinBox)
        row.addWidget(znSpinBox)
        
        planeNormalLayout.addLayout(row)
        
        row = self.newRow()
        row.addWidget(planeNormalGroup)
        
        # invert
        self.invertCheck = QtGui.QCheckBox("Invert selection")
        self.invertCheck.stateChanged.connect(self.changeInvert)
        
        row = self.newRow()
        row.addWidget(self.invertCheck)
        
        # show slice plane
        self.showSlicePlaneCheck = QtGui.QCheckBox("Show slice plane")
        self.showSlicePlaneCheck.stateChanged.connect(self.showPlaneChanged)
        
        row = self.newRow()
        row.addWidget(self.showSlicePlaneCheck)
    
    def refresh(self):
        """
        Called whenever new input is loaded.
        
        """
        # need to change min/max of sliders for x0,y0,z0
        pass
    
    def showPlaneChanged(self, state):
        """
        Show slice plane.
        
        """
        if self.showSlicePlaneCheck.isChecked():
            self.showSlicePlaneChecked = True
            self.showSlicePlane()
        else:
            self.showSlicePlaneChecked = False
            self.hideSlicePlane()
    
    def changeInvert(self, state):
        """
        Change invert.
        
        """
        if self.invertCheck.isChecked():
            self.invert = 1
        else:
            self.invert = 0
    
    def x0Changed(self, val):
        """
        x0 changed.
        
        """
        self.x0 = val
        self.showSlicePlane()
    
    def y0Changed(self, val):
        """
        y0 changed.
        
        """
        self.y0 = val
        self.showSlicePlane()
    
    def z0Changed(self, val):
        """
        z0 changed.
        
        """
        self.z0 = val
        self.showSlicePlane()
    
    def xnChanged(self, val):
        """
        xn changed.
        
        """
        self.xn = val
        self.showSlicePlane()
    
    def ynChanged(self, val):
        """
        yn changed.
        
        """
        self.yn = val
        self.showSlicePlane()
    
    def znChanged(self, val):
        """
        zn changed.
        
        """
        self.zn = val
        self.showSlicePlane()
    
    def showSlicePlane(self):
        """
        Update position of slice plane.
        
        """
        if not self.showSlicePlaneChecked:
            return
        
        # first remove it is already shown
        
        
        # args to pass
        p = (self.x0, self.y0, self.z0)
        n = (self.xn, self.yn, self.zn)
        
        # update actor
        self.slicePlane.update(p, n)
        
        # broadcast to renderers
        self.parent.filterTab.broadcastToRenderers("showSlicePlane", args=(self.slicePlane,))
    
    def hideSlicePlane(self):
        """
        Hide the slice plane.
        
        """
        # broadcast to renderers
        self.parent.filterTab.broadcastToRenderers("removeSlicePlane", globalBcast=True)
    
    def closeEvent(self, event):
        """
        Override closeEvent.
        
        """
        if self.showSlicePlaneChecked:
            self.showSlicePlaneCheck.setCheckState(QtCore.Qt.Unchecked)
        
        self.hide()

################################################################################
class AtomIdSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(AtomIdSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Atom ID"
        
        # only allow numbers, commas and hyphens
        rx = QtCore.QRegExp("[0-9]+(?:[-,]?[0-9]+)*")
        validator = QtGui.QRegExpValidator(rx, self)
        
        self.lineEdit = QtGui.QLineEdit()
        self.lineEdit.setValidator(validator)
        self.lineEdit.setToolTip("Comma separated list of atom IDs or ranges of atom IDs (hyphenated) that are visible (eg. '22,30-33' will show atom IDs 22, 30, 31, 32 and 33)")
        self.contentLayout.addRow("Visible IDs", self.lineEdit)

################################################################################
class CoordinationNumberSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(CoordinationNumberSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Coordination number"
        self.addProvidedScalar("Coordination number")
        
        self.minCoordNum = 0
        self.maxCoordNum = 100
        
        groupLayout = self.addFilteringGroupBox(slot=self.filteringToggled, checked=False)
        
        label = QtGui.QLabel("Min:")
        self.minCoordNumSpinBox = QtGui.QSpinBox()
        self.minCoordNumSpinBox.setSingleStep(1)
        self.minCoordNumSpinBox.setMinimum(0)
        self.minCoordNumSpinBox.setMaximum(999)
        self.minCoordNumSpinBox.setValue(self.minCoordNum)
        self.minCoordNumSpinBox.valueChanged.connect(self.setMinCoordNum)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.minCoordNumSpinBox)
        groupLayout.addLayout(row)
        
        label = QtGui.QLabel("Max:")
        self.maxCoordNumSpinBox = QtGui.QSpinBox()
        self.maxCoordNumSpinBox.setSingleStep(1)
        self.maxCoordNumSpinBox.setMinimum(0)
        self.maxCoordNumSpinBox.setMaximum(999)
        self.maxCoordNumSpinBox.setValue(self.maxCoordNum)
        self.maxCoordNumSpinBox.valueChanged.connect(self.setMaxCoordNum)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.maxCoordNumSpinBox)
        groupLayout.addLayout(row)
    
    def filteringToggled(self, arg):
        """
        Filtering toggled
        
        """
        self.filteringEnabled = arg
    
    def setMinCoordNum(self, val):
        """
        Set the minimum coordination number.
        
        """
        self.minCoordNum = val

    def setMaxCoordNum(self, val):
        """
        Set the maximum coordination number.
        
        """
        self.maxCoordNum = val

################################################################################
class VoronoiNeighboursSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        super(VoronoiNeighboursSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Voronoi neighbours"
        self.addProvidedScalar("Voronoi neighbours")
        
        self.minVoroNebs = 0
        self.maxVoroNebs = 999
        
        groupLayout = self.addFilteringGroupBox(slot=self.filteringToggled, checked=False)
        
        label = QtGui.QLabel("Min:")
        self.minVoroNebsSpin = QtGui.QSpinBox()
        self.minVoroNebsSpin.setSingleStep(1)
        self.minVoroNebsSpin.setMinimum(0)
        self.minVoroNebsSpin.setMaximum(999)
        self.minVoroNebsSpin.setValue(self.minVoroNebs)
        self.minVoroNebsSpin.valueChanged[int].connect(self.setMinVoroNebs)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.minVoroNebsSpin)
        groupLayout.addLayout(row)
        
        label = QtGui.QLabel("Max:")
        self.maxVoroNebsSpin = QtGui.QSpinBox()
        self.maxVoroNebsSpin.setSingleStep(1)
        self.maxVoroNebsSpin.setMinimum(0)
        self.maxVoroNebsSpin.setMaximum(999)
        self.maxVoroNebsSpin.setValue(self.maxVoroNebs)
        self.maxVoroNebsSpin.valueChanged[int].connect(self.setMaxVoroNebs)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.maxVoroNebsSpin)
        groupLayout.addLayout(row)
    
    def filteringToggled(self, arg):
        """
        Filtering toggled
        
        """
        self.filteringEnabled = arg
    
    def setMinVoroNebs(self, val):
        """
        Set the minimum Voronoi neighbours.
        
        """
        self.minVoroNebs = val

    def setMaxVoroNebs(self, val):
        """
        Set the maximum Voronoi neighbours.
        
        """
        self.maxVoroNebs = val

################################################################################
class VoronoiVolumeSettingsDialog(GenericSettingsDialog):
    """
    Settings for Voronoi volume filter
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(VoronoiVolumeSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Voronoi volume"
        self.addProvidedScalar("Voronoi volume")
        
        self.minVoroVol = 0.0
        self.maxVoroVol = 9999.99
        
        groupLayout = self.addFilteringGroupBox(slot=self.filteringToggled, checked=False)
        
        label = QtGui.QLabel("Min:")
        self.minVoroVolSpin = QtGui.QDoubleSpinBox()
        self.minVoroVolSpin.setSingleStep(0.01)
        self.minVoroVolSpin.setMinimum(0.0)
        self.minVoroVolSpin.setMaximum(9999.99)
        self.minVoroVolSpin.setValue(self.minVoroVol)
        self.minVoroVolSpin.valueChanged[float].connect(self.setMinVoroVol)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.minVoroVolSpin)
        groupLayout.addLayout(row)
        
        label = QtGui.QLabel("Max:")
        self.maxVoroVolSpin = QtGui.QDoubleSpinBox()
        self.maxVoroVolSpin.setSingleStep(0.01)
        self.maxVoroVolSpin.setMinimum(0.0)
        self.maxVoroVolSpin.setMaximum(9999.99)
        self.maxVoroVolSpin.setValue(self.maxVoroVol)
        self.maxVoroVolSpin.valueChanged[float].connect(self.setMaxVoroVol)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.maxVoroVolSpin)
        groupLayout.addLayout(row)
    
    def filteringToggled(self, arg):
        """
        Filtering toggled
        
        """
        self.filteringEnabled = arg
    
    def setMinVoroVol(self, val):
        """
        Set the minimum Voronoi volume.
        
        """
        self.minVoroVol = val

    def setMaxVoroVol(self, val):
        """
        Set the maximum Voronoi volume.
        
        """
        self.maxVoroVol = val

################################################################################
class BondOrderSettingsDialog(GenericSettingsDialog):
    """
    Settings for bond order filter
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(BondOrderSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Bond order"
        self.addProvidedScalar("Q4")
        self.addProvidedScalar("Q6")
        
        self.minQ4 = 0.0
        self.maxQ4 = 99.99
        self.minQ6 = 0.0
        self.maxQ6 = 99.99
        self.maxBondDistance = 4.0
        self.filterQ4Enabled = False
        self.filterQ6Enabled = False
        
        label = QtGui.QLabel("Max bond distance:")
        self.maxBondDistanceSpin = QtGui.QDoubleSpinBox()
        self.maxBondDistanceSpin.setSingleStep(0.01)
        self.maxBondDistanceSpin.setMinimum(2.0)
        self.maxBondDistanceSpin.setMaximum(9.99)
        self.maxBondDistanceSpin.setValue(self.maxBondDistance)
        self.maxBondDistanceSpin.valueChanged[float].connect(self.setMaxBondDistance)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.maxBondDistanceSpin)
        
        self.newRow()
        groupLayout = self.addFilteringGroupBox(title="Filter Q4", slot=self.filterQ4Toggled, checked=False)
        
        label = QtGui.QLabel("Min:")
        self.minQ4Spin = QtGui.QDoubleSpinBox()
        self.minQ4Spin.setSingleStep(0.01)
        self.minQ4Spin.setMinimum(0.0)
        self.minQ4Spin.setMaximum(9999.99)
        self.minQ4Spin.setValue(self.minQ4)
        self.minQ4Spin.valueChanged[float].connect(self.setMinQ4)
         
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.minQ4Spin)
        groupLayout.addLayout(row)
         
        label = QtGui.QLabel("Max:")
        self.maxQ4Spin = QtGui.QDoubleSpinBox()
        self.maxQ4Spin.setSingleStep(0.01)
        self.maxQ4Spin.setMinimum(0.0)
        self.maxQ4Spin.setMaximum(9999.99)
        self.maxQ4Spin.setValue(self.maxQ4)
        self.maxQ4Spin.valueChanged[float].connect(self.setMaxQ4)
         
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.maxQ4Spin)
        groupLayout.addLayout(row)
        
        self.newRow()
        groupLayout = self.addFilteringGroupBox(title="Filter Q6", slot=self.filterQ6Toggled, checked=False)
        
        label = QtGui.QLabel("Min:")
        self.minQ6Spin = QtGui.QDoubleSpinBox()
        self.minQ6Spin.setSingleStep(0.01)
        self.minQ6Spin.setMinimum(0.0)
        self.minQ6Spin.setMaximum(9999.99)
        self.minQ6Spin.setValue(self.minQ6)
        self.minQ6Spin.valueChanged[float].connect(self.setMinQ6)
         
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.minQ6Spin)
        groupLayout.addLayout(row)
         
        label = QtGui.QLabel("Max:")
        self.maxQ6Spin = QtGui.QDoubleSpinBox()
        self.maxQ6Spin.setSingleStep(0.01)
        self.maxQ6Spin.setMinimum(0.0)
        self.maxQ6Spin.setMaximum(9999.99)
        self.maxQ6Spin.setValue(self.maxQ6)
        self.maxQ6Spin.valueChanged[float].connect(self.setMaxQ6)
         
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.maxQ6Spin)
        groupLayout.addLayout(row)
        
        self.addLinkToHelpPage("usage/analysis/filters/bond_order.html")
    
    def filterQ4Toggled(self, arg):
        """
        Filter Q4 toggled
        
        """
        self.filterQ4Enabled = arg
    
    def filterQ6Toggled(self, arg):
        """
        Filter Q4 toggled
        
        """
        self.filterQ6Enabled = arg
    
    def setMaxBondDistance(self, val):
        """
        Set the max bond distance
        
        """
        self.maxBondDistance = val
    
    def setMinQ4(self, val):
        """
        Set the minimum value for Q4
        
        """
        self.minQ4 = val

    def setMaxQ4(self, val):
        """
        Set the maximum value for Q4
        
        """
        self.maxQ4 = val
    
    def setMinQ6(self, val):
        """
        Set the minimum value for Q6
        
        """
        self.minQ6 = val

    def setMaxQ6(self, val):
        """
        Set the maximum value for Q6
        
        """
        self.maxQ6 = val

################################################################################
class AcnaSettingsDialog(GenericSettingsDialog):
    """
    Settings for adaptive common neighbour analysis
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(AcnaSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "ACNA"
        self.addProvidedScalar("ACNA")
        
        self.maxBondDistance = 5.0
        self.filteringEnabled = False
        
        self.maxBondDistanceSpin = QtGui.QDoubleSpinBox()
        self.maxBondDistanceSpin.setSingleStep(0.1)
        self.maxBondDistanceSpin.setMinimum(2.0)
        self.maxBondDistanceSpin.setMaximum(9.99)
        self.maxBondDistanceSpin.setValue(self.maxBondDistance)
        self.maxBondDistanceSpin.valueChanged[float].connect(self.setMaxBondDistance)
        self.maxBondDistanceSpin.setToolTip("This is used for spatially decomposing the system. "
                                            "This should be set large enough that the required neighbours will be included.")
        self.contentLayout.addRow("Max bond distance", self.maxBondDistanceSpin)
        
        filterer = self.parent.filterer
        
        self.addHorizontalDivider()
        
        # filter check
        filterByStructureCheck = QtGui.QCheckBox()
        filterByStructureCheck.setChecked(self.filteringEnabled)
        filterByStructureCheck.setToolTip("Filter atoms by structure type")
        filterByStructureCheck.stateChanged.connect(self.filteringToggled)
        self.contentLayout.addRow("<b>Filter by structure</b>", filterByStructureCheck)
        
        # filter options group
        self.structureChecks = {}
        for i, structure in enumerate(filterer.knownStructures):
            cb = QtGui.QCheckBox()
            cb.setChecked(True)
            cb.stateChanged.connect(functools.partial(self.visToggled, i))
            self.contentLayout.addRow(structure, cb)
            self.structureChecks[structure] = cb
            
            if not self.filteringEnabled:
                cb.setEnabled(False)
        
        self.structureVisibility = np.ones(len(filterer.knownStructures), dtype=np.int32)
        
        self.addLinkToHelpPage("usage/analysis/filters/acna.html")
    
    def visToggled(self, index, checkState):
        """
        Visibility toggled for one structure type
        
        """
        if checkState == QtCore.Qt.Unchecked:
            self.structureVisibility[index] = 0
        
        else:
            self.structureVisibility[index] = 1
    
    def filteringToggled(self, state):
        """
        Filtering toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filteringEnabled = False
            
            # disable structure checks
            for key in self.structureChecks:
                cb = self.structureChecks[key]
                cb.setEnabled(False)
        
        else:
            self.filteringEnabled = True
            
            # enable structure checks
            for key in self.structureChecks:
                cb = self.structureChecks[key]
                cb.setEnabled(True)
    
    def setMaxBondDistance(self, val):
        """
        Set the max bond distance
        
        """
        self.maxBondDistance = val

################################################################################

class GenericScalarFilterSettingsDialog(GenericSettingsDialog):
    """
    Settings for generic scalar value filterer
    
    """
    def __init__(self, mainWindow, filterType, title, parent=None):
        super(GenericScalarFilterSettingsDialog, self).__init__(title, parent)
        
        self.filterType = filterType
        
        self.minVal = -10000.0
        self.maxVal = 10000.0
        
        self.minValSpinBox = QtGui.QDoubleSpinBox()
        self.minValSpinBox.setSingleStep(0.1)
        self.minValSpinBox.setMinimum(-99999.0)
        self.minValSpinBox.setMaximum(99999.0)
        self.minValSpinBox.setValue(self.minVal)
        self.minValSpinBox.valueChanged.connect(self.setMinVal)
        self.contentLayout.addRow("Minimum", self.minValSpinBox)
        
        self.maxValSpinBox = QtGui.QDoubleSpinBox()
        self.maxValSpinBox.setSingleStep(0.1)
        self.maxValSpinBox.setMinimum(-99999.0)
        self.maxValSpinBox.setMaximum(99999.0)
        self.maxValSpinBox.setValue(self.maxVal)
        self.maxValSpinBox.valueChanged.connect(self.setMaxVal)
        self.contentLayout.addRow("Maximum", self.maxValSpinBox)
    
    def setMinVal(self, val):
        """
        Set the minimum value.
        
        """
        self.minVal = val

    def setMaxVal(self, val):
        """
        Set the maximum value.
        
        """
        self.maxVal = val
