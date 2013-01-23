
"""
Settings for filters

@author: Chris Scott

"""
import sys

from PyQt4 import QtGui, QtCore

import utilities
from utilities import iconPath
import genericForm

try:
    import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)




################################################################################
class GenericSettingsDialog(QtGui.QDialog):
    def __init__(self, title, parent):
        super(GenericSettingsDialog, self).__init__(parent)
        
        self.setModal(0)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(iconPath("preferences-system.svg")))
#        self.resize(500,300)
        
        dialogLayout = QtGui.QVBoxLayout()
        dialogLayout.setAlignment(QtCore.Qt.AlignTop)
#        dialogLayout.setContentsMargins(0, 0, 0, 0)
#        dialogLayout.setSpacing(0)
        
        tabWidget = QtGui.QTabWidget()
        
        # filter settings
        self.contentLayout = QtGui.QVBoxLayout()
        self.contentLayout.setAlignment(QtCore.Qt.AlignTop)
        self.contentLayout.setContentsMargins(0, 0, 0, 0)
        self.contentLayout.setSpacing(0)
        
        contentWidget = QtGui.QWidget() #QtGui.QGroupBox(title)
#        contentWidget.setAlignment(QtCore.Qt.AlignCenter)
        contentWidget.setLayout(self.contentLayout)
        
#        dialogLayout.addWidget(contentWidget)
        tabWidget.addTab(contentWidget, "Filter")
        
        # display settings
        self.displaySettingsLayout = QtGui.QVBoxLayout()
        self.displaySettingsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.displaySettingsLayout.setContentsMargins(0, 0, 0, 0)
        self.displaySettingsLayout.setSpacing(0)
        
        displaySettingsWidget = QtGui.QWidget()
        displaySettingsWidget.setLayout(self.displaySettingsLayout)
        
        tabWidget.addTab(displaySettingsWidget, "Display")
        
        dialogLayout.addWidget(tabWidget)
        self.setLayout(dialogLayout)
        
        # buttons
        closeButton = QtGui.QPushButton("Hide")
        closeButton.setAutoDefault(1)
        self.connect(closeButton, QtCore.SIGNAL('clicked()'), self.close)
        
        buttonWidget = QtGui.QWidget()
        buttonLayout = QtGui.QHBoxLayout(buttonWidget)
        buttonLayout.setAlignment(QtCore.Qt.AlignCenter)
#        buttonLayout.addStretch()
        buttonLayout.addWidget(closeButton)
        
        dialogLayout.addWidget(buttonWidget)
    
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
class SpecieSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
        self.filterType = "Specie"
        
        self.specieList = []
        self.specieBoxes = {}
        self.specieRows = {}
        self.visibleSpecieList = []
        self.allSpeciesSelected = True
        
        self.allSpeciesBox = QtGui.QCheckBox("All")
        self.allSpeciesBox.setChecked(1)
        self.connect(self.allSpeciesBox, QtCore.SIGNAL('stateChanged(int)'), self.allSpeciesBoxChanged)
        row = self.newRow()
        row.addWidget(self.allSpeciesBox)
        
        self.newRow()
        
        self.refresh()
    
    def allSpeciesBoxChanged(self, val):
        """
        
        
        """
        if self.allSpeciesBox.isChecked():
            self.allSpeciesSelected = True
            
            for specie in self.specieList:
                self.specieBoxes[specie].setChecked(1)
            
        else:
            self.allSpeciesSelected = False
        
#        self.changedSpecie(0)
        
    def refresh(self):
        """
        Refresh the specie list
        
        """
        inputSpecieList = self.mainWindow.inputState.specieList
        refSpecieList = self.mainWindow.refState.specieList
        
        for spec in refSpecieList:
            if spec not in self.specieList:
                self.specieList.append(spec)
                self.addSpecieCheck(spec)
                if self.allSpeciesSelected:
                    self.specieBoxes[spec].setChecked(1)
        
        for spec in inputSpecieList:
            if spec not in self.specieList:                
                self.specieList.append(spec)
                self.addSpecieCheck(spec)
                if self.allSpeciesSelected:
                    self.specieBoxes[spec].setChecked(1)
        
        self.changedSpecie(0)

    def addSpecieCheck(self, specie):
        """
        Add check box for the given specie
        
        """
        self.specieBoxes[specie] = QtGui.QCheckBox(str(specie))
        
        self.connect(self.specieBoxes[specie], QtCore.SIGNAL('stateChanged(int)'), self.changedSpecie)
        
        row = self.newRow()
        row.addWidget(self.specieBoxes[specie])
        
        self.specieRows[specie] = row
    
    def changedSpecie(self, val):
        """
        Changed visibility of a specie.
        
        """
        self.visibleSpecieList = []
        for specie in self.specieList:
            if self.specieBoxes[specie].isChecked():
                self.visibleSpecieList.append(specie)
        
        if len(self.visibleSpecieList) != len(self.specieList):
            self.allSpeciesBox.setChecked(0)
            self.allSpeciesSelected = False


################################################################################
class CropSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
        self.filterType = "Crop"
        
        self.xmin = 0.0
        self.xmax = 0.0
        self.ymin = 0.0
        self.ymax = 0.0
        self.zmin = 0.0
        self.zmax = 0.0
        self.xEnabled = 0
        self.yEnabled = 0
        self.zEnabled = 0
        
        label = QtGui.QLabel( " X Min " )
        label2 = QtGui.QLabel( " X Max " )
        self.xCropCheckBox = QtGui.QCheckBox(" X Crop Enabled")
        self.xCropCheckBox.setChecked(0)
        self.connect( self.xCropCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.changedXEnabled )
        self.xMinRangeSpinBox = QtGui.QDoubleSpinBox()
        self.xMinRangeSpinBox.setSingleStep(0.1)
        self.xMinRangeSpinBox.setMinimum(-9999.0)
        self.xMinRangeSpinBox.setMaximum(9999.0)
        self.xMinRangeSpinBox.setValue(self.xmin)
        self.connect(self.xMinRangeSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setXMin)
        self.xMaxRangeSpinBox = QtGui.QDoubleSpinBox()
        self.xMaxRangeSpinBox.setSingleStep(0.1)
        self.xMaxRangeSpinBox.setMinimum(-9999.0)
        self.xMaxRangeSpinBox.setMaximum(9999.0)
        self.xMaxRangeSpinBox.setValue(self.xmax)
        self.connect(self.xMaxRangeSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setXMax)
        row = self.newRow()
        row.addWidget( self.xCropCheckBox )
        row = self.newRow()
        row.addWidget( label )
        row.addWidget( self.xMinRangeSpinBox )
        row.addWidget( label2 )
        row.addWidget( self.xMaxRangeSpinBox )
        
        self.newRow()
        
        label = QtGui.QLabel( " Y Min " )
        label2 = QtGui.QLabel( " Y Max " )
        self.yCropCheckBox = QtGui.QCheckBox(" Y Crop Enabled")
        self.yCropCheckBox.setChecked(0)
        self.connect( self.yCropCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.changedYEnabled )
        self.yMinRangeSpinBox = QtGui.QDoubleSpinBox()
        self.yMinRangeSpinBox.setSingleStep(0.1)
        self.yMinRangeSpinBox.setMinimum(-9999.0)
        self.yMinRangeSpinBox.setMaximum(9999.0)
        self.yMinRangeSpinBox.setValue(self.ymin)
        self.connect(self.yMinRangeSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setYMin)
        self.yMaxRangeSpinBox = QtGui.QDoubleSpinBox()
        self.yMaxRangeSpinBox.setSingleStep(0.1)
        self.yMaxRangeSpinBox.setMinimum(-9999.0)
        self.yMaxRangeSpinBox.setMaximum(9999.0)
        self.yMaxRangeSpinBox.setValue(self.ymax)
        self.connect(self.yMaxRangeSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setYMax)
        row = self.newRow()
        row.addWidget( self.yCropCheckBox )
        row = self.newRow()
        row.addWidget( label )
        row.addWidget( self.yMinRangeSpinBox )
        row.addWidget( label2 )
        row.addWidget( self.yMaxRangeSpinBox )
        
        self.newRow()
        
        label = QtGui.QLabel( " Z Min " )
        label2 = QtGui.QLabel( " Z Max " )
        self.zCropCheckBox = QtGui.QCheckBox(" Z Crop Enabled")
        self.zCropCheckBox.setChecked(0)
        self.connect( self.zCropCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.changedZEnabled )
        self.zMinRangeSpinBox = QtGui.QDoubleSpinBox()
        self.zMinRangeSpinBox.setSingleStep(0.1)
        self.zMinRangeSpinBox.setMinimum(-9999.0)
        self.zMinRangeSpinBox.setMaximum(9999.0)
        self.zMinRangeSpinBox.setValue(self.zmin)
        self.connect(self.zMinRangeSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setZMin)
        self.zMaxRangeSpinBox = QtGui.QDoubleSpinBox()
        self.zMaxRangeSpinBox.setSingleStep(0.1)
        self.zMaxRangeSpinBox.setMinimum(-9999.0)
        self.zMaxRangeSpinBox.setMaximum( 9999.0)
        self.zMaxRangeSpinBox.setValue(self.zmax)
        self.connect(self.zMaxRangeSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setZMax)
        row = self.newRow()
        row.addWidget( self.zCropCheckBox )
        row = self.newRow()
        row.addWidget( label )
        row.addWidget( self.zMinRangeSpinBox )
        row.addWidget( label2 )
        row.addWidget( self.zMaxRangeSpinBox )
        
        self.newRow()
        
        self.setToLatticeButton = QtGui.QPushButton('Set to lattice')
        self.setToLatticeButton.setAutoDefault(0)
        self.setToLatticeButton.setStatusTip('Set crop to lattice dimensions')
        self.connect(self.setToLatticeButton, QtCore.SIGNAL('clicked()'), self.setCropToLattice)
        row = self.newRow(align='Center')
        row.addWidget(self.setToLatticeButton)
        
    def setCropToLattice( self ):
        self.xMinRangeSpinBox.setValue( 0.0 )
        self.xMaxRangeSpinBox.setValue( self.mainWindow.inputState.cellDims[0] )
        self.yMinRangeSpinBox.setValue( 0.0 )
        self.yMaxRangeSpinBox.setValue( self.mainWindow.inputState.cellDims[1] )
        self.zMinRangeSpinBox.setValue( 0.0 )
        self.zMaxRangeSpinBox.setValue( self.mainWindow.inputState.cellDims[2] )
    
    def changedXEnabled( self ):
        if self.xCropCheckBox.isChecked():
            self.xEnabled = 1
        else:
            self.xEnabled = 0
    
    def changedYEnabled( self ):
        if self.yCropCheckBox.isChecked():
            self.yEnabled = 1
        else:
            self.yEnabled = 0
    
    def changedZEnabled( self ):
        if self.zCropCheckBox.isChecked():
            self.zEnabled = 1
        else:
            self.zEnabled = 0
    
    def setXMin( self, val ):
        self.xmin = val
    
    def setXMax( self, val ):
        self.xmax = val
    
    def setYMin( self, val ):
        self.ymin = val
    
    def setYMax( self, val ):
        self.ymax = val
    
    def setZMin( self, val ):
        self.zmin = val
    
    def setZMax( self, val ):
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
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
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
        self.xCentreSpinBox.valueChanged.connect(self.xCentreChanged)
        
        self.yCentreSpinBox = QtGui.QDoubleSpinBox()
        self.yCentreSpinBox.setSingleStep(0.01)
        self.yCentreSpinBox.setMinimum(-9999.0)
        self.yCentreSpinBox.setMaximum( 9999.0)
        self.yCentreSpinBox.setValue(self.yCentre)
        self.yCentreSpinBox.valueChanged.connect(self.yCentreChanged)
        
        self.zCentreSpinBox = QtGui.QDoubleSpinBox()
        self.zCentreSpinBox.setSingleStep(0.01)
        self.zCentreSpinBox.setMinimum(-9999.0)
        self.zCentreSpinBox.setMaximum( 9999.0)
        self.zCentreSpinBox.setValue(self.zCentre)
        self.zCentreSpinBox.valueChanged.connect(self.zCentreChanged)
        
        label = QtGui.QLabel("Centre: ")
        comma = QtGui.QLabel(" , ")
        comma2 = QtGui.QLabel(" , ")
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.xCentreSpinBox)
        row.addWidget(comma)
        row.addWidget(self.yCentreSpinBox)
        row.addWidget(comma2)
        row.addWidget(self.zCentreSpinBox)
        
        # set to centre
        self.setToLatticeButton = QtGui.QPushButton('Set to lattice centre')
        self.setToLatticeButton.setAutoDefault(0)
        self.setToLatticeButton.setStatusTip('Set to lattice centre')
        self.setToLatticeButton.clicked.connect(self.setToLattice)
        row = self.newRow(align='Center')
        row.addWidget(self.setToLatticeButton)
        
        # radius
        self.radiusSpinBox = QtGui.QDoubleSpinBox()
        self.radiusSpinBox.setSingleStep(0.01)
        self.radiusSpinBox.setMinimum(0.0)
        self.radiusSpinBox.setMaximum(9999.0)
        self.radiusSpinBox.setValue(self.radius)
        self.radiusSpinBox.valueChanged.connect(self.radiusChanged)
        
        label = QtGui.QLabel("Radius: ")
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.radiusSpinBox)
        
        # invert selection
        self.invertCheckBox = QtGui.QCheckBox("Invert selection")
        self.invertCheckBox.setChecked(0)
        self.invertCheckBox.stateChanged.connect(self.invertChanged)
        
        row = self.newRow()
        row.addWidget(self.invertCheckBox)
    
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
        self.xCentreSpinBox.setValue(self.mainWindow.inputState.cellDims[0] / 2.0)
        self.yCentreSpinBox.setValue(self.mainWindow.inputState.cellDims[1] / 2.0)
        self.zCentreSpinBox.setValue(self.mainWindow.inputState.cellDims[2] / 2.0)
    
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
class PointDefectsSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
        self.filterType = "Point defects"
        
        # settings
        self.vacancyRadius = 1.3
        self.specieList = []
        self.visibleSpecieList = []
        self.specieRows = {}
        self.specieBoxes = {}
        self.allSpeciesSelected = True
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
        self.calculateVolumes = 0
        self.drawConvexHulls = 0
        self.hideDefects = 0
        self.identifySplitInts = 1
        
        # vacancy radius option
        label = QtGui.QLabel("Vacancy radius ")
        self.vacRadSpinBox = QtGui.QDoubleSpinBox()
        self.vacRadSpinBox.setSingleStep(0.01)
        self.vacRadSpinBox.setMinimum(0.01)
        self.vacRadSpinBox.setMaximum(10.0)
        self.vacRadSpinBox.setValue(self.vacancyRadius)
        self.connect(self.vacRadSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.vacRadChanged)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.vacRadSpinBox)
        
        self.newRow()
        
        # defect type options
        label = QtGui.QLabel("Visible types:")
        row = self.newRow()
        row.addWidget(label)
        
        self.intTypeCheckBox = QtGui.QCheckBox(" Interstitials")
        self.intTypeCheckBox.setChecked(1)
        self.connect( self.intTypeCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.intVisChanged )
        row = self.newRow()
        row.addWidget(self.intTypeCheckBox)
        
        self.vacTypeCheckBox = QtGui.QCheckBox(" Vacancies   ")
        self.vacTypeCheckBox.setChecked(1)
        self.connect( self.vacTypeCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.vacVisChanged )
        row = self.newRow()
        row.addWidget(self.vacTypeCheckBox)
        
        self.antTypeCheckBox = QtGui.QCheckBox(" Antisites    ")
        self.antTypeCheckBox.setChecked(1)
        self.connect( self.antTypeCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.antVisChanged )
        row = self.newRow()
        row.addWidget(self.antTypeCheckBox)
        
        self.newRow()
        
        # identify split ints check box
        self.identifySplitsCheck = QtGui.QCheckBox(" Identify split interstitials")
        self.identifySplitsCheck.setChecked(1)
        self.identifySplitsCheck.stateChanged.connect(self.identifySplitsChanged)
        row = self.newRow()
        row.addWidget(self.identifySplitsCheck)
        
        self.newRow()
        
        # find clusters group box
        self.findClustersGroupBox = QtGui.QGroupBox("Find clusters")
        self.findClustersGroupBox.setCheckable(True)
        self.findClustersGroupBox.setChecked(False)
        self.findClustersGroupBox.setAlignment(QtCore.Qt.AlignHCenter)
        self.findClustersGroupBox.toggled.connect(self.findClustersChanged)
        
        findClustersLayout = QtGui.QVBoxLayout(self.findClustersGroupBox)
        findClustersLayout.setAlignment(QtCore.Qt.AlignTop)
        findClustersLayout.setContentsMargins(0, 0, 0, 0)
        findClustersLayout.setSpacing(0)
        
        row = self.newRow()
        row.addWidget(self.findClustersGroupBox)
        
        # find clusters check box
#        self.findClustersCheckBox = QtGui.QCheckBox(" Find clusters")
#        self.findClustersCheckBox.setChecked(0)
#        self.connect(self.findClustersCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.findClustersChanged)
#        row = self.newRow()
#        row.addWidget(self.findClustersCheckBox)
        
        # neighbour rad spin box
        label = QtGui.QLabel("Neighbour radius ")
        self.nebRadSpinBox = QtGui.QDoubleSpinBox()
        self.nebRadSpinBox.setSingleStep(0.01)
        self.nebRadSpinBox.setMinimum(0.01)
        self.nebRadSpinBox.setMaximum(100.0)
        self.nebRadSpinBox.setValue(self.neighbourRadius)
        self.connect(self.nebRadSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.nebRadChanged)
        
        row = genericForm.FormRow()
        row.addWidget(label)
        row.addWidget(self.nebRadSpinBox)
        findClustersLayout.addWidget(row)
        
        # minimum size spin box
        label = QtGui.QLabel("Minimum cluster size ")
        self.minNumSpinBox = QtGui.QSpinBox()
        self.minNumSpinBox.setMinimum(1)
        self.minNumSpinBox.setMaximum(1000)
        self.minNumSpinBox.setValue(self.minClusterSize)
        self.connect(self.minNumSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.minNumChanged)
        
        row = genericForm.FormRow()
        row.addWidget(label)
        row.addWidget(self.minNumSpinBox)
        findClustersLayout.addWidget(row)
        
        # maximum size spin box
        label = QtGui.QLabel("Maximum cluster size ")
        self.maxNumSpinBox = QtGui.QSpinBox()
        self.maxNumSpinBox.setMinimum(-1)
        self.maxNumSpinBox.setMaximum(999999)
        self.maxNumSpinBox.setValue(self.maxClusterSize)
        self.connect(self.maxNumSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.maxNumChanged)
        
        row = genericForm.FormRow()
        row.addWidget(label)
        row.addWidget(self.maxNumSpinBox)
        findClustersLayout.addWidget(row)
        
        self.newRow()
        
        label = QtGui.QLabel("Visible species:")
        row = self.newRow()
        row.addWidget(label)
        
        self.allSpeciesBox = QtGui.QCheckBox("All")
        self.allSpeciesBox.setChecked(1)
        self.connect(self.allSpeciesBox, QtCore.SIGNAL('stateChanged(int)'), self.allSpeciesBoxChanged)
        row = self.newRow()
        row.addWidget(self.allSpeciesBox)
        
        # draw hulls group box
        self.drawHullsGroupBox = QtGui.QGroupBox(" Draw convex hulls")
        self.drawHullsGroupBox.setCheckable(True)
        self.drawHullsGroupBox.setChecked(False)
        self.drawHullsGroupBox.setAlignment(QtCore.Qt.AlignHCenter)
        self.drawHullsGroupBox.toggled.connect(self.drawHullsChanged)
        
        drawHullsLayout = QtGui.QVBoxLayout(self.drawHullsGroupBox)
        drawHullsLayout.setAlignment(QtCore.Qt.AlignTop)
        drawHullsLayout.setContentsMargins(0, 0, 0, 0)
        drawHullsLayout.setSpacing(0)
        
        row = self.newDisplayRow()
        row.addWidget(self.drawHullsGroupBox)
        
        # hull colour
        label = QtGui.QLabel("Hull colour  ")
        
        col = QtGui.QColor(self.hullCol[0]*255.0, self.hullCol[1]*255.0, self.hullCol[2]*255.0)
        self.hullColourButton = QtGui.QPushButton("")
        self.hullColourButton.setFixedWidth(50)
        self.hullColourButton.setFixedHeight(30)
        self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
        self.connect(self.hullColourButton, QtCore.SIGNAL("clicked()"), self.showColourDialog)
        
        row = genericForm.FormRow()
        row.addWidget(label)
        row.addWidget(self.hullColourButton)
        drawHullsLayout.addWidget(row)
        
        # hull opacity
        label = QtGui.QLabel("Hull opacity ")
        
        self.hullOpacitySpinBox = QtGui.QDoubleSpinBox()
        self.hullOpacitySpinBox.setSingleStep(0.01)
        self.hullOpacitySpinBox.setMinimum(0.01)
        self.hullOpacitySpinBox.setMaximum(1.0)
        self.hullOpacitySpinBox.setValue(self.hullOpacity)
        self.connect(self.hullOpacitySpinBox, QtCore.SIGNAL('valueChanged(double)'), self.hullOpacityChanged)
        
        row = genericForm.FormRow()
        row.addWidget(label)
        row.addWidget(self.hullOpacitySpinBox)
        drawHullsLayout.addWidget(row)
        
        # hide atoms
        self.hideAtomsCheckBox = QtGui.QCheckBox(" Hide defects")
        self.hideAtomsCheckBox.stateChanged.connect(self.hideDefectsChanged)
        
        row = genericForm.FormRow()
        row.addWidget(self.hideAtomsCheckBox)
        drawHullsLayout.addWidget(row)
        
        # calculate volumes check box
        self.calcVolsCheckBox = QtGui.QCheckBox(" Calculate volumes")
        self.calcVolsCheckBox.setChecked(0)
        self.connect(self.calcVolsCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.calcVolsChanged)
        
        self.newDisplayRow()
        
        row = self.newDisplayRow()
        row.addWidget(self.calcVolsCheckBox)
        
        self.refresh()
    
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
    
    def findClustersChanged(self, findClusters):
        """
        Change find volumes setting.
        
        """
        if findClusters:
            self.findClusters = 1
        else:
            self.findClusters = 0
    
    def vacRadChanged(self, val):
        """
        Update vacancy radius
        
        """
        self.vacancyRadius = val
    
    def intVisChanged(self):
        """
        Change visibility of interstitials
        
        """
        if self.intTypeCheckBox.isChecked():
            self.showInterstitials = 1
        else:
            self.showInterstitials = 0
    
    def vacVisChanged(self):
        """
        Change visibility of vacancies
        
        """
        if self.vacTypeCheckBox.isChecked():
            self.showVacancies = 1
        else:
            self.showVacancies = 0
    
    def antVisChanged(self):
        """
        Change visibility of antisites
        
        """
        if self.antTypeCheckBox.isChecked():
            self.showAntisites = 1
        else:
            self.showAntisites = 0
    
    def allSpeciesBoxChanged(self, val):
        """
        
        
        """
        if self.allSpeciesBox.isChecked():
            self.allSpeciesSelected = True
            
            for specie in self.specieList:
                self.specieBoxes[specie].setChecked(1)
            
        else:
            self.allSpeciesSelected = False
        
#        self.changedSpecie(0)
        
    def refresh(self):
        """
        Refresh the specie list
        
        """
        refSpecieList = self.mainWindow.refState.specieList
        inputSpecieList = self.mainWindow.inputState.specieList
        
        for spec in refSpecieList:
            if spec not in self.specieList:
                self.specieList.append(spec)
                self.addSpecieCheck(spec)
                if self.allSpeciesSelected:
                    self.specieBoxes[spec].setChecked(1)
        
        for spec in inputSpecieList:
            if spec not in self.specieList:                
                self.specieList.append(spec)
                self.addSpecieCheck(spec)
                if self.allSpeciesSelected:
                    self.specieBoxes[spec].setChecked(1)
        
        self.changedSpecie(0)

    def addSpecieCheck(self, specie):
        """
        Add check box for the given specie
        
        """
        self.specieBoxes[specie] = QtGui.QCheckBox(str(specie))
        
        self.connect(self.specieBoxes[specie], QtCore.SIGNAL('stateChanged(int)'), self.changedSpecie)
        
        row = self.newRow()
        row.addWidget(self.specieBoxes[specie])
        
        self.specieRows[specie] = row
        
    def changedSpecie(self, val):
        """
        Changed visibility of a specie.
        
        """
        self.visibleSpecieList = []
        for specie in self.specieList:
            if self.specieBoxes[specie].isChecked():
                self.visibleSpecieList.append(specie)
        
        if len(self.visibleSpecieList) != len(self.specieList):
            self.allSpeciesBox.setChecked(0)
            self.allSpeciesSelected = False
    
    def drawHullsChanged(self, drawHulls):
        """
        Change draw hulls setting.
        
        """
        if drawHulls:
            self.drawConvexHulls = 1
        else:
            self.drawConvexHulls = 0
    
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
    
    def calcVolsChanged(self, val):
        """
        Changed calc vols.
        
        """
        if self.calcVolsCheckBox.isChecked():
            self.calculateVolumes = 1
        else:
            self.calculateVolumes = 0

################################################################################
class ClusterSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
        self.filterType = "Clusters"
        
        self.minClusterSize = 8
        self.maxClusterSize = -1
        self.drawConvexHulls = 0
        self.neighbourRadius = 5.0
        self.calculateVolumes = 0
        self.hullCol = [0]*3
        self.hullCol[2] = 1
        self.hullOpacity = 0.5
        self.hideAtoms = 0
        
        # neighbour rad spin box
        label = QtGui.QLabel("Neighbour radius ")
        self.nebRadSpinBox = QtGui.QDoubleSpinBox()
        self.nebRadSpinBox.setSingleStep(0.01)
        self.nebRadSpinBox.setMinimum(0.01)
        self.nebRadSpinBox.setMaximum(100.0)
        self.nebRadSpinBox.setValue(self.neighbourRadius)
        self.connect(self.nebRadSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.nebRadChanged)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.nebRadSpinBox)
        
        # minimum size spin box
        label = QtGui.QLabel("Minimum cluster size ")
        self.minNumSpinBox = QtGui.QSpinBox()
        self.minNumSpinBox.setMinimum(1)
        self.minNumSpinBox.setMaximum(1000)
        self.minNumSpinBox.setValue(self.minClusterSize)
        self.connect(self.minNumSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.minNumChanged)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.minNumSpinBox)
        
        # maximum size spin box
        label = QtGui.QLabel("Maximum cluster size ")
        self.maxNumSpinBox = QtGui.QSpinBox()
        self.maxNumSpinBox.setMinimum(-1)
        self.maxNumSpinBox.setMaximum(999999)
        self.maxNumSpinBox.setValue(self.maxClusterSize)
        self.connect(self.maxNumSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.maxNumChanged)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.maxNumSpinBox)
                
        # draw hulls group box
        self.drawHullsGroupBox = QtGui.QGroupBox(" Draw convex hulls")
        self.drawHullsGroupBox.setCheckable(True)
        self.drawHullsGroupBox.setChecked(False)
        self.drawHullsGroupBox.setAlignment(QtCore.Qt.AlignHCenter)
        self.drawHullsGroupBox.toggled.connect(self.drawHullsChanged)
        
        drawHullsLayout = QtGui.QVBoxLayout(self.drawHullsGroupBox)
        drawHullsLayout.setAlignment(QtCore.Qt.AlignTop)
        drawHullsLayout.setContentsMargins(0, 0, 0, 0)
        drawHullsLayout.setSpacing(0)
        
        row = self.newDisplayRow()
        row.addWidget(self.drawHullsGroupBox)
        
        # hull colour
        label = QtGui.QLabel("Hull colour  ")
        
        col = QtGui.QColor(self.hullCol[0]*255.0, self.hullCol[1]*255.0, self.hullCol[2]*255.0)
        self.hullColourButton = QtGui.QPushButton("")
        self.hullColourButton.setFixedWidth(50)
        self.hullColourButton.setFixedHeight(30)
        self.hullColourButton.setStyleSheet("QPushButton { background-color: %s }" % col.name())
        self.connect(self.hullColourButton, QtCore.SIGNAL("clicked()"), self.showColourDialog)
        
        row = genericForm.FormRow()
        row.addWidget(label)
        row.addWidget(self.hullColourButton)
        drawHullsLayout.addWidget(row)
        
        # hull opacity
        label = QtGui.QLabel("Hull opacity ")
        
        self.hullOpacitySpinBox = QtGui.QDoubleSpinBox()
        self.hullOpacitySpinBox.setSingleStep(0.01)
        self.hullOpacitySpinBox.setMinimum(0.01)
        self.hullOpacitySpinBox.setMaximum(1.0)
        self.hullOpacitySpinBox.setValue(self.hullOpacity)
        self.connect(self.hullOpacitySpinBox, QtCore.SIGNAL('valueChanged(double)'), self.hullOpacityChanged)
        
        row = genericForm.FormRow()
        row.addWidget(label)
        row.addWidget(self.hullOpacitySpinBox)
        drawHullsLayout.addWidget(row)
        
        # hide atoms
        self.hideAtomsCheckBox = QtGui.QCheckBox(" Hide atoms")
        self.hideAtomsCheckBox.stateChanged.connect(self.hideAtomsChanged)
        
        row = genericForm.FormRow()
        row.addWidget(self.hideAtomsCheckBox)
        drawHullsLayout.addWidget(row)
        
        # calculate volumes check box
        self.calcVolsCheckBox = QtGui.QCheckBox(" Calculate volumes")
        self.calcVolsCheckBox.setChecked(0)
        self.connect(self.calcVolsCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.calcVolsChanged)
        
        self.newDisplayRow()
        
        row = self.newDisplayRow()
        row.addWidget(self.calcVolsCheckBox)
    
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
    
    def calcVolsChanged(self, val):
        """
        Changed calc vols.
        
        """
        if self.calcVolsCheckBox.isChecked():
            self.calculateVolumes = 1
        else:
            self.calculateVolumes = 0
    
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
    
    def drawHullsChanged(self, drawHulls):
        """
        Change draw hulls setting.
        
        """
        if drawHulls:
            self.drawConvexHulls = 1
        else:
            self.drawConvexHulls = 0


################################################################################
class DisplacementSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
        self.filterType = "Displacement"
        
        self.minDisplacement = 1.3
        self.maxDisplacement = 1000.0
        
        label = QtGui.QLabel("Min displacement ")
        self.minDisplacementSpinBox = QtGui.QDoubleSpinBox()
        self.minDisplacementSpinBox.setSingleStep(0.1)
        self.minDisplacementSpinBox.setMinimum(0.0)
        self.minDisplacementSpinBox.setMaximum(9999.0)
        self.minDisplacementSpinBox.setValue(self.minDisplacement)
        self.connect(self.minDisplacementSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setMinDisplacement)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.minDisplacementSpinBox)
        
        label = QtGui.QLabel("Max displacement ")
        self.maxDisplacementSpinBox = QtGui.QDoubleSpinBox()
        self.maxDisplacementSpinBox.setSingleStep(0.1)
        self.maxDisplacementSpinBox.setMinimum(0.0)
        self.maxDisplacementSpinBox.setMaximum(9999.0)
        self.maxDisplacementSpinBox.setValue(self.maxDisplacement)
        self.connect(self.maxDisplacementSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setMaxDisplacement)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.maxDisplacementSpinBox)
    
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
class KESettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
        self.filterType = "Kinetic energy"
        
        self.minKE = -1000.0
        self.maxKE = 1000.0
        
        label = QtGui.QLabel("Min KE ")
        self.minKESpinBox = QtGui.QDoubleSpinBox()
        self.minKESpinBox.setSingleStep(0.1)
        self.minKESpinBox.setMinimum(-9999.0)
        self.minKESpinBox.setMaximum(9999.0)
        self.minKESpinBox.setValue(self.minKE)
        self.connect(self.minKESpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setMinKE)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.minKESpinBox)
        
        label = QtGui.QLabel("Max KE ")
        self.maxKESpinBox = QtGui.QDoubleSpinBox()
        self.maxKESpinBox.setSingleStep(0.1)
        self.maxKESpinBox.setMinimum(-9999.0)
        self.maxKESpinBox.setMaximum(9999.0)
        self.maxKESpinBox.setValue(self.maxKE)
        self.connect(self.maxKESpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setMaxKE)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.maxKESpinBox)
    
    def setMinKE(self, val):
        """
        Set the minimum KE.
        
        """
        self.minKE = val

    def setMaxKE(self, val):
        """
        Set the maximum KE.
        
        """
        self.maxKE = val


################################################################################
class PESettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
        self.filterType = "Potential energy"
        
        self.minPE = -1000.0
        self.maxPE = 1000.0
        
        label = QtGui.QLabel("Min PE ")
        self.minPESpinBox = QtGui.QDoubleSpinBox()
        self.minPESpinBox.setSingleStep(0.1)
        self.minPESpinBox.setMinimum(-9999.0)
        self.minPESpinBox.setMaximum(9999.0)
        self.minPESpinBox.setValue(self.minPE)
        self.connect(self.minPESpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setMinPE)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.minPESpinBox)
        
        label = QtGui.QLabel("Max PE ")
        self.maxPESpinBox = QtGui.QDoubleSpinBox()
        self.maxPESpinBox.setSingleStep(0.1)
        self.maxPESpinBox.setMinimum(-9999.0)
        self.maxPESpinBox.setMaximum(9999.0)
        self.maxPESpinBox.setValue(self.maxPE)
        self.connect(self.maxPESpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setMaxPE)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.maxPESpinBox)
    
    def setMinPE(self, val):
        """
        Set the minimum PE.
        
        """
        self.minPE = val

    def setMaxPE(self, val):
        """
        Set the maximum PE.
        
        """
        self.maxPE = val


################################################################################
class ChargeSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
        self.filterType = "Charge"
        
        self.minCharge = -100.0
        self.maxCharge = 100.0
        
        label = QtGui.QLabel("Min charge ")
        self.minChargeSpinBox = QtGui.QDoubleSpinBox()
        self.minChargeSpinBox.setSingleStep(0.1)
        self.minChargeSpinBox.setMinimum(-999.0)
        self.minChargeSpinBox.setMaximum(999.0)
        self.minChargeSpinBox.setValue(self.minCharge)
        self.connect(self.minChargeSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setMinCharge)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.minChargeSpinBox)
        
        label = QtGui.QLabel("Max charge ")
        self.maxChargeSpinBox = QtGui.QDoubleSpinBox()
        self.maxChargeSpinBox.setSingleStep(0.1)
        self.maxChargeSpinBox.setMinimum(-999.0)
        self.maxChargeSpinBox.setMaximum(999.0)
        self.maxChargeSpinBox.setValue(self.maxCharge)
        self.connect(self.maxChargeSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.setMaxCharge)
        
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(self.maxChargeSpinBox)
    
    def setMinCharge(self, val):
        """
        Set the minimum charge.
        
        """
        self.minPE = val

    def setMaxCharge(self, val):
        """
        Set the maximum charge.
        
        """
        self.maxPE = val
