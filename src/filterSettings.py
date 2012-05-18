
"""
Settings for filters

@author: Chris Scott

"""
import sys

from PyQt4 import QtGui, QtCore, Qt

import utilities
from utilities import iconPath
import genericForm
import globalsModule

try:
    import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)




################################################################################
class GenericSettingsDialog(QtGui.QDialog):
    def __init__(self, title, parent):
        QtGui.QDockWidget.__init__(self, parent=parent)
        
        self.setModal(0)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(iconPath("preferences-system.svg")))
#        self.resize(500,300)
        
        dialogLayout = QtGui.QVBoxLayout()
        dialogLayout.setAlignment(QtCore.Qt.AlignTop)
#        dialogLayout.setContentsMargins(0, 0, 0, 0)
#        dialogLayout.setSpacing(0)
        
        self.contentLayout = QtGui.QVBoxLayout()
        self.contentLayout.setAlignment(QtCore.Qt.AlignTop)
        self.contentLayout.setContentsMargins(0, 0, 0, 0)
        self.contentLayout.setSpacing(0)
        
        contentWidget = QtGui.QGroupBox(title)
        contentWidget.setAlignment(QtCore.Qt.AlignCenter)
        contentWidget.setLayout(self.contentLayout)
        
        dialogLayout.addWidget(contentWidget)
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
        
        row = genericForm.FormRow(align=align)
        self.contentLayout.addWidget(row)
        
        return row
    
    def removeRow(self,row):
        self.contentLayout.removeWidget(row)  
    
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
        print "REFRESHING CROP",self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax
    
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
        
        # check if qconvex programme located
        self.qconvex = utilities.checkForExe("qconvex")
        
        if self.qconvex:
            self.mainWindow.console.write("'qconvex' executable located at: %s" % (self.qconvex,))
        
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
        
        # find clusters check box
        self.findClustersCheckBox = QtGui.QCheckBox(" Find clusters")
        self.findClustersCheckBox.setChecked(0)
        self.connect(self.findClustersCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.findClustersChanged)
        row = self.newRow()
        row.addWidget(self.findClustersCheckBox)
        
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
        
        self.newRow()
        
        label = QtGui.QLabel("Visible species:")
        row = self.newRow()
        row.addWidget(label)
        
        self.allSpeciesBox = QtGui.QCheckBox("All")
        self.allSpeciesBox.setChecked(1)
        self.connect(self.allSpeciesBox, QtCore.SIGNAL('stateChanged(int)'), self.allSpeciesBoxChanged)
        row = self.newRow()
        row.addWidget(self.allSpeciesBox)
        
#        self.newRow()
        
        self.refresh()
    
    def nebRadChanged(self, val):
        """
        Change neighbour radius.
        
        """
        self.neighbourRadius = val
    
    def findClustersChanged(self):
        """
        Change find volumes setting.
        
        """
        if self.findClustersCheckBox.isChecked():
            if not self.qconvex:
                utilities.warnExeNotFound(self, "qconvex")
                self.findClustersCheckBox.setCheckState(0)
                return
            
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


################################################################################
class ClusterSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title, parent)
        
        self.filterType = "Clusters"
        
        # check if qconvex programme located
        self.qconvex = utilities.checkForExe("qconvex")
        
        if self.qconvex:
            self.mainWindow.console.write("'qconvex' executable located at: %s" % (self.qconvex,))
        
        self.minClusterSize = 5
        self.drawConvexHulls = 0
        self.neighbourRadius = 3.5
        self.calculateVolumes = 0
        
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
        
        self.newRow()
        
        # draw hull check box
        self.drawHullsCheckBox = QtGui.QCheckBox(" Draw convex hulls")
        self.drawHullsCheckBox.setChecked(0)
        self.connect( self.drawHullsCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.drawHullsChanged )
        
        row = self.newRow()
        row.addWidget(self.drawHullsCheckBox)
        
        # calculate volumes check box
        self.calcVolsCheckBox = QtGui.QCheckBox(" Calculate volumes")
        self.calcVolsCheckBox.setChecked(0)
        self.calcVolsCheckBox.setCheckable(0)
        self.connect(self.calcVolsCheckBox, QtCore.SIGNAL('stateChanged(int)'), self.calcVolsChanged)
        
        row = self.newRow()
        row.addWidget(self.calcVolsCheckBox)

    def calcVolsChanged(self, val):
        """
        Changed calc vols.
        
        """
        if self.calcVolsCheckBox.isChecked():
            if not self.drawHullsCheckBox.isChecked():
                self.calcVolsCheckBox.setCheckState(0)
            
            else:
                self.calculateVolumes = 1
        
        else:
            self.calculateVolumes = 0
    
    def minNumChanged(self, val):
        """
        Change min cluster size.
        
        """
        self.minClusterSize = val
    
    def nebRadChanged(self, val):
        """
        Change neighbour radius.
        
        """
        self.neighbourRadius = val
    
    def drawHullsChanged(self):
        """
        Change draw hulls setting.
        
        """
        if self.drawHullsCheckBox.isChecked():
            if not self.qconvex:
                utilities.warnExeNotFound(self, "qconvex")
                self.drawHullsCheckBox.setCheckState(0)
                return
            
            self.drawConvexHulls = 1
            self.calcVolsCheckBox.setCheckable(1)
        
        else:
            self.drawConvexHulls = 0
            self.calcVolsCheckBox.setCheckState(0)
            self.calcVolsCheckBox.setCheckable(0)


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
