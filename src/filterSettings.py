
"""
Settings for filters

@author: Chris Scott

"""

from PyQt4 import QtGui, QtCore, Qt

import resources
from utilities import iconPath
import genericForm





################################################################################
class GenericSettingsDialog(QtGui.QDialog):
    def __init__(self, title):
        QtGui.QDockWidget.__init__(self)
        
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
        pass


################################################################################
class SpecieSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title)
        
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
        
        newSpecieList = []
        for spec in inputSpecieList:
            newSpecieList.append(spec)
        
        # compare
        if not len(self.specieList):
            self.specieList = newSpecieList
            
            for spec in self.specieList:
                self.addSpecieCheck(spec)
                self.specieBoxes[spec].setChecked(1)
        
        for spec in self.specieList:
            if spec not in newSpecieList:
                print "NEED TO REMOVE SPEC", spec
                
                # remove from specie list
#                index = self.specieList.index(spec)
#                self.specieList.pop(index)
#                
#                # remove from visible specie list
#                if spec in self.visibleSpecieList:
#                    index = self.visibleSpecieList.index(spec)
#                    self.visibleSpecieList.pop(index)
#                
#                # remove row
#                self.specieRows[spec].removeWidget(self.specieBoxes[spec])
#                self.removeRow(self.specieRows[spec])
#                del self.specieRows[spec]
#                del self.specieBoxes[spec]
        
        for spec in newSpecieList:
            if spec not in self.specieList:
                print "NEED TO ADD SPEC", spec
                
                self.specieList.append(spec)
                self.addSpecieCheck(spec)
                if self.allSpeciesSelected:
                    self.specieBoxes[spec].setChecked(1)
        
        print "REFRESHED SPEC LIST", self.specieList
        
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
        
        print "VIS SPEC LIST", self.visibleSpecieList


################################################################################
class CropSettingsDialog(GenericSettingsDialog):
    def __init__(self, mainWindow, title, parent=None):
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        GenericSettingsDialog.__init__(self, title)
        
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
        
        GenericSettingsDialog.__init__(self, title)
        
        self.filterType = "Point defects"
        
        # settings
        self.vacancyRadius = 1.3
        self.visibleSpecieList = []
        self.showInterstitials = 1
        self.showAntisites = 1
        self.showVacancies = 1
        
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
        
        label = QtGui.QLabel("Visible species:")
        row = self.newRow()
        row.addWidget(label)
        
        
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
