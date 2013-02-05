
"""
Options for filter lists.

@author: Chris Scott

"""
import sys
import functools

from PyQt4 import QtGui, QtCore

from ..visutils import utilities
from ..visutils.utilities import iconPath
from . import genericForm

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class BondsOptionsWindow(QtGui.QDialog):
    """
    Bond options for filter list.
    
    """
    def __init__(self, mainWindow, parent=None):
        super(BondsOptionsWindow, self).__init__(parent)
        
        self.parent = parent
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Filter list colouring options")
        self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        self.mainWindow = mainWindow
        
        # current species set
        self.currentSpecies = set()
        self.bondChecksList = []
        self.bondPairDrawStatus = []
        self.bondPairsList = []
        self.NBondPairs = 0
        
        # options
        self.drawBonds = False
        
        # layout
        layout = QtGui.QVBoxLayout(self)
#        layout.setSpacing(0)
#        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        
        # draw bonds group box
        self.drawBondsGroup = QtGui.QGroupBox("Draw bonds")
        self.drawBondsGroup.setCheckable(True)
        self.drawBondsGroup.setChecked(False)
        self.drawBondsGroup.setAlignment(QtCore.Qt.AlignCenter)
        self.drawBondsGroup.toggled.connect(self.drawBondsToggled)
        
        layout.addWidget(self.drawBondsGroup)
        
        self.groupLayout = QtGui.QVBoxLayout()
#        self.groupLayout.setSpacing(0)
#        self.groupLayout.setContentsMargins(0, 0, 0, 0)
        self.groupLayout.setAlignment(QtCore.Qt.AlignTop)
        
        self.drawBondsGroup.setLayout(self.groupLayout)
    
    def drawBondsToggled(self, state):
        """
        Draw bonds changed.
        
        """
        self.drawBonds = state
        
        if self.drawBonds:
            self.parent.bondsOptionsButton.setText("Bonds options: On")
        
        else:
            self.parent.bondsOptionsButton.setText("Bonds options: Off")
    
    def addSpecie(self, sym):
        """
        Add specie to bonding options.
        
        """
        # first test if already added
        if sym in self.currentSpecies:
            print "SPEC ALREADY ADDED", sym, self.currentSpecies
            return
        
        # add to set
        self.currentSpecies.add(sym)
        
        # now add line(s) to group box
        for symb in self.currentSpecies:
            # check box
            check = QtGui.QCheckBox()
            check.stateChanged.connect(functools.partial(self.checkStateChanged, len(self.bondPairDrawStatus)))
            self.bondChecksList.append(check)
            
            # draw status
            self.bondPairDrawStatus.append(False)
            
            # label
            label = QtGui.QLabel("%s - %s" % (sym, symb))
            
            # pair
            pair = (sym, symb)
            self.bondPairsList.append(pair)
            
            # row
            row = QtGui.QHBoxLayout()
            row.addWidget(check)
            row.addWidget(label)
            
            self.groupLayout.addLayout(row)
            
            self.NBondPairs += 1
    
    def checkStateChanged(self, index, state):
        """
        Check state changed.
        
        """
        if state == 2:
            self.bondPairDrawStatus[index] = True
        
        else:
            self.bondPairDrawStatus[index] = False
    
    def refresh(self):
        """
        Refresh available bonds.
        
        Should be called whenever a new input is loaded!?
        If the species are the same don't change anything!?
        
        """
        inputState = self.mainWindow.inputState
        
        for specie in inputState.specieList:
            self.addSpecie(specie)



################################################################################

class ColouringOptionsWindow(QtGui.QDialog):
    """
    Window for displaying colouring options for filter list
    
    """
    def __init__(self, parent=None):
        super(ColouringOptionsWindow, self).__init__(parent)
        
        self.parent = parent
        self.setModal(0)
#        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Filter list colouring options")
        self.setWindowIcon(QtGui.QIcon(iconPath("painticon.png")))
        
        # defaults
        self.colourBy = "Specie"
        self.heightAxis = 1
        self.minVal = 0.0
        self.maxVal = 1.0
        self.solidColour = QtGui.QColor(255, 0, 0)
        self.solidColourRGB = (float(self.solidColour.red()) / 255.0, 
                               float(self.solidColour.green()) / 255.0,
                               float(self.solidColour.blue()) / 255.0)
        self.scalarBarText = "Height in Y (A)"
        self.atomPropertyType = "Kinetic energy"
        
        # layout
        windowLayout = QtGui.QVBoxLayout(self)
        
        # combo box
        self.colouringCombo = QtGui.QComboBox()
        self.colouringCombo.addItem("Specie")
        self.colouringCombo.addItem("Height")
        self.colouringCombo.addItem("Solid colour")
        self.colouringCombo.addItem("Atom property")
#        self.colouringCombo.addItem("Scalar")
        self.colouringCombo.currentIndexChanged.connect(self.colourByChanged)
        
        windowLayout.addWidget(self.colouringCombo)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget(self)
        
        # specie widget
        self.specieOptions = genericForm.GenericForm(self, 0, "Specie colouring options")
        
        self.stackedWidget.addWidget(self.specieOptions)
        
        # height widget
        heightOptions = genericForm.GenericForm(self, 0, "Height colouring options")
        
        # axis
        axisCombo = QtGui.QComboBox()
        axisCombo.addItem("Height in X")
        axisCombo.addItem("Height in Y")
        axisCombo.addItem("Height in Z")
        axisCombo.setCurrentIndex(1)
        axisCombo.currentIndexChanged.connect(self.axisChanged)
        
        row = heightOptions.newRow()
        row.addWidget(axisCombo)
        
        # min/max
        self.minValSpinBox = QtGui.QDoubleSpinBox()
        self.minValSpinBox.setSingleStep(0.1)
        self.minValSpinBox.setMinimum(-9999.0)
        self.minValSpinBox.setMaximum(9999.0)
        self.minValSpinBox.setValue(0)
        self.minValSpinBox.valueChanged.connect(self.minValChanged)
        
        self.maxValSpinBox = QtGui.QDoubleSpinBox()
        self.maxValSpinBox.setSingleStep(0.1)
        self.maxValSpinBox.setMinimum(-9999.0)
        self.maxValSpinBox.setMaximum(9999.0)
        self.maxValSpinBox.setValue(1)
        self.maxValSpinBox.valueChanged.connect(self.maxValChanged)
        
        label = QtGui.QLabel( " Min " )
        label2 = QtGui.QLabel( " Max " )
        
        row = heightOptions.newRow()
        row.addWidget(label)
        row.addWidget(self.minValSpinBox)
        
        row = heightOptions.newRow()
        row.addWidget(label2)
        row.addWidget(self.maxValSpinBox)
        
        # set to lattice
        setHeightToLatticeButton = QtGui.QPushButton("Set to lattice")
        setHeightToLatticeButton.setAutoDefault(0)
        setHeightToLatticeButton.clicked.connect(self.setHeightToLattice)
        
        row = heightOptions.newRow()
        row.addWidget(setHeightToLatticeButton)
        
        # scalar bar text
        self.scalarBarTextEdit = QtGui.QLineEdit("Height in Y (A)")
        self.scalarBarTextEdit.textChanged.connect(self.scalarBarTextChanged)
        
        label = QtGui.QLabel("Scalar bar title:")
        row = heightOptions.newRow()
        row.addWidget(label)
        
        row = heightOptions.newRow()
        row.addWidget(self.scalarBarTextEdit)
        
        self.stackedWidget.addWidget(heightOptions)
        
        # solid colour widget
        solidColourOptions = genericForm.GenericForm(self, 0, "Solid colour options")
        
        # solid colour button
        self.colourButton = QtGui.QPushButton("")
        self.colourButton.setFixedWidth(60)
        self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.solidColour.name())
        self.colourButton.clicked.connect(self.changeSolidColour)
        
        row = solidColourOptions.newRow()
        row.addWidget(self.colourButton)
        
        self.stackedWidget.addWidget(solidColourOptions)
        
        # atom property widget
        atomPropertyOptions = genericForm.GenericForm(self, 0, "Atom property options")
        
        # type
        self.propertyTypeCombo = QtGui.QComboBox()
        self.propertyTypeCombo.addItems(("Kinetic energy", "Potential energy", "Charge"))
        self.propertyTypeCombo.currentIndexChanged.connect(self.propertyTypeChanged)
        row = atomPropertyOptions.newRow()
        row.addWidget(self.propertyTypeCombo)
        
        # min/max
        self.propertyMinSpin = QtGui.QDoubleSpinBox()
        self.propertyMinSpin.setSingleStep(0.1)
        self.propertyMinSpin.setMinimum(-9999.0)
        self.propertyMinSpin.setMaximum(9999.0)
        self.propertyMinSpin.setValue(0)
        
        self.propertyMaxSpin = QtGui.QDoubleSpinBox()
        self.propertyMaxSpin.setSingleStep(0.1)
        self.propertyMaxSpin.setMinimum(-9999.0)
        self.propertyMaxSpin.setMaximum(9999.0)
        self.propertyMaxSpin.setValue(1)
        
        label = QtGui.QLabel( " Min " )
        label2 = QtGui.QLabel( " Max " )
        
        row = atomPropertyOptions.newRow()
        row.addWidget(label)
        row.addWidget(self.propertyMinSpin)
        
        row = atomPropertyOptions.newRow()
        row.addWidget(label2)
        row.addWidget(self.propertyMaxSpin)
        
        # set to scalar range
        setToPropertyRangeButton = QtGui.QPushButton("Set to scalar range")
        setToPropertyRangeButton.setAutoDefault(0)
        setToPropertyRangeButton.clicked.connect(self.setToPropertyRange)
        
        row = atomPropertyOptions.newRow()
        row.addWidget(setToPropertyRangeButton)
        
        # scalar bar text
        self.scalarBarTextEdit3 = QtGui.QLineEdit("<insert title>")
        
        label = QtGui.QLabel("Scalar bar title:")
        row = atomPropertyOptions.newRow()
        row.addWidget(label)
        row = atomPropertyOptions.newRow()
        row.addWidget(self.scalarBarTextEdit3)
        
        self.stackedWidget.addWidget(atomPropertyOptions)
        
        # scalar widget
        self.scalarOptions = genericForm.GenericForm(self, 0, "Scalar colour options")
        
        # min/max
        self.scalarMinSpin = QtGui.QDoubleSpinBox()
        self.scalarMinSpin.setSingleStep(0.1)
        self.scalarMinSpin.setMinimum(-9999.0)
        self.scalarMinSpin.setMaximum(9999.0)
        self.scalarMinSpin.setValue(0)
        
        self.scalarMaxSpin = QtGui.QDoubleSpinBox()
        self.scalarMaxSpin.setSingleStep(0.1)
        self.scalarMaxSpin.setMinimum(-9999.0)
        self.scalarMaxSpin.setMaximum(9999.0)
        self.scalarMaxSpin.setValue(1)
        
        label = QtGui.QLabel( " Min " )
        label2 = QtGui.QLabel( " Max " )
        
        row = self.scalarOptions.newRow()
        row.addWidget(label)
        row.addWidget(self.scalarMinSpin)
        
        row = self.scalarOptions.newRow()
        row.addWidget(label2)
        row.addWidget(self.scalarMaxSpin)
        
        # set to scalar range
        setToScalarRangeButton = QtGui.QPushButton("Set to scalar range")
        setToScalarRangeButton.setAutoDefault(0)
        setToScalarRangeButton.clicked.connect(self.setToScalarRange)
        
        row = self.scalarOptions.newRow()
        row.addWidget(setToScalarRangeButton)
        
        # scalar bar text
        self.scalarBarTextEdit2 = QtGui.QLineEdit("<insert title>")
        
        label = QtGui.QLabel("Scalar bar title:")
        row = self.scalarOptions.newRow()
        row.addWidget(label)
        row = self.scalarOptions.newRow()
        row.addWidget(self.scalarBarTextEdit2)
        
        self.stackedWidget.addWidget(self.scalarOptions)
        
        windowLayout.addWidget(self.stackedWidget)
    
    def propertyTypeChanged(self, val):
        """
        Property type changed.
        
        """
        self.atomPropertyType = str(self.propertyTypeCombo.currentText())
        
        self.parent.colouringOptionsButton.setText("Colouring options: %s" % self.atomPropertyType)
    
    def setToPropertyRange(self):
        """
        Set min/max to scalar range.
        
        """
        lattice = self.parent.mainWindow.inputState
        
        if self.atomPropertyType == "Kinetic energy":
            minVal = min(lattice.KE)
            maxVal = max(lattice.KE)
        
        elif self.atomPropertyType == "Potential energy":
            minVal = min(lattice.PE)
            maxVal = max(lattice.PE)
        
        else:
            minVal = min(lattice.charge)
            maxVal = max(lattice.charge)
        
        if minVal == maxVal:
            maxVal += 1
        
        self.propertyMinSpin.setValue(minVal)
        self.propertyMaxSpin.setValue(maxVal)
    
    def setToScalarRange(self):
        """
        Set min/max to scalar range.
        
        """
        scalars = self.parent.filterer.scalars
        
        if len(scalars):
            minVal = min(scalars)
            maxVal = max(scalars)
            if minVal == maxVal:
                maxVal += 1
            
            print "MIN,MAX", minVal, maxVal
            
            self.scalarMinSpin.setValue(minVal)
            self.scalarMaxSpin.setValue(maxVal)
    
    def refreshScalarColourOption(self, scalarType):
        """
        Refresh colour by scalar options.
        
        """
        if self.colouringCombo.count() == 5 and self.colouringCombo.currentText() == scalarType:
            print "SAME"
        
        else:
            if self.colouringCombo.currentIndex() == 4:
                print "SET ZERO"
                self.colouringCombo.setCurrentIndex(0)
            
            self.colouringCombo.removeItem(4)
            
            if len(scalarType):
                self.colouringCombo.addItem(scalarType)
                print "ADD", scalarType
    
    def scalarBarTextChanged(self, text):
        """
        Scalar bar text changed.
        
        """
        self.scalarBarText = str(text)
    
    def changeSolidColour(self):
        """
        Change solid colour.
        
        """
        col = QtGui.QColorDialog.getColor()
        
        if col.isValid():
            self.solidColour = col
            self.colourButton.setStyleSheet("QPushButton { background-color: %s }" % self.solidColour.name())
            
            self.solidColourRGB = (float(self.solidColour.red()) / 255.0, 
                                   float(self.solidColour.green()) / 255.0,
                                   float(self.solidColour.blue()) / 255.0)   
    
    def setHeightToLattice(self):
        """
        Set height to lattice.
        
        """
        self.minValSpinBox.setValue(0.0)
        self.maxValSpinBox.setValue(self.parent.mainWindow.refState.cellDims[self.heightAxis])
    
    def maxValChanged(self, val):
        """
        Max height changed.
        
        """
        self.maxVal = val
    
    def minValChanged(self, val):
        """
        Min height changed.
        
        """
        self.minVal = val
    
    def axisChanged(self, index):
        """
        Changed axis.
        
        """
        self.heightAxis = index
        
        axis = ["X", "Y", "Z"]
        self.scalarBarTextEdit.setText("Height in %s (A)" % axis[index])
    
    def colourByChanged(self, index):
        """
        Colour by changed.
        
        """
        self.colourBy = str(self.colouringCombo.currentText())
        
        if self.colourBy == "Atom property":
            colourByText = str(self.propertyTypeCombo.currentText())
        else:
            colourByText = self.colourBy
        
        self.parent.colouringOptionsButton.setText("Colouring options: %s" % colourByText)
        
        self.stackedWidget.setCurrentIndex(index)
    
    def closeEvent(self, event):
        """
        Close event.
        
        """
        self.parent.colouringOptionsOpen = False
        self.hide()

