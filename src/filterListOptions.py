
"""
Options for filter lists.

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
        
        # layout
        windowLayout = QtGui.QVBoxLayout(self)
        
        # combo box
        self.colouringCombo = QtGui.QComboBox()
        self.colouringCombo.addItem("Specie")
        self.colouringCombo.addItem("Height")
        self.colouringCombo.addItem("Solid colour")
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
        if self.colouringCombo.count() == 4 and self.colouringCombo.currentText() == scalarType:
            print "SAME"
        
        else:
            if self.colouringCombo.currentIndex() == 3:
                print "SET ZERO"
                self.colouringCombo.setCurrentIndex(0)
            
            self.colouringCombo.removeItem(3)
            
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
        
        self.parent.colouringOptionsButton.setText("Colouring options: %s" % self.colourBy)
        
        self.stackedWidget.setCurrentIndex(index)
    
    def closeEvent(self, event):
        """
        Close event.
        
        """
        self.parent.colouringOptionsOpen = False
        self.hide()

