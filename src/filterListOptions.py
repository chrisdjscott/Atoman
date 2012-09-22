
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
#        self.resize(400,300)
        
        # defaults
        self.colourBy = "Specie"
        self.heightAxis = 1
        self.minVal = 0.0
        self.maxVal = 1.0
        
        windowLayout = QtGui.QVBoxLayout(self)
#        windowLayout.setAlignment(QtCore.Qt.AlignTop)
#        windowLayout.setContentsMargins(0, 0, 0, 0)
#        windowLayout.setSpacing(0)
        
        # combo box
        self.colouringCombo = QtGui.QComboBox()
        self.colouringCombo.addItem("Specie")
        self.colouringCombo.addItem("Height")
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
        axisCombo.addItem("Height in x")
        axisCombo.addItem("Height in y")
        axisCombo.addItem("Height in z")
        axisCombo.setCurrentIndex(1)
        axisCombo.currentIndexChanged.connect(self.axisChanged)
        
        row = heightOptions.newRow()
        row.addWidget(axisCombo)
        
        # min/max
        self.minHeightSpinBox = QtGui.QDoubleSpinBox()
        self.minHeightSpinBox.setSingleStep(0.1)
        self.minHeightSpinBox.setMinimum(-9999.0)
        self.minHeightSpinBox.setMaximum(9999.0)
        self.minHeightSpinBox.setValue(0)
        self.minHeightSpinBox.valueChanged.connect(self.minValChanged)
        
        self.maxHeightSpinBox = QtGui.QDoubleSpinBox()
        self.maxHeightSpinBox.setSingleStep(0.1)
        self.maxHeightSpinBox.setMinimum(-9999.0)
        self.maxHeightSpinBox.setMaximum(9999.0)
        self.maxHeightSpinBox.setValue(1)
        self.maxHeightSpinBox.valueChanged.connect(self.maxValChanged)
        
        label = QtGui.QLabel( " Min " )
        label2 = QtGui.QLabel( " Max " )
        
        row = heightOptions.newRow()
        row.addWidget(label)
        row.addWidget(self.minHeightSpinBox)
        
        row = heightOptions.newRow()
        row.addWidget(label2)
        row.addWidget(self.maxHeightSpinBox)
        
        # set to lattice
        setHeightToLatticeButton = QtGui.QPushButton("Set to lattice")
        setHeightToLatticeButton.setAutoDefault(0)
        setHeightToLatticeButton.clicked.connect(self.setHeightToLattice)
        
        row = heightOptions.newRow()
        row.addWidget(setHeightToLatticeButton)
        
        self.stackedWidget.addWidget(heightOptions)
        
        windowLayout.addWidget(self.stackedWidget)
    
    def setHeightToLattice(self):
        """
        Set height to lattice.
        
        """
        self.minHeightSpinBox.setValue(0.0)
        self.maxHeightSpinBox.setValue(self.parent.mainWindow.refState.cellDims[self.heightAxis])
    
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

