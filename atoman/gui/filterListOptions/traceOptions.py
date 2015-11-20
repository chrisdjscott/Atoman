
"""
Trace options
-------------

This feature is currently in development...

The trace options settings allow you to track the movement of the visible 
atoms within the filter list.  This will calculate the displacement of all
visible atoms and draw a tube/bond from their initial locations to their
current positions. 

* The "Bond thickness" option specifies the thickness of the bond during 
  onscreen rendering ("VTK") or offline/POV-Ray rendering ("POV")
* The "Number of sides" option determines how many sides make up the tube
  used to render the bond.  A higher setting will look better but will be 
  much slower to render and interact with.

"""

import logging

from PySide import QtGui, QtCore


################################################################################

class TraceOptionsWindow(QtGui.QDialog):
    """
    Window for setting trace options.
    
    """
    modified = QtCore.Signal(str)
    
    def __init__(self, mainWindow, parent=None):
        super(TraceOptionsWindow, self).__init__(parent)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.logger = logging.getLogger(__name__)
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Trace options") # filter list id should be in here
#        self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        # defaults
        self.bondThicknessVTK = 0.4
        self.bondThicknessPOV = 0.4
        self.bondNumSides = 5
        self.drawTraceVectors = False
        
        # layout
        windowLayout = QtGui.QVBoxLayout(self)
        
        # draw trace vector settings
        self.drawVectorsGroup = QtGui.QGroupBox("Draw trace vectors")
        self.drawVectorsGroup.setCheckable(True)
        self.drawVectorsGroup.setChecked(False)
        self.drawVectorsGroup.toggled.connect(self.drawTraceVectorsChanged)
        
        grpLayout = QtGui.QVBoxLayout(self.drawVectorsGroup)
        grpLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # thickness
        bondThicknessGroup = QtGui.QGroupBox("Bond thickness")
        bondThicknessGroup.setAlignment(QtCore.Qt.AlignCenter)
        bondThicknessLayout = QtGui.QVBoxLayout()
        bondThicknessGroup.setLayout(bondThicknessLayout)
        grpLayout.addWidget(bondThicknessGroup)
        
        # vtk
        vtkThickSpin = QtGui.QDoubleSpinBox()
        vtkThickSpin.setMinimum(0.01)
        vtkThickSpin.setMaximum(10)
        vtkThickSpin.setSingleStep(0.01)
        vtkThickSpin.setValue(self.bondThicknessVTK)
        vtkThickSpin.valueChanged.connect(self.vtkThickChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("VTK:"))
        row.addWidget(vtkThickSpin)
        bondThicknessLayout.addLayout(row)
        
        # pov
        povThickSpin = QtGui.QDoubleSpinBox()
        povThickSpin.setMinimum(0.01)
        povThickSpin.setMaximum(10)
        povThickSpin.setSingleStep(0.01)
        povThickSpin.setValue(self.bondThicknessPOV)
        povThickSpin.valueChanged.connect(self.povThickChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("POV:"))
        row.addWidget(povThickSpin)
        bondThicknessLayout.addLayout(row)
        
        # num sides group
        numSidesGroup = QtGui.QGroupBox("Number of sides")
        numSidesGroup.setAlignment(QtCore.Qt.AlignCenter)
        numSidesLayout = QtGui.QVBoxLayout()
        numSidesGroup.setLayout(numSidesLayout)
        grpLayout.addWidget(numSidesGroup)
        
        # num sides
        numSidesSpin = QtGui.QSpinBox()
        numSidesSpin.setMinimum(3)
        numSidesSpin.setMaximum(999)
        numSidesSpin.setSingleStep(1)
        numSidesSpin.setValue(self.bondNumSides)
        numSidesSpin.valueChanged.connect(self.numSidesChanged)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(numSidesSpin)
        numSidesLayout.addLayout(row)
        
        windowLayout.addWidget(self.drawVectorsGroup)
        
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        windowLayout.addWidget(buttonBox)
    
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
    
    def drawTraceVectorsChanged(self, drawVectors):
        """
        Draw trace vectors toggled
        
        """
        self.drawTraceVectors = drawVectors
        
        if self.drawTraceVectors:
            text = "Trace options: On"
        else:
            text = "Trace options: Off"
        
        self.modified.emit(text)
