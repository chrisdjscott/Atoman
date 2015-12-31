
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
from __future__ import absolute_import
from __future__ import unicode_literals

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
        
        self.setWindowTitle("Trace options")  # filter list id should be in here
#        self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))
        
        # defaults
        self.bondThicknessVTK = 0.2
        self.bondThicknessPOV = 0.2
        self.bondNumSides = 6
        self.drawTraceVectors = False
        self.drawAsArrows = True
        
        # for compatibility with vectors
        self.vectorScaleFactor = 1
        self.vectorResolution = self.bondNumSides
        
        # layout
        layout = QtGui.QFormLayout(self)
        
        # draw trace settings
        self.drawTraceCheck = QtGui.QCheckBox()
        if self.drawTraceVectors:
            self.drawTraceCheck.setCheckState(QtCore.Qt.Checked)
        else:
            self.drawTraceCheck.setCheckState(QtCore.Qt.Unchecked)
        self.drawTraceCheck.stateChanged.connect(self.drawTraceVectorsChanged)
        layout.addRow("Draw trace vectors", self.drawTraceCheck)
        
        # draw as arrows
        self.arrowsCheck = QtGui.QCheckBox()
        if self.drawAsArrows:
            self.arrowsCheck.setCheckState(QtCore.Qt.Checked)
        else:
            self.arrowsCheck.setCheckState(QtCore.Qt.Unchecked)
        self.arrowsCheck.setEnabled(self.drawTraceVectors)
        self.arrowsCheck.stateChanged.connect(self.drawAsArrowsChanged)
        layout.addRow("Draw as arrows", self.arrowsCheck)
        
        # bond thickness
        self.vtkThickSpin = QtGui.QDoubleSpinBox()
        self.vtkThickSpin.setMinimum(0.01)
        self.vtkThickSpin.setMaximum(10)
        self.vtkThickSpin.setSingleStep(0.01)
        self.vtkThickSpin.setValue(self.bondThicknessVTK)
        self.vtkThickSpin.valueChanged.connect(self.vtkThickChanged)
        self.vtkThickSpin.setEnabled(self.drawTraceVectors)
        layout.addRow("Bond thickness (VTK)", self.vtkThickSpin)
        self.povThickSpin = QtGui.QDoubleSpinBox()
        self.povThickSpin.setMinimum(0.01)
        self.povThickSpin.setMaximum(10)
        self.povThickSpin.setSingleStep(0.01)
        self.povThickSpin.setValue(self.bondThicknessPOV)
        self.povThickSpin.valueChanged.connect(self.povThickChanged)
        self.povThickSpin.setEnabled(self.drawTraceVectors)
        layout.addRow("Bond thickness (POV)", self.povThickSpin)
        
        # num sides
        self.numSidesSpin = QtGui.QSpinBox()
        self.numSidesSpin.setMinimum(3)
        self.numSidesSpin.setMaximum(999)
        self.numSidesSpin.setSingleStep(1)
        self.numSidesSpin.setValue(self.bondNumSides)
        self.numSidesSpin.setEnabled(self.drawTraceVectors)
        self.numSidesSpin.valueChanged.connect(self.numSidesChanged)
        layout.addRow("Bond number of sides", self.numSidesSpin)
        
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)
        
    def numSidesChanged(self, val):
        """Number of sides changed."""
        self.bondNumSides = val
        self.vectorResolution = val
    
    def vtkThickChanged(self, val):
        """VTK thickness changed."""
        self.bondThicknessVTK = val
    
    def povThickChanged(self, val):
        """POV thickness changed."""
        self.bondThicknessPOV = val
    
    def drawAsArrowsChanged(self, state):
        """Draw as arrows changed."""
        self.drawAsArrows = False if state == QtCore.Qt.Unchecked else True
    
    def drawTraceVectorsChanged(self, state):
        """Draw trace vectors toggled."""
        self.drawTraceVectors = False if state == QtCore.Qt.Unchecked else True
        self.vtkThickSpin.setEnabled(self.drawTraceVectors)
        self.povThickSpin.setEnabled(self.drawTraceVectors)
        self.numSidesSpin.setEnabled(self.drawTraceVectors)
        self.arrowsCheck.setEnabled(self.drawTraceVectors)
        text = "Trace options: On" if self.drawTraceVectors else "Trace options: Off"
        self.modified.emit(text)
