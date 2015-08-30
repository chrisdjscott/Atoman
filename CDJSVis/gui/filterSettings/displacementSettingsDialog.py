
"""
Contains GUI forms for the displacement filter.

"""
from PySide import QtGui, QtCore

from . import base


################################################################################

class DisplacementSettingsDialog(base.GenericSettingsDialog):
    """
    Displacement filter settings form.
    
    """
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
        self.filteringEnabled = False
        
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
