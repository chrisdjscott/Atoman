
"""
Contains GUI forms for the displacement filter.

"""
from PySide import QtGui, QtCore

from . import base
from ...filtering.filters import displacementFilter


################################################################################

class DisplacementSettingsDialog(base.GenericSettingsDialog):
    """
    Displacement filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(DisplacementSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Displacement"
        self.addProvidedScalar("Displacement")
        
        # settings
        self._settings = displacementFilter.DisplacementFilterSettings()
        
        # filtering options
        self.addCheckBox("filteringEnabled", toolTip="Filter atoms by displacement", label="<b>Enable filtering</b>",
                         extraSlot=self.filteringToggled)
        
        self.minDispSpin = self.addDoubleSpinBox("minDisplacement", minVal=0, maxVal=9999, step=0.1,
                                                 toolTip="Minimum displacement for an atom to be visible", label="Minimum",
                                                 settingEnabled="filteringEnabled")
        
        self.maxDispSpin = self.addDoubleSpinBox("maxDisplacement", minVal=0, maxVal=9999, step=0.1,
                                                 toolTip="Maximum displacement for an atom to be visible", label="Maximum",
                                                 settingEnabled="filteringEnabled")
        
        
        # draw displacement vector settings
        self.addCheckBox("drawDisplacementVectors", toolTip="Draw displacement vectors (movement) of atoms",
                         label="<b>Draw displacement vectors</b>", extraSlot=self.drawVectorsChanged, displayLayout=True)
        
        # vtk thickness
        self.vtkThickSpin = self.addDoubleSpinBox("bondThicknessVTK", minVal=0.01, maxVal=10, step=0.1,
                                                  toolTip="Thickness of lines showing defect movement (VTK)",
                                                  label="Bond thickness (VTK)", settingEnabled="drawDisplacementVectors",
                                                  displayLayout=True)
        
        # pov thickness
        self.povThickSpin = self.addDoubleSpinBox("bondThicknessPOV", minVal=0.01, maxVal=10, step=0.1,
                                                  toolTip="Thickness of lines showing defect movement (POV-Ray)",
                                                  label="Bond thickness (POV)", settingEnabled="drawDisplacementVectors",
                                                  displayLayout=True)
        
        # num sides
        self.numSidesSpin = self.addSpinBox("bondNumSides", minVal=3, maxVal=999,
                                            toolTip="Number of sides when rendering displacement vectors (more looks better but is slower)",
                                            label="Bond number of sides", settingEnabled="drawDisplacementVectors", displayLayout=True)
    
    def drawVectorsChanged(self, enabled):
        """Draw displacement vectors toggled."""
        self.logger.debug("Draw displacement vectors: %r", enabled)
        self.vtkThickSpin.setEnabled(enabled)
        self.povThickSpin.setEnabled(enabled)
        self.numSidesSpin.setEnabled(enabled)
    
    def filteringToggled(self, enabled):
        """Filtering toggled."""
        self.minDispSpin.setEnabled(enabled)
        self.maxDispSpin.setEnabled(enabled)
