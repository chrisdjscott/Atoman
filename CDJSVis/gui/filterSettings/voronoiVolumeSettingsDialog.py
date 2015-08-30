
"""
Contains GUI forms for the Voronoi volume filter.

"""
from PySide import QtGui

from . import base


################################################################################
class VoronoiVolumeSettingsDialog(base.GenericSettingsDialog):
    """
    Settings for Voronoi volume filter
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(VoronoiVolumeSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Voronoi volume"
        self.addProvidedScalar("Voronoi volume")
        
        self.minVoroVol = 0.0
        self.maxVoroVol = 9999.99
        
        groupLayout = self.addFilteringGroupBox(slot=self.filteringToggled, checked=False)
        
        label = QtGui.QLabel("Min:")
        self.minVoroVolSpin = QtGui.QDoubleSpinBox()
        self.minVoroVolSpin.setSingleStep(0.01)
        self.minVoroVolSpin.setMinimum(0.0)
        self.minVoroVolSpin.setMaximum(9999.99)
        self.minVoroVolSpin.setValue(self.minVoroVol)
        self.minVoroVolSpin.valueChanged[float].connect(self.setMinVoroVol)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.minVoroVolSpin)
        groupLayout.addLayout(row)
        
        label = QtGui.QLabel("Max:")
        self.maxVoroVolSpin = QtGui.QDoubleSpinBox()
        self.maxVoroVolSpin.setSingleStep(0.01)
        self.maxVoroVolSpin.setMinimum(0.0)
        self.maxVoroVolSpin.setMaximum(9999.99)
        self.maxVoroVolSpin.setValue(self.maxVoroVol)
        self.maxVoroVolSpin.valueChanged[float].connect(self.setMaxVoroVol)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.maxVoroVolSpin)
        groupLayout.addLayout(row)
    
    def filteringToggled(self, arg):
        """
        Filtering toggled
        
        """
        self.filteringEnabled = arg
    
    def setMinVoroVol(self, val):
        """
        Set the minimum Voronoi volume.
        
        """
        self.minVoroVol = val

    def setMaxVoroVol(self, val):
        """
        Set the maximum Voronoi volume.
        
        """
        self.maxVoroVol = val
