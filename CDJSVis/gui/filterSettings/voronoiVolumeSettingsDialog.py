
"""
Contains GUI forms for the Voronoi volume filter.

"""
from PySide import QtGui, QtCore

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
        self.filteringEnabled = False
        
        # filter check
        filterCheck = QtGui.QCheckBox()
        filterCheck.setChecked(self.filteringEnabled)
        filterCheck.setToolTip("Filter by Voronoi volume")
        filterCheck.stateChanged.connect(self.filteringToggled)
        self.contentLayout.addRow("<b>Filter by Voronoi volume</b>", filterCheck)
        
        self.minVoroVolSpin = QtGui.QDoubleSpinBox()
        self.minVoroVolSpin.setSingleStep(0.01)
        self.minVoroVolSpin.setMinimum(0.0)
        self.minVoroVolSpin.setMaximum(9999.99)
        self.minVoroVolSpin.setValue(self.minVoroVol)
        self.minVoroVolSpin.valueChanged[float].connect(self.setMinVoroVol)
        self.minVoroVolSpin.setEnabled(self.filteringEnabled)
        self.minVoroVolSpin.setToolTip("Minimum visible Voronoi volume")
        self.contentLayout.addRow("Minimum", self.minVoroVolSpin)
        
        self.maxVoroVolSpin = QtGui.QDoubleSpinBox()
        self.maxVoroVolSpin.setSingleStep(0.01)
        self.maxVoroVolSpin.setMinimum(0.0)
        self.maxVoroVolSpin.setMaximum(9999.99)
        self.maxVoroVolSpin.setValue(self.maxVoroVol)
        self.maxVoroVolSpin.valueChanged[float].connect(self.setMaxVoroVol)
        self.maxVoroVolSpin.setEnabled(self.filteringEnabled)
        self.maxVoroVolSpin.setToolTip("Maximum visible Voronoi volume")
        self.contentLayout.addRow("Maximum", self.maxVoroVolSpin)
    
    def filteringToggled(self, state):
        """
        Filtering toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filteringEnabled = False
        
        else:
            self.filteringEnabled = True
        
        self.minVoroVolSpin.setEnabled(self.filteringEnabled)
        self.maxVoroVolSpin.setEnabled(self.filteringEnabled)
    
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
