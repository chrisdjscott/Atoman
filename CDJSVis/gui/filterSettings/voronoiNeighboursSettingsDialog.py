
"""
Contains GUI forms for the Voronoi neighbours filter.

"""
from PySide import QtGui, QtCore

from . import base


################################################################################
class VoronoiNeighboursSettingsDialog(base.GenericSettingsDialog):
    """
    Voronoi neighbours filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(VoronoiNeighboursSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Voronoi neighbours"
        self.addProvidedScalar("Voronoi neighbours")
        
        self.minVoroNebs = 0
        self.maxVoroNebs = 999
        self.filteringEnabled = False
        
        # filter check
        filterCheck = QtGui.QCheckBox()
        filterCheck.setChecked(self.filteringEnabled)
        filterCheck.setToolTip("Filter by Voronoi neighbours")
        filterCheck.stateChanged.connect(self.filteringToggled)
        self.contentLayout.addRow("<b>Filter by Voronoi neighbours</b>", filterCheck)
        
        self.minVoroNebsSpin = QtGui.QSpinBox()
        self.minVoroNebsSpin.setSingleStep(1)
        self.minVoroNebsSpin.setMinimum(0)
        self.minVoroNebsSpin.setMaximum(999)
        self.minVoroNebsSpin.setValue(self.minVoroNebs)
        self.minVoroNebsSpin.valueChanged[int].connect(self.setMinVoroNebs)
        self.minVoroNebsSpin.setEnabled(self.filteringEnabled)
        self.minVoroNebsSpin.setToolTip("Minimum number of Voronoi neighbours")
        self.contentLayout.addRow("Minimum", self.minVoroNebsSpin)
        
        
        self.maxVoroNebsSpin = QtGui.QSpinBox()
        self.maxVoroNebsSpin.setSingleStep(1)
        self.maxVoroNebsSpin.setMinimum(0)
        self.maxVoroNebsSpin.setMaximum(999)
        self.maxVoroNebsSpin.setValue(self.maxVoroNebs)
        self.maxVoroNebsSpin.valueChanged[int].connect(self.setMaxVoroNebs)
        self.maxVoroNebsSpin.setEnabled(self.filteringEnabled)
        self.maxVoroNebsSpin.setToolTip("Maximum number of Voronoi neighbours")
        self.contentLayout.addRow("Maximum", self.maxVoroNebsSpin)
    
    def filteringToggled(self, state):
        """
        Filtering toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filteringEnabled = False
        
        else:
            self.filteringEnabled = True
        
        self.minVoroNebsSpin.setEnabled(self.filteringEnabled)
        self.maxVoroNebsSpin.setEnabled(self.filteringEnabled)
    
    def setMinVoroNebs(self, val):
        """
        Set the minimum Voronoi neighbours.
        
        """
        self.minVoroNebs = val

    def setMaxVoroNebs(self, val):
        """
        Set the maximum Voronoi neighbours.
        
        """
        self.maxVoroNebs = val
