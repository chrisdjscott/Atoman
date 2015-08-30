
"""
Contains GUI forms for the Voronoi neighbours filter.

"""
from PySide import QtGui

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
        
        groupLayout = self.addFilteringGroupBox(slot=self.filteringToggled, checked=False)
        
        label = QtGui.QLabel("Min:")
        self.minVoroNebsSpin = QtGui.QSpinBox()
        self.minVoroNebsSpin.setSingleStep(1)
        self.minVoroNebsSpin.setMinimum(0)
        self.minVoroNebsSpin.setMaximum(999)
        self.minVoroNebsSpin.setValue(self.minVoroNebs)
        self.minVoroNebsSpin.valueChanged[int].connect(self.setMinVoroNebs)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.minVoroNebsSpin)
        groupLayout.addLayout(row)
        
        label = QtGui.QLabel("Max:")
        self.maxVoroNebsSpin = QtGui.QSpinBox()
        self.maxVoroNebsSpin.setSingleStep(1)
        self.maxVoroNebsSpin.setMinimum(0)
        self.maxVoroNebsSpin.setMaximum(999)
        self.maxVoroNebsSpin.setValue(self.maxVoroNebs)
        self.maxVoroNebsSpin.valueChanged[int].connect(self.setMaxVoroNebs)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.maxVoroNebsSpin)
        groupLayout.addLayout(row)
    
    def filteringToggled(self, arg):
        """
        Filtering toggled
        
        """
        self.filteringEnabled = arg
    
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
