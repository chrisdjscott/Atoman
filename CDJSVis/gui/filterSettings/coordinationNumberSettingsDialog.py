
"""
Contains GUI forms for the coordination number filter.

"""
from PySide import QtGui

from . import base


################################################################################

class CoordinationNumberSettingsDialog(base.GenericSettingsDialog):
    """
    Coordination number settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(CoordinationNumberSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Coordination number"
        self.addProvidedScalar("Coordination number")
        
        self.minCoordNum = 0
        self.maxCoordNum = 100
        
        groupLayout = self.addFilteringGroupBox(slot=self.filteringToggled, checked=False)
        
        label = QtGui.QLabel("Min:")
        self.minCoordNumSpinBox = QtGui.QSpinBox()
        self.minCoordNumSpinBox.setSingleStep(1)
        self.minCoordNumSpinBox.setMinimum(0)
        self.minCoordNumSpinBox.setMaximum(999)
        self.minCoordNumSpinBox.setValue(self.minCoordNum)
        self.minCoordNumSpinBox.valueChanged.connect(self.setMinCoordNum)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.minCoordNumSpinBox)
        groupLayout.addLayout(row)
        
        label = QtGui.QLabel("Max:")
        self.maxCoordNumSpinBox = QtGui.QSpinBox()
        self.maxCoordNumSpinBox.setSingleStep(1)
        self.maxCoordNumSpinBox.setMinimum(0)
        self.maxCoordNumSpinBox.setMaximum(999)
        self.maxCoordNumSpinBox.setValue(self.maxCoordNum)
        self.maxCoordNumSpinBox.valueChanged.connect(self.setMaxCoordNum)
        
        row = QtGui.QHBoxLayout()
        row.addWidget(label)
        row.addWidget(self.maxCoordNumSpinBox)
        groupLayout.addLayout(row)
    
    def filteringToggled(self, arg):
        """
        Filtering toggled
        
        """
        self.filteringEnabled = arg
    
    def setMinCoordNum(self, val):
        """
        Set the minimum coordination number.
        
        """
        self.minCoordNum = val

    def setMaxCoordNum(self, val):
        """
        Set the maximum coordination number.
        
        """
        self.maxCoordNum = val
