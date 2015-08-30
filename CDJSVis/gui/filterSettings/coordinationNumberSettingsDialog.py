
"""
Contains GUI forms for the coordination number filter.

"""
from PySide import QtGui, QtCore

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
        self.filteringEnabled = False
        
        # filter check
        filterCheck = QtGui.QCheckBox()
        filterCheck.setChecked(self.filteringEnabled)
        filterCheck.setToolTip("Filter by coordination number")
        filterCheck.stateChanged.connect(self.filteringToggled)
        self.contentLayout.addRow("<b>Filter by coordination</b>", filterCheck)
        
        self.minCoordNumSpinBox = QtGui.QSpinBox()
        self.minCoordNumSpinBox.setSingleStep(1)
        self.minCoordNumSpinBox.setMinimum(0)
        self.minCoordNumSpinBox.setMaximum(999)
        self.minCoordNumSpinBox.setValue(self.minCoordNum)
        self.minCoordNumSpinBox.valueChanged.connect(self.setMinCoordNum)
        self.minCoordNumSpinBox.setToolTip("Minimum visible coordination number")
        self.minCoordNumSpinBox.setEnabled(self.filteringEnabled)
        self.contentLayout.addRow("Minimum", self.minCoordNumSpinBox)
        
        self.maxCoordNumSpinBox = QtGui.QSpinBox()
        self.maxCoordNumSpinBox.setSingleStep(1)
        self.maxCoordNumSpinBox.setMinimum(0)
        self.maxCoordNumSpinBox.setMaximum(999)
        self.maxCoordNumSpinBox.setValue(self.maxCoordNum)
        self.maxCoordNumSpinBox.valueChanged.connect(self.setMaxCoordNum)
        self.maxCoordNumSpinBox.setToolTip("Maximum visible coordination number")
        self.maxCoordNumSpinBox.setEnabled(self.filteringEnabled)
        self.contentLayout.addRow("Maximum", self.maxCoordNumSpinBox)
    
    def filteringToggled(self, state):
        """
        Filtering toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filteringEnabled = False
        
        else:
            self.filteringEnabled = True
        
        self.minCoordNumSpinBox.setEnabled(self.filteringEnabled)
        self.maxCoordNumSpinBox.setEnabled(self.filteringEnabled)
    
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
