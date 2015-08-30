
"""
Contains GUI forms for a generic scalar filter.

"""
from PySide import QtGui

from . import base


################################################################################

class GenericScalarSettingsDialog(base.GenericSettingsDialog):
    """
    Settings for generic scalar value filterer
    
    """
    def __init__(self, mainWindow, filterType, title, parent=None):
        super(GenericScalarSettingsDialog, self).__init__(title, parent)
        
        self.filterType = filterType
        
        self.minVal = -10000.0
        self.maxVal = 10000.0
        
        self.minValSpinBox = QtGui.QDoubleSpinBox()
        self.minValSpinBox.setSingleStep(0.1)
        self.minValSpinBox.setMinimum(-99999.0)
        self.minValSpinBox.setMaximum(99999.0)
        self.minValSpinBox.setValue(self.minVal)
        self.minValSpinBox.valueChanged.connect(self.setMinVal)
        self.contentLayout.addRow("Minimum", self.minValSpinBox)
        
        self.maxValSpinBox = QtGui.QDoubleSpinBox()
        self.maxValSpinBox.setSingleStep(0.1)
        self.maxValSpinBox.setMinimum(-99999.0)
        self.maxValSpinBox.setMaximum(99999.0)
        self.maxValSpinBox.setValue(self.maxVal)
        self.maxValSpinBox.valueChanged.connect(self.setMaxVal)
        self.contentLayout.addRow("Maximum", self.maxValSpinBox)
    
    def setMinVal(self, val):
        """
        Set the minimum value.
        
        """
        self.minVal = val

    def setMaxVal(self, val):
        """
        Set the maximum value.
        
        """
        self.maxVal = val
