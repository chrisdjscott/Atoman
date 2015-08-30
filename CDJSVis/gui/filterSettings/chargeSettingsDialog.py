
"""
Contains GUI forms for the charge filter.

"""
from PySide import QtGui

from . import base


################################################################################

class ChargeSettingsDialog(base.GenericSettingsDialog):
    """
    GUI form for the charge filter.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(ChargeSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Charge"
        
        self.minCharge = -100.0
        self.maxCharge = 100.0
        
        self.minChargeSpinBox = QtGui.QDoubleSpinBox()
        self.minChargeSpinBox.setSingleStep(0.1)
        self.minChargeSpinBox.setMinimum(-999.0)
        self.minChargeSpinBox.setMaximum(999.0)
        self.minChargeSpinBox.setValue(self.minCharge)
        self.minChargeSpinBox.valueChanged.connect(self.setMinCharge)
        self.contentLayout.addRow("Min charge", self.minChargeSpinBox)
        
        self.maxChargeSpinBox = QtGui.QDoubleSpinBox()
        self.maxChargeSpinBox.setSingleStep(0.1)
        self.maxChargeSpinBox.setMinimum(-999.0)
        self.maxChargeSpinBox.setMaximum(999.0)
        self.maxChargeSpinBox.setValue(self.maxCharge)
        self.maxChargeSpinBox.valueChanged.connect(self.setMaxCharge)
        self.contentLayout.addRow("Max charge", self.maxChargeSpinBox)
    
    def setMinCharge(self, val):
        """
        Set the minimum charge.
        
        """
        self.minCharge = val

    def setMaxCharge(self, val):
        """
        Set the maximum charge.
        
        """
        self.maxCharge = val
