
"""
Contains GUI forms for the charge filter.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import functools

from PyQt5 import QtWidgets


from . import base
from ...filtering.filters import chargeFilter


################################################################################

class ChargeSettingsDialog(base.GenericSettingsDialog):
    """
    GUI form for the charge filter.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(ChargeSettingsDialog, self).__init__(title, parent, "Charge")
        
        self._settings = chargeFilter.ChargeFilterSettings()
        
        self.minChargeSpinBox = QtWidgets.QDoubleSpinBox()
        self.minChargeSpinBox.setSingleStep(0.1)
        self.minChargeSpinBox.setMinimum(-999.0)
        self.minChargeSpinBox.setMaximum(999.0)
        self.minChargeSpinBox.setValue(self._settings.getSetting("minCharge"))
        self.minChargeSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "minCharge"))
        self.contentLayout.addRow("Min charge", self.minChargeSpinBox)
        
        self.maxChargeSpinBox = QtWidgets.QDoubleSpinBox()
        self.maxChargeSpinBox.setSingleStep(0.1)
        self.maxChargeSpinBox.setMinimum(-999.0)
        self.maxChargeSpinBox.setMaximum(999.0)
        self.maxChargeSpinBox.setValue(self._settings.getSetting("maxCharge"))
        self.maxChargeSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "maxCharge"))
        self.contentLayout.addRow("Max charge", self.maxChargeSpinBox)
