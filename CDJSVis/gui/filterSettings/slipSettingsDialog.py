
"""
Contains GUI forms for the slip filter.

"""
from PySide import QtGui, QtCore

from . import base
from ...filtering.filters import slipFilter


################################################################################

class SlipSettingsDialog(base.GenericSettingsDialog):
    """
    Settings form for the slip filter.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(SlipSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Slip"
        self.addProvidedScalar("Slip")
        
        # settings
        self._settings = slipFilter.SlipFilterSettings()
        
        # filtering options
        self.addCheckBox("filteringEnabled", toolTip="Filter atoms by slip", label="<b>Enable filtering</b>", extraSlot=self.filteringToggled)
        
        self.minSlipSpin = self.addDoubleSpinBox("minSlip", minVal=0, maxVal=9999, step=0.1, toolTip="Minimum visible slip",
                                                 label="Minimum", settingEnabled="filteringEnabled")
        
        self.maxSlipSpin = self.addDoubleSpinBox("maxSlip", minVal=0, maxVal=9999, step=0.1, toolTip="Maximum visible slip",
                                                 label="Maximum", settingEnabled="filteringEnabled")
    
    def filteringToggled(self, enabled):
        """
        Filtering toggled
        
        """
        self.minSlipSpin.setEnabled(enabled)
        self.maxSlipSpin.setEnabled(enabled)
