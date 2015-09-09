
"""
Contains GUI forms for the slip filter.

"""
from . import base
from ...filtering.filters import slipFilter


################################################################################

class SlipSettingsDialog(base.GenericSettingsDialog):
    """
    Settings form for the slip filter.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(SlipSettingsDialog, self).__init__(title, parent, "Slip")
        
        self.addProvidedScalar("Slip")
        
        # settings
        self._settings = slipFilter.SlipFilterSettings()
        
        # neighbour cut-off spin box
        toolTip = "<FONT COLOR=black>When comparing the relative displacements of neighbours, consider atoms to be neighbours "
        toolTip += "if their separation in the reference lattice was less than this value.</FONT>"
        self.addDoubleSpinBox("neighbourCutOff", label="Neighbour cut-off", minVal=2.0, maxVal=9.99, step=0.1, toolTip=toolTip)
        
        toolTip = "<FONT COLOR=black>When calculating the slip of an atom relative to one of its neighbours, ignore the slip "
        toolTip += "constribution from the neighbour if the magnitude is less than this value.</FONT>"
        self.addDoubleSpinBox("slipTolerance", label="Slip tolerance", minVal=0.0, maxVal=9.99, step=0.1, toolTip=toolTip)
        
        self.addHorizontalDivider()
        
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
