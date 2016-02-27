
"""
Contains GUI forms for the bond order filter.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from . import base
from ...filtering.filters import bondOrderFilter


################################################################################

class BondOrderSettingsDialog(base.GenericSettingsDialog):
    """
    Settings for bond order filter
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(BondOrderSettingsDialog, self).__init__(title, parent, "Bond order")
        
        self.addProvidedScalar("Q4")
        self.addProvidedScalar("Q6")
        
        self._settings = bondOrderFilter.BondOrderFilterSettings()
        
        # max bond distance spin box
        toolTip = "This is used for spatially decomposing the system. "
        toolTip += "This should be set large enough that the required neighbours will be included."
        self.addDoubleSpinBox("maxBondDistance", label="Max bond distance", minVal=2.0, maxVal=9.99, step=0.1, toolTip=toolTip)
        
        self.addHorizontalDivider()
        
        # filter Q4 check box
        self.addCheckBox("filterQ4Enabled", toolTip="Filter atoms by Q4", label="<b>Filter Q4</b>", extraSlot=self.filterQ4Toggled)
        
        # filter Q4 spin boxes
        self.minQ4Spin = self.addDoubleSpinBox("minQ4", label="Minimum", minVal=0, maxVal=9999, step=0.1,
                                               toolTip="Minimum visible Q4 value", settingEnabled="filterQ4Enabled")
        self.maxQ4Spin = self.addDoubleSpinBox("maxQ4", label="Maximum", minVal=0, maxVal=9999, step=0.1,
                                               toolTip="Maximum visible Q4 value", settingEnabled="filterQ4Enabled")
        
        # filter Q6 check box
        self.addCheckBox("filterQ6Enabled", toolTip="Filter atoms by Q6", label="<b>Filter Q6</b>", extraSlot=self.filterQ6Toggled)
        
        # filter Q6 spin boxes
        self.minQ6Spin = self.addDoubleSpinBox("minQ6", label="Minimum", minVal=0, maxVal=9999, step=0.1,
                                               toolTip="Minimum visible Q6 value", settingEnabled="filterQ6Enabled")
        self.maxQ6Spin = self.addDoubleSpinBox("maxQ6", label="Maximum", minVal=0, maxVal=9999, step=0.1,
                                               toolTip="Maximum visible Q6 value", settingEnabled="filterQ6Enabled")
        
        self.addLinkToHelpPage("usage/analysis/filters/bond_order.html")
    
    def filterQ4Toggled(self, enabled):
        """
        Filter Q4 toggled
        
        """
        self.minQ4Spin.setEnabled(enabled)
        self.maxQ4Spin.setEnabled(enabled)
    
    def filterQ6Toggled(self, enabled):
        """
        Filter Q6 toggled
        
        """
        self.minQ6Spin.setEnabled(enabled)
        self.maxQ6Spin.setEnabled(enabled)
