
"""
Contains GUI forms for the slip filter.

"""
from PySide import QtGui, QtCore

from . import base


################################################################################

class SlipSettingsDialog(base.GenericSettingsDialog):
    """
    Settings form for the slip filter.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(SlipSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Slip"
        self.addProvidedScalar("Slip")
        
        self.minSlip = 0.0
        self.maxSlip = 9999.0
        self.filteringEnabled = False
        
        # filtering options
        filterCheck = QtGui.QCheckBox()
        filterCheck.setChecked(self.filteringEnabled)
        filterCheck.setToolTip("Filter atoms by slip")
        filterCheck.stateChanged.connect(self.filteringToggled)
        self.contentLayout.addRow("<b>Enable filtering</b>", filterCheck)
        
        self.minSlipSpin = QtGui.QDoubleSpinBox()
        self.minSlipSpin.setSingleStep(0.1)
        self.minSlipSpin.setMinimum(0.0)
        self.minSlipSpin.setMaximum(9999.0)
        self.minSlipSpin.setValue(self.minSlip)
        self.minSlipSpin.valueChanged.connect(self.setMinSlip)
        self.minSlipSpin.setEnabled(False)
        self.contentLayout.addRow("Min", self.minSlipSpin)
        
        self.maxSlipSpin = QtGui.QDoubleSpinBox()
        self.maxSlipSpin.setSingleStep(0.1)
        self.maxSlipSpin.setMinimum(0.0)
        self.maxSlipSpin.setMaximum(9999.0)
        self.maxSlipSpin.setValue(self.maxSlip)
        self.maxSlipSpin.valueChanged.connect(self.setMaxSlip)
        self.maxSlipSpin.setEnabled(False)
        self.contentLayout.addRow("Max", self.maxSlipSpin)
    
    def filteringToggled(self, state):
        """
        Filtering toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filteringEnabled = False
            
            self.minSlipSpin.setEnabled(False)
            self.maxSlipSpin.setEnabled(False)
        
        else:
            self.filteringEnabled = True
            
            self.minSlipSpin.setEnabled(True)
            self.maxSlipSpin.setEnabled(True)
    
    def setMinSlip(self, val):
        """
        Set the minimum slip.
        
        """
        self.minSlip = val

    def setMaxSlip(self, val):
        """
        Set the maximum slip.
        
        """
        self.maxSlip = val
