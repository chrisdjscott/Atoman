
"""
Contains GUI forms for the bond order filter.

"""
from PySide import QtGui, QtCore

from . import base


################################################################################

class BondOrderSettingsDialog(base.GenericSettingsDialog):
    """
    Settings for bond order filter
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(BondOrderSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Bond order"
        self.addProvidedScalar("Q4")
        self.addProvidedScalar("Q6")
        
        self.minQ4 = 0.0
        self.maxQ4 = 99.99
        self.minQ6 = 0.0
        self.maxQ6 = 99.99
        self.maxBondDistance = 4.0
        self.filterQ4Enabled = False
        self.filterQ6Enabled = False
        
        # max bond distance spin box
        self.maxBondDistanceSpin = QtGui.QDoubleSpinBox()
        self.maxBondDistanceSpin.setSingleStep(0.01)
        self.maxBondDistanceSpin.setMinimum(2.0)
        self.maxBondDistanceSpin.setMaximum(9.99)
        self.maxBondDistanceSpin.setValue(self.maxBondDistance)
        self.maxBondDistanceSpin.valueChanged[float].connect(self.setMaxBondDistance)
        self.maxBondDistanceSpin.setToolTip("This is used for spatially decomposing the system. "
                                            "This should be set large enough that the required neighbours will be included.")
        self.contentLayout.addRow("Max bond distance", self.maxBondDistanceSpin)
        
        self.addHorizontalDivider()
        
        # filter Q4
        filterQ4Check = QtGui.QCheckBox()
        filterQ4Check.setChecked(self.filterQ4Enabled)
        filterQ4Check.setToolTip("Filter atoms by Q4")
        filterQ4Check.stateChanged.connect(self.filterQ4Toggled)
        self.contentLayout.addRow("<b>Filter Q4</b>", filterQ4Check)
        
        self.minQ4Spin = QtGui.QDoubleSpinBox()
        self.minQ4Spin.setSingleStep(0.01)
        self.minQ4Spin.setMinimum(0.0)
        self.minQ4Spin.setMaximum(9999.99)
        self.minQ4Spin.setValue(self.minQ4)
        self.minQ4Spin.valueChanged[float].connect(self.setMinQ4)
        self.minQ4Spin.setEnabled(self.filterQ4Enabled)
        self.minQ4Spin.setToolTip("Minimum visible Q4 value")
        self.contentLayout.addRow("Minimum", self.minQ4Spin)
         
        self.maxQ4Spin = QtGui.QDoubleSpinBox()
        self.maxQ4Spin.setSingleStep(0.01)
        self.maxQ4Spin.setMinimum(0.0)
        self.maxQ4Spin.setMaximum(9999.99)
        self.maxQ4Spin.setValue(self.maxQ4)
        self.maxQ4Spin.valueChanged[float].connect(self.setMaxQ4)
        self.maxQ4Spin.setEnabled(self.filterQ4Enabled)
        self.maxQ4Spin.setToolTip("Maximum visible Q4 value")
        self.contentLayout.addRow("Maximum", self.maxQ4Spin)
        
        # filter Q6
        filterQ6Check = QtGui.QCheckBox()
        filterQ6Check.setChecked(self.filterQ6Enabled)
        filterQ6Check.setToolTip("Filter atoms by Q6")
        filterQ6Check.stateChanged.connect(self.filterQ6Toggled)
        self.contentLayout.addRow("<b>Filter Q6</b>", filterQ6Check)
        
        self.minQ6Spin = QtGui.QDoubleSpinBox()
        self.minQ6Spin.setSingleStep(0.01)
        self.minQ6Spin.setMinimum(0.0)
        self.minQ6Spin.setMaximum(9999.99)
        self.minQ6Spin.setValue(self.minQ6)
        self.minQ6Spin.valueChanged[float].connect(self.setMinQ6)
        self.minQ6Spin.setEnabled(self.filterQ6Enabled)
        self.minQ6Spin.setToolTip("Minimum visible Q6 value")
        self.contentLayout.addRow("Minimum", self.minQ6Spin)
         
        self.maxQ6Spin = QtGui.QDoubleSpinBox()
        self.maxQ6Spin.setSingleStep(0.01)
        self.maxQ6Spin.setMinimum(0.0)
        self.maxQ6Spin.setMaximum(9999.99)
        self.maxQ6Spin.setValue(self.maxQ6)
        self.maxQ6Spin.valueChanged[float].connect(self.setMaxQ6)
        self.maxQ6Spin.setEnabled(self.filterQ6Enabled)
        self.maxQ6Spin.setToolTip("Maximum visible Q6 value")
        self.contentLayout.addRow("Maximum", self.maxQ6Spin)
        
        self.addLinkToHelpPage("usage/analysis/filters/bond_order.html")
    
    def filterQ4Toggled(self, state):
        """
        Filter Q4 toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filterQ4Enabled = False
        
        else:
            self.filterQ4Enabled = True
        
        self.minQ4Spin.setEnabled(self.filterQ4Enabled)
        self.maxQ4Spin.setEnabled(self.filterQ4Enabled)
    
    def filterQ6Toggled(self, state):
        """
        Filter Q4 toggled
        
        """
        if state == QtCore.Qt.Unchecked:
            self.filterQ6Enabled = False
        
        else:
            self.filterQ6Enabled = True
        
        self.minQ6Spin.setEnabled(self.filterQ6Enabled)
        self.maxQ6Spin.setEnabled(self.filterQ6Enabled)
    
    def setMaxBondDistance(self, val):
        """
        Set the max bond distance
        
        """
        self.maxBondDistance = val
    
    def setMinQ4(self, val):
        """
        Set the minimum value for Q4
        
        """
        self.minQ4 = val

    def setMaxQ4(self, val):
        """
        Set the maximum value for Q4
        
        """
        self.maxQ4 = val
    
    def setMinQ6(self, val):
        """
        Set the minimum value for Q6
        
        """
        self.minQ6 = val

    def setMaxQ6(self, val):
        """
        Set the maximum value for Q6
        
        """
        self.maxQ6 = val
