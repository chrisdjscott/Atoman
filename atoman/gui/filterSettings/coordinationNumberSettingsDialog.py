
"""
Contains GUI forms for the coordination number filter.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import functools

from PySide2 import QtCore, QtWidgets


from . import base
from ...filtering.filters import coordinationNumberFilter


################################################################################

class CoordinationNumberSettingsDialog(base.GenericSettingsDialog):
    """
    Coordination number settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(CoordinationNumberSettingsDialog, self).__init__(title, parent, "Coordination number")
        
        self.setMinimumWidth(350)
        
        self.addProvidedScalar("Coordination number")
        
        self._settings = coordinationNumberFilter.CoordinationNumberFilterSettings()
        
        # filter check
        filterCheck = QtWidgets.QCheckBox()
        filterCheck.setChecked(self._settings.getSetting("filteringEnabled"))
        filterCheck.setToolTip("Filter by coordination number")
        filterCheck.stateChanged.connect(self.filteringToggled)
        self.contentLayout.addRow("<b>Filter by coordination</b>", filterCheck)
        
        self.minCoordNumSpinBox = QtWidgets.QSpinBox()
        self.minCoordNumSpinBox.setSingleStep(1)
        self.minCoordNumSpinBox.setMinimum(0)
        self.minCoordNumSpinBox.setMaximum(999)
        self.minCoordNumSpinBox.setValue(self._settings.getSetting("minCoordNum"))
        self.minCoordNumSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "minCoordNum"))
        self.minCoordNumSpinBox.setToolTip("Minimum visible coordination number")
        self.minCoordNumSpinBox.setEnabled(self._settings.getSetting("filteringEnabled"))
        self.contentLayout.addRow("Minimum", self.minCoordNumSpinBox)
        
        self.maxCoordNumSpinBox = QtWidgets.QSpinBox()
        self.maxCoordNumSpinBox.setSingleStep(1)
        self.maxCoordNumSpinBox.setMinimum(0)
        self.maxCoordNumSpinBox.setMaximum(999)
        self.maxCoordNumSpinBox.setValue(self._settings.getSetting("maxCoordNum"))
        self.maxCoordNumSpinBox.valueChanged.connect(functools.partial(self._settings.updateSetting, "maxCoordNum"))
        self.maxCoordNumSpinBox.setToolTip("Maximum visible coordination number")
        self.maxCoordNumSpinBox.setEnabled(self._settings.getSetting("filteringEnabled"))
        self.contentLayout.addRow("Maximum", self.maxCoordNumSpinBox)
    
    def filteringToggled(self, state):
        """
        Filtering toggled
        
        """
        enabled = False if state == QtCore.Qt.Unchecked else True
        self._settings.updateSetting("filteringEnabled", enabled)
        self.minCoordNumSpinBox.setEnabled(enabled)
        self.maxCoordNumSpinBox.setEnabled(enabled)
