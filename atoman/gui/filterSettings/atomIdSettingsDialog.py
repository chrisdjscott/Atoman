
"""
Contains GUI forms for the atom ID filter.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from PySide2 import QtGui, QtCore, QtWidgets


from . import base
from ...filtering.filters import atomIdFilter


################################################################################

class AtomIdSettingsDialog(base.GenericSettingsDialog):
    """
    Atom ID filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(AtomIdSettingsDialog, self).__init__(title, parent, "Atom ID")
        
        self._settings = atomIdFilter.AtomIdFilterSettings()
        
        # only allow numbers, commas and hyphens
        rx = QtCore.QRegExp("[0-9]+(?:[-,]?[0-9]+)*")
        validator = QtGui.QRegExpValidator(rx, self)
        
        self.lineEdit = QtWidgets.QLineEdit(self._settings.getSetting("filterString"))
        self.lineEdit.setValidator(validator)
        self.lineEdit.setToolTip("Comma separated list of atom IDs or ranges of atom IDs (hyphenated) that are visible (eg. '22,30-33' will show atom IDs 22, 30, 31, 32 and 33)")
        self.lineEdit.editingFinished.connect(self.editingFinished)
        self.contentLayout.addRow("Visible IDs", self.lineEdit)
    
    def editingFinished(self):
        """Line edit has been editted."""
        self.logger.debug("Editing finished: '%s'", self.lineEdit.text())
        self._settings.updateSetting("filterString", str(self.lineEdit.text()))
