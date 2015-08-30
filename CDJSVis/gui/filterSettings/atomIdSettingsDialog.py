
"""
Contains GUI forms for the atom ID filter.

"""
from PySide import QtGui, QtCore

from . import base


################################################################################

class AtomIdSettingsDialog(base.GenericSettingsDialog):
    """
    Atom ID filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(AtomIdSettingsDialog, self).__init__(title, parent)
        
        self.filterType = "Atom ID"
        
        # only allow numbers, commas and hyphens
        rx = QtCore.QRegExp("[0-9]+(?:[-,]?[0-9]+)*")
        validator = QtGui.QRegExpValidator(rx, self)
        
        self.lineEdit = QtGui.QLineEdit()
        self.lineEdit.setValidator(validator)
        self.lineEdit.setToolTip("Comma separated list of atom IDs or ranges of atom IDs (hyphenated) that are visible (eg. '22,30-33' will show atom IDs 22, 30, 31, 32 and 33)")
        self.contentLayout.addRow("Visible IDs", self.lineEdit)
