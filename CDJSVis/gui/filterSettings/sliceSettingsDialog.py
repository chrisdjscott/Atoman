
"""
Contains GUI forms for the slice filter.

"""
from PySide import QtGui, QtCore

from . import base
from ...rendering import slicePlane
from ...filtering.filters import sliceFilter


################################################################################

class SliceSettingsDialog(base.GenericSettingsDialog):
    """
    Slice filter settings form.
    
    """
    def __init__(self, mainWindow, title, parent=None):
        super(SliceSettingsDialog, self).__init__(title, parent, "Slice")
        
        # slice plane
        self.slicePlane = slicePlane.SlicePlane(self.pipelinePage)
        
        # settings
        self._settings = sliceFilter.SliceFilterSettings()
        
        # defaults
        lattice = self.pipelinePage.inputState
        self._settings.updateSetting("x0", lattice.cellDims[0] / 2.0)
        self._settings.updateSetting("y0", lattice.cellDims[1] / 2.0)
        self._settings.updateSetting("z0", lattice.cellDims[2] / 2.0)
        self.showSlicePlaneChecked = False
        
        # show slice plane
        self.showSlicePlaneCheck = QtGui.QCheckBox()
        self.showSlicePlaneCheck.stateChanged.connect(self.showPlaneChanged)
        self.showSlicePlaneCheck.setToolTip("Show the slice plane as a visual aid")
        self.contentLayout.addRow("Show slice plane", self.showSlicePlaneCheck)
        
        self.addHorizontalDivider()
        
        # plane centre
        x0SpinBox = QtGui.QDoubleSpinBox()
        x0SpinBox.setSingleStep(1)
        x0SpinBox.setMinimum(-1000)
        x0SpinBox.setMaximum(1000)
        x0SpinBox.setValue(self._settings.getSetting("x0"))
        x0SpinBox.setToolTip("Plane centre x value")
        x0SpinBox.valueChanged.connect(self.x0Changed)
        
        y0SpinBox = QtGui.QDoubleSpinBox()
        y0SpinBox.setSingleStep(1)
        y0SpinBox.setMinimum(-1000)
        y0SpinBox.setMaximum(1000)
        y0SpinBox.setValue(self._settings.getSetting("y0"))
        y0SpinBox.setToolTip("Plane centre y value")
        y0SpinBox.valueChanged.connect(self.y0Changed)
        
        z0SpinBox = QtGui.QDoubleSpinBox()
        z0SpinBox.setSingleStep(1)
        z0SpinBox.setMinimum(-1000)
        z0SpinBox.setMaximum(1000)
        z0SpinBox.setValue(self._settings.getSetting("z0"))
        z0SpinBox.setToolTip("Plane centre z value")
        z0SpinBox.valueChanged.connect(self.z0Changed)
        
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(x0SpinBox)
        hbox.addWidget(QtGui.QLabel(","))
        hbox.addWidget(y0SpinBox)
        hbox.addWidget(QtGui.QLabel(","))
        hbox.addWidget(z0SpinBox)
        self.contentLayout.addRow("Place centre", hbox)
        
        # plane normal 
        xnSpinBox = QtGui.QDoubleSpinBox()
        xnSpinBox.setSingleStep(0.1)
        xnSpinBox.setMinimum(-1000)
        xnSpinBox.setMaximum(1000)
        xnSpinBox.setValue(self._settings.getSetting("xn"))
        xnSpinBox.setToolTip("Plane normal x value")
        xnSpinBox.valueChanged.connect(self.xnChanged)
        
        ynSpinBox = QtGui.QDoubleSpinBox()
        ynSpinBox.setSingleStep(0.1)
        ynSpinBox.setMinimum(-1000)
        ynSpinBox.setMaximum(1000)
        ynSpinBox.setValue(self._settings.getSetting("yn"))
        ynSpinBox.setToolTip("Plane normal y value")
        ynSpinBox.valueChanged.connect(self.ynChanged)
        
        znSpinBox = QtGui.QDoubleSpinBox()
        znSpinBox.setSingleStep(0.1)
        znSpinBox.setMinimum(-1000)
        znSpinBox.setMaximum(1000)
        znSpinBox.setValue(self._settings.getSetting("zn"))
        znSpinBox.setToolTip("Plane normal z value")
        znSpinBox.valueChanged.connect(self.znChanged)
        
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(xnSpinBox)
        hbox.addWidget(QtGui.QLabel(","))
        hbox.addWidget(ynSpinBox)
        hbox.addWidget(QtGui.QLabel(","))
        hbox.addWidget(znSpinBox)
        self.contentLayout.addRow("Place normal", hbox)
        
        self.addHorizontalDivider()
        
        # invert
        self.invertCheck = QtGui.QCheckBox()
        self.invertCheck.stateChanged.connect(self.changeInvert)
        self.invertCheck.setToolTip("Invert the selection of atoms")
        self.contentLayout.addRow("Invert selection", self.invertCheck)
    
    def refresh(self):
        """
        Called whenever new input is loaded.
        
        """
        # need to change min/max of sliders for x0,y0,z0
        pass
    
    def showPlaneChanged(self, state):
        """
        Show slice plane.
        
        """
        if self.showSlicePlaneCheck.isChecked():
            self.showSlicePlaneChecked = True
            self.showSlicePlane()
        else:
            self.showSlicePlaneChecked = False
            self.hideSlicePlane()
    
    def changeInvert(self, state):
        """
        Change invert.
        
        """
        checked = False if state == QtCore.Qt.Unchecked else True
        self._settings.updateSetting("invert", checked)
    
    def x0Changed(self, val):
        """
        x0 changed.
        
        """
        self._settings.updateSetting("x0", val)
        self.showSlicePlane()
    
    def y0Changed(self, val):
        """
        y0 changed.
        
        """
        self._settings.updateSetting("y0", val)
        self.showSlicePlane()
    
    def z0Changed(self, val):
        """
        z0 changed.
        
        """
        self._settings.updateSetting("z0", val)
        self.showSlicePlane()
    
    def xnChanged(self, val):
        """
        xn changed.
        
        """
        self._settings.updateSetting("xn", val)
        self.showSlicePlane()
    
    def ynChanged(self, val):
        """
        yn changed.
        
        """
        self._settings.updateSetting("yn", val)
        self.showSlicePlane()
    
    def znChanged(self, val):
        """
        zn changed.
        
        """
        self._settings.updateSetting("zn", val)
        self.showSlicePlane()
    
    def showSlicePlane(self):
        """
        Update position of slice plane.
        
        """
        if not self.showSlicePlaneChecked:
            return
        
        # first remove it is already shown
        
        
        # args to pass
        p = (self._settings.getSetting("x0"), self._settings.getSetting("y0"), self._settings.getSetting("z0"))
        n = (self._settings.getSetting("xn"), self._settings.getSetting("yn"), self._settings.getSetting("zn"))
        
        # update actor
        self.slicePlane.update(p, n)
        
        # broadcast to renderers
        self.parent.filterTab.broadcastToRenderers("showSlicePlane", args=(self.slicePlane,))
    
    def hideSlicePlane(self):
        """
        Hide the slice plane.
        
        """
        # broadcast to renderers
        self.parent.filterTab.broadcastToRenderers("removeSlicePlane", globalBcast=True)
    
    def closeEvent(self, event):
        """
        Override closeEvent.
        
        """
        if self.showSlicePlaneChecked:
            self.showSlicePlaneCheck.setCheckState(QtCore.Qt.Unchecked)
            self.showSlicePlaneChecked = False
        
        self.hide()
