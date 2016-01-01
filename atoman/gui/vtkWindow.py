
"""
The VTK Window

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import vtk
from PySide import QtCore

from .QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk.qt
vtk.qt.PyQtImpl = "PySide"


class VTKRenWinInteractOverride(vtk.vtkGenericRenderWindowInteractor):
    """
    Customised interactor to get rid of OpenGL warnings.
    
    """
    def Initialize(self):
        self.Initialized = 1
        self.Enable()


class VTKWindow(QVTKRenderWindowInteractor):
    """
    The VTK window
    
    """
    leftButtonReleased = QtCore.Signal(QtCore.QEvent)
    rightButtonReleased = QtCore.Signal(QtCore.QEvent)
    
    def __init__(self, parent=None, wflags=QtCore.Qt.WindowFlags(), **kw):
        super(VTKWindow, self).__init__(parent=parent, wflags=wflags, **kw)
        
        # disable mouse wheel option
        try:
            self._disableMouseWheel = kw["disable_mouse_wheel"]
        except KeyError:
            self._disableMouseWheel = False
    
    def changeDisableMouseWheel(self, disableMouseWheel):
        """Enable/disable mouse wheel event"""
        self._disableMouseWheel = disableMouseWheel
    
    def wheelEvent(self, ev):
        """Override mouse wheel event, disabling if required."""
        if not self._disableMouseWheel:
            super(VTKWindow, self).wheelEvent(ev)
    
    def mouseReleaseEvent(self, ev):
        """Override mouse release event as not working."""
        super(VTKWindow, self).mouseReleaseEvent(ev)
        if self._ActiveButton == QtCore.Qt.LeftButton:
            self.leftButtonReleased.emit(ev)
        elif self._ActiveButton == QtCore.Qt.RightButton:
            self.rightButtonReleased.emit(ev)
