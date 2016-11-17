
"""
The VTK Window

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import vtk
from PyQt5 import QtCore

from .QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
# from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
try:
    import vtk.qt
    vtk.qt.PyQtImpl = "PyQt5"
except:
    pass


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
    leftButtonPressed = QtCore.pyqtSignal(QtCore.QEvent)
    leftButtonReleased = QtCore.pyqtSignal(QtCore.QEvent)
    rightButtonPressed = QtCore.pyqtSignal(QtCore.QEvent)
    rightButtonReleased = QtCore.pyqtSignal(QtCore.QEvent)
    mouseMoved = QtCore.pyqtSignal(QtCore.QEvent)
    
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
    
    def mousePressEvent(self, ev):
        """Override mouse press event."""
        super(VTKWindow, self).mousePressEvent(ev)
        if self._ActiveButton == QtCore.Qt.LeftButton:
            self.leftButtonPressed.emit(ev)
        elif self._ActiveButton == QtCore.Qt.RightButton:
            self.rightButtonPressed.emit(ev)
    
    def mouseReleaseEvent(self, ev):
        """Override mouse release event as not working."""
        super(VTKWindow, self).mouseReleaseEvent(ev)
        if self._ActiveButton == QtCore.Qt.LeftButton:
            self.leftButtonReleased.emit(ev)
        elif self._ActiveButton == QtCore.Qt.RightButton:
            self.rightButtonReleased.emit(ev)
    
    def mouseMoveEvent(self, ev):
        """Override mouse move event."""
        super(VTKWindow, self).mouseMoveEvent(ev)
        self.mouseMoved.emit(ev)
