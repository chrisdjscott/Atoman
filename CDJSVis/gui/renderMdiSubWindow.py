
"""
Mdi sub window for displaying VTK render window.

@author: Chris Scott

"""
import os
import sys
import shutil
import platform
import tempfile

from PyQt4 import QtGui, QtCore
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import numpy as np

from ..visutils.utilities import iconPath, resourcePath
from . import dialogs
from ..visclibs import picker_c
from ..rendering import renderer
try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class RendererWindow(QtGui.QWidget):
    """
    Renderer sub window.
    
    Holds VTK render window.
    
    """
    def __init__(self, mainWindow, index, parent=None):
        super(RendererWindow, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.setWindowTitle("Render window %d" % index)
        
        # layout
        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # toolbar
        toolbar = QtGui.QToolBar()
        layout.addWidget(toolbar)
        
        # button to displace lattice frame
        showCellAction = self.createAction("Toggle cell", slot=self.toggleCellFrame, icon="cell_icon.svg", 
                                           tip="Toggle cell frame visibility")
        
        # button to display axes
        showAxesAction = self.createAction("Toggle axes", slot=self.toggleAxes, icon="axis_icon2.svg", 
                                           tip="Toggle axes visiblity")
        
        # reset camera to cell
        setCamToCellAction = self.createAction("Reset to cell", slot=self.setCameraToCell, icon="set_cam_cell.svg", 
                                           tip="Reset camera to cell")
        
        self.addActions(toolbar, (showCellAction, showAxesAction, setCamToCellAction))
        
        # VTK render window
        self.vtkRenWin = vtk.vtkRenderWindow()
        
        # VTK interactor
        self.vtkRenWinInteract = QVTKRenderWindowInteractor(self, rw=self.vtkRenWin)
        
        
        # interactor style
        self.vtkRenWinInteract._Iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # add observers
        self.vtkRenWinInteract._Iren.AddObserver("LeftButtonPressEvent", self.leftButtonPressed)
        self.vtkRenWinInteract._Iren.AddObserver("MouseMoveEvent", self.mouseMoved)
        self.vtkRenWinInteract._Iren.AddObserver("LeftButtonReleaseEvent", self.leftButtonReleased)
        
        # add picker
        self.vtkPicker = vtk.vtkCellPicker()
        self.vtkPicker.SetTolerance(0.000001)
        self.vtkPicker.AddObserver("EndPickEvent", self.endPickEvent)
        self.vtkRenWinInteract.SetPicker(self.vtkPicker)
        
        # vtk renderer
        self.vtkRen = vtk.vtkRenderer()
        self.vtkRen.SetBackground(1, 1, 1)
        
        self.vtkRenWin.AddRenderer(self.vtkRen)
        
#        self.vtkRenWinInteract.Initialize()
        self.vtkRenWinInteract.Start()
        
        layout.addWidget(self.vtkRenWinInteract)
        
        # renderer
        self.renderer = renderer.Renderer(self)
        
        # which filter list is it associated with
        label = QtGui.QLabel("Analysis pipeline:")
        self.analysisPipelineCombo = QtGui.QComboBox()
        
        row = QtGui.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.addWidget(label)
        row.addWidget(self.analysisPipelineCombo)
        
        layout.addLayout(row)
        
        
        
    
    def endPickEvent(self, obj, event):
        """
        End of vtk pick event.
        
        """
        if self.vtkPicker.GetCellId() < 0:
            pass
        
        else:
            pickPos = self.vtkPicker.GetPickPosition()
            pickPos_np = np.asarray(pickPos, dtype=np.float64)
            
            # find which atom was picked...
            
            # loop over filter lists
            filterLists = self.mainWindow.mainToolbar.filterPage.filterLists
            
            minSepIndex = -1
            minSep = 9999999.0
            minSepType = None
            minSepScalarType = None
            minSepScalar = None
            for filterList in filterLists:
                filterer = filterList.filterer
                
                visibleAtoms = filterer.visibleAtoms
                interstitials = filterer.interstitials
                vacancies = filterer.vacancies
                antisites = filterer.antisites
                onAntisites = filterer.onAntisites
                splitInts = filterer.splitInterstitials
                scalars = filterer.scalars
                scalarsType = filterer.scalarsType
                
                result = np.empty(3, np.float64)
                
                status = picker_c.pickObject(visibleAtoms, vacancies, interstitials, antisites, splitInts, pickPos_np, 
                                             self.mainWindow.inputState.pos, self.mainWindow.refState.pos, self.mainWindow.PBC, self.mainWindow.inputState.cellDims,
                                             self.mainWindow.refState.minPos, self.mainWindow.refState.maxPos, self.mainWindow.inputState.specie, 
                                             self.mainWindow.refState.specie, self.mainWindow.inputState.specieCovalentRadius, 
                                             self.mainWindow.refState.specieCovalentRadius, result)
                
                tmp_type, tmp_index, tmp_sep = result
                
                if tmp_index >= 0 and tmp_sep < minSep:
                    minSep = tmp_sep
                    minSepType = int(tmp_type)
                    
                    if minSepType == 0:
                        minSepIndex = visibleAtoms[int(tmp_index)]
                    else:
                        minSepIndex = int(tmp_index)
                        
                        if minSepType == 1:
                            defList = (vacancies,)
                        elif minSepType == 2:
                            defList = (interstitials,)
                        elif minSepType == 3:
                            defList = (antisites, onAntisites)
                        else:
                            defList = (splitInts,)
                    
                    if len(scalarsType):
                        minSepScalar = scalars[tmp_index]
                        minSepScalarType = scalarsType
                    else:
                        minSepScalar = None
                        minSepScalarType = None
            
#            print "MIN SEP", minSep, "TYPE", minSepType, "INDEX", minSepIndex
            
            if minSep < 0.1:
                if minSepType == 0:
                    atomInfoWindow = dialogs.AtomInfoWindow(self, minSepIndex, minSepScalar, minSepScalarType, parent=self)
                    atomInfoWindow.show()
                
                else:
                    defectInfoWindow = dialogs.DefectInfoWindow(self, minSepIndex, minSepType, defList, parent=self)
                    defectInfoWindow.show()
    
    def leftButtonPressed(self, obj, event):
        """
        Left mouse button pressed
        
        """
        self.mouseMotion = 0
        
        # left release event isn't working so have to pick by double click
        if self.vtkRenWinInteract.GetRepeatCount() == 1:
            pos = self.vtkRenWinInteract.GetEventPosition()
            self.vtkPicker.Pick(pos[0], pos[1], 0, self.vtkRen)
    
    def mouseMoved(self, obj, event):
        """
        Mouse moved
        
        """
        self.mouseMotion = 1
    
    def leftButtonReleased(self, obj, event):
        """
        Left button released.
        
        """
        print "LEFT RELEASE", self.mouseMotion
    
    def setCameraToCell(self):
        """
        Reset the camera to point at the cell
        
        """
        self.renderer.setCameraToCell()
    
    def toggleCellFrame(self):
        """
        Toggle lattice frame visibility
        
        """
        self.renderer.toggleLatticeFrame()
    
    def toggleAxes(self):
        """
        Toggle axes visibility
        
        """
        self.renderer.toggleAxes()
    
    def createAction(self, text, slot=None, shortcut=None, icon=None,
                     tip=None, checkable=False, signal="triggered()"):
        """
        Create an action
        
        """
        action = QtGui.QAction(text, self)
        
        if icon is not None:
            action.setIcon(QtGui.QIcon(iconPath(icon)))
        
        if shortcut is not None:
            action.setShortcut(shortcut)
        
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        
        if slot is not None:
            self.connect(action, QtCore.SIGNAL(signal), slot)
        
        if checkable:
            action.setCheckable(True)
        
        return action
    
    def addActions(self, target, actions):
        """
        Add a tuple of actions to the target.
        
        """
        for action in actions:
            if action is None:
                target.addSeparator()
            
            else:
                target.addAction(action)





