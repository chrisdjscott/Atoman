
"""
Mdi sub window for displaying VTK render window.

@author: Chris Scott

"""
import os
import sys

from PyQt4 import QtGui, QtCore
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import numpy as np

from ..visutils.utilities import iconPath, resourcePath
from ..visutils import utilities
from . import dialogs
from ..visclibs import picker_c
from ..rendering import renderer
from .outputDialog import OutputDialog
from ..rendering.text import vtkRenderWindowText
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
        self.rendererIndex = index
        
        self.setWindowTitle("Render window %d" % index)
        
        self.closed = False
        
        # layout
        layout = QtGui.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        self.currentPipelineIndex = 0
        self.currentPipelineString = "Pipeline 0"
        self.onScreenInfoActors = vtk.vtkActor2DCollection()
        
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
        
        # text selector
        openTextSelectorAction = self.createAction("On-screen info", self.showTextSelector, 
                                                   icon="preferences-desktop-font.svg", 
                                                   tip="Show on-screen text selector")
        
        # output dialog
        showOutputDialogAction = self.createAction("Output dialog", slot=self.showOutputDialog, icon="loadandsave-icon.svg",
                                                   tip="Open output dialog")
        
        self.addActions(toolbar, (showCellAction, showAxesAction, setCamToCellAction, None, 
                                  openTextSelectorAction, None, showOutputDialogAction))
        
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
        
        # do a post ref render if the ref is already loaded
        if self.mainWindow.refLoaded:
            self.renderer.postRefRender()
        
        # output dialog
        self.outputDialog = OutputDialog(self, self.mainWindow, None, index)
        
        # text selector
        self.textSelector = dialogs.OnScreenInfoDialog(self.mainWindow, index, parent=self)
        
        # which filter list is it associated with
        label = QtGui.QLabel("Analysis pipeline:")
        self.analysisPipelineCombo = QtGui.QComboBox()
        self.analysisPipelineCombo.currentIndexChanged.connect(self.pipelineChanged)
        self.initPipelines()
        
        row = QtGui.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.addWidget(label)
        row.addWidget(self.analysisPipelineCombo)
        
        layout.addLayout(row)
    
    def initPipelines(self):
        """
        Initialise pipeline combo.
        
        """
        if not hasattr(self.mainWindow, "mainToolbar"):
            return
        
        combo = self.mainWindow.mainToolbar.pipelineCombo
        
        for i in xrange(combo.count()):
            self.newPipeline(str(combo.itemText(i)))
    
    def getCurrentPipelinePage(self):
        """
        Return current pipeline page.
        
        """
        return self.mainWindow.mainToolbar.pipelineList[self.currentPipelineIndex]
    
    def getFilterLists(self):
        """
        Returns the filter lists for given pipeline.
        
        """
        if hasattr(self.mainWindow, "mainToolbar"):
            filterLists = self.getCurrentPipelinePage().filterLists
        else:
            filterLists = []
        
        return filterLists
    
    def gatherVisibleAtoms(self):
        """
        Gather visible atoms array.
        
        """
        return self.getCurrentPipelinePage().gatherVisibleAtoms()
    
    def removeActors(self):
        """
        Remove current actors.
        
        """
        filterLists = self.getFilterLists()
        
        for filterList in filterLists:
            filterer = filterList.filterer
            actorsCollection = filterer.actorsCollection
            
            actorsCollection.InitTraversal()
            actor = actorsCollection.GetNextItem()
            while actor is not None:
                try:
                    self.vtkRen.RemoveActor(actor)
                except:
                    pass
                
                actor = actorsCollection.GetNextItem()
            
            if filterer.scalarBarAdded:
                self.vtkRen.RemoveActor2D(filterer.scalarBar)
            
            self.vtkRenWinInteract.ReInitialize()
    
    def addActors(self):
        """
        Add current actors.
        
        """
        filterLists = self.getFilterLists()
        
        for filterList in filterLists:
            filterer = filterList.filterer
            actorsCollection = filterer.actorsCollection
            
            actorsCollection.InitTraversal()
            actor = actorsCollection.GetNextItem()
            while actor is not None:
                try:
                    self.vtkRen.AddActor(actor)
                except:
                    pass
                
                actor = actorsCollection.GetNextItem()
            
            if filterer.scalarBarAdded:
                self.vtkRen.AddActor2D(filterer.scalarBar)
            
            self.vtkRenWinInteract.ReInitialize()
    
    def pipelineChanged(self, index):
        """
        Current pipeline changed.
        
        """
        # remove actors
        self.removeActors()
        
        # update vars
        self.currentPipelineString = str(self.analysisPipelineCombo.currentText())
        self.currentPipelineIndex = index
        
        # get new actors
        self.addActors()
        
        # refresh text
        self.refreshOnScreenInfo()
    
    def showOutputDialog(self):
        """
        Show output dialog.
        
        """
        if not self.mainWindow.refLoaded:
            return
        
        self.outputDialog.hide()
        self.outputDialog.show()
    
    def removePipeline(self, index):
        """
        Remove given pipeline.
        
        """
        self.analysisPipelineCombo.removeItem(index)
        
        # update index and string
        self.currentPipelineString = str(self.analysisPipelineCombo.currentText())
        self.currentPipelineIndex = self.analysisPipelineCombo.currentIndex()
        
    
    def newPipeline(self, name):
        """
        Add new pipeline to the combo.
        
        """
        self.analysisPipelineCombo.addItem(name)
    
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
            filterLists = self.getFilterLists()
            
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
            
            print "MIN SEP", minSep, "TYPE", minSepType, "INDEX", minSepIndex
            
            if minSep < 0.1:
                if minSepType == 0:
                    atomInfoWindow = dialogs.AtomInfoWindow(self.mainWindow, minSepIndex, minSepScalar, minSepScalarType, parent=self)
                    atomInfoWindow.show()
                
                else:
                    defectInfoWindow = dialogs.DefectInfoWindow(self.mainWindow, minSepIndex, minSepType, defList, parent=self)
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
    
    def showTextSelector(self):
        """
        Show the text selector.
        
        """
        if not self.mainWindow.refLoaded:
            return
        
        self.textSelector.hide()
        self.textSelector.show()
    
    def removeOnScreenInfo(self):
        """
        Remove on screen info.
        
        """
        self.onScreenInfoActors.InitTraversal()
        actor = self.onScreenInfoActors.GetNextItem()
        while actor is not None:
            try:
                self.vtkRen.RemoveActor(actor)
            except:
                pass
            
            actor = self.onScreenInfoActors.GetNextItem()
        
        self.vtkRenWinInteract.ReInitialize()
        
        self.onScreenInfoActors = vtk.vtkActor2DCollection()
    
    def refreshOnScreenInfo(self):
        """
        Refresh the on-screen information.
        
        """
        if not self.mainWindow.refLoaded:
            return
        
        textSel = self.textSelector
        selectedText = textSel.selectedText
        textSettings = textSel.textSettings
        
        self.onScreenInfo = {}
        self.removeOnScreenInfo()
        
        if not selectedText.count():
            return
        
        # atom count doesn't change
        if "Atom count" not in self.onScreenInfo:
            self.onScreenInfo["Atom count"] = "%d atoms" % self.mainWindow.inputState.NAtoms
        
        # sim time doesn't change
        if "Simulation time" not in self.onScreenInfo:
            self.onScreenInfo["Simulation time"] = utilities.simulationTimeLine(self.mainWindow.inputState.simTime)
        
        # barrier doesn't change
        if "Energy barrier" not in self.onScreenInfo and self.mainWindow.inputState.barrier is not None:
            self.onScreenInfo["Energy barrier"] = "Barrier: %f eV" % self.mainWindow.inputState.barrier
        
        # filter lists
        filterLists = self.getFilterLists()
        
        # visible counts always recalculated
        visCountActive = False
        visCount = 0
        for filterList in filterLists:
            if filterList.visible and not filterList.defectFilterSelected:
                visCountActive = True
                visCount += filterList.filterer.NVis
        
        if visCountActive:
            self.onScreenInfo["Visible count"] = "%d visible" % visCount
        
            visSpecCount = np.zeros(len(self.mainWindow.inputState.specieList), np.int32)
            for filterList in filterLists:
                if filterList.visible and not filterList.defectFilterSelected and filterList.filterer.NVis:
                    if len(visSpecCount) == len(filterList.filterer.visibleSpecieCount):
                        visSpecCount = np.add(visSpecCount, filterList.filterer.visibleSpecieCount)
        
            specieList = self.mainWindow.inputState.specieList
            self.onScreenInfo["Visible specie count"] = []
            for i, cnt in enumerate(visSpecCount):
                self.onScreenInfo["Visible specie count"].append("%d %s" % (cnt, specieList[i]))
        
        # defects counts
        defectFilterActive = False
        NVac = 0
        NInt = 0
        NAnt = 0
        showVacs = False
        showInts = False
        showAnts = False
        for filterList in filterLists:
            if filterList.visible and filterList.defectFilterSelected:
                defectFilterActive = True
                
                NVac += filterList.filterer.NVac
                NInt += filterList.filterer.NInt
                NAnt += filterList.filterer.NAnt
                
                # defects settings
                defectsSettings = filterList.currentSettings[0]
                
                if defectsSettings.showVacancies:
                    showVacs = True
                
                if defectsSettings.showInterstitials:
                    showInts = True
                
                if defectsSettings.showAntisites:
                    showAnts = True
        
        if defectFilterActive:
            # defect specie counters
            vacSpecCount = np.zeros(len(self.mainWindow.refState.specieList), np.int32)
            intSpecCount = np.zeros(len(self.mainWindow.inputState.specieList), np.int32)
            antSpecCount = np.zeros((len(self.mainWindow.refState.specieList), len(self.mainWindow.inputState.specieList)), np.int32)
            splitSpecCount = np.zeros((len(self.mainWindow.inputState.specieList), len(self.mainWindow.inputState.specieList)), np.int32)
            for filterList in filterLists:
                if filterList.visible and filterList.defectFilterSelected and filterList.filterer.NVis:
                    if len(vacSpecCount) == len(filterList.filterer.vacancySpecieCount):
                        vacSpecCount = np.add(vacSpecCount, filterList.filterer.vacancySpecieCount)
                        intSpecCount = np.add(intSpecCount, filterList.filterer.interstitialSpecieCount)
                        antSpecCount = np.add(antSpecCount, filterList.filterer.antisiteSpecieCount)
                        splitSpecCount = np.add(splitSpecCount, filterList.filterer.splitIntSpecieCount)
            
            # now add to dict
            self.onScreenInfo["Defect count"] = []
            
            if showVacs:
                self.onScreenInfo["Defect count"].append("%d vacancies" % (NVac,))
            
            if showInts:
                self.onScreenInfo["Defect count"].append("%d interstitials" % (NInt,))
            
            if showAnts:
                self.onScreenInfo["Defect count"].append("%d antisites" % (NAnt,))
            
            specListInput = self.mainWindow.inputState.specieList
            specListRef = self.mainWindow.refState.specieList
            specRGBInput = self.mainWindow.inputState.specieRGB
            specRGBRef = self.mainWindow.refState.specieRGB
            
            self.onScreenInfo["Defect specie count"] = []
            
            if showVacs:
                for i, cnt in enumerate(vacSpecCount):
                    self.onScreenInfo["Defect specie count"].append(["%d %s vacancies" % (cnt, specListRef[i]), specRGBRef[i]])
            
            if showInts:
                for i, cnt in enumerate(intSpecCount):
                    self.onScreenInfo["Defect specie count"].append(["%d %s interstitials" % (cnt, specListInput[i]), specRGBInput[i]])
                
                if defectsSettings.identifySplitInts:
                    for i in xrange(len(specListInput)):
                        for j in xrange(i, len(specListInput)):
                            if j == i:
                                N = splitSpecCount[i][j]
                                rgb = specRGBInput[i]
                            else:
                                N = splitSpecCount[i][j] + splitSpecCount[j][i]
                                rgb = (specRGBInput[i] + specRGBInput[j]) / 2.0
                            
                            self.onScreenInfo["Defect specie count"].append(["%d %s-%s split ints" % (N, specListInput[i], specListInput[j]), rgb])
            
            if showAnts:
                for i in xrange(len(specListRef)):
                    for j in xrange(len(specListInput)):
                        if i == j:
                            continue
                        self.onScreenInfo["Defect specie count"].append(["%d %s on %s antisites" % (antSpecCount[i][j], specListInput[j], specListRef[i]), specRGBRef[i]])
        
        # alignment/position stuff
        topyLeft = self.vtkRenWinInteract.height() - 5
        topxLeft = 5
        topyRight = self.vtkRenWinInteract.height() - 5
        topxRight = self.vtkRenWinInteract.width() - 220
        
        # loop over selected text
        for i in xrange(selectedText.count()):
            item = selectedText.item(i)
            item = str(item.text())
            settings = textSettings[item]
            
            try:
                line = self.onScreenInfo[item]
                
                if item == "Visible specie count":
                    for j, specline in enumerate(line):
                        r, g, b = self.mainWindow.inputState.specieRGB[j]
                        
                        if settings.textPosition == "Top left":
                            xpos = topxLeft
                            ypos = topyLeft
                        else:
                            xpos = topxRight
                            ypos = topyRight
                        
                        # add actor
                        actor = vtkRenderWindowText(specline, 20, xpos, ypos, r, g, b)
                        
                        if settings.textPosition == "Top left":
                            topyLeft -= 20
                        else:
                            topyRight -= 20
                        
                        self.onScreenInfoActors.AddItem(actor)
                
                elif item == "Defect count":
                    for specline in line:
                        r = g = b = 0
                        
                        if settings.textPosition == "Top left":
                            xpos = topxLeft
                            ypos = topyLeft
                        else:
                            xpos = topxRight
                            ypos = topyRight
                        
                        # add actor
                        actor = vtkRenderWindowText(specline, 20, xpos, ypos, r, g, b)
                        
                        if settings.textPosition == "Top left":
                            topyLeft -= 20
                        else:
                            topyRight -= 20
                        
                        self.onScreenInfoActors.AddItem(actor)
                
                elif item == "Defect specie count":
                    for array in line:
                        lineToAdd = array[0]
                        r, g, b = array[1]
                        
                        if settings.textPosition == "Top left":
                            xpos = topxLeft
                            ypos = topyLeft
                        else:
                            xpos = topxRight
                            ypos = topyRight
                        
                        # add actor
                        actor = vtkRenderWindowText(lineToAdd, 20, xpos, ypos, r, g, b)
                        
                        if settings.textPosition == "Top left":
                            topyLeft -= 20
                        else:
                            topyRight -= 20
                        
                        self.onScreenInfoActors.AddItem(actor)
                
                else:
                    r = g = b = 0
                    
                    if settings.textPosition == "Top left":
                        xpos = topxLeft
                        ypos = topyLeft
                    else:
                        xpos = topxRight
                        ypos = topyRight
                    
                    # add actor
                    actor = vtkRenderWindowText(line, 20, xpos, ypos, r, g, b)
                    
                    if settings.textPosition == "Top left":
                        topyLeft -= 20
                    else:
                        topyRight -= 20
                    
                    self.onScreenInfoActors.AddItem(actor)
            
            except KeyError:
                pass
#                print "WARNING: '%s' not in onScreenInfo dict" % item
        
        # add to render window
        self.onScreenInfoActors.InitTraversal()
        actor = self.onScreenInfoActors.GetNextItem()
        while actor is not None:
            try:
                self.vtkRen.AddActor(actor)
            except:
                pass
            
            actor = self.onScreenInfoActors.GetNextItem()
        
        self.vtkRenWinInteract.ReInitialize()
    
    def closeEvent(self, event):
        """
        Override close event.
        
        """
        reply = QtGui.QMessageBox.question(self, 'Message', "Are you sure you want to close this window", 
                                           QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
        if reply == QtGui.QMessageBox.Yes:
            self.closed = True
            self.mainWindow.renderWindowClosed()
            event.accept()
        else:
            event.ignore()



