
"""
The slice plane helper

@author: Chris Scott

"""
import vtk


################################################################################
class SlicePlane(object):
    """
    Slice plane.
    
    """
    def __init__(self, ren, renWinInteract, mainWindow):
        self.ren = ren
        self.renWinInteract = renWinInteract
        self.actor = vtk.vtkActor()
        self.source = vtk.vtkPlaneSource()
        self.mapper = vtk.vtkPolyDataMapper()
        
        self.mainWindow = mainWindow
    
        self.visible = False
    
    def show(self, p, n):
        """
        Show the slice plane in given position.
        
        """
        if self.visible:
            self.ren.RemoveActor(self.actor)
            self.visible = False
        
        inputState = self.mainWindow.inputState
        
        # source
        self.source.SetOrigin(-50, -50, 0)
        self.source.SetPoint1(inputState.cellDims[0] + 50, -50, 0)
        self.source.SetPoint2(-50, inputState.cellDims[1] + 50, 0)
        self.source.SetNormal(n)
        self.source.SetCenter(p)
        self.source.SetXResolution(100)
        self.source.SetYResolution(100)
        
        # mapper
        self.mapper.SetInputConnection(self.source.GetOutputPort())
        
        # actor
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetDiffuseColor(1, 0, 0)
        self.actor.GetProperty().SetSpecular(0.4)
        self.actor.GetProperty().SetSpecularPower(10)
        self.actor.GetProperty().SetOpacity(0.7)
        self.actor.GetProperty().SetLineWidth(2.0)
        self.actor.GetProperty().EdgeVisibilityOn()
        
        # add to ren
        self.ren.AddActor(self.actor)
        self.renWinInteract.ReInitialize()
        
        self.visible = True
    
    def hide(self):
        """
        Remove the actor.
        
        """
        if self.visible:
            self.ren.RemoveActor(self.actor)
            self.renWinInteract.ReInitialize()
            self.visible = False

