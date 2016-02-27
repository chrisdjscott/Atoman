
"""
The slice plane helper

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import vtk


################################################################################
class SlicePlane(vtk.vtkActor):
    """
    Slice plane.
    
    """
    def __init__(self, pipelinePage):
        self.source = vtk.vtkPlaneSource()
        self.mapper = vtk.vtkPolyDataMapper()
        
        self.pipelinePage = pipelinePage
    
    def update(self, p, n):
        """
        Show the slice plane in given position.
        
        """
        inputState = self.pipelinePage.inputState
        
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
        self.SetMapper(self.mapper)
        self.GetProperty().SetDiffuseColor(1, 0, 0)
        self.GetProperty().SetSpecular(0.4)
        self.GetProperty().SetSpecularPower(10)
        self.GetProperty().SetOpacity(0.7)
        self.GetProperty().SetLineWidth(2.0)
        self.GetProperty().EdgeVisibilityOn()
    
#    def hide(self):
#        """
#        Remove the actor.
#        
#        """
#        if self.visible:
#            self.ren.RemoveActor(self.actor)
#            self.renWinInteract.ReInitialize()
#            self.visible = False

