
"""
Cell outline

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import vtk


################################################################################

class CellOutline(object):
    def __init__(self, ren):
        
        self.ren = ren
        self.source = vtk.vtkOutlineSource()
        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
        self.visible = 0
        self.currentColour = (0, 0, 0)
    
    def add(self, a, b):
        """
        Add the lattice cell.
        
        """
        # first remove if already visible
        if self.visible:
            self.remove()
        
        # now add it
        self.source.SetBounds(a[0], b[0], a[1], b[1], a[2], b[2])
        
        self.mapper.SetInputConnection(self.source.GetOutputPort())
        
        self.actor.SetMapper(self.mapper)
        self.setColour(self.currentColour)
        
        self.ren.AddActor(self.actor)
        
        self.visible = 1
    
    def remove(self):
        """
        Remove the cell outline.
        
        """
        self.ren.RemoveActor(self.actor)
        
        self.visible = 0
    
    def setColour(self, colour):
        """
        Set colour.
        
        """
        self.actor.GetProperty().SetColor(colour)
        
        self.currentColour = colour

