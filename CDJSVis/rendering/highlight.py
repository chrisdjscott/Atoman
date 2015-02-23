
"""
Module for highlighting picked atoms/defects

@author: Chris Scott

"""
import vtk


################################################################################

class AtomHighlighter(vtk.vtkActor):
    """
    Atom highlighter
    
    """
    def __init__(self, pos, radius, rgb=[0.62, 0, 0.77]):
        self.source = vtk.vtkSphereSource()
        self.mapper = vtk.vtkPolyDataMapper()
        
        # source
        self.source.SetCenter(pos)
        self.source.SetRadius(radius)
        
        # mapper
        self.mapper.SetInputConnection(self.source.GetOutputPort())
        
        # actor
        self.SetMapper(self.mapper)
        self.GetProperty().SetColor(rgb)

################################################################################

class VacancyHighlighter(vtk.vtkActor):
    """
    Vacancy highlighter
    
    """
    def __init__(self, pos, radius, rgb=[0.62, 0, 0.77]):
        self.source = vtk.vtkCubeSource()
        self.mapper = vtk.vtkPolyDataMapper()
        
        # length of sides
        self.source.SetXLength(radius)
        self.source.SetYLength(radius)
        self.source.SetZLength(radius)
        
        # centre
        self.source.SetCenter(pos)
        
        # mapper
        self.mapper.SetInputConnection(self.source.GetOutputPort())
        
        # actor
        self.SetMapper(self.mapper)
        self.GetProperty().SetColor(rgb)
    
################################################################################

class AntisiteHighlighter(vtk.vtkActor):
    """
    Antisite highlighter
    
    """
    def __init__(self, pos, radius, rgb=[0.62, 0, 0.77]):
        self.source = vtk.vtkCubeSource()
        self.mapper = vtk.vtkPolyDataMapper()
        
        # length of sides
        self.source.SetXLength(radius)
        self.source.SetYLength(radius)
        self.source.SetZLength(radius)
        
        # centre
        self.source.SetCenter(pos)
        
        # edges filter
        edges = vtk.vtkExtractEdges()
        edges.SetInputConnection(self.source.GetOutputPort())
        
        # tube filter
        tubes = vtk.vtkTubeFilter()
        tubes.SetInputConnection(edges.GetOutputPort())
        tubes.SetRadius(0.11)
        tubes.SetNumberOfSides(5)
        
        # mapper
        self.mapper.SetInputConnection(tubes.GetOutputPort())
        
        # actor
        self.SetMapper(self.mapper)
        self.GetProperty().SetColor(rgb)
