
"""
Module for rendering antisites

"""
import logging

import vtk

from .. import utils

################################################################################

class AntisiteRenderer(object):
    """
    Render a set of antisites.
    
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def render(self, points, scalarsArray, radiusArray, nspecies, colouringOptions, atomScaleFactor, lut):
        """
        Render the given antisites (wire frame).
        
        """
        self._logger.debug("Rendering antisites: colour by '%s'", colouringOptions.colourBy)
        
        # poly data
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(scalarsArray)
        polydata.GetPointData().SetScalars(radiusArray)
        
        # source
        cubeSource = vtk.vtkCubeSource()
        edges = vtk.vtkExtractEdges()
        edges.SetInputConnection(cubeSource.GetOutputPort())
        glyphSource = vtk.vtkTubeFilter()
        glyphSource.SetInputConnection(edges.GetOutputPort())
        glyphSource.SetRadius(0.05)
        glyphSource.SetVaryRadius(0)
        glyphSource.SetNumberOfSides(5)
        glyphSource.UseDefaultNormalOn()
        glyphSource.SetDefaultNormal(.577, .577, .577)
        
        # glyph
        glyph = vtk.vtkGlyph3D()
        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            glyph.SetSource(glyphSource.GetOutput())
            glyph.SetInput(antsPolyData)
        else:
            glyph.SetSourceConnection(glyphSource.GetOutputPort())
            glyph.SetInputData(polydata)
        glyph.SetScaleFactor(atomScaleFactor * 2.0)
        glyph.SetScaleModeToScaleByScalar()
        glyph.ClampingOff()
          
        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("colours")
        utils.setMapperScalarRange(mapper, colouringOptions, nspecies)
        
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        return utils.ActorObject(actor)
