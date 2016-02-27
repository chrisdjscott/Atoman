
"""
Module for rendering antisites

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

import vtk

from . import baseRenderer
from . import povrayWriters
from .. import utils


class AntisiteRenderer(baseRenderer.BaseRenderer):
    """
    Render a set of antisites.
    
    """
    def __init__(self):
        super(AntisiteRenderer, self).__init__()
        self._logger = logging.getLogger(__name__)
    
    def render(self, pointsData, scalarsArray, radiusArray, nspecies, colouringOptions, atomScaleFactor, lut):
        """
        Render the given antisites (wire frame).
        
        """
        self._logger.debug("Rendering antisites: colour by '%s'", colouringOptions.colourBy)
        
        # points
        points = vtk.vtkPoints()
        points.SetData(pointsData.getVTK())
        
        # poly data
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(scalarsArray.getVTK())
        polydata.GetPointData().SetScalars(radiusArray.getVTK())
        
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
            glyph.SetInput(polydata)
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
        
        # store attributes
        self._actor = utils.ActorObject(actor)
        self._data["Points"] = pointsData
        self._data["Scalars"] = scalarsArray
        self._data["Radius"] = radiusArray
        self._data["LUT"] = lut
        self._data["Scale factor"] = atomScaleFactor
    
    def writePovray(self, filename):
        """Write antisites to POV-Ray file."""
        self._logger.debug("Writing antisites POV-Ray file")
        
        # povray writer
        writer = povrayWriters.PovrayAntisitesWriter()
        writer.write(filename, self._data["Points"], self._data["Scalars"], self._data["Radius"],
                     self._data["Scale factor"], self._data["LUT"])
