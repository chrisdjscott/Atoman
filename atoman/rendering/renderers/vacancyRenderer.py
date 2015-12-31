
"""
Module for rendering vacancies

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

import vtk

from . import baseRenderer
from . import povrayWriters
from .. import utils


class VacancyRenderer(baseRenderer.BaseRenderer):
    """
    Render a set of vacancies.
    
    """
    def __init__(self):
        super(VacancyRenderer, self).__init__()
        self._logger = logging.getLogger(__name__)
    
    def render(self, pointsData, scalarsArray, radiusArray, nspecies, colouringOptions, atomScaleFactor, lut, settings):
        """
        Render the given antisites (wire frame).
        
        """
        self._logger.debug("Rendering vacancies: colour by '%s'", colouringOptions.colourBy)
        
        # points
        points = vtk.vtkPoints()
        points.SetData(pointsData.getVTK())
        
        # poly data
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(scalarsArray.getVTK())
        polydata.GetPointData().SetScalars(radiusArray.getVTK())
        
        # source
        glyphSource = vtk.vtkCubeSource()
        
        # glyph
        glyph = vtk.vtkGlyph3D()
        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            glyph.SetSource(glyphSource.GetOutput())
            glyph.SetInput(polydata)
        else:
            glyph.SetSourceConnection(glyphSource.GetOutputPort())
            glyph.SetInputData(polydata)
        scaleVacs = 2.0 * settings.getSetting("vacScaleSize")
        glyph.SetScaleFactor(atomScaleFactor * scaleVacs)
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
        actor.GetProperty().SetSpecular(settings.getSetting("vacSpecular"))
        actor.GetProperty().SetSpecularPower(settings.getSetting("vacSpecularPower"))
        actor.GetProperty().SetOpacity(settings.getSetting("vacOpacity"))
        
        # store attributes
        self._actor = utils.ActorObject(actor)
        self._data["Points"] = pointsData
        self._data["Scalars"] = scalarsArray
        self._data["Radius"] = radiusArray
        self._data["LUT"] = lut
        self._data["Scale factor"] = atomScaleFactor
        self._data["Vacancy opacity"] = settings.getSetting("vacOpacity")
    
    def writePovray(self, filename):
        """Write atoms to POV-Ray file."""
        self._logger.debug("Writing vacancies POV-Ray file")
        
        # povray writer
        writer = povrayWriters.PovrayVacanciesWriter()
        writer.write(filename, self._data["Points"], self._data["Scalars"], self._data["Radius"],
                     self._data["Scale factor"], self._data["LUT"], self._data["Vacancy opacity"])
