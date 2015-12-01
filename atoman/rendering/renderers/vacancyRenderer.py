
"""
Module for rendering vacancies

"""
import logging

import vtk

from .. import utils

################################################################################

class VacancyRenderer(object):
    """
    Render a set of vacancies.
    
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def render(self, points, scalarsArray, radiusArray, nspecies, colouringOptions, atomScaleFactor, lut, settings):
        """
        Render the given antisites (wire frame).
        
        """
        self._logger.debug("Rendering vacancies: colour by '%s'", colouringOptions.colourBy)
        
        # poly data
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetPointData().AddArray(scalarsArray)
        polydata.GetPointData().SetScalars(radiusArray)
        
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
        
        return utils.ActorObject(actor)
