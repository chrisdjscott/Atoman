
"""
Module for rendering atoms

"""
import logging

import vtk

from .. import utils

################################################################################

class AtomRenderer(object):
    """
    Render a set of atoms.
    
    """
    def __init__(self, shape="sphere"):
        self._logger = logging.getLogger(__name__)
        self._shape = shape
    
    def render(self, atomPoints, scalarsArray, radiusArray, nspecies, colouringOptions, atomScaleFactor, lut, resolution):
        """
        Render the given atoms.
        
        Explain...
        
        """
        self._logger.debug("Rendering atoms: shape is '%s', colour by: '%s'", self._shape, colouringOptions.colourBy)
        
        # poly data
        atomsPolyData = vtk.vtkPolyData()
        atomsPolyData.SetPoints(atomPoints)
        atomsPolyData.GetPointData().AddArray(scalarsArray)
        atomsPolyData.GetPointData().SetScalars(radiusArray)
        
        # glyph source
        atomsGlyphSource = vtk.vtkSphereSource() #TODO: depends on self._shape
        atomsGlyphSource.SetPhiResolution(resolution)
        atomsGlyphSource.SetThetaResolution(resolution)
        atomsGlyphSource.SetRadius(1.0)
        
        # glyph
        atomsGlyph = vtk.vtkGlyph3D()
        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            atomsGlyph.SetSource(atomsGlyphSource.GetOutput())
            atomsGlyph.SetInput(atomsPolyData)
        else:
            atomsGlyph.SetSourceConnection(atomsGlyphSource.GetOutputPort())
            atomsGlyph.SetInputData(atomsPolyData)
        atomsGlyph.SetScaleFactor(atomScaleFactor)
        atomsGlyph.SetScaleModeToScaleByScalar()
        atomsGlyph.ClampingOff()
          
        # mapper
        atomsMapper = vtk.vtkPolyDataMapper()
        atomsMapper.SetInputConnection(atomsGlyph.GetOutputPort())
        atomsMapper.SetLookupTable(lut)
        atomsMapper.SetScalarModeToUsePointFieldData()
        atomsMapper.SelectColorArray("colours")
        utils.setMapperScalarRange(atomsMapper, colouringOptions, nspecies)
        
        # glyph mapper
        # glyphMapper = vtk.vtkGlyph3DMapper()
        # if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
        #     glyphMapper.SetInputConnection(atomsPolyData.GetProducerPort())
        # else:
        #     glyphMapper.SetInputData(atomsPolyData)
        # glyphMapper.SetSourceConnection(atomsGlyphSource.GetOutputPort())
        # glyphMapper.SetScaleFactor(displayOptions.atomScaleFactor)
        # glyphMapper.SetScaleModeToScaleByMagnitude()
        # glyphMapper.ClampingOff()
        # glyphMapper.SetLookupTable(lut)
        # glyphMapper.SetScalarModeToUsePointFieldData()
        # glyphMapper.SelectColorArray("colours")
        # setMapperScalarRange(glyphMapper, colouringOptions, NSpecies)
        # atomsMapper = glyphMapper
        
        # actor
        atomsActor = vtk.vtkActor()
        atomsActor.SetMapper(atomsMapper)
        atomsActor.GetProperty().SetSpecular(0.4)
        atomsActor.GetProperty().SetSpecularPower(50)
        
        return utils.ActorObject(atomsActor)
