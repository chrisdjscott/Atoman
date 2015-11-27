
"""
Module for rendering vectors

"""
import logging

import vtk

from .. import utils

################################################################################

class VectorRenderer(object):
    """
    Render vectors as arrows
    
    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def render(self, atomPoints, scalarsArray, vectorsArray, nspecies, colouringOptions, vectorsOptions, lut):
        """
        Render vectors.
        
        """
        self._logger.debug("Rendering vectors")
        
        # polydata
        arrowPolyData = vtk.vtkPolyData()
        arrowPolyData.SetPoints(atomPoints)
        arrowPolyData.GetPointData().SetScalars(scalarsArray)
        arrowPolyData.GetPointData().SetVectors(vectorsArray)
    
        # arrow source
        arrowSource = vtk.vtkArrowSource()
        arrowSource.SetShaftResolution(vectorsOptions.vectorResolution)
        arrowSource.SetTipResolution(vectorsOptions.vectorResolution)
        arrowSource.Update()
        
        # glyph mapper
        arrowGlyph = vtk.vtkGlyph3DMapper()
        arrowGlyph.OrientOn()
        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            arrowGlyph.SetInputConnection(arrowPolyData.GetProducerPort())
        else:
            arrowGlyph.SetInputData(arrowPolyData)
        arrowGlyph.SetSourceConnection(arrowSource.GetOutputPort())
        arrowGlyph.SetScaleModeToScaleByMagnitude()
        arrowGlyph.SetScaleArray("vectors")
        arrowGlyph.SetScalarModeToUsePointFieldData()
        arrowGlyph.SelectColorArray("colours")
        arrowGlyph.SetScaleFactor(vectorsOptions.vectorScaleFactor)
        arrowMapper = arrowGlyph
        arrowMapper.SetLookupTable(lut)
        utils.setMapperScalarRange(arrowMapper, colouringOptions, nspecies)
    
        # actor
        arrowActor = vtk.vtkActor()
        arrowActor.SetMapper(arrowMapper)
        
        return utils.ActorObject(arrowActor)
