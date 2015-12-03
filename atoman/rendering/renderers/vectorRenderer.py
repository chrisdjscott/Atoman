
"""
Module for rendering vectors

"""
import logging

import vtk

from . import baseRenderer
from .. import utils

################################################################################

class VectorRenderer(baseRenderer.BaseRenderer):
    """
    Render vectors as arrows
    
    """
    def __init__(self):
        super(VectorRenderer, self).__init__()
        self._logger = logging.getLogger(__name__)
    
    def render(self, pointsData, scalarsArray, vectorsArray, nspecies, colouringOptions, vectorsOptions, lut):
        """
        Render vectors.
        
        """
        self._logger.debug("Rendering vectors")
        
        # points
        points = vtk.vtkPoints()
        points.SetData(pointsData.getVTK())
        
        # polydata
        arrowPolyData = vtk.vtkPolyData()
        arrowPolyData.SetPoints(points)
        arrowPolyData.GetPointData().SetScalars(scalarsArray.getVTK())
        arrowPolyData.GetPointData().SetVectors(vectorsArray.getVTK())
    
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
        
        # store attributes
        self._actor = utils.ActorObject(arrowActor)
        self._data["Points"] = pointsData
        self._data["Scalars"] = scalarsArray
        self._data["Vectors"] = vectorsArray
