
"""
Module for rendering Voronoi cells

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

import vtk
import numpy as np

from . import baseRenderer
from . import povrayWriters
from .. import utils
from ...algebra import vectors


class VoronoiRenderer(baseRenderer.BaseRenderer):
    """
    Render Voronoi cells.
    
    """
    def __init__(self):
        super(VoronoiRenderer, self).__init__()
        self._logger = logging.getLogger(__name__)
    
    def render(self, inputState, visibleAtoms, scalarsArray, lut, voro, voronoiOptions, colouringOptions):
        """
        Render Voronoi cells for visible atoms
        
        """
        self._logger.debug("Rendering Voronoi cells (%d visible atoms)", len(visibleAtoms))
        
        # object for combining poly datas
        appendPolyData = vtk.vtkAppendPolyData()
        
        # loop over the visible atoms
        for visIndex, index in enumerate(visibleAtoms):
            # check we are working with the same atom!
            inp_pos = inputState.atomPos(index)
            out_pos = voro.getInputAtomPos(index)
            sep = vectors.separation(inp_pos, out_pos, inputState.cellDims, np.ones(3, np.int32))
            if sep > 1e-4:
                raise RuntimeError("Voronoi ordering is different")
            
            # faces
            faces = voro.atomFaces(index)
            if faces is None:
                continue
            
            # scalar value for this atom
            scalar = scalarsArray[visIndex]
            
            # points (vertices)
            points = vtk.vtkPoints()
            scalars = vtk.vtkFloatArray()
            for point in voro.atomVertices(index):
                points.InsertNextPoint(point)
                scalars.InsertNextValue(scalar)
            
            # make polygons
            facePolygons = vtk.vtkCellArray()
            for face in faces:
                polygon = vtk.vtkPolygon()
                
                # set number of vertices
                polygon.GetPointIds().SetNumberOfIds(len(face))
                
                # add vertices (indexes)
                for i, index in enumerate(face):
                    polygon.GetPointIds().SetId(i, index)
                
                # add the polygon to the set
                facePolygons.InsertNextCell(polygon)
            
            # polydata object
            regionPolyData = vtk.vtkPolyData()
            regionPolyData.SetPoints(points)
            regionPolyData.SetPolys(facePolygons)
            regionPolyData.GetPointData().SetScalars(scalars)
            
            # append the poly data
            if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
                appendPolyData.AddInputConnection(regionPolyData.GetProducerPort())
            else:
                appendPolyData.AddInputData(regionPolyData)
        appendPolyData.Update()
        
        # remove any duplicate points
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(appendPolyData.GetOutputPort())
        cleanFilter.Update()
        
        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cleanFilter.GetOutputPort())
        mapper.SetLookupTable(lut)
        utils.setMapperScalarRange(mapper, colouringOptions, len(inputState.specieList))
        
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(voronoiOptions.opacity)
        
        # store data
        self._actor = utils.ActorObject(actor)
        self._data["LUT"] = lut
        self._data["Voronoi"] = voro
        self._data["Scalars"] = scalarsArray
        self._data["Visible atoms"] = visibleAtoms
        self._data["Lattice"] = inputState
        self._data["Opacity"] = voronoiOptions.opacity
    
    def writePovray(self, filename):
        """Write voronoi cells to POV-Ray file."""
        self._logger.debug("Writing Voronoi cells POV-Ray file")
        
        # povray writer
        writer = povrayWriters.PovrayVoronoiWriter()
        writer.write(filename, self._data["Visible atoms"], self._data["Lattice"], self._data["Scalars"],
                     self._data["LUT"], self._data["Voronoi"], self._data["Opacity"])
