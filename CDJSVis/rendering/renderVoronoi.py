
"""
Module for rendering Voronoi cells

@author: Chris Scott

"""
import time

import numpy as np
import vtk

from ..visutils import vectors
from .utils import setupLUT, getScalar, setMapperScalarRange


################################################################################

def makePolygon(indexes):
    """
    Make a polygon from given indexes.
    
    Need to do scalars for colour too????
    
    """
    polygon = vtk.vtkPolygon()
    
    # set number of vertices
    polygon.GetPointIds().SetNumberOfIds(len(indexes))
    
    # add vertices (indexes)
    for i, index in enumerate(indexes):
        polygon.GetPointIds().SetId(i, index)
    
    return polygon

################################################################################

def getActorsForVoronoiCells(visibleAtoms, inputState, voronoi, colouringOptions, voronoiOptions, actorsCollection, log=None):
    """
    Return actors for Voronoi cells
    
    """
    renderVoroTime = time.time()
    
    print "RENDER VORONOI"
    
    # setup LUT
    lut = setupLUT(inputState.specieList, inputState.specieRGB, colouringOptions)
    
    # looks like we will have to make an actor for each atom
    # NOT IDEAL!
    for index in visibleAtoms:
        # check we are working with the same atom!
        inp_pos = inputState.atomPos(index)
        out_pos = voronoi.getInputAtomPos(index)
        
        sep = vectors.separation(inp_pos, out_pos, inputState.cellDims, np.ones(3, np.int32))
        assert sep < 1e-4, "ERROR: VORO OUTPUT ORDERING DIFFERENT (%f)" % sep
        
        # scalar val for this atom
        scalar = getScalar(colouringOptions, inputState, index)
        
        # points (vertices)
        points = vtk.vtkPoints()
        scalars = vtk.vtkFloatArray()
        for point in voronoi.atomVertices(index):
            points.InsertNextPoint(point)
            scalars.InsertNextValue(scalar)
        
        # make polygons
        facePolygons = vtk.vtkCellArray()
        for face in voronoi.atomFaces(index):
            facePolygon = makePolygon(face)
            facePolygons.InsertNextCell(facePolygon)
        
        # polydata object
        regionPolyData = vtk.vtkPolyData()
        regionPolyData.SetPoints(points)
        regionPolyData.SetPolys(facePolygons)
        regionPolyData.GetPointData().SetScalars(scalars)
        
        # mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInput(regionPolyData)
        mapper.SetLookupTable(lut)
        setMapperScalarRange(mapper, colouringOptions, len(inputState.specieList))
        
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(voronoiOptions.opacity)
        
        actorsCollection.AddItem(actor)
        
    renderVoroTime = time.time() - renderVoroTime
    
    print "RENDER VORO TIME", renderVoroTime
