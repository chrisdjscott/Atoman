
"""
Module for rendering Voronoi cells

@author: Chris Scott

"""
import numpy as np
import vtk

from ..visutils import vectors


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

def getActorsForVoronoiCells(visibleAtoms, inputState, vorRegionList, log=None):
    """
    Return actors for Voronoi cells
    
    """
    print "RENDER VORONOI"
    
    print "ASSUMING ORDERING IS THE SAME!!!!"
    
    for index in visibleAtoms:
        # this region
        vorDict = vorRegionList[index]
        
        # check we are working with the same atom!
        inp_pos = inputState.atomPos(index)
        out_pos = vorDict["original"]
        
        sep = vectors.separation(inp_pos, out_pos, inputState.cellDims, np.ones(3, np.int32))
        assert sep < 1e-4, "ERROR: VORO OUTPUT ORDERING DIFFERENT (%f)" % sep
        
        # make polygons
        
        
        return
    





################################################################################
def getActorsForHullFacets(facets, pos, mainWindow, actorsCollection, settings):
    """
    Render convex hull facets
    
    """
    # probably want to pass some settings through too eg colour, opacity etc
    
    
    points = vtk.vtkPoints()
    for i in xrange(len(pos) / 3):
        points.InsertNextPoint(pos[3*i], pos[3*i+1], pos[3*i+2])
    
    # create triangles
    triangles = vtk.vtkCellArray()
    for i in xrange(len(facets)):
        triangle = makeTriangle(facets[i])
        triangles.InsertNextCell(triangle)
    
    # polydata object
    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)
    
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(trianglePolyData)
    
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(settings.hullOpacity)
    actor.GetProperty().SetColor(settings.hullCol[0], settings.hullCol[1], settings.hullCol[2])
    
    actorsCollection.AddItem(actor)
