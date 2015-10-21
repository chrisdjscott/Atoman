
"""
Module for rendering Voronoi cells

@author: Chris Scott

"""
import os
import time
import logging

import numpy as np
import vtk

from ..algebra import vectors
from . import utils
from .utils import setupLUT, getScalar, setMapperScalarRange, getScalarsType
from ..filtering import clusters


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

def getActorsForVoronoiCells(visibleAtoms, inputState, voronoi, colouringOptions, voronoiOptions, actorsDict, povfile, scalarsDict, log=None):
    """
    Return actors for Voronoi cells
    
    """
    logger = logging.getLogger(__name__)
    
    renderVoroTime = time.time()
    
    logger.debug("Rendering Voronoi volumes")
    
    # scalar type
    scalarType = getScalarsType(colouringOptions)
    
    # scalars array
    if scalarType == 5:
        scalarsArray = scalarsDict[colouringOptions.colourBy]
    else:
        scalarsArray = np.array([], dtype=np.float64)
    
    # setup LUT
    lut = setupLUT(inputState.specieList, inputState.specieRGB, colouringOptions)
    
    # looks like we will have to make an actor for each atom
    # NOT IDEAL!
    actorsDictLocal = {}
    for visIndex, index in enumerate(visibleAtoms):
        # check we are working with the same atom!
        inp_pos = inputState.atomPos(index)
        out_pos = voronoi.getInputAtomPos(index)
        
        sep = vectors.separation(inp_pos, out_pos, inputState.cellDims, np.ones(3, np.int32))
        if sep > 1e-4:
            logger.error("Voro output ordering different (%f)", sep)
        
        # faces
        faces = voronoi.atomFaces(index)
        
        if faces is None:
            continue
        
        if len(scalarsArray):
            scalarVal=scalarsArray[visIndex]
        else:
            scalarVal = None
        
        # scalar val for this atom
        scalar = getScalar(colouringOptions, inputState, index, scalarVal=scalarVal)
        
        # points (vertices)
        points = vtk.vtkPoints()
        scalars = vtk.vtkFloatArray()
        for point in voronoi.atomVertices(index):
            points.InsertNextPoint(point)
            scalars.InsertNextValue(scalar)
        
        # make polygons
        facePolygons = vtk.vtkCellArray()
        for face in faces:
            facePolygon = makePolygon(face)
            facePolygons.InsertNextCell(facePolygon)
        
        # polydata object
        regionPolyData = vtk.vtkPolyData()
        regionPolyData.SetPoints(points)
        regionPolyData.SetPolys(facePolygons)
        regionPolyData.GetPointData().SetScalars(scalars)
        
        # mapper
        mapper = vtk.vtkPolyDataMapper()
        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            mapper.SetInput(regionPolyData)
        else:
            mapper.SetInputData(regionPolyData)
        mapper.SetLookupTable(lut)
        setMapperScalarRange(mapper, colouringOptions, len(inputState.specieList))
        
        # actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(voronoiOptions.opacity)
        
        actorsDictLocal["Voronoi cell {0}".format(visIndex)] = utils.ActorObject(actor)
        
        # colour for povray file
        rgb = np.empty(3, np.float64)
        lut.GetColor(scalar, rgb)
        
        # write to POV-Ray file
#         writePOVRayVoroVolume(voronoi.atomVertices(index), voronoi.atomFaces(index), povfile, rgb, voronoiOptions)
        
        # this is messy...
        # find convex hull of vertices (so can get triangles...)
        # then write to POV file
        pos = np.asarray(voronoi.atomVertices(index))
        pos = pos.flatten()
        facets = clusters.findConvexHullFacets(len(pos) / 3, pos)
        writePOVRayVoroVolumeTriangles(facets, pos, povfile, voronoiOptions, rgb)
    
    actorsDict["Voronoi cells"] = actorsDictLocal
    
    renderVoroTime = time.time() - renderVoroTime
    
    logger.debug("  Render Voronoi time: %f", renderVoroTime)

################################################################################
def writePOVRayVoroVolumeTriangles(facets, pos, filename, settings, colour_rgb):
    """
    Write hull to POV-Ray file.
    
    """
    if len(pos) / 3 < 3:
        pass
    
    else:
        if os.path.exists(filename):
            fh = open(filename, "a")
        
        else:
            fh = open(filename, "w")
        
        # how many vertices
        vertices = set()
        vertexMapper = {}
        NVertices = 0
        for facet in facets:
            for j in xrange(3):
                if facet[j] not in vertices:
                    vertices.add(facet[j])
                    vertexMapper[facet[j]] = NVertices
                    NVertices += 1
        
        # construct mesh
        lines = []
        nl = lines.append
        
        nl("mesh2 {")
        nl("  vertex_vectors {")
        nl("    %d," % NVertices)
        
        count = 0
        for key, value in sorted(vertexMapper.iteritems(), key=lambda (k, v): (v, k)):
            if count == NVertices - 1:
                string = ""
            
            else:
                string = ","
            
            nl("    <%f,%f,%f>%s" % (- pos[3*key], pos[3*key+1], pos[3*key+2], string))
            
            count += 1
        
        nl("  }")
        nl("  face_indices {")
        nl("    %d," % len(facets))
        
        count = 0
        for facet in facets:
            if count == len(facets) - 1:
                string = ""
            
            else:
                string = ","
            
            nl("    <%d,%d,%d>%s" % (vertexMapper[facet[0]], vertexMapper[facet[1]], vertexMapper[facet[2]], string))
            
            count += 1
        
        nl("  }")
        nl("  pigment { color rgbt <%f,%f,%f,%f> }" % (colour_rgb[0], colour_rgb[1], colour_rgb[2], 1.0 - settings.opacity))
        nl("  finish { diffuse 0.4 ambient 0.25 phong 0.9 }")
        nl("}")
        nl("")
        
        fh.write("\n".join(lines))

################################################################################

def writePOVRayVoroVolume(vertices, faces, povfile, colour_rgb, voronoiOptions):
    """
    Write Voronoi volume to POV-Ray file
    
    """
    lines = []
    nl = lines.append
    
    # loop over faces
    for face in faces:
        num_points = len(face)
        
        nl("polygon {")
        nl("  %d," % num_points)
        
        line = ""
        for index in face:
            point = vertices[index]
            
            if len(line):
                line += ","
            else:
                line += "  "
            
            line += "<%f,%f,%f>" % (-point[0], point[1], point[2])
        
        nl(line)
        
        #TODO: make these options...
        nl("  texture {")
        nl("    finish { diffuse 0.4 ambient 0.25 phong 0.9 }")
        nl("    pigment { color rgbt <%f,%f,%f,%f> }" % (colour_rgb[0], colour_rgb[1], colour_rgb[2], 1.0 - voronoiOptions.opacity))
        nl("  }")
        nl("}")
    
    nl("")
    
    if os.path.exists(povfile):
        f = open(povfile, "a")
    else:
        f = open(povfile, "w")
    
    f.write("\n".join(lines))
    
    f.close()
