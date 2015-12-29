
"""
Classes for writing POV-Ray files from renderers.

"""
import copy

import numpy as np

from ...filtering import clusters
from ...filtering import _clusters


class PovrayAtomsWriter(object):
    """
    Write POV-Ray atoms to file.
    
    """
    def write(self, filename, pointsArray, scalarsArray, radiusArray, scaleFactor, lut, mode="a"):
        """Write to POV-Ray file."""
        # numpy arrays of data
        points = pointsArray.getNumpy()
        scalars = scalarsArray.getNumpy()
        radii = radiusArray.getNumpy()
        
        # array for holding rgb
        rgb = np.empty(3, np.float64)
        
        # open file to write
        # TODO: write in C to improve performance
        with open(filename, mode) as fh:
            # loop over visible atoms
            for i in xrange(len(points)):
                # colour for povray atom
                lut.GetColor(scalars[i], rgb)
                
                # atom line
                line = "sphere { <%f,%f,%f>, %f " % (-points[i][0], points[i][1], points[i][2], radii[i] * scaleFactor)
                line += "pigment { color rgb <%f,%f,%f> } " % (rgb[0], rgb[1], rgb[2])
                line += "finish { ambient %f phong %f } }\n" % (0.25, 0.9)
                
                # povray atom
                fh.write(line)


class PovrayBondsWriter(object):
    """
    Write POV-Ray bonds to file.
    
    """
    def write(self, filename, pointsArray, vectorsArray, scalarsArray, lut, bondThickness, mode="a"):
        """Write to POV-Ray file."""
        # numpy arrays of data
        points = pointsArray.getNumpy()
        scalars = scalarsArray.getNumpy()
        vectors = vectorsArray.getNumpy()
        
        # array for holding rgb
        rgb = np.empty(3, np.float64)
        
        # TODO: make these options
        phong = 0.9
        metallic = ""
        transparency = 0.0
        
        # open file to write
        # TODO: write in C to improve performance
        with open(filename, mode) as fh:
            # loop over visible atoms
            for i in xrange(len(points)):
                # colour for povray bond
                lut.GetColor(scalars[i], rgb)
                
                # end point positions
                posa = points[i]
                posb = posa + vectors[i]
                
                # write to file
                fh.write("cylinder { <%f,%f,%f>,<%f,%f,%f>, %f\n" % (-posa[0], posa[1], posa[2], -posb[0], posb[1],
                                                                     posb[2], bondThickness))
                fh.write("           pigment { color rgbt <%f,%f,%f,%f> }\n" % (rgb[0], rgb[1], rgb[2], transparency))
                fh.write("           finish { phong %f %s } }\n" % (phong, metallic))


class PovrayClustersWriter(object):
    """
    Write clusters to POV-Ray file.
    
    """
    def write(self, filename, clusterList, neighbourRadius, hullOpacity, hullColour, mode="a"):
        """Write to POV-Ray file."""
        # open file for writing
        with open(filename, mode) as filehandle:
            # loop over clusters making poly data
            for clusterIndex, cluster in enumerate(clusterList):
                # get the positions for this cluster
                clusterPos = cluster.makeClusterPos()
                
                # lattice
                lattice = cluster.getLattice()
                
                # get settings and prepare to render (unapply PBCs)
                appliedPBCs = np.zeros(7, np.int32)
                _clusters.prepareClusterToDrawHulls(len(cluster), clusterPos, lattice.cellDims, lattice.PBC,
                                                    appliedPBCs, neighbourRadius)
                
                # render this clusters facets
                self.writeClusterFacets(len(cluster), clusterPos, lattice, neighbourRadius, hullColour, hullOpacity,
                                        filehandle)
                
                # handle PBCs
                if len(cluster) > 1:
                    # move the cluster across each PBC that it overlaps
                    while max(appliedPBCs) > 0:
                        # send the cluster across PBCs
                        tmpClusterPos = copy.deepcopy(clusterPos)
                        clusters.applyPBCsToCluster(tmpClusterPos, lattice.cellDims, appliedPBCs)
                        
                        # render the modified clusters facets
                        self.writeClusterFacets(len(cluster), tmpClusterPos, lattice, neighbourRadius, hullColour,
                                                hullOpacity, filehandle)
    
    def writeClusterFacets(self, clusterSize, clusterPos, lattice, neighbourRadius, hullColour, hullOpacity, fh):
        """Write clusters facets to povray file."""
        # get facets
        facets = clusters.findConvexHullFacets(clusterSize, clusterPos)
        
        if facets is not None:
            # TODO: make sure not facets more than neighbour rad from cell
            facets = clusters.checkFacetsPBCs(facets, clusterPos, 2.0 * neighbourRadius, lattice.PBC, lattice.cellDims)
            
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
                string = "" if count == NVertices - 1 else ","
                nl("    <%f,%f,%f>%s" % (- clusterPos[3 * key], clusterPos[3 * key + 1], clusterPos[3 * key + 2],
                                         string))
                count += 1
            nl("  }")
            nl("  face_indices {")
            nl("    %d," % len(facets))
            for count, facet in enumerate(facets):
                string = "" if count == len(facets) - 1 else ","
                nl("    <%d,%d,%d>%s" % (vertexMapper[facet[0]], vertexMapper[facet[1]], vertexMapper[facet[2]],
                                         string))
            nl("  }")
            nl("  pigment { color rgbt <%f,%f,%f,%f> }" % (hullColour[0], hullColour[1], hullColour[2],
                                                           1.0 - hullOpacity))
            nl("  finish { diffuse 0.4 ambient 0.25 phong 0.9 }")
            nl("}")
            nl("")
            
            fh.write("\n".join(lines))
