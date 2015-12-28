
"""
Classes for writing POV-Ray files from renderers.

"""
import numpy as np


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
