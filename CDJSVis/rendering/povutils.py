
"""
Utility methods for povray rendering.

@author: Chris Scott

"""





################################################################################

def povrayBond(posa, posb, thickness, rgb, t, phong=0.9, metallic=""):
    """
    Return string for rendering atom in POV-Ray.
    
    """
    line = "cylinder { <%f,%f,%f>,<%f,%f,%f>, %f pigment { color rgbt <%f,%f,%f,%f> } finish { phong %f %s } }\n" % (-posa[0], posa[1], posa[2], 
                                                                                                                     -posb[0], posb[1], posb[2],
                                                                                                                     thickness, 
                                                                                                                     rgb[0], rgb[1], rgb[2], t, 
                                                                                                                     phong, metallic)
    
    return line
