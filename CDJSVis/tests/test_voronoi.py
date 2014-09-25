
"""
Unit tests for Voronoi tessellation

    - test calculate Voronoi
    - test VoronoiResult class
    
    ***** test that sum of voro volumes == volume of lattice *****
    

"""
import os
import unittest
import tempfile
import shutil
   
from ..state import latticeReaders
from ..filtering import voronoi
   

################################################################################

def log_output(*args, **kwargs):
    print args[0]

def log_warning(*args, **kwargs):
    print "DISPLAY WARNING: %s" % args[0]

def log_error(*args, **kwargs):
    print "DISPLAY ERROR: %s" % args[0]

################################################################################
   
def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "testing", path)
   
################################################################################
   
# class TestComputeVoronoi(unittest.TestCase):
#     """
#     Test computeVoronoi
#        
#     """
#     def setUp(self):
#         # tmp dir
#         self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
#         
#         # create reader
#         reader = latticeReaders.LbomdDatReader(self.tmpLocation, log_output, log_warning, log_error)
#         
#         self.lattice = reader.readFile(path_to_file("testVolume.dat"))
#     
#     def tearDown(self):
#         # remove tmp dir
#         shutil.rmtree(self.tmpLocation)
#         
#         self.lattice = None
#     
#     def test_voronoiSuccess(self):
#         """
#         Voronoi (pyvoro) successful
#            
#         """
#         vor = voronoi.computeVoronoi(lattice, voronoiOptions, PBC, log)
        
        
        
        


