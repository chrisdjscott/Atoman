  
"""
Unit tests for filterer module
   
"""
import os
import unittest
import tempfile
import shutil
from .. import clusters

################################################################################

def log_output(*args, **kwargs):
    print args[0]

def log_warning(*args, **kwargs):
    print "DISPLAY WARNING: %s" % args[0]

def log_error(*args, **kwargs):
    print "DISPLAY ERROR: %s" % args[0]

################################################################################
   
def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing", path)
   
################################################################################
   
class TestClusters(unittest.TestCase):
    """
    Test filterer
       
    """
    def setUp(self):
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
    
    def tearDown(self):
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
    
    def test_convexHullFacetArea(self):
        """
        Convex hull facet area
           
        """
        try:
            import pyhull
        except ImportError:
            self.skipTest("Pyhull (optional) is not available so cannot compute facet area.")
        
        # make points
        N = 8
        pts = [0,0,0,
               3,0,0,
               0,3,0,
               0,0,3,
               3,3,0,
               0,3,3,
               3,0,3,
               3,3,3]
        
        # calc volume
        volume, facetArea = clusters.findConvexHullVolume(N, pts)
        
        self.assertAlmostEqual(facetArea, 54.0)
    
    def test_convexHullVolume(self):
        """
        Convex hull volume
           
        """
        # make points
        N = 8
        pts = [0,0,0,
               3,0,0,
               0,3,0,
               0,0,3,
               3,3,0,
               0,3,3,
               3,0,3,
               3,3,3]
        
        # calc volume
        volume, facetArea = clusters.findConvexHullVolume(N, pts)
        
        self.assertAlmostEqual(volume, 27.0)
     

