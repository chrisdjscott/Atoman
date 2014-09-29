
"""
Unit tests for Voronoi tessellation

    - test Voronoi class

"""
import os
import unittest
import tempfile
import shutil

import numpy as np

from ..state.latticeReaders import LbomdDatReader, basic_displayError, basic_displayWarning, basic_log
from ..filtering import _voronoi

################################################################################
   
def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "testing", path)
   
################################################################################
   
class TestVoronoi(unittest.TestCase):
    """
    Test Voronoi
        
    """
    def setUp(self):
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
         
        # create reader
        reader = LbomdDatReader(self.tmpLocation, basic_log, basic_displayWarning, basic_displayError)
         
        status, self.lattice = reader.readFile(path_to_file("lattice.dat"))
        if status:
            self.fail("Error reading in Lattice")
     
    def tearDown(self):
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
         
        self.lattice = None
     
    def test_voronoiSumVolumes(self):
        """
        Voronoi volume sum
            
        """
        lattice = self.lattice
        
        vor = _voronoi.Voronoi()
        
        PBC = np.ones(3, np.int32)
        useRadii = 0
        vor.computeVoronoi(lattice.pos, lattice.minPos, lattice.maxPos, lattice.cellDims, PBC, lattice.specie, 
                           lattice.specieCovalentRadius, 10, useRadii)
        
        vols = vor.atomVolumesArray()
        volsum = np.sum(vols)
        
        V = self.lattice.volume()
        
        self.assertAlmostEqual(volsum, V)
    
    def test_voronoiSumVolumesRadii(self):
        """
        Voronoi volume sum (radii)
            
        """
        lattice = self.lattice
        
        vor = _voronoi.Voronoi()
        
        PBC = np.ones(3, np.int32)
        useRadii = 1
        vor.computeVoronoi(lattice.pos, lattice.minPos, lattice.maxPos, lattice.cellDims, PBC, lattice.specie, 
                           lattice.specieCovalentRadius, 10, useRadii)
        
        vols = vor.atomVolumesArray()
        volsum = np.sum(vols)
        
        V = self.lattice.volume()
        
        self.assertAlmostEqual(volsum, V)
    
#     def test_resultOrder(self):
#         """
#         Voronoi result order
#         
#         """
        
################################################################################
   
class TestVoronoi2(unittest.TestCase):
    """
    Test Voronoi2
        
    """
    def setUp(self):
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
         
        # create reader
        reader = LbomdDatReader(self.tmpLocation, basic_log, basic_displayWarning, basic_displayError)
         
        status, self.lattice = reader.readFile(path_to_file("kenny_lattice.dat"))
        if status:
            self.fail("Error reading in Lattice")
     
    def tearDown(self):
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
         
        self.lattice = None
     
    def test_voronoiSumVolumes(self):
        """
        Voronoi volume sum 2
            
        """
        lattice = self.lattice
        
        vor = _voronoi.Voronoi()
        
        PBC = np.ones(3, np.int32)
        useRadii = 0
        vor.computeVoronoi(lattice.pos, lattice.minPos, lattice.maxPos, lattice.cellDims, PBC, lattice.specie, 
                           lattice.specieCovalentRadius, 10, useRadii)
        
        vols = vor.atomVolumesArray()
        volsum = np.sum(vols)
        
        V = self.lattice.volume()
        
        self.assertAlmostEqual(volsum, V)
    
    def test_voronoiSumVolumesRadii(self):
        """
        Voronoi volume sum (radii) 2
            
        """
        lattice = self.lattice
        
        vor = _voronoi.Voronoi()
        
        PBC = np.ones(3, np.int32)
        useRadii = 1
        vor.computeVoronoi(lattice.pos, lattice.minPos, lattice.maxPos, lattice.cellDims, PBC, lattice.specie, 
                           lattice.specieCovalentRadius, 10, useRadii)
        
        vols = vor.atomVolumesArray()
        volsum = np.sum(vols)
        
        V = self.lattice.volume()
        
        self.assertAlmostEqual(volsum, V)
