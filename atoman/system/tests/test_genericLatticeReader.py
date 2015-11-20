
"""
Unit tests for generic lattice reader

"""
import os
import unittest
import tempfile
import shutil

import numpy as np

from .. import latticeReaderGeneric
from ..lattice import Lattice


################################################################################

def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..",  "testing", path)

def updateProgress(a, b, msg):
    pass

def hideProgess():
    pass

################################################################################

class TestFileFormats(unittest.TestCase):
    """
    Test file formats
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
        # write default file formats file
        self.fffn = os.path.join(self.tmpLocation, "file_formats.IN")
        with open(self.fffn, "w") as fh:
            fh.write(latticeReaderGeneric._defaultFileFormatsFile)
        
        # file formats object
        self.ffs = latticeReaderGeneric.FileFormats()
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
        
        # remove refs
        self.ffs = None
        self.fffn = None
    
    def test_readFileFormats(self):
        """
        Read file formats
        
        """
        # raises exception if something goes wrong
        self.ffs.read(self.fffn)
        
        # check the formats are all in
        self.assertTrue("LBOMD Lattice" in self.ffs)
        self.assertTrue("LBOMD REF" in self.ffs)
        self.assertTrue("LBOMD XYZ" in self.ffs)
        self.assertTrue("LBOMD XYZ (Velocity)" in self.ffs)
        self.assertTrue("LBOMD XYZ (Charge)" in self.ffs)
        self.assertTrue("Indenter" in self.ffs)
        self.assertTrue("FAILSAFE" in self.ffs)

################################################################################

class TestReadLatticeGeneric(unittest.TestCase):
    """
    Test read lattice generic
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
        # write default file formats file
        fn = os.path.join(self.tmpLocation, "file_formats.IN")
        with open(fn, "w") as fh:
            fh.write(latticeReaderGeneric._defaultFileFormatsFile)
        
        # file formats object
        self.ffs = latticeReaderGeneric.FileFormats()
        self.ffs.read(fn)
        
        # lattice reader
        self.reader = latticeReaderGeneric.LatticeReaderGeneric(self.tmpLocation, updateProgress=updateProgress, hideProgress=hideProgess)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
        
        # remove refs
        self.ffs = None
        self.reader = None
    
    def test_readGenericLbomdLattice(self):
        """
        Generic reader: LBOMD Lattice
        
        """
        fn = path_to_file("kenny_lattice.dat")
        fmt = self.ffs.getFormat("LBOMD Lattice")
        
        status, state = self.reader.readFile(fn, fmt)
        
        self.assertEqual(status, 0)
        self.assertIsInstance(state, Lattice)
        self.assertEqual(state.NAtoms, 1140)
        self.assertTrue(np.allclose([26.3781222148, 26.3781222148, 26.3781222148], state.cellDims))
        self.assertTrue("Si" in state.specieList)
        self.assertTrue("B_" in state.specieList)
        self.assertTrue("O_" in state.specieList)
        indx = np.where(state.specieList == "Si")[0][0]
        self.assertEqual(state.specieCount[indx], 280)
        indx = np.where(state.specieList == "B_")[0][0]
        self.assertEqual(state.specieCount[indx], 120)
        indx = np.where(state.specieList == "O_")[0][0]
        self.assertEqual(state.specieCount[indx], 740)
    
    def test_readGenericLbomdRef(self):
        """
        Generic reader: LBOMD REF
        
        """
        fn = path_to_file("anim-ref-Hdiff.xyz.gz")
        fmt = self.ffs.getFormat("LBOMD REF")
        
        status, state = self.reader.readFile(fn, fmt)
        
        self.assertEqual(status, 0)
        self.assertIsInstance(state, Lattice)
        self.assertEqual(state.NAtoms, 16392)
        self.assertTrue(np.allclose([74.24, 74.24, 74.24], state.cellDims))
        self.assertTrue("Ga" in state.specieList)
        self.assertTrue("Pu" in state.specieList)
        self.assertTrue("H_" in state.specieList)
        indx = np.where(state.specieList == "Ga")[0][0]
        self.assertEqual(state.specieCount[indx], 819)
        indx = np.where(state.specieList == "Pu")[0][0]
        self.assertEqual(state.specieCount[indx], 15565)
        indx = np.where(state.specieList == "H_")[0][0]
        self.assertEqual(state.specieCount[indx], 8)
