
"""
Unit tests for latticeReaders module

"""
import os
import unittest
import tempfile
import shutil

from .. import latticeReaders
from ..lattice import Lattice


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

class TestLbomdDatReader(unittest.TestCase):
    """
    Test LBOMD DAT reader
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
        
        # create reader
        self.reader = latticeReaders.LbomdDatReader(self.tmpLocation, log_output, log_warning, log_error)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
        
        # remove ref to reader
        self.reader = None
    
    def test_readGzipLatticeInstance(self):
        """
        DAT: Read gzipped lattice returns Lattice
        
        """
        filename = path_to_file("postcascref.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIsInstance(state, Lattice)
    
    def test_readGzipStatusOk(self):
        """
        DAT: Read gzipped lattice status ok
        
        """
        filename = path_to_file("postcascref.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(status, 0)
    
    def test_readLatticeInstance(self):
        """
        DAT: Read lattice returns Lattice
        
        """
        filename = path_to_file("lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIsInstance(state, Lattice)
    
    def test_readStatusOk(self):
        """
        DAT: Read lattice status ok
        
        """
        filename = path_to_file("lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(status, 0)
    
    def test_NAtoms(self):
        """
        DAT: NAtoms
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.NAtoms, 1140)
    
    def test_cellDims(self):
        """
        DAT: Cell dimensions
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.cellDims[0], 26.3781222148)
        self.assertEqual(state.cellDims[1], 26.3781222148)
        self.assertEqual(state.cellDims[2], 26.3781222148)
    
    def test_NSpecies(self):
        """
        DAT: NSpecies
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(len(state.specieList), 3)
    
    def test_specieList(self):
        """
        DAT: Specie list
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIn("Si", state.specieList)
        self.assertIn("B_", state.specieList)
        self.assertIn("O_", state.specieList)
    
    def test_getSpecieIndex(self):
        """
        DAT: Get specie index
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.specieList[state.getSpecieIndex("Si")], "Si")
        self.assertEqual(state.specieList[state.getSpecieIndex("B_")], "B_")
        self.assertEqual(state.specieList[state.getSpecieIndex("O_")], "O_")
    
    def test_specieCount(self):
        """
        DAT: Specie count
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.specieCount[state.getSpecieIndex("Si")], 280)
        self.assertEqual(state.specieCount[state.getSpecieIndex("B_")], 120)
        self.assertEqual(state.specieCount[state.getSpecieIndex("O_")], 740)
    
################################################################################

class TestLbomdRefReader(unittest.TestCase):
    """
    Test LBOMD DAT reader
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
        
        # create reader
        self.reader = latticeReaders.LbomdRefReader(self.tmpLocation, log_output, log_warning, log_error)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
        
        # remove ref to reader
        self.reader = None
    
    def test_readGzipLatticeInstance(self):
        """
        REF: Read gzipped lattice returns Lattice
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIsInstance(state, Lattice)
    
    def test_readGzipStatusOk(self):
        """
        REF: Read gzipped lattice status ok
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(status, 0)
    
    def test_readLatticeInstance(self):
        """
        REF: Read lattice returns Lattice
        
        """
        filename = path_to_file("animation-reference.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIsInstance(state, Lattice)
    
    def test_readStatusOk(self):
        """
        REF: Read lattice status ok
        
        """
        filename = path_to_file("animation-reference.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(status, 0)
    
    def test_NAtoms(self):
        """
        REF: NAtoms
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.NAtoms, 16392)
    
    def test_cellDims(self):
        """
        REF: Cell dimensions
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.cellDims[0], 74.24)
        self.assertEqual(state.cellDims[1], 74.24)
        self.assertEqual(state.cellDims[2], 74.24)
    
    def test_NSpecies(self):
        """
        REF: NSpecies
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(len(state.specieList), 3)
    
    def test_specieList(self):
        """
        REF: Specie list
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIn("Pu", state.specieList)
        self.assertIn("Ga", state.specieList)
        self.assertIn("H_", state.specieList)
    
    def test_getSpecieIndex(self):
        """
        REF: Get specie index
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.specieList[state.getSpecieIndex("Pu")], "Pu")
        self.assertEqual(state.specieList[state.getSpecieIndex("Ga")], "Ga")
        self.assertEqual(state.specieList[state.getSpecieIndex("H_")], "H_")
    
    def test_specieCount(self):
        """
        REF: Specie count
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.specieCount[state.getSpecieIndex("Pu")], 15565)
        self.assertEqual(state.specieCount[state.getSpecieIndex("Ga")], 819)
        self.assertEqual(state.specieCount[state.getSpecieIndex("H_")], 8)





