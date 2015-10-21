
"""
Unit tests for latticeReaders module

"""
import os
import unittest
import tempfile
import shutil

from ..state import latticeReaders
from ..state.lattice import Lattice


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

def specie_index(specieList, symIn):
    index = None
    for count, sym in enumerate(specieList):
        if sym == symIn:
            index = count
            break
    
    return index

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
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
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
        READDAT gzipped returns Lattice
        
        """
        filename = path_to_file("postcascref.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIsInstance(state, Lattice)
    
    def test_readGzipStatusOk(self):
        """
        READDAT gzipped status ok
        
        """
        filename = path_to_file("postcascref.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(status, 0)
    
    def test_readLatticeInstance(self):
        """
        READDAT returns Lattice
        
        """
        filename = path_to_file("lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIsInstance(state, Lattice)
    
    def test_readStatusOk(self):
        """
        READDAT lattice status ok
        
        """
        filename = path_to_file("lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(status, 0)
    
    def test_NAtoms(self):
        """
        READDAT NAtoms
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.NAtoms, 1140)
    
    def test_cellDims(self):
        """
        READDAT cell dimensions
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.cellDims[0], 26.3781222148)
        self.assertEqual(state.cellDims[1], 26.3781222148)
        self.assertEqual(state.cellDims[2], 26.3781222148)
    
    def test_NSpecies(self):
        """
        READDAT NSpecies
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(len(state.specieList), 3)
    
    def test_specieList(self):
        """
        READDAT specie list
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIn("Si", state.specieList)
        self.assertIn("B_", state.specieList)
        self.assertIn("O_", state.specieList)
    
    def test_specieCount(self):
        """
        READDAT specie count
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Si")], 280)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "B_")], 120)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "O_")], 740)
    
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
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
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
        READREF gzipped returns Lattice
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIsInstance(state, Lattice)
    
    def test_readGzipStatusOk(self):
        """
        READREF gzipped status ok
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(status, 0)
    
    def test_readLatticeInstance(self):
        """
        READREF returns Lattice
        
        """
        filename = path_to_file("animation-reference.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIsInstance(state, Lattice)
    
    def test_readStatusOk(self):
        """
        READREF lattice status ok
        
        """
        filename = path_to_file("animation-reference.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(status, 0)
    
    def test_NAtoms(self):
        """
        READREF NAtoms
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.NAtoms, 16392)
    
    def test_cellDims(self):
        """
        READREF cell dimensions
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.cellDims[0], 74.24)
        self.assertEqual(state.cellDims[1], 74.24)
        self.assertEqual(state.cellDims[2], 74.24)
    
    def test_NSpecies(self):
        """
        READREF NSpecies
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(len(state.specieList), 3)
    
    def test_specieList(self):
        """
        READREF specie list
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIn("Pu", state.specieList)
        self.assertIn("Ga", state.specieList)
        self.assertIn("H_", state.specieList)
    
    def test_specieCount(self):
        """
        READREF specie count
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Pu")], 15565)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Ga")], 819)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "H_")], 8)

################################################################################

def skipIfStatus(status):
    if not status:
        return unittest.skip("Skipping XYZ test because readRef failed")

################################################################################

class TestLbomdXYZReader(unittest.TestCase):
    """
    Test LBOMD DAT reader
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
        # create reader
        refReader = latticeReaders.LbomdRefReader(self.tmpLocation, log_output, log_warning, log_error)
        refStatus, self.refState = refReader.readFile(path_to_file("anim-ref-Hdiff.xyz"))
        
        if refStatus:
            raise unittest.SkipTest("read ref failed ('%s'; %d; %r)" % (path_to_file("anim-ref-Hdiff.xyz"), refStatus, self.refState))
        
        self.reader = latticeReaders.LbomdXYZReader(self.tmpLocation, log_output, log_warning, log_error)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
        
        # remove ref to reader
        self.reader = None
        self.refState = None
    
    def test_readGzipLatticeInstance(self):
        """
        READXYZ gzipped returns Lattice
        
        """
        filename = path_to_file("input-HDiff.xyz")
        
        status, state = self.reader.readFile(filename, self.refState)
        
        self.assertIsInstance(state, Lattice)
    
    def test_readGzipStatusOk(self):
        """
        READXYZ gzipped status ok
        
        """
        filename = path_to_file("input-HDiff.xyz")
        
        status, state = self.reader.readFile(filename, self.refState)
        
        self.assertEqual(status, 0)
    
#     def test_readLatticeInstance(self):
#         """
#         READXYZ returns Lattice
#         
#         """
#         filename = path_to_file("animation-reference.xyz")
#         
#         status, state = self.reader.readFile(filename)
#         
#         self.assertIsInstance(state, Lattice)
#     
#     def test_readStatusOk(self):
#         """
#         READXYZ lattice status ok
#         
#         """
#         filename = path_to_file("animation-reference.xyz")
#         
#         status, state = self.reader.readFile(filename)
#         
#         self.assertEqual(status, 0)
    
    def test_NAtoms(self):
        """
        READXYZ NAtoms
        
        """
        filename = path_to_file("input-HDiff.xyz")
        
        status, state = self.reader.readFile(filename, self.refState)
        
        self.assertEqual(state.NAtoms, 16392)
    
    def test_cellDims(self):
        """
        READXYZ cell dimensions
        
        """
        filename = path_to_file("input-HDiff.xyz")
        
        status, state = self.reader.readFile(filename, self.refState)
        
        self.assertEqual(state.cellDims[0], 74.24)
        self.assertEqual(state.cellDims[1], 74.24)
        self.assertEqual(state.cellDims[2], 74.24)
    
    def test_NSpecies(self):
        """
        READXYZ NSpecies
        
        """
        filename = path_to_file("input-HDiff.xyz")
        
        status, state = self.reader.readFile(filename, self.refState)
        
        self.assertEqual(len(state.specieList), 3)
    
    def test_specieList(self):
        """
        READXYZ specie list
        
        """
        filename = path_to_file("input-HDiff.xyz")
        
        status, state = self.reader.readFile(filename, self.refState)
        
        self.assertIn("Pu", state.specieList)
        self.assertIn("Ga", state.specieList)
        self.assertIn("H_", state.specieList)
    
    def test_specieCount(self):
        """
        READXYZ specie count
        
        """
        filename = path_to_file("input-HDiff.xyz")
        
        status, state = self.reader.readFile(filename, self.refState)
        
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Pu")], 15565)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Ga")], 819)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "H_")], 8)



