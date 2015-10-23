
"""
Unit tests for Lattice object

"""
import os
import unittest
import tempfile
import shutil

from ..latticeReaders import LbomdDatReader, basic_displayError, basic_displayWarning, basic_log

################################################################################
   
def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing", path)
   
################################################################################
   
class TestLattice(unittest.TestCase):
    """
    Test Lattice
        
    """
    def setUp(self):
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
         
        # create reader
        reader = LbomdDatReader(self.tmpLocation, basic_log, basic_displayWarning, basic_displayError)
         
        status, self.lattice = reader.readFile(path_to_file("lattice.dat"))
        if status:
            self.fail("Error reading in Lattice")
     
    def tearDown(self):
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
         
        self.lattice = None
     
    def test_latticeVolume(self):
        """
        Lattice volume
            
        """
        lattice = self.lattice
        
        v1 = lattice.volume()
        v2 = lattice.cellDims[0] * lattice.cellDims[1] * lattice.cellDims[2]
        
        self.assertEqual(v1, v2)
