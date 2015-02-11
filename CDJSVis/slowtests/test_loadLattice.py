  
"""
Slow tests for loading systems
   
"""
import os
import tempfile
import shutil

import numpy as np

from . import base
from .. import mainWindow
from ..state.lattice import Lattice


################################################################################

def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "testing", path)

################################################################################

class TestLoadLattice(base.UsesQApplication):
    """
    Test loading a Lattice
       
    """
    def setUp(self):
        """
        Set up the test
        
        """
        super(TestLoadLattice, self).setUp()
        
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
        
        # main window
        self.mw = mainWindow.MainWindow(None)
        self.mw.preferences.renderingForm.maxAtomsAutoRun = 0
        self.mw.show()
    
    def tearDown(self):
        """
        Tidy up
        
        """
        super(TestLoadLattice, self).tearDown()
        
        # remove refs
        self.mw = None
        
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
    
    def test_loadLbomdDat(self):
        """
        GUI: Load LBOMD Lattice
           
        """
        self.mw.systemsDialog.load_system_form.readerForm.openFile(path_to_file("kenny_lattice.dat"))
        
        state = self.mw.mainToolbar.pipelineList[0].inputState
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
    
    def test_loadLbomdRef(self):
        """
        GUI: Load LBOMD REF
           
        """
        self.mw.systemsDialog.load_system_form.readerForm.openFile(path_to_file("anim-ref-Hdiff.xyz.gz"))
        
        state = self.mw.mainToolbar.pipelineList[0].inputState
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
    
#     def test_displayAtomsUnfiltered(self):
#         """
#         GUI: Display atoms unfiltered
#         
#         """
#         self.mw.systemsDialog.load_system_form.readerForm.openFile(path_to_file("kenny_lattice.dat"))
#         
#         state = self.mw.mainToolbar.pipelineList[0].inputState
#         self.assertIsInstance(state, Lattice)
#         self.mw.mainToolbar.pipelineList[0].runAllFilterLists()
        
