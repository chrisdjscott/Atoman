  
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

class TestReloadSystem(base.UsesQApplication):
    """
    Test reloading a system
       
    """
    def setUp(self):
        """
        Set up the test
        
        """
        super(TestReloadSystem, self).setUp()
        
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
        
        # main window
        self.mw = mainWindow.MainWindow(None, testing=True)
        self.mw.preferences.renderingForm.maxAtomsAutoRun = 0
        self.mw.show()
        
        # copy a lattice to tmpLocation
        self.fn = os.path.join(self.tmpLocation, "testLattice.dat")
        shutil.copy(path_to_file("kenny_lattice.dat"), self.fn)
        
        # load Lattice
        try:
            self.mw.systemsDialog.load_system_form.readerForm.openFile(self.fn)
            state = self.mw.mainToolbar.pipelineList[0].inputState
            err = False
            if not isinstance(state, Lattice):
                err = True
            elif state.NAtoms != 1140:
                err = True
            if err:
                self.fail("Loading Lattice failed")
        except:
            self.fail("Loading Lattice failed")
    
    def tearDown(self):
        """
        Tidy up
        
        """
        super(TestReloadSystem, self).tearDown()
        
        # remove refs
        self.fn = None
        self.mw.close()
        self.mw = None
        
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
    
    def test_reloadSystem(self):
        """
        GUI: Reload system
           
        """
        # just reload
        self.mw.systemsDialog.systems_list_widget.setCurrentRow(0)
        self.mw.systemsDialog.reload_system()
        
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
        
        # modify
        with open(self.fn) as f:
            lines = f.readlines()
        array = lines[2].strip().split()
        array[0] = "Fe"
        lines[2] = " ".join(array)
        lines[2] += "\n"
        with open(self.fn, "w") as f:
            f.write("".join(lines))
        
        # reload
        self.mw.systemsDialog.systems_list_widget.setCurrentRow(0)
        self.mw.systemsDialog.reload_system()
        
        state = self.mw.mainToolbar.pipelineList[0].inputState
        self.assertIsInstance(state, Lattice)
        self.assertEqual(state.NAtoms, 1140)
        self.assertTrue(np.allclose([26.3781222148, 26.3781222148, 26.3781222148], state.cellDims))
        self.assertTrue("Si" in state.specieList)
        self.assertTrue("B_" in state.specieList)
        self.assertTrue("O_" in state.specieList)
        self.assertTrue("Fe" in state.specieList)
        indx = np.where(state.specieList == "Si")[0][0]
        self.assertEqual(state.specieCount[indx], 279)
        indx = np.where(state.specieList == "B_")[0][0]
        self.assertEqual(state.specieCount[indx], 120)
        indx = np.where(state.specieList == "O_")[0][0]
        self.assertEqual(state.specieCount[indx], 740)
        indx = np.where(state.specieList == "Fe")[0][0]
        self.assertEqual(state.specieCount[indx], 1)
