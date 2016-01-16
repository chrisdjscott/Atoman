  
"""
Slow tests for filtering systems
   
"""
import os
import tempfile
import shutil

import numpy as np

from . import base
from ..gui import mainWindow
from ..system.lattice import Lattice


def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "testing", path)


class TestFilteringKennyLattice(base.UsesQApplication):
    """
    Test filtering a system
       
    """
    def setUp(self):
        """
        Set up the test
        
        """
        super(TestFilteringKennyLattice, self).setUp()
        
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
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
        super(TestFilteringKennyLattice, self).tearDown()
        
        # remove refs
        self.fn = None
        self.mw.close()
        self.mw = None
        
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
    
    def test_filterAtomID(self):
        """
        GUI: filter atom ID
           
        """
        # add the atom ID filter
        pp = self.mw.mainToolbar.pipelineList[0]
        flist = pp.filterLists[0]
        flist.addFilter(filterName="Atom ID")
        item = flist.listItems.item(0)
        item.filterSettings.lineEdit.setText("104,1,4-7")
        item.filterSettings.lineEdit.editingFinished.emit()
        
        # run the filter
        pp.runAllFilterLists()
        
        # check the result
        flt = flist.filterer
        atomids = (104, 1, 4, 5, 6, 7)
        self.assertEqual(len(flt.visibleAtoms), 6)
        for i in xrange(6):
            self.assertTrue(pp.inputState.atomID[flt.visibleAtoms[i]] in atomids)
