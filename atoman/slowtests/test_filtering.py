  
"""
Slow tests for filtering systems
   
"""
import os
import tempfile
import shutil

from PySide import QtCore

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
        except:
            self.fail("Loading Lattice failed")
        else:
            if err:
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


class TestFilteringGoldLattice(base.UsesQApplication):
    """
    Test filtering a system
       
    """
    def setUp(self):
        """
        Set up the test
        
        """
        super(TestFilteringGoldLattice, self).setUp()
        
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
        # main window
        self.mw = mainWindow.MainWindow(None, testing=True)
        self.mw.preferences.renderingForm.maxAtomsAutoRun = 0
        self.mw.show()
        
        # copy a lattice to tmpLocation
        self.fn = os.path.join(self.tmpLocation, "testLattice.dat")
        shutil.copy(path_to_file("lattice.dat"), self.fn)
        
        # load Lattice
        try:
            self.mw.systemsDialog.load_system_form.readerForm.openFile(self.fn)
            state = self.mw.mainToolbar.pipelineList[0].inputState
            err = False
            if not isinstance(state, Lattice):
                err = True
            elif state.NAtoms != 6912:
                err = True
        except:
            self.fail("Loading Lattice failed")
        else:
            if err:
                self.fail("Loading Lattice failed")
    
    def tearDown(self):
        """
        Tidy up
        
        """
        super(TestFilteringGoldLattice, self).tearDown()
        
        # remove refs
        self.fn = None
        self.mw.close()
        self.mw = None
        
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
    
    def test_filterAcna(self):
        """
        GUI: Filter ACNA
        
        """
        # add the filter
        pp = self.mw.mainToolbar.pipelineList[0]
        flist = pp.filterLists[0]
        flist.addFilter(filterName="ACNA")
        
        # run the filter
        pp.runAllFilterLists()
        
        # check the result
        flt = flist.filterer
        self.assertEqual(len(flt.visibleAtoms), 6912)
        self.assertTrue("ACNA" in flt.scalarsDict)
        for i in xrange(6912):
            self.assertEqual(flt.scalarsDict["ACNA"][i], 1)
        
        # check filtering
        item = flist.listItems.item(0)
        item.filterSettings.filteringToggled(QtCore.Qt.Checked)
        item.filterSettings.visToggled(1, QtCore.Qt.Unchecked)
        
        # run the filter
        pp.runAllFilterLists()
        
        # check the result
        self.assertEqual(len(flt.visibleAtoms), 0)
