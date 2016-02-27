
"""
Slow tests for filtering systems

"""
import os
import tempfile
import shutil

from PyQt5 import QtCore
from six.moves import range

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
        for i in range(6):
            self.assertTrue(pp.inputState.atomID[flt.visibleAtoms[i]] in atomids)
    
    def test_filterCharge(self):
        """
        GUI: filter charge
           
        """
        # add the atom ID filter
        pp = self.mw.mainToolbar.pipelineList[0]
        flist = pp.filterLists[0]
        flist.addFilter(filterName="Charge")
        item = flist.listItems.item(0)
        item.filterSettings.minChargeSpinBox.setValue(0.0)
        
        # run the filter
        pp.runAllFilterLists()
        
        # check the result
        flt = flist.filterer
        self.assertEqual(len(flt.visibleAtoms), 400)
        for i in range(len(flt.visibleAtoms)):
            self.assertGreaterEqual(pp.inputState.charge[flt.visibleAtoms[i]], 0)
    
    def test_coordinationNumber(self):
        """
        GUI: coordination number
        
        """
        # add the atom ID filter
        pp = self.mw.mainToolbar.pipelineList[0]
        flist = pp.filterLists[0]
        flist.addFilter(filterName="Coordination number")
        item = flist.listItems.item(0)
        item.filterSettings.minCoordNumSpinBox.setValue(0)
        item.filterSettings.maxCoordNumSpinBox.setValue(3)
        item.filterSettings.filteringToggled(QtCore.Qt.Checked)
        
        # run the filter
        pp.runAllFilterLists()
        
        # check the result
        flt = flist.filterer
        self.assertEqual(len(flt.visibleAtoms), 858)
        self.assertEqual(flt.visibleSpecieCount[pp.inputState.getSpecieIndex("Si")], 2)
        self.assertEqual(flt.visibleSpecieCount[pp.inputState.getSpecieIndex("B_")], 116)
        self.assertEqual(flt.visibleSpecieCount[pp.inputState.getSpecieIndex("O_")], 740)
        self.assertTrue("Coordination number" in flt.scalarsDict)
        scalars = flt.scalarsDict["Coordination number"]
        self.assertEqual(len(scalars), 858)
        for val in scalars:
            self.assertLessEqual(val, 3)
            self.assertGreaterEqual(val, 0)


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
        GUI: filter ACNA
        
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
        for i in range(6912):
            self.assertEqual(flt.scalarsDict["ACNA"][i], 1)
        
        # check filtering
        item = flist.listItems.item(0)
        item.filterSettings.filteringToggled(QtCore.Qt.Checked)
        item.filterSettings.visToggled(1, QtCore.Qt.Unchecked)
        
        # run the filter
        pp.runAllFilterLists()
        
        # check the result
        self.assertEqual(len(flt.visibleAtoms), 0)
    
    def test_filterBondOrder(self):
        """
        GUI: filter bond order
        
        """
        # add the filter
        pp = self.mw.mainToolbar.pipelineList[0]
        flist = pp.filterLists[0]
        flist.addFilter(filterName="Bond order")
        
        # run the filter
        pp.runAllFilterLists()
        
        # check the result
        flt = flist.filterer
        self.assertEqual(len(flt.visibleAtoms), 6912)
        self.assertTrue("Q4" in flt.scalarsDict)
        self.assertTrue("Q6" in flt.scalarsDict)
        for i in range(6912):
            self.assertAlmostEqual(flt.scalarsDict["Q4"][i], 0.191, places=3)
            self.assertAlmostEqual(flt.scalarsDict["Q6"][i], 0.575, places=3)


class TestFilteringPuGaRefXyz(base.UsesQApplication):
    """
    Test filtering a system
       
    """
    def setUp(self):
        """
        Set up the test
        
        """
        super(TestFilteringPuGaRefXyz, self).setUp()
        
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
        # main window
        self.mw = mainWindow.MainWindow(None, testing=True)
        self.mw.preferences.renderingForm.maxAtomsAutoRun = 0
        self.mw.show()
        
        # copy a lattices to tmpLocation
        self.reffn = os.path.join(self.tmpLocation, "ref.xyz.gz")
        shutil.copy(path_to_file("animation-reference.xyz.gz"), self.reffn)
        self.inpfn = os.path.join(self.tmpLocation, "inp.xyz")
        shutil.copy(path_to_file("PuGaH0080.xyz"), self.inpfn)
        
        # load Lattice
        pp = self.mw.mainToolbar.pipelineList[0]
        try:
            self.mw.systemsDialog.load_system_form.readerForm.openFile(self.reffn)
            state = pp.inputState
            err = False
            if not isinstance(state, Lattice):
                err = True
            elif state.NAtoms != 55296:
                err = True
            if not err:
                self.mw.systemsDialog.load_system_form.readerForm.openFile(self.inpfn)
                pp.inputChanged(1)
                state = pp.inputState
                err = False
                if not isinstance(state, Lattice):
                    err = True
                elif state.NAtoms != 55296:
                    err = True
        except:
            self.fail("Loading Lattices failed")
        else:
            if err:
                self.fail("Loading Lattices failed")
    
    def tearDown(self):
        """
        Tidy up
        
        """
        super(TestFilteringPuGaRefXyz, self).tearDown()
        
        # remove refs
        self.reffn = None
        self.inpfn = None
        self.mw.close()
        self.mw = None
        
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
    
    def test_filterSpecies(self):
        """
        GUI: filter species
        
        """
        # add the filter
        pp = self.mw.mainToolbar.pipelineList[0]
        flist = pp.filterLists[0]
        flist.addFilter(filterName="Species")
        item = flist.listItems.item(0)
        inputState = pp.inputState
        item.filterSettings._settings.updateSetting("visibleSpeciesList", ["Ga"])

        # run the filter
        pp.runAllFilterLists()
        
        # check the result
        flt = flist.filterer
        self.assertEqual(len(flt.visibleAtoms), 2764)
        for i in range(len(flt.visibleAtoms)):
            self.assertEqual(inputState.atomSym(flt.visibleAtoms[i]), "Ga")
    
    def test_filterSpeciesClusters(self):
        """
        GUI: filter species + cluster
        
        """
        # add the species filter
        pp = self.mw.mainToolbar.pipelineList[0]
        flist = pp.filterLists[0]
        flist.addFilter(filterName="Species")
        item = flist.listItems.item(0)
        inputState = pp.inputState
        item.filterSettings._settings.updateSetting("visibleSpeciesList", ["Ga"])
        
        # add the clusters filter
        flist.addFilter(filterName="Cluster")
        
        # run the filter
        pp.runAllFilterLists()
        
        # check the result
        flt = flist.filterer
        self.assertEqual(len(flt.visibleAtoms), 923)
        for i in range(len(flt.visibleAtoms)):
            self.assertEqual(inputState.atomSym(flt.visibleAtoms[i]), "Ga")
