  
"""
Slow tests for rendering bonds
   
"""
import os
import sys
import unittest
import tempfile
import shutil
   
from PySide import QtGui
   
from CDJSVis import mainWindow
from CDJSVis.lattice import Lattice

try:
    app = QtGui.QApplication(sys.argv)
except RuntimeError:
    pass

################################################################################

def log_output(*args, **kwargs):
    print args[0]

def log_warning(*args, **kwargs):
    print "DISPLAY WARNING: %s" % args[0]

def log_error(*args, **kwargs):
    print "DISPLAY ERROR: %s" % args[0]

################################################################################
   
def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "testing", path)
   
################################################################################
   
class TestRenderBonds(unittest.TestCase):
    """
    Test rendering bonds
       
    """
    def setUp(self):
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
        
        self.mw = mainWindow.MainWindow(None)
        self.mw.preferences.renderingForm.maxAtomsAutoRun = 0
        self.mw.show()
        
        status = self.mw.systemsDialog.load_system_form.readerForms["AUTO DETECT"].openFile(filename=path_to_file("kenny_lattice.dat"))
        if status:
            raise unittest.SkipTest("Load lattice failed (%d)" % status)
    
    def tearDown(self):
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
    
    def test_renderBonds(self):
        """
        Render bonds
           
        """
        # pipeline page
        pp = self.mw.mainToolbar.pipelineList[0]
        
        # filter list
        fl = pp.filterLists[0]
        
        # bonds settings
        bo = fl.bondsOptions
        
        # set draw bonds check
        bo.drawBonds = True
        
        # set Si-O_ and B_-O_
        for i, pair in enumerate(bo.bondPairsList):
            if "Si" in pair and "O_" in pair:
                bo.bondPairDrawStatus[i] = True
            elif "B_" in pair and "O_" in pair:
                bo.bondPairDrawStatus[i] = True
        
        for pair, status in zip(bo.bondPairsList, bo.bondPairDrawStatus):
            print "PAIR: %r; DRAW: %r" % (pair, status)
        
        # run filter lists
        pp.runAllFilterLists()
        self.assertEqual(0, 1, "OCH")


