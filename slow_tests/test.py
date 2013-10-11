  
"""
Unit tests for filterer module
   
"""
import os
import sys
import unittest
import tempfile
import shutil
   
from PySide import QtGui
   
from CDJSVis import mainWindow
from CDJSVis.lattice import Lattice

app = QtGui.QApplication(sys.argv)
   

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
   
class TestMW(unittest.TestCase):
    """
    Test filterer
       
    """
    def setUp(self):
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
        
        self.mw = mainWindow.MainWindow()
    
    def tearDown(self):
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
    
    def test_loadLattice(self):
        """
        Load lattice
           
        """
        status = self.mw.systemsDialog.load_system_form.lbomdDatWidget.openFile(filename=path_to_file("kenny_lattice.dat"))
        
        self.assertEqual(status, 0)
        self.assertIsInstance(self.mw.mainToolbar.pipelineList[0].inputState, Lattice)
