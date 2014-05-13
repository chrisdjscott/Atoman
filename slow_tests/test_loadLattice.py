  
"""
Slow tests for loading systems
   
"""
import os
import sys
import unittest
import tempfile
import shutil
   
from PySide import QtGui
   
from CDJSVis import mainWindow
from CDJSVis.state.lattice import Lattice

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
   
class TestLoadLattice(unittest.TestCase):
    """
    Test loading lattices
       
    """
    def setUp(self):
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="CDJSVisTest")
        
        self.mw = mainWindow.MainWindow(None)
        self.mw.preferences.renderingForm.maxAtomsAutoRun = 0
        self.mw.show()
    
    def tearDown(self):
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
    
    def test_loadLbomdDat(self):
        """
        Load LBOMD DAT file
           
        """
        status = self.mw.systemsDialog.load_system_form.readerForms["LBOMD DAT"].openFile(filename=path_to_file("kenny_lattice.dat"))
        
        self.assertEqual(status, 0)
        self.assertIsInstance(self.mw.mainToolbar.pipelineList[0].inputState, Lattice)
    
    def test_loadLbomdRef(self):
        """
        Load LBOMD Ref file
           
        """
        status = self.mw.systemsDialog.load_system_form.readerForms["LBOMD REF"].openFile(filename=path_to_file("anim-ref-Hdiff.xyz"))
        
        self.assertEqual(status, 0)
        self.assertIsInstance(self.mw.mainToolbar.pipelineList[0].inputState, Lattice)
    
    def test_loadLbomdXYZ(self):
        """
        Load LBOMD XYZ file
            
        """
        status = self.mw.systemsDialog.load_system_form.readerForms["LBOMD XYZ"].openFile(filename=path_to_file("anim-ref-Hdiff.xyz"), isRef=True)
        self.assertEqual(status, 0)
        
        status = self.mw.systemsDialog.load_system_form.readerForms["LBOMD XYZ"].openFile(filename=path_to_file("input-HDiff.xyz"))
        self.assertEqual(status, 0)
        self.assertIsInstance(self.mw.mainToolbar.pipelineList[0].inputState, Lattice)
    
    def test_loadLbomdDat_auto(self):
        """
        Load LBOMD DAT file (AUTO DETECT)
           
        """
        status = self.mw.systemsDialog.load_system_form.readerForms["AUTO DETECT"].openFile(filename=path_to_file("kenny_lattice.dat"))
        
        self.assertEqual(status, 0)
        self.assertIsInstance(self.mw.mainToolbar.pipelineList[0].inputState, Lattice)
    
    def test_loadLbomdRef_auto(self):
        """
        Load LBOMD Ref file (AUTO DETECT)
           
        """
        status = self.mw.systemsDialog.load_system_form.readerForms["AUTO DETECT"].openFile(filename=path_to_file("anim-ref-Hdiff.xyz"))
        
        self.assertEqual(status, 0)
        self.assertIsInstance(self.mw.mainToolbar.pipelineList[0].inputState, Lattice)
    
    def test_loadLbomdXYZ_auto(self):
        """
        Load LBOMD XYZ file (AUTO DETECT)
            
        """
        status = self.mw.systemsDialog.load_system_form.readerForms["AUTO DETECT"].openFile(filename=path_to_file("anim-ref-Hdiff.xyz"))
        self.assertEqual(status, 0)
        
        status = self.mw.systemsDialog.load_system_form.readerForms["AUTO DETECT"].openFile(filename=path_to_file("input-HDiff.xyz"))
        self.assertEqual(status, 0)
        self.assertIsInstance(self.mw.mainToolbar.pipelineList[0].inputState, Lattice)
