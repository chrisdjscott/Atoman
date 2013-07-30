
"""
The systems dialog.

From here you can load/generate a lattice;
view loaded lattices; set input/ref system, etc

@author: Chris Scott

"""
import sys

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from .genericForm import GenericForm
from . import latticeReaderForms
from . import latticeGeneratorForms

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class GenerateInputForm(GenericForm):
    """
    Form for generating a new system
    
    """
    def __init__(self, parent, mainWindow, mainToolbar):
        super(GenerateInputForm, self).__init__(parent, None, "Generate new system")
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.mainToolbar = mainToolbar
        
        # system type combo
        self.inputTypeCombo = QtGui.QComboBox()
        self.inputTypeCombo.addItem("Pu-Ga (L12 method)")
#         self.inputTypeCombo.addItem("FCC")
#         self.inputTypeCombo.addItem("Rock salt (eg. MgO)")
#         self.inputTypeCombo.addItem("Fluorite")
#         self.inputTypeCombo.addItem("Pyrochlore")
#         self.inputTypeCombo.addItem("6H")
        self.inputTypeCombo.currentIndexChanged.connect(self.setWidgetStack)
        
        row = self.newRow()
        row.addWidget(self.inputTypeCombo)
        
        self.show()
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget()
        
        row = self.newRow()
        row.addWidget(self.stackedWidget)
        
        # Pu3Ga generator
        self.pu3ga_generator = latticeGeneratorForms.Pu3GaLatticeGenerator(self, self.mainWindow)
        self.stackedWidget.addWidget(self.pu3ga_generator)
        
    
    def file_generated(self, lattice):
        """
        File generated
        
        """
        self.parent.file_generated(lattice)
    
    def setWidgetStack(self, index):
        """
        Set stack
        
        """
        self.stackedWidget.setCurrentIndex(index)
    

################################################################################

class LoadSystemForm(GenericForm):
    """
    Form for reading new lattice.
    
    """
    def __init__(self, parent, mainWindow, mainToolbar):
        super(LoadSystemForm, self).__init__(parent, None, "Load new system")
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.mainToolbar = mainToolbar
        
        # file type combo
        self.inputTypeCombo = QtGui.QComboBox()
        self.inputTypeCombo.addItem("LBOMD DAT")
        self.inputTypeCombo.addItem("LBOMD REF")
        self.inputTypeCombo.addItem("LBOMD XYZ")
        self.inputTypeCombo.currentIndexChanged.connect(self.setWidgetStack)
        
        row = self.newRow()
        row.addWidget(self.inputTypeCombo)
        
        # stacked widget
        self.widgetStack = QtGui.QStackedWidget()
        
        self.lbomdDatWidget = latticeReaderForms.LbomdDatReaderForm(self, self.mainToolbar, self.mainWindow, None, "ref")
        self.widgetStack.addWidget(self.lbomdDatWidget)
        
        self.lbomdRefWidget = latticeReaderForms.LbomdRefReaderForm(self, self.mainToolbar, self.mainWindow, None, "ref")
        self.widgetStack.addWidget(self.lbomdRefWidget)
        
        self.lbomdXyzWidget = latticeReaderForms.LbomdXYZReaderForm(self, self.mainToolbar, self.mainWindow, None, "ref")
        self.widgetStack.addWidget(self.lbomdXyzWidget)
        
        row = self.newRow()
        row.addWidget(self.widgetStack)
        
        self.show()
    
    def setWidgetStack(self, index):
        """
        Change load ref stack.
        
        """
#         ok = self.okToChangeFileType() 
        ok = True
        
        if ok:
            self.widgetStack.setCurrentIndex(index)
#             self.refTypeCurrentIndex = index
            
#             if index == 0:
#                 self.inputTypeCombo.setCurrentIndex(0)
#             
#             elif index == 1:
#                 self.inputTypeCombo.setCurrentIndex(2)
    
    def fileLoaded(self, fileType, state, filename, extension):
        """
        Called when a file is loaded
        
        """
        print "FILE HAS BEEN LOADED... DO SOMETHING"
        
        self.parent.file_loaded(state, filename, extension)
        

################################################################################

class SystemsDialog(QtGui.QDialog):
    """
    Systems dialog
    
    """
    def __init__(self, parent, mainWindow):
        super(SystemsDialog, self).__init__(parent)
        
        self.setWindowTitle("Systems dialog")
        self.setModal(False)
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        
        self.resize(80, 120)
        
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        # dict for storing loaded systems
        self.lattice_list = []
        self.filenames_list = []
        self.extensions_list = []
        
        # defaults
        self.ref_selected = False
        self.input_selected = True
        self.ref_index = None
        self.input_index = None
        
        # dialog layout
        dialog_layout = QtGui.QVBoxLayout(self)
        
        # box for list of loaded systems
        list_holder = GenericForm(self, None, "Loaded systems")
        list_holder.show()
        dialog_layout.addWidget(list_holder)
        
        # add list widget
        self.systems_list_widget = QtGui.QListWidget(self)
#         self.systems_list_widget.setFixedHeight(60)
        
        row = list_holder.newRow()
        row.addWidget(self.systems_list_widget)
        
        # set as ref button?
        set_ref_button = QtGui.QPushButton("Set as ref")
        set_ref_button.setToolTip("Set as ref")
        set_ref_button.clicked.connect(self.set_ref)
        
        # set as input button?
        set_input_button = QtGui.QPushButton("Set as input")
        set_input_button.setToolTip("Set as input")
        set_input_button.clicked.connect(self.set_input)
        
        # remove system button
        remove_system = QtGui.QPushButton(QtGui.QIcon(iconPath("list-remove.svg")), "")
        remove_system.setToolTip("Remove system")
        remove_system.clicked.connect(self.remove_system)
        
        row = list_holder.newRow()
        row.addWidget(set_ref_button)
        row.addWidget(set_input_button)
        row.addWidget(remove_system)
        
        # box for new system stuff
        new_holder = GenericForm(self, None, "New system")
        new_holder.show()
        dialog_layout.addWidget(new_holder)
        
        # load or generate combo
        self.new_type_combo = QtGui.QComboBox()
        self.new_type_combo.addItem("Load system")
        self.new_type_combo.addItem("Generate system")
        self.new_type_combo.currentIndexChanged.connect(self.set_new_system_stack)
        
        row = new_holder.newRow()
        row.addWidget(self.new_type_combo)
        
        # stacked widget
        self.new_system_stack = QtGui.QStackedWidget()
        
        row = new_holder.newRow()
        row.addWidget(self.new_system_stack)
        
        # load input form
        self.load_system_form = LoadSystemForm(self, self.mainWindow, self.mainToolbar)
        self.new_system_stack.addWidget(self.load_system_form)
        
        # generate input form
        self.generate_system_form = GenerateInputForm(self, self.mainWindow, self.mainToolbar)
        self.new_system_stack.addWidget(self.generate_system_form)
    
    def refresh_type_text(self):
        """
        Refresh input/ref marker
        
        """
        if not self.systems_list_widget.count():
            return
        
        for i in xrange(self.systems_list_widget.count()):
            item_text = "%s (%d atoms)" % (self.filenames_list[i], self.lattice_list[i].NAtoms)
            
            if i == self.ref_index:
                item_text += " [REF]"
            
            if i == self.input_index:
                item_text += " [INPUT]"
            
            item = self.systems_list_widget.item(i)
            
            item.setText(item_text)
    
    def set_ref(self, index=None):
        """
        Set reference lattice
        
        """
        if not self.systems_list_widget.count():
            return
        
        if index is None:
            index = self.systems_list_widget.currentIndex()
        
        if not type(index) is int:
            print "DEBUG", index.column(), index.row()
            index = index.row()
        
        print "SET REF LATTICE", index
        
        lattice = self.lattice_list[index]
        filename = self.filenames_list[index]
        extension = None
        
        self.mainWindow.postFileLoaded("ref", lattice, filename, extension)
        
        self.ref_selected = True
        self.ref_index = index
        
        self.refresh_type_text()
        
        self.check_ref_change_ok()
    
    def check_ref_change_ok(self):
        """
        Check it was ok to change the ref.
        
        """
        if self.input_index is None or self.ref_index is None:
            return
        
        ref = self.lattice_list[self.ref_index]
        inp = self.lattice_list[self.input_index]
        
        diff = False
        for i in xrange(3):
            if inp.cellDims[i] != ref.cellDims[i]:
                diff = True
                break
        
        if diff:
            self.mainWindow.console.write("WARNING: new ref has different cellDims: setting input = ref")
            
            self.set_input(self.ref_index)
    
    def check_input_change_ok(self):
        """
        Check it was ok to change the input.
        
        """
        if self.input_index is None or self.ref_index is None:
            return
        
        ref = self.lattice_list[self.ref_index]
        inp = self.lattice_list[self.input_index]
        
        diff = False
        for i in xrange(3):
            if inp.cellDims[i] != ref.cellDims[i]:
                diff = True
                break
        
        if diff:
            self.mainWindow.console.write("WARNING: new input has different cellDims: setting ref = input")
            
            self.set_ref(self.input_index)
    
    def set_input(self, index=None):
        """
        Set input lattice
        
        """
        if not self.systems_list_widget.count():
            return
        
        if index is None:
            index = self.systems_list_widget.currentIndex()
        
        if not type(index) is int:
            print "DEBUG", index.column(), index.row()
            index = index.row()
        
        print "SET INPUT LATTICE", index
        
        lattice = self.lattice_list[index]
        filename = self.filenames_list[index]
        extension = self.extensions_list[index]
        
        self.mainWindow.postFileLoaded("input", lattice, filename, extension)
        
        self.input_selected = True
        self.input_index = index
        
        self.refresh_type_text()
        
        self.check_input_change_ok()
    
    def file_generated(self, lattice):
        """
        File generated
        
        """
        print "FILE GENERATED", lattice, "generated.dat", "dat"
        
        index = self.add_lattice(lattice, "generated.dat", "dat")
        
        if not self.ref_selected:
            self.set_ref(index=index)
            self.set_input(index=index)
    
    def file_loaded(self, lattice, filename, extension):
        """
        Called after a file had been loaded (or generated too?)
        
        """
        print "FILE LOADED", lattice, filename, extension
        
        index = self.add_lattice(lattice, filename, extension)
        
        if not self.ref_selected:
            self.set_ref(index=index)
            self.set_input(index=index)
    
    def add_lattice(self, lattice, filename, extension):
        """
        Add lattice
        
        """
        index = len(self.lattice_list)
        
        self.lattice_list.append(lattice)
        self.filenames_list.append(filename)
        self.extensions_list.append(extension)
        
        list_item = QtGui.QListWidgetItem()
        list_item.setText("%s (%d atoms)" % (filename, lattice.NAtoms))
        
        self.systems_list_widget.addItem(list_item)
        
        return index
    
    def set_new_system_stack(self, index):
        """
        Set new system stack
        
        """
        print "SET STACK", index
        self.new_system_stack.setCurrentIndex(index)
    
    def remove_system(self):
        """
        Remove system(s) from list
        
        """
        if not self.systems_list_widget.count():
            return
        
        print "REMOVE SYSTEM", self.systems_list_widget.currentIndex()
        
        # do not allow removal of system that is tagged with ref or input
        



