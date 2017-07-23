
"""
Lattice generation
==================

In the *Generate new system* section of the *Input* tab of the main toolbar lattices
of different types and sizes can be generated.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from ..lattice_gen import lattice_gen_pu3ga
from ..lattice_gen import lattice_gen_fcc
from ..lattice_gen import lattice_gen_bcc
from ..lattice_gen import lattice_gen_fluorite
from ..lattice_gen import lattice_gen_rockSalt
from ..lattice_gen import lattice_gen_sic
from ..lattice_gen import lattice_gen_graphite
from ..lattice_gen import lattice_gen_diamond
from ..lattice_gen import lattice_gen_diamond_indenter

################################################################################

class GenericLatticeGeneratorForm(QtGui.QWidget):
    """
    Generic reader widget.

    """
    def __init__(self, parent, mainWindow, title):
        super(GenericLatticeGeneratorForm, self).__init__(parent)

        self.parent = parent
        self.mainWindow = mainWindow

        self.generatorArgs = None
        self.filename = "generated.dat"

        self.formLayout = QtGui.QFormLayout(self)

        self.show()

    def generateLattice(self):
        """
        Generate the lattice

        """
        status, lattice = self.generateLatticeMain()

        self.postGenerateLattice(status, lattice)

    def generateLatticeMain(self):
        """
        Main generate function to be overridden

        """
        return 1, None

    def postGenerateLattice(self, status, lattice):
        """
        Post generate lattice.

        """
        if not status and lattice is not None:
            self.parent.file_generated(lattice, self.filename)

    def add_specie_options(self, NSpecies):
        """
        Add specie options

        """
        pass

    def add_a0_option(self):
        """
        Add lattice constant option

        """
        latticeConstantSpin = QtGui.QDoubleSpinBox()
        latticeConstantSpin.setDecimals(5)
        latticeConstantSpin.setSingleStep(0.1)
        latticeConstantSpin.setMinimum(0.00001)
        latticeConstantSpin.setMaximum(99.99999)
        latticeConstantSpin.setValue(self.generatorArgs.a0)
        latticeConstantSpin.valueChanged.connect(self.latticeConstantChanged)
        latticeConstantSpin.setSuffix(" \u212B")
        latticeConstantSpin.setToolTip("Set the lattice constant")
        self.formLayout.addRow("Lattice constant", latticeConstantSpin)

    def latticeConstantChanged(self, val):
        """
        Lattice constant changed

        """
        self.generatorArgs.a0 = val

    def add_unit_cell_options(self):
        """
        Add unit cell options to layout

        """
        row = QtGui.QHBoxLayout()

        # num unit cells
        numUnitCellsXSpin = QtGui.QSpinBox()
        numUnitCellsXSpin.setMinimum(1)
        numUnitCellsXSpin.setMaximum(1000)
        numUnitCellsXSpin.setValue(self.generatorArgs.NCells[0])
        numUnitCellsXSpin.valueChanged.connect(self.numUnitCellsXChanged)
        numUnitCellsXSpin.setToolTip("Set the number of unit cells in x")
        row.addWidget(numUnitCellsXSpin)

        row.addWidget(QtGui.QLabel("x"))

        numUnitCellsYSpin = QtGui.QSpinBox()
        numUnitCellsYSpin.setMinimum(1)
        numUnitCellsYSpin.setMaximum(1000)
        numUnitCellsYSpin.setValue(self.generatorArgs.NCells[1])
        numUnitCellsYSpin.valueChanged.connect(self.numUnitCellsYChanged)
        numUnitCellsYSpin.setToolTip("Set the number of unit cells in y")
        row.addWidget(numUnitCellsYSpin)

        row.addWidget(QtGui.QLabel("x"))

        numUnitCellsZSpin = QtGui.QSpinBox()
        numUnitCellsZSpin.setMinimum(1)
        numUnitCellsZSpin.setMaximum(1000)
        numUnitCellsZSpin.setValue(self.generatorArgs.NCells[2])
        numUnitCellsZSpin.valueChanged.connect(self.numUnitCellsZChanged)
        numUnitCellsZSpin.setToolTip("Set the number of unit cells in z")
        row.addWidget(numUnitCellsZSpin)

        self.formLayout.addRow("Number of unit cells", row)

    def numUnitCellsXChanged(self, val):
        """
        Number of unit cells changed.

        """
        self.generatorArgs.NCells[0] = val

    def numUnitCellsYChanged(self, val):
        """
        Number of unit cells changed.

        """
        self.generatorArgs.NCells[1] = val

    def numUnitCellsZChanged(self, val):
        """
        Number of unit cells changed.

        """
        self.generatorArgs.NCells[2] = val

    def filenameChanged(self, text):
        """
        Filename has changed

        """
        self.filename = str(text)

    def add_filename_option(self):
        """
        Add filename option

        """
        filenameLineEdit = QtGui.QLineEdit(self.filename)
        filenameLineEdit.setFixedWidth(130)
        filenameLineEdit.textChanged.connect(self.filenameChanged)
        filenameLineEdit.setToolTip("Enter the display name")
        self.formLayout.addRow("Display name", filenameLineEdit)

    def add_pbc_options(self):
        """
        Add pbc options

        """
        # periodic boundaries
        PBCXCheckBox = QtGui.QCheckBox("x   ")
        PBCXCheckBox.setChecked(QtCore.Qt.Checked)
        PBCXCheckBox.setToolTip("Set periodic boundaries in x")
        PBCYCheckBox = QtGui.QCheckBox("y   ")
        PBCYCheckBox.setChecked(QtCore.Qt.Checked)
        PBCYCheckBox.setToolTip("Set periodic boundaries in y")
        PBCZCheckBox = QtGui.QCheckBox("z   ")
        PBCZCheckBox.setChecked(QtCore.Qt.Checked)
        PBCZCheckBox.setToolTip("Set periodic boundaries in z")

        PBCXCheckBox.stateChanged.connect(self.PBCXChanged)
        PBCYCheckBox.stateChanged.connect(self.PBCYChanged)
        PBCZCheckBox.stateChanged.connect(self.PBCZChanged)

        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.addWidget(PBCXCheckBox)
        rowLayout.addWidget(PBCYCheckBox)
        rowLayout.addWidget(PBCZCheckBox)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(PBCXCheckBox)
        hbox.addWidget(PBCYCheckBox)
        hbox.addWidget(PBCZCheckBox)
        self.formLayout.addRow("Periodic boundaries", hbox)

    def PBCXChanged(self, checked):
        """
        PBC changed.

        """
        if checked == QtCore.Qt.Checked:
            self.generatorArgs.pbcx = True

        else:
            self.generatorArgs.pbcx = False

    def PBCYChanged(self, checked):
        """
        PBC changed.

        """
        if checked == QtCore.Qt.Checked:
            self.generatorArgs.pbcy = True

        else:
            self.generatorArgs.pbcy = False

    def PBCZChanged(self, checked):
        """
        PBC changed.

        """
        if checked == QtCore.Qt.Checked:
            self.generatorArgs.pbcz = True

        else:
            self.generatorArgs.pbcz = False

    def add_generate_button(self):
        """
        Add generate button

        """
        generateButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/run-build.png")), "Generate lattice")
        generateButton.clicked.connect(self.generateLattice)
        generateButton.setToolTip("Generate the lattice")

        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(generateButton)
        hbox.addStretch(1)
        self.formLayout.addRow(hbox)

################################################################################

class Pu3GaLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    Pu3Ga lattice generator

    """
    def __init__(self, parent, mainWindow):
        super(Pu3GaLatticeGeneratorForm, self).__init__(parent, mainWindow, "Pu3Ga lattice generator")

        self.generatorArgs = lattice_gen_pu3ga.Args()

        self.add_filename_option()

        self.add_unit_cell_options()

        self.add_a0_option()

        self.add_pbc_options()

        # percent Ga
        percGaSpin = QtGui.QDoubleSpinBox()
        percGaSpin.setSingleStep(0.1)
        percGaSpin.setMinimum(0.0)
        percGaSpin.setMaximum(25.0)
        percGaSpin.setValue(self.generatorArgs.percGa)
        percGaSpin.valueChanged.connect(self.percGaChanged)
        percGaSpin.setSuffix(" at.%")
        percGaSpin.setToolTip("Set the Ga concentration")
        self.formLayout.addRow("Ga concentration", percGaSpin)

        # generate button
        self.add_generate_button()

    def percGaChanged(self, val):
        """
        Percent Ga changed

        """
        self.generatorArgs.percGa = val

    def generateLatticeMain(self):
        """
        Generate lattice

        """
        generator = lattice_gen_pu3ga.Pu3GaLatticeGenerator()

        status, lattice = generator.generateLattice(self.generatorArgs)

        return status, lattice

################################################################################

class FCCLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    FCC lattice generator

    """
    def __init__(self, parent, mainWindow):
        super(FCCLatticeGeneratorForm, self).__init__(parent, mainWindow, "FCC lattice generator")

        self.generatorArgs = lattice_gen_fcc.Args()

        self.add_filename_option()

        # specie
        self.specie_text = QtGui.QLineEdit(self.generatorArgs.sym)
        self.specie_text.setFixedWidth(30)
        self.specie_text.setMaxLength(2)
        self.specie_text.textEdited.connect(self.specie_text_edited)
        self.specie_text.setToolTip("Set the species")
        self.formLayout.addRow("Species", self.specie_text)

        self.add_unit_cell_options()

        self.add_a0_option()

        self.add_pbc_options()

        # generate button
        self.add_generate_button()

    def specie_text_edited(self, text):
        """
        Specie text edited

        """
        self.generatorArgs.sym = str(text)

    def generateLatticeMain(self):
        """
        Generate lattice

        """
        generator = lattice_gen_fcc.FCCLatticeGenerator()

        status, lattice = generator.generateLattice(self.generatorArgs)

        return status, lattice

################################################################################

class BCCLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    BCC lattice generator

    """
    def __init__(self, parent, mainWindow):
        super(BCCLatticeGeneratorForm, self).__init__(parent, mainWindow, "BCC lattice generator")

        self.generatorArgs = lattice_gen_bcc.Args()

        self.add_filename_option()

        # specie
        self.specie_text = QtGui.QLineEdit(self.generatorArgs.sym)
        self.specie_text.setFixedWidth(30)
        self.specie_text.setMaxLength(2)
        self.specie_text.textEdited.connect(self.specie_text_edited)
        self.specie_text.setToolTip("Set the species")
        self.formLayout.addRow("Species", self.specie_text)

        self.add_unit_cell_options()

        self.add_a0_option()

        self.add_pbc_options()

        # generate button
        self.add_generate_button()

    def specie_text_edited(self, text):
        """
        Specie text edited

        """
        self.generatorArgs.sym = str(text)

    def generateLatticeMain(self):
        """
        Generate lattice

        """
        generator = lattice_gen_bcc.BCCLatticeGenerator()

        status, lattice = generator.generateLattice(self.generatorArgs)

        return status, lattice

################################################################################

class FluoriteLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    Fluorite lattice generator

    """
    def __init__(self, parent, mainWindow):
        super(FluoriteLatticeGeneratorForm, self).__init__(parent, mainWindow, "Fluorite lattice generator")

        self.generatorArgs = lattice_gen_fluorite.Args()

        self.add_filename_option()

        # specie 1
        self.specie1_text = QtGui.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        self.specie1_text.setToolTip("Set the symbol of the first species")

        # charge 1
        charge1Spin = QtGui.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        charge1Spin.setToolTip("Set the charge of the first species")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.specie1_text)
        hbox.addWidget(charge1Spin)
        self.formLayout.addRow("Species 1", hbox)

        # specie 2
        self.specie2_text = QtGui.QLineEdit(self.generatorArgs.sym2)
        self.specie2_text.setFixedWidth(30)
        self.specie2_text.setMaxLength(2)
        self.specie2_text.textEdited.connect(self.specie2_text_edited)
        self.specie2_text.setToolTip("Set the symbol of the second species")

        # charge 2
        charge2Spin = QtGui.QDoubleSpinBox()
        charge2Spin.setMinimum(-99.99)
        charge2Spin.setValue(self.generatorArgs.charge2)
        charge2Spin.valueChanged.connect(self.charge2_changed)
        charge2Spin.setToolTip("Set the charge of the second species")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.specie2_text)
        hbox.addWidget(charge2Spin)
        self.formLayout.addRow("Species 2", hbox)

        self.add_unit_cell_options()

        self.add_a0_option()

        self.add_pbc_options()

        # generate button
        self.add_generate_button()

    def charge1_changed(self, val):
        """
        Charge 1 changed

        """
        self.generatorArgs.charge1 = val

    def charge2_changed(self, val):
        """
        Charge 2 changed

        """
        self.generatorArgs.charge2 = val

    def specie1_text_edited(self, text):
        """
        Specie 1 text edited

        """
        self.generatorArgs.sym1 = str(text)

    def specie2_text_edited(self, text):
        """
        Specie 2 text edited

        """
        self.generatorArgs.sym2 = str(text)

    def generateLatticeMain(self):
        """
        Generate lattice

        """
        generator = lattice_gen_fluorite.FluoriteLatticeGenerator()

        status, lattice = generator.generateLattice(self.generatorArgs)

        return status, lattice

################################################################################

class RockSaltLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    Rock Salt lattice generator

    """
    def __init__(self, parent, mainWindow):
        super(RockSaltLatticeGeneratorForm, self).__init__(parent, mainWindow, "Rock Salt lattice generator")

        self.generatorArgs = lattice_gen_rockSalt.Args()

        self.add_filename_option()

        # specie 1
        self.specie1_text = QtGui.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        self.specie1_text.setToolTip("Set the symbol of the first species")

        # charge 1
        charge1Spin = QtGui.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        charge1Spin.setToolTip("Set the charge of the first species")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.specie1_text)
        hbox.addWidget(charge1Spin)
        self.formLayout.addRow("Species 1", hbox)

        # specie 2
        self.specie2_text = QtGui.QLineEdit(self.generatorArgs.sym2)
        self.specie2_text.setFixedWidth(30)
        self.specie2_text.setMaxLength(2)
        self.specie2_text.textEdited.connect(self.specie2_text_edited)
        self.specie2_text.setToolTip("Set the symbol of the second species")

        # charge 2
        charge2Spin = QtGui.QDoubleSpinBox()
        charge2Spin.setMinimum(-99.99)
        charge2Spin.setValue(self.generatorArgs.charge2)
        charge2Spin.valueChanged.connect(self.charge2_changed)
        charge2Spin.setToolTip("Set the charge of the second species")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.specie2_text)
        hbox.addWidget(charge2Spin)
        self.formLayout.addRow("Species 2", hbox)

        self.add_unit_cell_options()

        self.add_a0_option()

        self.add_pbc_options()

        # generate button
        self.add_generate_button()

    def charge1_changed(self, val):
        """
        Charge 1 changed

        """
        self.generatorArgs.charge1 = val

    def charge2_changed(self, val):
        """
        Charge 2 changed

        """
        self.generatorArgs.charge2 = val

    def specie1_text_edited(self, text):
        """
        Specie 1 text edited

        """
        self.generatorArgs.sym1 = str(text)

    def specie2_text_edited(self, text):
        """
        Specie 2 text edited

        """
        self.generatorArgs.sym2 = str(text)

    def generateLatticeMain(self):
        """
        Generate lattice

        """
        generator = lattice_gen_rockSalt.RockSaltLatticeGenerator()

        status, lattice = generator.generateLattice(self.generatorArgs)

        return status, lattice

################################################################################

class SiC4HLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    SiC 4H lattice generator

    """
    def __init__(self, parent, mainWindow):
        super(SiC4HLatticeGeneratorForm, self).__init__(parent, mainWindow, "CSi 4H lattice generator")

        self.generatorArgs = lattice_gen_sic.Args()

        self.add_filename_option()

        # specie 1
        self.specie1_text = QtGui.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        self.specie1_text.setToolTip("Set the symbol of the first species")

        # charge 1
        charge1Spin = QtGui.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        charge1Spin.setToolTip("Set the charge of the first species")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.specie1_text)
        hbox.addWidget(charge1Spin)
        self.formLayout.addRow("Species 1", hbox)

        # specie 2
        self.specie2_text = QtGui.QLineEdit(self.generatorArgs.sym2)
        self.specie2_text.setFixedWidth(30)
        self.specie2_text.setMaxLength(2)
        self.specie2_text.textEdited.connect(self.specie2_text_edited)
        self.specie2_text.setToolTip("Set the symbol of the second species")

        # charge 2
        charge2Spin = QtGui.QDoubleSpinBox()
        charge2Spin.setMinimum(-99.99)
        charge2Spin.setValue(self.generatorArgs.charge2)
        charge2Spin.valueChanged.connect(self.charge2_changed)
        charge2Spin.setToolTip("Set the charge of the second species")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.specie2_text)
        hbox.addWidget(charge2Spin)
        self.formLayout.addRow("Species 2", hbox)

        self.add_unit_cell_options()

        self.add_a0_option()

        self.add_pbc_options()

        # generate button
        self.add_generate_button()

    def charge1_changed(self, val):
        """
        Charge 1 changed

        """
        self.generatorArgs.charge1 = val

    def charge2_changed(self, val):
        """
        Charge 2 changed

        """
        self.generatorArgs.charge2 = val

    def specie1_text_edited(self, text):
        """
        Specie 1 text edited

        """
        self.generatorArgs.sym1 = str(text)

    def specie2_text_edited(self, text):
        """
        Specie 2 text edited

        """
        self.generatorArgs.sym2 = str(text)

    def generateLatticeMain(self):
        """
        Generate lattice

        """
        generator = lattice_gen_sic.SiC4HLatticeGenerator()

        status, lattice = generator.generateLattice(self.generatorArgs)

        return status, lattice


#######################################################################################

class GraphiteLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    Graphite lattice generator

    """
    def __init__(self, parent, mainWindow):
        super(GraphiteLatticeGeneratorForm, self).__init__(parent, mainWindow, "Graphite lattice generator")

        self.generatorArgs = lattice_gen_graphite.Args()

        self.add_filename_option()

        # specie
        self.specie1_text = QtGui.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        self.specie1_text.setToolTip("Set the atom symbol")

        # charge
        charge1Spin = QtGui.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        charge1Spin.setToolTip("Set the atom charge")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.specie1_text)
        hbox.addWidget(charge1Spin)
        self.formLayout.addRow("Species", hbox)

        # unit cell options
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("No. unit cells:"))

        # num unit cells
        numUnitCellsXSpin = QtGui.QSpinBox()
        numUnitCellsXSpin.setMinimum(1)
        numUnitCellsXSpin.setMaximum(1000)
        numUnitCellsXSpin.setValue(self.generatorArgs.NCells[0])
        numUnitCellsXSpin.valueChanged.connect(self.numUnitCellsXChanged)
        numUnitCellsXSpin.setToolTip("Set the number of unit cells in x")
        row.addWidget(numUnitCellsXSpin)

        row.addWidget(QtGui.QLabel("x"))

        numUnitCellsYSpin = QtGui.QSpinBox()
        numUnitCellsYSpin.setMinimum(1)
        numUnitCellsYSpin.setMaximum(1000)
        numUnitCellsYSpin.setValue(self.generatorArgs.NCells[1])
        numUnitCellsYSpin.valueChanged.connect(self.numUnitCellsYChanged)
        numUnitCellsYSpin.setToolTip("Set the number of unit cells in y")
        row.addWidget(numUnitCellsYSpin)

        row.addWidget(QtGui.QLabel("x"))

        numUnitCellsZSpin = QtGui.QSpinBox()
        numUnitCellsZSpin.setMinimum(1)
        numUnitCellsZSpin.setMaximum(1000)
        numUnitCellsZSpin.setValue(self.generatorArgs.NCells[2])
        numUnitCellsZSpin.valueChanged.connect(self.numUnitCellsZChanged)
        numUnitCellsZSpin.setToolTip("Set the number of unit cells in z")
        row.addWidget(numUnitCellsZSpin)

        self.formLayout.addRow(row)

        # output total lattice size before generating
        infogrid = QtGui.QGridLayout()
        self.latsize_x = 1
        self.latsize_y = 2
        self.latsize_z = 3
        self.latsize_text = QtGui.QLabel()
        infogrid.addWidget(self.latsize_text,0,0)
        
        # Show number of atoms, before generating
        self.lat_numatoms = 3
        self.lat_numatoms_text = QtGui.QLabel()
        infogrid.addWidget(self.lat_numatoms_text,1,0)     
        self.formLayout.addRow("Lattice dimensions",infogrid)

        # Lattice constants
        # Lattice parameter presets combo
        ParamCombo = QtGui.QComboBox()
        ParamCombo.addItem("AIREBO")
        ParamCombo.addItem("ReaxFF May2016")
        ParamCombo.addItem("Custom")
        ParamCombo.currentIndexChanged.connect(self.ParamComboChanged)
        ParamCombo.setToolTip("Set lattice parameter presets")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(ParamCombo)
        self.formLayout.addRow("Parameter presets", hbox)

        # Lattice 'a' parameter
        self.latticeAConstSpin = QtGui.QDoubleSpinBox()
        self.latticeAConstSpin.setDecimals(5)
        self.latticeAConstSpin.setSingleStep(0.1)
        self.latticeAConstSpin.setMinimum(0.00001)
        self.latticeAConstSpin.setMaximum(99.99999)
        self.latticeAConstSpin.setValue(self.generatorArgs.a0)
        self.latticeAConstSpin.setEnabled( False )
        self.latticeAConstSpin.valueChanged.connect(self.latticeAConstChanged)
        self.latticeAConstSpin.setSuffix(" \u212B")
        self.latticeAConstSpin.setToolTip("Set the lattice 'a' constant (select custom to edit this)")
        self.formLayout.addRow("Lattice 'a' constant", self.latticeAConstSpin)

        # Lattice 'c' parameter
        self.latticeCConstSpin = QtGui.QDoubleSpinBox()
        self.latticeCConstSpin.setDecimals(5)
        self.latticeCConstSpin.setSingleStep(0.1)
        self.latticeCConstSpin.setMinimum(0.00001)
        self.latticeCConstSpin.setMaximum(99.99999)
        self.latticeCConstSpin.setValue(self.generatorArgs.c0)
        self.latticeCConstSpin.setEnabled( False )
        self.latticeCConstSpin.valueChanged.connect(self.latticeCConstChanged)
        self.latticeCConstSpin.setSuffix(" \u212B")
        self.latticeCConstSpin.setToolTip("Set the lattice 'c' constant (select custom to edit this)")
        self.formLayout.addRow("Lattice 'c' constant", self.latticeCConstSpin)

        # Graphite layer stacking input
        rx = QtCore.QRegExp("([a-c]?[A-C]?)*")
        validator = QtGui.QRegExpValidator(rx, self)

        self.GrahiteLayerStacking = QtGui.QLineEdit()
        self.GrahiteLayerStacking.setValidator(validator)
        self.GrahiteLayerStacking.setText('ab')
        self.GrahiteLayerStacking.setToolTip("Graphene layer stacking pattern. (String consisting of 'a', 'b' and 'c' only.) ")
        self.GrahiteLayerStacking.textChanged.connect(self.GrahiteLayerStackingChanged)
        self.formLayout.addRow("Grahite Layer Stacking", self.GrahiteLayerStacking)

        # checkboxes for periodic boundaries
        self.add_pbc_options()

        # generate button
        self.add_generate_button()
        
        # update lattice dimension and no. atoms text to correct initial values
        self.UpdateLatticeSizeText()
        
    def UpdateLatticeSizeText(self):
        """
        Updates the Lattice dimensions and num atoms string in the generator form.

        """
        self.latsize_x = 1.732050808 * self.latticeAConstSpin.value() * self.generatorArgs.NCells[0]
        self.latsize_y = self.latticeAConstSpin.value() * self.generatorArgs.NCells[1]
        self.latsize_z = self.latticeCConstSpin.value() * self.generatorArgs.NCells[2] * len(self.GrahiteLayerStacking.text())
        self.lat_numatoms = int(4 * len(self.GrahiteLayerStacking.text()) * 
                                self.generatorArgs.NCells[0] * 
                                self.generatorArgs.NCells[1] * 
                                self.generatorArgs.NCells[2])
        
        self.latsize_text.setText('{:.1f}'.format(self.latsize_x) + " \u212B x " + 
                                  '{:.1f}'.format(self.latsize_y) + " \u212B x " + 
                                  '{:.1f}'.format(self.latsize_z) + " \u212B ")
        self.lat_numatoms_text.setText(str(self.lat_numatoms) + " Atoms" )

    def GrahiteLayerStackingChanged(self, val):
        """
        Lattice layer stacking changed

        """
        self.generatorArgs.stacking = val
        self.UpdateLatticeSizeText()

    def latticeAConstChanged(self, val):
        """
        Lattice constant changed

        """
        self.generatorArgs.a0 = val
        self.UpdateLatticeSizeText()

    def latticeCConstChanged(self, val):
        """
        Lattice constant changed

        """
        self.generatorArgs.c0 = val
        self.UpdateLatticeSizeText()

    def ParamComboChanged(self, index):
        """
        Parameter presets combo changed

        """

        # AIREBO
        if(index == 0):
            self.generatorArgs.a0 = 2.4175
            self.generatorArgs.c0 = 3.358
            self.latticeAConstSpin.setValue(2.4175)
            self.latticeAConstSpin.setEnabled( False )
            self.latticeCConstSpin.setValue(3.358)
            self.latticeCConstSpin.setEnabled( False )
        # ReaxFF May 2016
        if(index == 1):
            self.generatorArgs.a0 = 2.433
            self.generatorArgs.c0 = 3.2567
            self.latticeAConstSpin.setValue(2.433)
            self.latticeAConstSpin.setEnabled( False )
            self.latticeCConstSpin.setValue(3.2567)
            self.latticeCConstSpin.setEnabled( False )

        # Custom
        if(index == 2):
            self.latticeAConstSpin.setEnabled( True )
            self.latticeCConstSpin.setEnabled( True )

    def charge1_changed(self, val):
        """
        Charge 1 changed

        """
        self.generatorArgs.charge1 = val

    def specie1_text_edited(self, text):
        """
        Specie 1 text edited

        """
        self.generatorArgs.sym1 = str(text)
        
    def numUnitCellsXChanged(self, val):
        """
        Number of unit cells changed.

        """
        self.generatorArgs.NCells[0] = val
        self.UpdateLatticeSizeText()

    def numUnitCellsYChanged(self, val):
        """
        Number of unit cells changed.

        """
        self.generatorArgs.NCells[1] = val
        self.UpdateLatticeSizeText()

    def numUnitCellsZChanged(self, val):
        """
        Number of unit cells changed.

        """
        self.generatorArgs.NCells[2] = val
        self.UpdateLatticeSizeText()

    def generateLatticeMain(self):
        """
        Generate lattice

        """
        generator = lattice_gen_graphite.GraphiteLatticeGenerator()

        status, lattice = generator.generateLattice(self.generatorArgs)

        return status, lattice


################################################################################

class DiamondLatticeGeneratorForm(GenericLatticeGeneratorForm):
    """
    Diamond lattice generator

    """
    def __init__(self, parent, mainWindow):
        super(DiamondLatticeGeneratorForm, self).__init__(parent, mainWindow, "Diamond lattice generator")

        self.generatorArgs = lattice_gen_diamond.Args()

        self.add_filename_option()

        # specie 1
        self.specie1_text = QtGui.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        self.specie1_text.setToolTip("Set the symbol of the first species")

        # charge 1
        charge1Spin = QtGui.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        charge1Spin.setToolTip("Set the charge of the first species")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.specie1_text)
        hbox.addWidget(charge1Spin)
        self.formLayout.addRow("Species 1", hbox)

        # unit cell options
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("No. unit cells:"))

        # num unit cells
        numUnitCellsXSpin = QtGui.QSpinBox()
        numUnitCellsXSpin.setMinimum(1)
        numUnitCellsXSpin.setMaximum(1000)
        numUnitCellsXSpin.setValue(self.generatorArgs.NCells[0])
        numUnitCellsXSpin.valueChanged.connect(self.numUnitCellsXChanged)
        numUnitCellsXSpin.setToolTip("Set the number of unit cells in x")
        row.addWidget(numUnitCellsXSpin)

        row.addWidget(QtGui.QLabel("x"))

        numUnitCellsYSpin = QtGui.QSpinBox()
        numUnitCellsYSpin.setMinimum(1)
        numUnitCellsYSpin.setMaximum(1000)
        numUnitCellsYSpin.setValue(self.generatorArgs.NCells[1])
        numUnitCellsYSpin.valueChanged.connect(self.numUnitCellsYChanged)
        numUnitCellsYSpin.setToolTip("Set the number of unit cells in y")
        row.addWidget(numUnitCellsYSpin)

        row.addWidget(QtGui.QLabel("x"))

        numUnitCellsZSpin = QtGui.QSpinBox()
        numUnitCellsZSpin.setMinimum(1)
        numUnitCellsZSpin.setMaximum(1000)
        numUnitCellsZSpin.setValue(self.generatorArgs.NCells[2])
        numUnitCellsZSpin.valueChanged.connect(self.numUnitCellsZChanged)
        numUnitCellsZSpin.setToolTip("Set the number of unit cells in z")
        row.addWidget(numUnitCellsZSpin)

        self.formLayout.addRow(row)

        # output total lattice size before generating
        infogrid = QtGui.QGridLayout()
        self.latsize_x = 1
        self.latsize_y = 2
        self.latsize_z = 3
        self.latsize_text = QtGui.QLabel()
        infogrid.addWidget(self.latsize_text,0,0)
        
        # Show number of atoms, before generating
        self.lat_numatoms = 3
        self.lat_numatoms_text = QtGui.QLabel()
        infogrid.addWidget(self.lat_numatoms_text,1,0)     
        self.formLayout.addRow("Lattice dimensions",infogrid)
        

        # Lattice constants

        # Lattice parameter presets combo
        ParamCombo = QtGui.QComboBox()
        ParamCombo.addItem("AIREBO")
        ParamCombo.addItem("ReaxFF May2016")
        ParamCombo.addItem("Custom")
        ParamCombo.currentIndexChanged.connect(self.ParamComboChanged)
        ParamCombo.setToolTip("Set lattice parameter presets")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(ParamCombo)
        self.formLayout.addRow("Parameter presets", hbox)

        # Lattice 'a' parameter
        self.latticeAConstSpin = QtGui.QDoubleSpinBox()
        self.latticeAConstSpin.setDecimals(8)
        self.latticeAConstSpin.setSingleStep(0.1)
        self.latticeAConstSpin.setMinimum(0.00001)
        self.latticeAConstSpin.setMaximum(99.99999)
        self.latticeAConstSpin.setValue(self.generatorArgs.a0)
        self.latticeAConstSpin.setEnabled( False )
        self.latticeAConstSpin.valueChanged.connect(self.latticeAConstChanged)
        self.latticeAConstSpin.setSuffix(" \u212B")
        self.latticeAConstSpin.setToolTip("Set the lattice 'a' constant (select custom to edit this)")
        self.formLayout.addRow("Lattice 'a' constant", self.latticeAConstSpin)

        self.add_pbc_options()

        # generate button
        self.add_generate_button()
        
        # update lattice dimension and no. atoms text to correct initial values
        self.UpdateLatticeSizeText()
        
    def UpdateLatticeSizeText(self):
        """
        Updates the Lattice dimensions and num atoms string in the generator form.

        """
        self.latsize_x = self.latticeAConstSpin.value() * self.generatorArgs.NCells[0]
        self.latsize_y = self.latticeAConstSpin.value() * self.generatorArgs.NCells[1]
        self.latsize_z = self.latticeAConstSpin.value() * self.generatorArgs.NCells[2]
        self.lat_numatoms = int(8 * self.generatorArgs.NCells[0] * 
                                self.generatorArgs.NCells[1] * 
                                self.generatorArgs.NCells[2])
        
        self.latsize_text.setText('{:.1f}'.format(self.latsize_x) + " \u212B x " + 
                                  '{:.1f}'.format(self.latsize_y) + " \u212B x " + 
                                  '{:.1f}'.format(self.latsize_z) + " \u212B ")
        self.lat_numatoms_text.setText(str(self.lat_numatoms) + " Atoms" )

    def numUnitCellsXChanged(self, val):
        """
        Number of unit cells changed.

        """
        self.generatorArgs.NCells[0] = val
        self.UpdateLatticeSizeText()

    def numUnitCellsYChanged(self, val):
        """
        Number of unit cells changed.

        """
        self.generatorArgs.NCells[1] = val
        self.UpdateLatticeSizeText()

    def numUnitCellsZChanged(self, val):
        """
        Number of unit cells changed.

        """
        self.generatorArgs.NCells[2] = val
        self.UpdateLatticeSizeText()

    def charge1_changed(self, val):
        """
        Charge 1 changed

        """
        self.generatorArgs.charge1 = val

    def specie1_text_edited(self, text):
        """
        Specie 1 text edited

        """
        self.generatorArgs.sym1 = str(text)

    def latticeAConstChanged(self, val):
        """
        Lattice constant changed

        """
        self.generatorArgs.a0 = val
        self.UpdateLatticeSizeText()

    def ParamComboChanged(self, index):
        """
        Parameter presets combo changed

        """

        # AIREBO
        if(index == 0):
            self.generatorArgs.a0 = 3.556717
            self.latticeAConstSpin.setValue(3.556717)
            self.latticeAConstSpin.setEnabled( False )
        # ReaxFF May 2016
        if(index == 1):
            self.generatorArgs.a0 = 3.54723712
            self.latticeAConstSpin.setValue(3.54723712)
            self.latticeAConstSpin.setEnabled( False )
        # Custom
        if(index == 2):
            self.latticeAConstSpin.setEnabled( True )

    def generateLatticeMain(self):
        """
        Generate lattice

        """
        generator = lattice_gen_diamond.DiamondLatticeGenerator()

        status, lattice = generator.generateLattice(self.generatorArgs)

        return status, lattice


################################################################################

class DiamondIndenterGeneratorForm(GenericLatticeGeneratorForm):
    """
    Diamond Indenter generator

    """
    def __init__(self, parent, mainWindow):
        super(DiamondIndenterGeneratorForm, self).__init__(parent, mainWindow, "Diamond Indenter generator")

        self.generatorArgs = lattice_gen_diamond_indenter.Args()

        self.add_filename_option()

        # specie 1
        self.specie1_text = QtGui.QLineEdit(self.generatorArgs.sym1)
        self.specie1_text.setFixedWidth(30)
        self.specie1_text.setMaxLength(2)
        self.specie1_text.textEdited.connect(self.specie1_text_edited)
        self.specie1_text.setEnabled( False )
        self.specie1_text.setToolTip("Set the symbol of the first species")

        # charge 1
        charge1Spin = QtGui.QDoubleSpinBox()
        charge1Spin.setMinimum(-99.99)
        charge1Spin.setValue(self.generatorArgs.charge1)
        charge1Spin.valueChanged.connect(self.charge1_changed)
        charge1Spin.setEnabled( False )
        charge1Spin.setToolTip("Set the charge of the first species")

        # Display atom
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.specie1_text)
        hbox.addWidget(charge1Spin)
        self.formLayout.addRow("Carbon: ", hbox)

        # Indenter atom layers
        AtomLayersSpin = QtGui.QSpinBox()
        AtomLayersSpin.setMinimum(1)
        AtomLayersSpin.setMaximum(10000)
        AtomLayersSpin.setValue(self.generatorArgs.AtomLayers)
        AtomLayersSpin.valueChanged.connect(self.AtomLayersSpinChanged)
        AtomLayersSpin.setToolTip("Set the number of layers of carbon atoms in the indenter")

        self.formLayout.addRow("Carbon Atom Layers: ", AtomLayersSpin)

        # Layers to cut from tip
        AtomLayersCutSpin = QtGui.QSpinBox()
        AtomLayersCutSpin.setMinimum(0)
        AtomLayersCutSpin.setMaximum(10000)
        AtomLayersCutSpin.setValue(self.generatorArgs.TipCutLayers)
        AtomLayersCutSpin.valueChanged.connect(self.AtomLayersCutSpinChanged)
        AtomLayersCutSpin.setToolTip("Set the number of layers of carbon atoms to cut off the tip")

        self.formLayout.addRow("Layers cut from tip: ", AtomLayersCutSpin)
        
        # Layers to cut from corners
        AtomLayersCornerCutSpin = QtGui.QSpinBox()
        AtomLayersCornerCutSpin.setMinimum(0)
        AtomLayersCornerCutSpin.setMaximum(10000)
        AtomLayersCornerCutSpin.setValue(self.generatorArgs.CornerSliceLayers)
        AtomLayersCornerCutSpin.valueChanged.connect(self.AtomLayersCornerCutSpinChanged)
        AtomLayersCornerCutSpin.setToolTip("Set the number of layers of carbon atoms to cut off the corners")

        self.formLayout.addRow("Layers cut from corners: ", AtomLayersCornerCutSpin)


        # Lattice constants

        # Lattice parameter presets combo
        ParamCombo = QtGui.QComboBox()
        ParamCombo.addItem("AIREBO")
        ParamCombo.addItem("ReaxFF May2016")
        ParamCombo.addItem("Custom")
        ParamCombo.currentIndexChanged.connect(self.ParamComboChanged)
        ParamCombo.setToolTip("Set lattice parameter presets")

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(ParamCombo)
        self.formLayout.addRow("Parameter presets", hbox)

        # Lattice 'a' parameter
        self.latticeAConstSpin = QtGui.QDoubleSpinBox()
        self.latticeAConstSpin.setDecimals(8)
        self.latticeAConstSpin.setSingleStep(0.1)
        self.latticeAConstSpin.setMinimum(0.00001)
        self.latticeAConstSpin.setMaximum(99.99999)
        self.latticeAConstSpin.setValue(self.generatorArgs.a0)
        self.latticeAConstSpin.setEnabled( False )
        self.latticeAConstSpin.valueChanged.connect(self.latticeAConstChanged)
        self.latticeAConstSpin.setSuffix(" \u212B")
        self.latticeAConstSpin.setToolTip("Set the lattice 'a' constant (select custom to edit this)")
        self.formLayout.addRow("Lattice 'a' constant", self.latticeAConstSpin)


        # generate button
        self.add_generate_button()

    def charge1_changed(self, val):
        """
        Charge 1 changed

        """
        self.generatorArgs.charge1 = val

    def AtomLayersSpinChanged(self, val):
        """
        Number of atom layers changed

        """
        self.generatorArgs.AtomLayers = val

    def AtomLayersCutSpinChanged(self, val):
        """
        Number of atom layers changed

        """
        self.generatorArgs.TipCutLayers = val
        
    def AtomLayersCornerCutSpinChanged(self, val):
        """
        Number of atom layers changed

        """
        self.generatorArgs.CornerSliceLayers = val

    def specie1_text_edited(self, text):
        """
        Specie 1 text edited

        """
        self.generatorArgs.sym1 = str(text)
        
    def latticeAConstChanged(self, val):
        """
        Lattice constant changed

        """
        self.generatorArgs.a0 = val

    def ParamComboChanged(self, index):
        """
        Parameter presets combo changed

        """

        # AIREBO
        if(index == 0):
            self.generatorArgs.a0 = 3.556717
            self.latticeAConstSpin.setValue(3.556717)
            self.latticeAConstSpin.setEnabled( False )
        # ReaxFF May 2016
        if(index == 1):
            self.generatorArgs.a0 = 3.54723712
            self.latticeAConstSpin.setValue(3.54723712)
            self.latticeAConstSpin.setEnabled( False )
        # Custom
        if(index == 2):
            self.latticeAConstSpin.setEnabled( True )

    def generateLatticeMain(self):
        """
        Generate lattice

        """
        generator = lattice_gen_diamond_indenter.DiamondIndenterGenerator()

        status, lattice = generator.generateLattice(self.generatorArgs)

        return status, lattice
