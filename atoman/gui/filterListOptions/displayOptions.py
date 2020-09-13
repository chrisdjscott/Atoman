
"""
Display options
---------------

Display options for a filter list.

* "Atom size scale factor" scales the radii of the atoms by the selected amount
* The "Sphere resolution" settings determine how the atoms (spheres) are drawn.
  There are three defaults: "low", "medium" and "high, or you can enter the
  settings manually.  In the formula "N" is the number of visible spheres.

"""
from __future__ import absolute_import
from __future__ import unicode_literals

import functools

from PySide2 import QtCore, QtWidgets


from .. import genericForm
import six


################################################################################

class DisplayOptionsWindow(QtWidgets.QDialog):
    """
    Display options dialog.

    """
    def __init__(self, mainWindow, parent=None):
        super(DisplayOptionsWindow, self).__init__(parent)

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.parent = parent
        self.mainWindow = mainWindow

        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setMinimumWidth(250)

        self.setWindowTitle("Filter list %d display options" % self.parent.tab)
#        self.setWindowIcon(QtGui.QIcon(iconPath("bonding.jpg")))

        # settings
        settings = QtCore.QSettings()

        # default options (read from settings if appropriate)
        self.atomScaleFactor = 1.0
        self.resA = float(settings.value("display/resA", 250.0))
        self.resB = float(settings.value("display/resB", 0.36))

        self.resDefaults = {
            "medium": (250, 0.36),
            "high": (330, 0.36),
            "low": (170, 0.36),
        }

        # layout
        layout = QtWidgets.QVBoxLayout(self)

        # group box
        scaleFactorGroup = genericForm.GenericForm(self, None, "Atom size scale factor")
        scaleFactorGroup.show()

        # scale factor
        self.atomScaleFactorSpin = QtWidgets.QDoubleSpinBox()
        self.atomScaleFactorSpin.setMinimum(0.1)
        self.atomScaleFactorSpin.setMaximum(2.0)
        self.atomScaleFactorSpin.setSingleStep(0.1)
        self.atomScaleFactorSpin.setValue(self.atomScaleFactor)

        row = scaleFactorGroup.newRow()
        row.addWidget(self.atomScaleFactorSpin)

        self.atomScaleFactorSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.atomScaleFactorSlider.setMinimum(1)
        self.atomScaleFactorSlider.setMaximum(20)
        self.atomScaleFactorSlider.setSingleStep(1)
        self.atomScaleFactorSlider.setValue(int(self.atomScaleFactor * 10))

        self.atomScaleFactorSpin.valueChanged.connect(self.atomScaleSpinChanged)
        self.atomScaleFactorSlider.valueChanged.connect(self.atomScaleSliderChanged)

        row = scaleFactorGroup.newRow()
        row.addWidget(self.atomScaleFactorSlider)

        layout.addWidget(scaleFactorGroup)

        # group box for resolution settings
        resGroupBox = genericForm.GenericForm(self, None, "Sphere resolution")
        resGroupBox.show()

        label = QtWidgets.QLabel("res = a.N^(-b)")
        row = resGroupBox.newRow()
        row.addWidget(label)

        label = QtWidgets.QLabel("a = ")
        self.resASpin = QtWidgets.QDoubleSpinBox()
        self.resASpin.setMinimum(1)
        self.resASpin.setMaximum(500)
        self.resASpin.setSingleStep(1)
        self.resASpin.valueChanged.connect(self.resAChanged)
        row = resGroupBox.newRow()
        row.addWidget(label)
        row.addWidget(self.resASpin)

        label = QtWidgets.QLabel("b = ")
        self.resBSpin = QtWidgets.QDoubleSpinBox()
        self.resBSpin.setMinimum(0.01)
        self.resBSpin.setMaximum(1)
        self.resBSpin.setSingleStep(0.01)
        self.resBSpin.valueChanged.connect(self.resBChanged)
        row = resGroupBox.newRow()
        row.addWidget(label)
        row.addWidget(self.resBSpin)

        # defaults buttons
        self.defaultButtonsDict = {}
        for setting in self.resDefaults:
            settingButton = QtWidgets.QPushButton(setting, parent=self)
            settingButton.setToolTip("Use default: %s" % setting)
            settingButton.clicked.connect(functools.partial(self.applyDefault, setting))
            settingButton.setAutoDefault(0)
            settingButton.setCheckable(1)
            settingButton.setChecked(0)
            row = resGroupBox.newRow()
            row.addWidget(settingButton)
            self.defaultButtonsDict[setting] = settingButton

        # set values
        self.resASpin.setValue(self.resA)
        self.resBSpin.setValue(self.resB)

        # store as default
        storeDefaultButton = QtWidgets.QPushButton("Store as default", parent=self)
        storeDefaultButton.setToolTip("Store settings as default values")
        storeDefaultButton.setAutoDefault(0)
        storeDefaultButton.clicked.connect(self.storeResSettings)
        row = resGroupBox.newRow()
        row.addWidget(storeDefaultButton)

        layout.addWidget(resGroupBox)

        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def storeResSettings(self):
        """
        Store current settings as default

        """
        settings = QtCore.QSettings()
        settings.setValue("display/resA", self.resA)
        settings.setValue("display/resB", self.resB)

    def applyDefault(self, setting):
        """
        Use default resA

        """
        resA, resB = self.resDefaults[setting]
        self.resASpin.setValue(resA)
        self.resBSpin.setValue(resB)

        # make sure this one is checked
        self.defaultButtonsDict[setting].setChecked(1)

    def resAChanged(self, val):
        """
        a changed

        """
        self.resA = val

        for setting, values in six.iteritems(self.resDefaults):
            aval, bval = values
            if aval == self.resA and bval == self.resB:
                self.defaultButtonsDict[setting].setChecked(1)
            else:
                self.defaultButtonsDict[setting].setChecked(0)

    def resBChanged(self, val):
        """
        b changed

        """
        self.resB = val

    def atomScaleSpinChanged(self, val):
        """
        Atom scale factor spin box changed.

        """
        self.atomScaleFactor = val
        self.atomScaleFactorSlider.setValue(int(val * 10))

    def atomScaleSliderChanged(self, val):
        """
        Atom scale factor slider changed.

        """
        self.atomScaleFactor = float(val) / 10.0
        self.atomScaleFactorSpin.setValue(self.atomScaleFactor)
