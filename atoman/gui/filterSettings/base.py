
"""
Settings for filters.

Dialogs must be named like: FilterNameSettingsDialog
where FilterName is the (capitalised) name of the
filter with no spaces. Eg "Point defects" becomes
"PointDefectsSettingsDialog".

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import logging

from PySide import QtGui, QtCore

from ...visutils.utilities import iconPath
import functools


################################################################################
class GenericSettingsDialog(QtGui.QDialog):
    def __init__(self, title, parent, filterType):
        super(GenericSettingsDialog, self).__init__(parent)

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.parent = parent
        self.mainWindow = self.parent.mainWindow
        self.pipelinePage = self.parent.filterTab
        self.filterType = filterType

        # logger
        loggerName = __name__
        words = str(filterType).title().split()
        dialogName = "%sSettingsDialog" % "".join(words)
        moduleName = dialogName[:1].lower() + dialogName[1:]
        array = loggerName.split(".")
        array[-1] = moduleName
        loggerName = ".".join(array)
        self.logger = logging.getLogger(loggerName)

        # get tab and filter id's
        array = title.split("(")[1].split(")")[0].split()
        self.listID = int(array[1])
        self.filterID = int(array[3])

        self.setModal(0)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)

        self.setWindowTitle(title)
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/configure.png")))
#        self.resize(500,300)

        dialogLayout = QtGui.QVBoxLayout()
        dialogLayout.setAlignment(QtCore.Qt.AlignTop)

        tabWidget = QtGui.QTabWidget()

        # layout/widget
        self.contentLayout = QtGui.QFormLayout()
        contentWidget = QtGui.QWidget()
        contentWidget.setLayout(self.contentLayout)

        tabWidget.addTab(contentWidget, "Calculate")

        # display settings
        self.displaySettingsLayout = QtGui.QFormLayout()
        displaySettingsWidget = QtGui.QWidget()
        displaySettingsWidget.setLayout(self.displaySettingsLayout)

        tabWidget.addTab(displaySettingsWidget, "Display")

        dialogLayout.addWidget(tabWidget)
        self.setLayout(dialogLayout)

        # button box
        self.buttonBox = QtGui.QDialogButtonBox()

        # add close button
        closeButton = self.buttonBox.addButton(self.buttonBox.Close)
        closeButton.setDefault(True)
        self.buttonBox.rejected.connect(self.close)

        dialogLayout.addWidget(self.buttonBox)

        # filtering enabled by default
#         self.filteringEnabled = True

        # help page
        self.helpPage = None

        # does this filter provide scalars
        self._providedScalars = []

        self._settings = None

    def addSpinBox(self, setting, minVal=None, maxVal=None, step=None, toolTip=None, label=None, settingEnabled=None, displayLayout=False):
        """
        Add a QSpinBox to the content layout with the given label.

        """
        # spin box
        spin = QtGui.QSpinBox()

        # optional configuration
        if minVal is not None:
            spin.setMinimum(minVal)
        if maxVal is not None:
            spin.setMaximum(maxVal)
        if step is not None:
            spin.setSingleStep(step)
        if toolTip is not None:
            spin.setToolTip(toolTip)
        if settingEnabled is not None:
            spin.setEnabled(self._settings.getSetting(settingEnabled))

        # set initial value
        spin.setValue(self._settings.getSetting(setting))

        # connect slot to updateSetting
        spin.valueChanged.connect(functools.partial(self._settings.updateSetting, setting))

        # optionally add to content layout
        if label is not None:
            if displayLayout:
                self.displaySettingsLayout.addRow(label, spin)
            else:
                self.contentLayout.addRow(label, spin)

        return spin

    def addDoubleSpinBox(self, setting, minVal=None, maxVal=None, step=None, toolTip=None, label=None, settingEnabled=None, displayLayout=False):
        """
        Add a QDoubleSpinBox to the content layout with the given label.

        """
        # spin box
        spin = QtGui.QDoubleSpinBox()

        # optional configuration
        if minVal is not None:
            spin.setMinimum(minVal)
        if maxVal is not None:
            spin.setMaximum(maxVal)
        if step is not None:
            spin.setSingleStep(step)
        if toolTip is not None:
            spin.setToolTip(toolTip)
        if settingEnabled is not None:
            spin.setEnabled(self._settings.getSetting(settingEnabled))

        # set initial value
        spin.setValue(self._settings.getSetting(setting))

        # connect slot to updateSetting
        spin.valueChanged.connect(functools.partial(self._settings.updateSetting, setting))

        # optionally add to content layout
        if label is not None:
            if displayLayout:
                self.displaySettingsLayout.addRow(label, spin)
            else:
                self.contentLayout.addRow(label, spin)

        return spin

    def addCheckBox(self, setting, toolTip=None, label=None, extraSlot=None, displayLayout=False, settingEnabled=None):
        """
        Add a check box.

        """
        # check box
        check = QtGui.QCheckBox()

        # initial check status
        check.setChecked(self._settings.getSetting(setting))

        # optional configuration
        if toolTip is not None:
            check.setToolTip(toolTip)
        if settingEnabled is not None:
            check.setEnabled(self._settings.getSetting(settingEnabled))

        # connect stateChanged signal
        def slot(state):
            enabled = False if state == QtCore.Qt.Unchecked else True
            self._settings.updateSetting(setting, enabled)
            if extraSlot is not None:
                extraSlot(enabled)

        check.stateChanged.connect(slot)

        # optionally add to content layout
        if label is not None:
            if displayLayout:
                self.displaySettingsLayout.addRow(label, check)
            else:
                self.contentLayout.addRow(label, check)

        return check

    def addComboBox(self, setting, items, toolTip=None, label=None, displayLayout=False, settingEnabled=None):
        """
        Add a combo box.

        """
        # combo box
        combo = QtGui.QComboBox()

        # add items
        combo.addItems(items)

        # set current index
        combo.setCurrentIndex(self._settings.getSetting(setting))

        # optional configuration
        if toolTip is not None:
            combo.setToolTip(toolTip)
        if settingEnabled is not None:
            combo.setEnabled(self._settings.getSetting(settingEnabled))

        # connect currentIndexChanged signal
        combo.currentIndexChanged.connect(functools.partial(self._settings.updateSetting, setting))

        # optionally add to content layout
        if label is not None:
            if displayLayout:
                self.displaySettingsLayout.addRow(label, combo)
            else:
                self.contentLayout.addRow(label, combo)

        return combo

    def getSettings(self):
        """Return the settings object."""
        return self._settings

    def addProvidedScalar(self, name):
        """Add scalar option."""
        self._providedScalars.append(name)

    def getProvidedScalars(self):
        """Return a list of scalars provided by this filter."""
        return self._providedScalars

    def addHorizontalDivider(self, displaySettings=False):
        """Add horizontal divider (QFrame)."""
        line = QtGui.QFrame()
        line.setFrameShape(QtGui.QFrame.HLine)
        line.setFrameShadow(QtGui.QFrame.Sunken)
        if displaySettings:
            self.displaySettingsLayout.addRow(line)
        else:
            self.contentLayout.addRow(line)

    def addLinkToHelpPage(self, page):
        """Add button with link to help page."""
        helpButton = self.buttonBox.addButton(self.buttonBox.Help)
        helpButton.setAutoDefault(False)
        helpButton.setToolTip("<p>Show help page (opens in browser)</p>")
        self.buttonBox.helpRequested.connect(self.loadHelpPage)

        self.helpPage = page

    def loadHelpPage(self):
        """Load the help page."""
        self.logger.debug("Help requested: '%s'", self.helpPage)

        if self.helpPage is None:
            return

        self.mainWindow.showHelp(relativeUrl=self.helpPage)

    def closeEvent(self, event):
        self.hide()

    def refresh(self):
        """
        Called whenever a new input is loaded.

        Should be overridden if required.

        """
        pass
