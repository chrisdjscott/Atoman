
"""
The output tab for the main toolbar

@author: Chris Scott

"""
import os
import sys
import shutil
import subprocess
import copy
import logging
import math
import functools
import datetime
import time

import numpy as np
from PySide import QtGui, QtCore

from ..visutils import utilities
from ..visutils import threading_vis
from ..visutils.utilities import iconPath
from . import genericForm
from ..state import _output as output_c
from ..plotting import rdf as rdf_c
from ..algebra import _vectors as vectors_c
from ..plotting import plotDialog
from . import utils

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################
class OutputDialog(QtGui.QDialog):
    def __init__(self, parent, mainWindow, width, index):
        super(OutputDialog, self).__init__(parent)
        
        self.parent = parent
        self.rendererWindow = parent
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.width = width
        
        self.setWindowTitle("Output - Render window %d" % index)
        self.setModal(0)
        
        # size
        self.resize(QtCore.QSize(350, 600))
        
        # layout
        outputTabLayout = QtGui.QVBoxLayout(self)
        outputTabLayout.setContentsMargins(0, 0, 0, 0)
        outputTabLayout.setSpacing(0)
        outputTabLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # add tab bar
        self.outputTypeTabBar = QtGui.QTabWidget(self)
        self.outputTypeTabBar.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        # add tabs to tab bar
        
        # image tab
        imageTabWidget = QtGui.QWidget()
        imageTabLayout = QtGui.QVBoxLayout(imageTabWidget)
        imageTabLayout.setContentsMargins(0, 0, 0, 0)
        
        self.imageTab = ImageTab(self, self.mainWindow, self.width)
        imageTabLayout.addWidget(self.imageTab)
        
        self.outputTypeTabBar.addTab(imageTabWidget, "Image")
        
        # file tab
        fileTabWidget = QtGui.QWidget()
        fileTabLayout = QtGui.QVBoxLayout(fileTabWidget)
        fileTabLayout.setContentsMargins(0, 0, 0, 0)
        
        self.fileTab = FileTab(self, self.mainWindow, self.width)
        fileTabLayout.addWidget(self.fileTab)
        self.outputTypeTabBar.addTab(fileTabWidget, "File")
        
        # plot tab
        self.plotTab = PlotTab(self.mainWindow, self.rendererWindow, parent=self)
        self.outputTypeTabBar.addTab(self.plotTab, "Plot")
        
        # add tab bar to layout
        outputTabLayout.addWidget(self.outputTypeTabBar)

################################################################################

class ScalarsHistogramOptionsForm(genericForm.GenericForm):
    """
    Main options form for scalars histograms
    
    """
    def __init__(self, parent, mainWindow, rendererWindow):
        super(ScalarsHistogramOptionsForm, self).__init__(parent, 0, "Histogram plot options")
        
        self.mainWindow = mainWindow
        self.rendererWindow = rendererWindow
        
        # current number of scalar plots
        self.numScalarsPlots = 0
        
        # current plots
        self.currentPlots = {}
        
        # add combo box
        self.scalarsCombo = QtGui.QComboBox()
        self.scalarsCombo.currentIndexChanged[int].connect(self.scalarsComboChanged)
        self.newRow().addWidget(self.scalarsCombo)
        
        # add stacked widget
        self.stackedWidget = QtGui.QStackedWidget()
        self.newRow().addWidget(self.stackedWidget)
        
        self.logger = logging.getLogger(__name__+".ScalarsHistogramOptionsForm")
    
    def scalarsComboChanged(self, index):
        """
        Scalars combo changed
        
        """
        self.stackedWidget.setCurrentIndex(index)
    
    def removeScalarPlotOptions(self):
        """
        Remove scalar plot options
        
        """
        self.logger.debug("Removing scalar plot options")
        for scalarsID in self.currentPlots.keys():
            self.logger.debug(" Removing: '%s'", scalarsID)
            form = self.currentPlots.pop(scalarsID)
            self.stackedWidget.removeWidget(form)
            form.deleteLater()
            self.scalarsCombo.removeItem(0)
        
        self.numScalarsPlots = 0
    
    def addAtomPropertyPlotOptions(self):
        """
        Add atom property plot options
        
        """
        self.logger.debug("Adding atom property plot options")
        
        # get current pipeline page
        pp = self.rendererWindow.getCurrentPipelinePage()
        ppindex = pp.pipelineIndex
        lattice = pp.inputState
        
        # add PE plot option
        scalarsArray = lattice.PE
        if np.min(scalarsArray) == 0 == np.max(scalarsArray):
            self.logger.debug(" Skipping PE: all zero")
        else:
            self.logger.debug(" Adding PE plot")
            scalarsName = "Potential energy"
            scalarsID = "%s (%d)" % (scalarsName, ppindex)
            self.addScalarPlotOptions(scalarsID, scalarsName, scalarsArray)
        
        # add KE plot option
        scalarsArray = lattice.KE
        if np.min(scalarsArray) == 0 == np.max(scalarsArray):
            self.logger.debug(" Skipping KE: all zero")
        else:
            self.logger.debug(" Adding KE plot")
            scalarsName = "Kinetic energy"
            scalarsID = "%s (%d)" % (scalarsName, ppindex)
            self.addScalarPlotOptions(scalarsID, scalarsName, scalarsArray)
        
        # add charge plot option
        scalarsArray = lattice.charge
        if np.min(scalarsArray) == 0 == np.max(scalarsArray):
            self.logger.debug(" Skipping charge: all zero")
        else:
            self.logger.debug(" Adding charge plot")
            scalarsName = "Charge"
            scalarsID = "%s (%d)" % (scalarsName, ppindex)
            self.addScalarPlotOptions(scalarsID, scalarsName, scalarsArray)
    
    def addScalarPlotOptions(self, scalarsID, scalarsName, scalarsArray):
        """
        Add plot for scalar 'name'
        
        """
        # don't add duplicates (should never happen anyway)
        if scalarsID in self.currentPlots:
            return
        
        # don't add empty arrays
        if not len(scalarsArray):
            return
        
        self.logger.debug("Adding scalar plot option: '%s'", scalarsID)
        
        # create form
        form = GenericHistogramPlotForm(self, scalarsID, scalarsName, scalarsArray)
        
        # add to stacked widget
        self.stackedWidget.addWidget(form)
        
        # add to combo box
        self.scalarsCombo.addItem(scalarsID)
        
        # store in dict
        self.currentPlots[scalarsID] = form
        
        # number of plots
        self.numScalarsPlots += 1
    
    def refreshScalarPlotOptions(self):
        """
        Refresh plot options
        
        * Called after pipeline page has run filters/single filter has run
        * loops over all filter lists under pipeline page, adding plots for all scalars
        * plots named after pipeline index and filter list index
        * also called when renderWindow pipeline index changes
        * also called when filter lists are cleared, etc...?
        
        """
        self.logger.debug("Refreshing plot options")
        
        # remove old options
        self.removeScalarPlotOptions()
        
        # get current pipeline page
        pp = self.rendererWindow.getCurrentPipelinePage()
        ppindex = pp.pipelineIndex
        
        # get filter lists
        filterLists = pp.filterLists
        
        # first add atom properties (KE, PE, charge)
        self.addAtomPropertyPlotOptions()
        
        # loop over filter lists, adding scalars
        self.logger.debug("Looping over filter lists (%d)", len(filterLists))
        for filterList in filterLists:
            # make unique name for pipeline page/filter list combo
            findex = filterList.tab
            filterListID = "%d-%d" % (ppindex, findex)
            self.logger.debug("Filter list %d; id '%s'", filterList.tab, filterListID)
            
            # loop over scalars in scalarsDict on filterer
            for scalarsName, scalarsArray in filterList.filterer.scalarsDict.iteritems():
                # make unique id
                scalarsID = "%s (%s)" % (scalarsName, filterListID)
                
                # add
                self.addScalarPlotOptions(scalarsID, scalarsName, scalarsArray)
        
        # hide if no plots, otherwise show
        if self.numScalarsPlots > 0:
            self.show()
        else:
            self.hide()

################################################################################

class PlotTab(QtGui.QWidget):
    """
    Plot tab
    
    """
    def __init__(self, mainWindow, rendererWindow, parent=None):
        super(PlotTab, self).__init__(parent)
        
        self.mainWindow = mainWindow
        self.rendererWindow = rendererWindow
        
        # layout
        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        
        # rdf
        row = self.newRow()
        self.rdfForm = RDFForm(self, self.mainWindow)
        row.addWidget(self.rdfForm)
        
        # scalars histograms
        self.scalarsForm = ScalarsHistogramOptionsForm(self, mainWindow, rendererWindow)
        row = self.newRow()
        row.addWidget(self.scalarsForm)
        
        self.layout.addStretch(1)
        
        # logging
        self.logger = logging.getLogger(__name__+".PlotTab")
    
    def newRow(self):
        """
        New row
        
        """
        row = genericForm.FormRow()
        self.layout.addWidget(row)
        
        return row

################################################################################

class GenericHistogramPlotForm(genericForm.GenericForm):
    """
    Plot options for a histogram of scalar values
    
    """
    def __init__(self, parent, scalarsID, scalarsName, scalarsArray):
        super(GenericHistogramPlotForm, self).__init__(parent, 0, "%s plot options" % scalarsID)
        
        self.parent = parent
        self.scalarsID = scalarsID
        self.scalarsName = scalarsName
        self.scalarsArray = scalarsArray
        self.logger = logging.getLogger(__name__+".GenericHistogramPlotForm")
        
        # scalar min/max
        self.scalarMin = np.min(scalarsArray)
        self.scalarMax = np.max(scalarsArray)
        
        # default 
        self.useNumBins = True
        self.numBins = 10
        self.binWidth = 1.0
        self.showAsFraction = False
        
        # min/max labels
        row = self.newRow()
        row.addWidget(QtGui.QLabel("Min: %f" % self.scalarMin))
        row = self.newRow()
        row.addWidget(QtGui.QLabel("Max: %f" % self.scalarMax))
        
        # num bins/bin width combo
        binCombo = QtGui.QComboBox()
        binCombo.addItem("Number of bins:")
        binCombo.addItem("Bin width:")
        binCombo.currentIndexChanged[int].connect(self.binComboChanged)
        
        # bin stack
        self.binStack = QtGui.QStackedWidget()
        
        # number of bins spin
        numBinsSpin = QtGui.QSpinBox()
        numBinsSpin.setMinimum(2)
        numBinsSpin.setMaximum(999)
        numBinsSpin.setSingleStep(1)
        numBinsSpin.setValue(self.numBins)
        numBinsSpin.valueChanged.connect(self.numBinsChanged)
        self.binStack.addWidget(numBinsSpin)
        
        # bin width spin
        binWidthSpin = QtGui.QDoubleSpinBox()
        binWidthSpin.setMinimum(0.01)
        binWidthSpin.setMaximum(99.99)
        binWidthSpin.setSingleStep(0.1)
        binWidthSpin.setValue(self.binWidth)
        binWidthSpin.valueChanged.connect(self.binWidthChanged)
        self.binStack.addWidget(binWidthSpin)
        
        # row
        row = self.newRow()
        row.addWidget(binCombo)
        row.addWidget(self.binStack)
        
        # show as fraction option
        showAsFractionCheck = QtGui.QCheckBox("Show as fraction")
        showAsFractionCheck.setCheckState(QtCore.Qt.Unchecked)
        showAsFractionCheck.stateChanged.connect(self.showAsFractionChanged)
        row = self.newRow()
        row.addWidget(showAsFractionCheck)
        
        # plot button
        plotButton = QtGui.QPushButton(QtGui.QIcon(iconPath("Plotter.png")), "Plot")
        plotButton.clicked.connect(self.makePlot)
        row = self.newRow()
        row.addWidget(plotButton)
        
        # show
        self.show()
    
    def binWidthChanged(self, val):
        """
        Bin width changed
        
        """
        self.binWidth = val
    
    def binComboChanged(self, index):
        """
        Bin combo changed
        
        """
        if index == 0:
            self.useNumBins = True
            self.binStack.setCurrentIndex(index)
        elif index == 1:
            self.useNumBins = False
            self.binStack.setCurrentIndex(index)
        else:
            self.logger.error("Bin combo index error (%d)", index)
    
    def showAsFractionChanged(self, checkState):
        """
        Show as fraction changed
        
        """
        if checkState == QtCore.Qt.Unchecked:
            self.showAsFraction = False
        else:
            self.showAsFraction = True
    
    def numBinsChanged(self, val):
        """
        Number of bins changed
        
        """
        self.numBins = val
    
    def makePlot(self):
        """
        Do the plot
        
        """
        self.logger.debug("Plotting '%s'", self.scalarsID)
        
        scalars = self.scalarsArray
        minVal = math.floor(self.scalarMin)
        maxVal = math.ceil(self.scalarMax)
        
        if maxVal == minVal:
            self.logger.error("Max val == min val; not plotting histogram")
            return
        
        # number of bins
        if self.useNumBins:
            numBins = self.numBins
        else:
            binWidth = self.binWidth
            
            # max
            maxVal = minVal
            while maxVal < self.scalarMax:
                maxVal += binWidth
            
            # num bins
            numBins = math.ceil((maxVal - minVal) / binWidth)
        
        # settings dict
        settingsDict = {}
        settingsDict["xlabel"] = self.scalarsName
        
        # make plot dialog
        if self.showAsFraction:
            # compute histogram
            hist, binEdges = np.histogram(scalars, numBins, range=(minVal, maxVal))
             
            # make fraction
            fracHist = hist / float(len(scalars))
            
            # bin width
            binWidth = (maxVal - minVal) / numBins
            
            # y label
            settingsDict["ylabel"] = "Fraction"
            
            # bar plot
            dlg = plotDialog.PlotDialog(self, self.parent.mainWindow, "%s histogram" % self.scalarsID, "bar", 
                                        (binEdges[:-1], fracHist), {"width": binWidth,}, settingsDict=settingsDict)
        
        else:
            # y label
            settingsDict["ylabel"] = "N"
            
            # histogram plot
            dlg = plotDialog.PlotDialog(self, self.parent.mainWindow, "%s histogram" % self.scalarsID, "hist", 
                                        (scalars, numBins), {"range": (minVal, maxVal),}, settingsDict=settingsDict)
        
        # show dialog
        dlg.show()

################################################################################

class RDFForm(genericForm.GenericForm):
    """
    RDF output form.
    
    """
    def __init__(self, parent, mainWindow):
        super(RDFForm, self).__init__(parent, 0, "RDF plot options")
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.rendererWindow = self.parent.rendererWindow
        self.logger = logging.getLogger(__name__+".RDFForm")
        
        # defaults
        self.spec1 = "ALL"
        self.spec2 = "ALL"
        self.binMin = 2.0
        self.binMax = 10.0
        self.NBins = 100
        
        # bond type
        label = QtGui.QLabel("Bond type:")
        row = self.newRow()
        row.addWidget(label)
        
        self.spec1Combo = QtGui.QComboBox()
        self.spec1Combo.addItem("ALL")
        self.spec1Combo.currentIndexChanged[str].connect(self.spec1Changed)
        row.addWidget(self.spec1Combo)
        
        label = QtGui.QLabel(" - ")
        row.addWidget(label)
        
        self.spec2Combo = QtGui.QComboBox()
        self.spec2Combo.addItem("ALL")
        self.spec2Combo.currentIndexChanged[str].connect(self.spec2Changed)
        row.addWidget(self.spec2Combo)
        
        # bin range
        label = QtGui.QLabel("Bin range:")
        row = self.newRow()
        row.addWidget(label)
        
        binMinSpin = QtGui.QDoubleSpinBox()
        binMinSpin.setMinimum(0.0)
        binMinSpin.setMaximum(500.0)
        binMinSpin.setSingleStep(0.01)
        binMinSpin.setValue(self.binMin)
        binMinSpin.valueChanged.connect(self.binMinChanged)
        row.addWidget(binMinSpin)
        
        label = QtGui.QLabel(" - ")
        row.addWidget(label)
        
        binMaxSpin = QtGui.QDoubleSpinBox()
        binMaxSpin.setMinimum(0.0)
        binMaxSpin.setMaximum(500.0)
        binMaxSpin.setSingleStep(0.01)
        binMaxSpin.setValue(self.binMax)
        binMaxSpin.valueChanged.connect(self.binMaxChanged)
        row.addWidget(binMaxSpin)
        
        # num bins
        label = QtGui.QLabel("Number of bins:")
        row = self.newRow()
        row.addWidget(label)
        
        numBinsSpin = QtGui.QSpinBox()
        numBinsSpin.setMinimum(2)
        numBinsSpin.setMaximum(100000)
        numBinsSpin.setSingleStep(1)
        numBinsSpin.setValue(self.NBins)
        numBinsSpin.valueChanged.connect(self.numBinsChanged)
        row.addWidget(numBinsSpin)
        
        # plot button
        plotButton = QtGui.QPushButton(QtGui.QIcon(iconPath("Plotter.png")), "Plot")
        plotButton.clicked.connect(self.plotRDF)
        row = self.newRow()
        row.addWidget(plotButton)
        
        # show
        self.show()
    
    def refresh(self):
        """
        Should be called whenver a new input is loaded.
        
        Refreshes the combo boxes with input specie list.
        
        """
        # lattice
        specieList = self.rendererWindow.getCurrentInputState().specieList
        
        # store current so can try to reselect
        spec1CurrentText = str(self.spec1Combo.currentText())
        spec2CurrentText = str(self.spec2Combo.currentText())
        
        # clear and rebuild combo box
        self.spec1Combo.clear()
        self.spec2Combo.clear()
        
        self.spec1Combo.addItem("ALL")
        self.spec2Combo.addItem("ALL")
        
        count = 1
        match1 = False
        match2 = False
        for sym in specieList:
            self.spec1Combo.addItem(sym)
            self.spec2Combo.addItem(sym)
            
            if sym == spec1CurrentText:
                self.spec1Combo.setCurrentIndex(count)
                match1 = True
            
            if sym == spec2CurrentText:
                self.spec2Combo.setCurrentIndex(count)
                match2 = True
            
            count += 1
        
        if not match1:
            self.spec1Combo.setCurrentIndex(0)
        
        if not match2:
            self.spec2Combo.setCurrentIndex(0)
    
    def plotRDF(self):
        """
        Plot RDF.
        
        """
        self.logger.info("Plotting RDF for visible atoms")
        
        # lattice and pipeline page
        inputLattice = self.rendererWindow.getCurrentInputState()
        pp = self.rendererWindow.getCurrentPipelinePage()
        
        # check system size
        warnDims = []
        if pp.PBC[0] and self.binMax > inputLattice.cellDims[0] / 2.0:
            warnDims.append("x")
        if pp.PBC[1] and self.binMax > inputLattice.cellDims[1] / 2.0:
            warnDims.append("y")
        if pp.PBC[2] and self.binMax > inputLattice.cellDims[2] / 2.0:
            warnDims.append("z")
        
        if len(warnDims):
            msg = "The maximum radius you have requested is greater than half the box length in the %s direction(s)!" % ", ".join(warnDims)
            self.mainWindow.displayError(msg)
            return
        
        # first gather vis atoms
        visibleAtoms = self.rendererWindow.gatherVisibleAtoms()
                    
        if not len(visibleAtoms):
            self.mainWindow.displayWarning("No visible atoms: cannot calculate RDF")
            return
        
        # then determine species
        specieList = inputLattice.specieList
        
        if self.spec1 == "ALL":
            spec1Index = -1
        else:
            spec1Index = int(np.where(specieList == self.spec1)[0][0])
        
        if self.spec2 == "ALL":
            spec2Index = -1
        else:
            spec2Index = int(np.where(specieList == self.spec2)[0][0])
        
        # prelims
        rdfArray = np.zeros(self.NBins, np.float64)
        
        # show progress dialog
        progDiag = utils.showProgressDialog("Calculating RDF", "Calculating RDF...", self)
        
        # num threads
        ompNumThreads = self.mainWindow.preferences.generalForm.openmpNumThreads
        
        try:
            # then calculate
            rdf_c.calculateRDF(visibleAtoms, inputLattice.specie, inputLattice.pos, spec1Index, spec2Index, inputLattice.minPos,
                               inputLattice.maxPos, inputLattice.cellDims, pp.PBC, self.binMin, self.binMax, self.NBins,
                               rdfArray, ompNumThreads)
        
        finally:
            utils.cancelProgressDialog(progDiag)
        
        # then plot
        interval = (self.binMax - self.binMin) / float(self.NBins)
        xn = np.arange(self.binMin + interval / 2.0, self.binMax, interval, dtype=np.float64)
        
        # prepare to plot
        settingsDict = {}
        settingsDict["title"] = "Radial distribution function"
        settingsDict["xlabel"] = "Bond length (A)"
        settingsDict["ylabel"] = "%s - %s G(r)" % (self.spec1, self.spec2)
        
        # show plot dialog
        dialog = plotDialog.PlotDialog(self, self.mainWindow, "Radial distribution function ", 
                                       "plot", (xn, rdfArray), {"linewidth": 2, "label": None},
                                       settingsDict=settingsDict)
        dialog.show()
    
    def numBinsChanged(self, val):
        """
        Num bins changed.
        
        """
        self.NBins = val
    
    def binMinChanged(self, val):
        """
        Bin min changed.
        
        """
        self.binMin = val
    
    def binMaxChanged(self, val):
        """
        Bin max changed.
        
        """
        self.binMax = val
    
    def spec1Changed(self, text):
        """
        Spec 1 changed.
        
        """
        self.spec1 = str(text)
    
    def spec2Changed(self, text):
        """
        Spec 2 changed.
        
        """
        self.spec2 = str(text)
    

################################################################################

class FileTab(QtGui.QWidget):
    """
    File output tab.
    
    """
    def __init__(self, parent, mainWindow, width):
        super(FileTab, self).__init__(parent)
        
        self.parent = parent
        self.rendererWindow = parent.rendererWindow
        self.mainWindow = mainWindow
        self.width = width
        
        # initial values
        self.outputFileType = "LATTICE"
        self.writeFullLattice = True
        
        # layout
        mainLayout = QtGui.QVBoxLayout(self)
        mainLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # name group
        fileNameGroup = genericForm.GenericForm(self, 0, "Output file options")
        fileNameGroup.show()
        
        # file type
        outputTypeCombo = QtGui.QComboBox()
        outputTypeCombo.addItem("LATTICE")
#         outputTypeCombo.addItem("LBOMD REF")
#         outputTypeCombo.addItem("LBOMD XYZ")
#         outputTypeCombo.addItem("LBOMD FAILSAFE")
        outputTypeCombo.currentIndexChanged[str].connect(self.outputTypeChanged)
        
        label = QtGui.QLabel("File type: ")
        
        row = fileNameGroup.newRow()
        row.addWidget(label)
        row.addWidget(outputTypeCombo)
        
        # option to write full lattice
        fullLatticeCheck = QtGui.QCheckBox("Write full lattice (not just visible)")
        fullLatticeCheck.setCheckState(QtCore.Qt.Checked)
        fullLatticeCheck.stateChanged.connect(self.fullLatticeCheckChanged)
        
        row = fileNameGroup.newRow()
        row.addWidget(fullLatticeCheck)
        
        # file name, save image button
        row = fileNameGroup.newRow()
        
        label = QtGui.QLabel("File name: ")
        self.outputFileName = QtGui.QLineEdit("lattice.dat")
        self.outputFileName.setFixedWidth(120)
        saveFileButton = QtGui.QPushButton(QtGui.QIcon(iconPath("image-x-generic.svg")), "")
        saveFileButton.setToolTip("Save to file")
        saveFileButton.clicked.connect(self.saveToFile)
        
        row.addWidget(label)
        row.addWidget(self.outputFileName)
        row.addWidget(saveFileButton)
        
        # dialog
        row = fileNameGroup.newRow()
        
        saveFileDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Save to file")
        saveFileDialogButton.setToolTip("Save to file")
        saveFileDialogButton.setCheckable(0)
        saveFileDialogButton.setFixedWidth(150)
        saveFileDialogButton.clicked.connect(self.saveToFileDialog)
        
        row.addWidget(saveFileDialogButton)
        
        # overwrite
        self.overwriteCheck = QtGui.QCheckBox("Overwrite")
        
        row = fileNameGroup.newRow()
        row.addWidget(self.overwriteCheck)
        
        mainLayout.addWidget(fileNameGroup)
    
    def fullLatticeCheckChanged(self, val):
        """
        Full lattice check changed.
        
        """
        if val == QtCore.Qt.Unchecked:
            self.writeFullLattice = False
        else:
            self.writeFullLattice = True
    
    def saveToFile(self):
        """
        Save current system to file.
        
        """
        filename = str(self.outputFileName.text())
        
        if not len(filename):
            return
        
        if os.path.exists(filename) and not self.overwriteCheck.isChecked():
            self.mainWindow.displayWarning("File already exists: not overwriting")
            return
        
        # lattice object
        lattice = self.rendererWindow.getCurrentInputState()
        
        # gather vis atoms
        visibleAtoms = self.rendererWindow.gatherVisibleAtoms()
        
        # write in C lib
        output_c.writeLattice(filename, visibleAtoms, lattice.cellDims, lattice.specieList, lattice.specie, lattice.pos, lattice.charge, self.writeFullLattice)
    
    def saveToFileDialog(self):
        """
        Open dialog.
        
        """
#         filename = QtGui.QFileDialog.getSaveFileName(self, 'Save File', '.', options=QtGui.QFileDialog.DontUseNativeDialog)[0]
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save File', '.')[0]
        
        if len(filename):
            self.outputFileName.setText(str(filename))
            self.saveToFile()
    
    def outputTypeChanged(self, fileType):
        """
        Output type changed.
        
        """
        self.outputFileType = str(fileType)

################################################################################

class ImageTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(ImageTab, self).__init__(parent)
        
        self.logger = logging.getLogger(__name__+".ImageTab")
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.width = width
        self.rendererWindow = self.parent.rendererWindow
        
        # initial values
        self.renderType = "VTK"
        self.imageFormat = "jpg"
#        self.overlayImage = False
        
        imageTabLayout = QtGui.QVBoxLayout(self)
#        imageTabLayout.setContentsMargins(0, 0, 0, 0)
#        imageTabLayout.setSpacing(0)
        imageTabLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # Add the generic image options at the top
        group = QtGui.QGroupBox("Image options")
        group.setAlignment(QtCore.Qt.AlignHCenter)
        
        groupLayout = QtGui.QVBoxLayout(group)
        groupLayout.setContentsMargins(0, 0, 0, 0)
        groupLayout.setSpacing(0)
        
        # render type (povray or vtk)
        renderTypeButtonGroup = QtGui.QButtonGroup(self)
        renderTypeButtonGroup.setExclusive(1)
        renderTypeButtonGroup.buttonClicked[int].connect(self.setRenderType)
        
        self.POVButton = QtGui.QPushButton(QtGui.QIcon(iconPath("pov-icon.svg")), "POV-Ray")
        self.POVButton.setCheckable(1)
        self.POVButton.setChecked(0)
        
        self.VTKButton = QtGui.QPushButton(QtGui.QIcon(iconPath("vtk-icon.svg")), "VTK")
        self.VTKButton.setCheckable(1)
        self.VTKButton.setChecked(1)
        
        renderTypeButtonGroup.addButton(self.VTKButton)
        renderTypeButtonGroup.addButton(self.POVButton)
        
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignTop)
        rowLayout.addWidget(self.VTKButton)
        rowLayout.addWidget(self.POVButton)
        
        groupLayout.addWidget(row)
        
        # image format
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        imageFormatButtonGroup = QtGui.QButtonGroup(self)
        imageFormatButtonGroup.setExclusive(1)
        imageFormatButtonGroup.buttonClicked[int].connect(self.setImageFormat)
        
        self.JPEGCheck = QtGui.QCheckBox("JPEG")
        self.JPEGCheck.setChecked(1)
        self.PNGCheck = QtGui.QCheckBox("PNG")
        self.TIFFCheck = QtGui.QCheckBox("TIFF")
        
        imageFormatButtonGroup.addButton(self.JPEGCheck)
        imageFormatButtonGroup.addButton(self.PNGCheck)
        imageFormatButtonGroup.addButton(self.TIFFCheck)
        
        rowLayout.addWidget(self.JPEGCheck)
        rowLayout.addWidget(self.PNGCheck)
        rowLayout.addWidget(self.TIFFCheck)
        
        groupLayout.addWidget(row)
        
        # additional (POV-Ray) options
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        groupLayout.addWidget(row)
        
        imageTabLayout.addWidget(group)
        
        # tab bar for different types of image output
        self.imageTabBar = QtGui.QTabWidget(self)
        self.imageTabBar.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        # add tabs to tab bar
        singleImageTabWidget = QtGui.QWidget()
        singleImageTabLayout = QtGui.QVBoxLayout(singleImageTabWidget)
        singleImageTabLayout.setContentsMargins(0, 0, 0, 0)
        self.singleImageTab = SingleImageTab(self, self.mainWindow, self.width)
        singleImageTabLayout.addWidget(self.singleImageTab)
        self.imageTabBar.addTab(singleImageTabWidget, "Single")
        
        imageSequenceTabWidget = QtGui.QWidget()
        imageSequenceTabLayout = QtGui.QVBoxLayout(imageSequenceTabWidget)
        imageSequenceTabLayout.setContentsMargins(0, 0, 0, 0)
        self.imageSequenceTab = ImageSequenceTab(self, self.mainWindow, self.width)
        imageSequenceTabLayout.addWidget(self.imageSequenceTab)
        self.imageTabBar.addTab(imageSequenceTabWidget, "Sequence")
        
        imageRotateTabWidget = QtGui.QWidget()
        imageRotateTabLayout = QtGui.QVBoxLayout(imageRotateTabWidget)
        imageRotateTabLayout.setContentsMargins(0, 0, 0, 0)
        self.imageRotateTab = ImageRotateTab(self, self.mainWindow, self.width)
        imageRotateTabLayout.addWidget(self.imageRotateTab)
        self.imageTabBar.addTab(imageRotateTabWidget, "Rotate")
        
        imageTabLayout.addWidget(self.imageTabBar)
    
    def setImageFormat(self, val):
        """
        Set the image format.
        
        """
        if self.JPEGCheck.isChecked():
            self.imageFormat = "jpg"
        
        elif self.PNGCheck.isChecked():
            self.imageFormat = "png"
        
        elif self.TIFFCheck.isChecked():
            self.imageFormat = "tif"
    
    def setRenderType(self, val):
        """
        Set current render type
        
        """
        if self.POVButton.isChecked():
            settings = self.mainWindow.preferences.povrayForm
            
            if not utilities.checkForExe(settings.pathToPovray):
                self.POVButton.setChecked(0)
                self.VTKButton.setChecked(1)
                utilities.warnExeNotFound(self, "%s (POV-Ray)" % (settings.pathToPovray,))
            
            else:
                self.renderType = "POV"
                self.imageFormat = "png"
                self.PNGCheck.setChecked(1)
        
        elif self.VTKButton.isChecked():
            self.renderType = "VTK"
            self.imageFormat = "jpg"
            self.JPEGCheck.setChecked(1)
    
    def createMovieLogger(self, level, message):
        """
        Log message for create movie object
        
        """
        logger = logging.getLogger(__name__)
        method = getattr(logger, level, None)
        if method is not None:
            method(message)
    
    def createMovie(self, saveDir, saveText, createMovieBox):
        """
        Create movie.
        
        """
        settings = self.mainWindow.preferences.ffmpegForm
        ffmpeg = utilities.checkForExe(settings.pathToFFmpeg)
        if not ffmpeg:
            utilities.warnExeNotFound(self, "%s (FFmpeg)" % (settings.pathToFFmpeg,))
            return 2
        
        CWD = os.getcwd()
        try:
            os.chdir(saveDir)
        except OSError:
            return 1
        
        try:
            # settings
            settings = self.mainWindow.preferences.ffmpegForm
            framerate = createMovieBox.framerate
            bitrate = settings.bitrate
            outputprefix = createMovieBox.prefix
            outputsuffix = createMovieBox.suffix
            
            saveText = os.path.basename(saveText)
            
            self.logger.info("Creating movie file: %s.%s", outputprefix, outputsuffix)
            
            # movie generator object
            generator = MovieGenerator()
            generator.log.connect(self.createMovieLogger)
            generator.allDone.connect(generator.deleteLater)
            
            # create movie
#             generator.run(os.getcwd(), ffmpeg, framerate, saveText, self.imageFormat, bitrate, outputprefix, outputsuffix)
            
            # runnable for sending to thread pool
            runnable = threading_vis.GenericRunnable(generator, args=(os.getcwd(), ffmpeg, framerate, saveText, 
                                                                      self.imageFormat, bitrate, outputprefix, 
                                                                      outputsuffix))
            runnable.setAutoDelete(False)
            
            # add to thread pool
            QtCore.QThreadPool.globalInstance().start(runnable)
        
        finally:
            os.chdir(CWD)

################################################################################

class MovieGenerator(QtCore.QObject):
    """
    Call ffmpeg to generate a movie
    
    """
    log = QtCore.Signal(str, str)
    allDone = QtCore.Signal()
    
    def __init__(self):
        super(MovieGenerator, self).__init__()
    
    def run(self, workDir, ffmpeg, framerate, saveText, imageFormat, bitrate, outputPrefix, outputSuffix):
        """
        Create movie
        
        """
        owd = os.getcwd()
        os.chdir(workDir)
        
        try:
            command = "'%s' -r %d -y -i %s.%s -r %d -b %dk '%s.%s'" % (ffmpeg, framerate, saveText, 
                                                                       imageFormat, 25, bitrate, 
                                                                       outputPrefix, outputSuffix)
            
            self.log.emit("debug", 'Command: "%s"' % command)
            
            ffmpegTime = time.time()
            
            process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
             
            output, stderr = process.communicate()
            status = process.poll()
            if status:
                self.log.emit("error", "FFmpeg failed (%d)" % status)
                self.log.emit("error", output)
                self.log.emit("error", stderr)
            
            ffmpegTime = time.time() - ffmpegTime
            self.log.emit("debug", "FFmpeg time taken: %f s" % ffmpegTime)
        
        finally:
            os.chdir(owd)
            self.allDone.emit()

################################################################################
class SingleImageTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(SingleImageTab, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.width = width
        self.rendererWindow = self.parent.rendererWindow
        
        # initial values
        self.overwriteImage = 0
        self.openImage = 1
        
        # layout
        mainLayout = QtGui.QVBoxLayout(self)
#        mainLayout.setContentsMargins(0, 0, 0, 0)
#        mainLayout.setSpacing(0)
        mainLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # file name, save image button
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        
        label = QtGui.QLabel("File name")
        self.imageFileName = QtGui.QLineEdit("image")
        self.imageFileName.setFixedWidth(120)
        saveImageButton = QtGui.QPushButton(QtGui.QIcon(iconPath("image-x-generic.svg")), "")
        saveImageButton.setToolTip("Save image")
        saveImageButton.clicked.connect(functools.partial(self.saveSingleImage, True))
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.imageFileName)
        rowLayout.addWidget(saveImageButton)
        
        mainLayout.addWidget(row)
        
        # dialog
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        
        saveImageDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Save image")
        saveImageDialogButton.setToolTip("Save image")
        saveImageDialogButton.setCheckable(0)
        saveImageDialogButton.setFixedWidth(150)
        saveImageDialogButton.clicked.connect(self.saveSingleImageDialog)
        
        rowLayout.addWidget(saveImageDialogButton)
        
        mainLayout.addWidget(row)
        
        # options
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.overwriteCheck = QtGui.QCheckBox("Overwrite")
        self.overwriteCheck.stateChanged[int].connect(self.overwriteCheckChanged)
        
        self.openImageCheck = QtGui.QCheckBox("Open image")
        self.openImageCheck.setChecked(True)
        self.openImageCheck.stateChanged[int].connect(self.openImageCheckChanged)
        
        rowLayout.addWidget(self.overwriteCheck)
        rowLayout.addWidget(self.openImageCheck)
        
        mainLayout.addWidget(row)
        
    def saveSingleImageDialog(self):
        """
        Open dialog to get save file name
        
        """
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save File', '.')[0]
        
        if len(filename):
            self.imageFileName.setText(str(filename))
            self.saveSingleImage(showProgress=True)
    
    def saveSingleImage(self, showProgress=False):
        """
        Screen capture.
        
        """
        if self.parent.renderType == "POV":
            settings = self.mainWindow.preferences.povrayForm
            povray = utilities.checkForExe(settings.pathToPovray)
            if not povray:
                utilities.warnExeNotFound(self, "%s (POV-Ray)" % (settings.pathToPovray,))
                return
        
        else:
            povray = ""
        
        filename = str(self.imageFileName.text())
        
        if not len(filename):
            return
        
        # show progress dialog
        if showProgress and self.parent.renderType == "POV":
            progress = QtGui.QProgressDialog(parent=self)
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setWindowTitle("Busy")
            progress.setLabelText("Running POV-Ray...")
            progress.setRange(0, 0)
            progress.setMinimumDuration(0)
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            progress.show()
        
        filename = self.rendererWindow.renderer.saveImage(self.parent.renderType, self.parent.imageFormat, 
                                                          filename, self.overwriteImage, povray=povray)
        
        # hide progress dialog
        if showProgress and self.parent.renderType == "POV":
            QtGui.QApplication.restoreOverrideCursor()
            progress.cancel()
        
        if filename is None:
            print "SAVE IMAGE FAILED"
            return
        
        # open image viewer
        if self.openImage:
            dirname = os.path.dirname(filename)
            if not dirname:
                dirname = os.getcwd()
            
            self.mainWindow.imageViewer.changeDir(dirname)
            self.mainWindow.imageViewer.showImage(filename)
            self.mainWindow.imageViewer.hide()
            self.mainWindow.imageViewer.show()
    
    def openImageCheckChanged(self, val):
        """
        Open image
        
        """
        if self.openImageCheck.isChecked():
            self.openImage = 1
        
        else:
            self.openImage = 0
    
    def overwriteCheckChanged(self, val):
        """
        Overwrite file
        
        """
        if self.overwriteCheck.isChecked():
            self.overwriteImage = 1
        
        else:
            self.overwriteImage = 0

################################################################################
class CreateMovieBox(QtGui.QGroupBox):
    """
    Create movie settings
    
    """
    def __init__(self, parent=None):
        super(CreateMovieBox, self).__init__(parent)
        
        self.setTitle("Create movie")
        self.setCheckable(True)
        self.setChecked(True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        
        # defaults
        self.framerate = 10
        self.prefix = "movie"
        self.suffix = "flv"
        
        # layout
        self.contentLayout = QtGui.QVBoxLayout(self)
        self.contentLayout.setContentsMargins(0, 0, 0, 0)
        self.contentLayout.setSpacing(0)
        
        # framerate
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("Framerate:")
        rowLayout.addWidget(label)
        
        framerateSpin = QtGui.QSpinBox()
        framerateSpin.setMinimum(1)
        framerateSpin.setMaximum(10000)
        framerateSpin.setValue(self.framerate)
        framerateSpin.valueChanged.connect(self.framerateChanged)
        rowLayout.addWidget(framerateSpin)
        
        label = QtGui.QLabel(" fps")
        rowLayout.addWidget(label)
        
        # file prefix
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("File prefix:")
        rowLayout.addWidget(label)
        
        prefixLineEdit = QtGui.QLineEdit(self.prefix)
        prefixLineEdit.setFixedWidth(130)
        prefixLineEdit.textChanged.connect(self.prefixChanged)
        rowLayout.addWidget(prefixLineEdit)
        
        # container
        rowLayout = self.newRow()
        
        label = QtGui.QLabel("Container:")
        rowLayout.addWidget(label)
        
        containerCombo = QtGui.QComboBox()
        containerCombo.addItem("flv")
        containerCombo.addItem("mpg")
#         containerCombo.addItem("mp4")
        containerCombo.addItem("avi")
#         containerCombo.addItem("mov")
        containerCombo.currentIndexChanged[str].connect(self.suffixChanged)
        rowLayout.addWidget(containerCombo)
    
    def suffixChanged(self, text):
        """
        Suffix changed
        
        """
        self.suffix = str(text)
    
    def framerateChanged(self, val):
        """
        Framerate changed.
        
        """
        self.framerate = val
    
    def prefixChanged(self, text):
        """
        Prefix changed.
        
        """
        self.prefix = str(text)
    
    def newRow(self, align=None):
        """
        New row
        
        """
        row = genericForm.FormRow(align=align)
        self.contentLayout.addWidget(row)
        
        return row

################################################################################
class ImageSequenceTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(ImageSequenceTab, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.width = width
        self.rendererWindow = self.parent.rendererWindow
        
        self.logger = logging.getLogger(__name__+".ImageSequenceTab")
        
        # initial values
        self.numberFormat = "%04d"
        self.minIndex = 0
        self.maxIndex = 10
        self.interval = 1
        self.fileprefixText = "guess"
        self.overwrite = False
        self.flickerFlag = False
        self.rotateAfter = False
#         self.createMovie = 1
        
        # layout
        mainLayout = QtGui.QVBoxLayout(self)
#        mainLayout.setContentsMargins(0, 0, 0, 0)
#        mainLayout.setSpacing(0)
        mainLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # output directory
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        label = QtGui.QLabel("Output folder")
        self.outputFolder = QtGui.QLineEdit("sequencer")
        self.outputFolder.setFixedWidth(120)
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.outputFolder)
        
        mainLayout.addWidget(row)
        
        # file prefix
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        label = QtGui.QLabel("File prefix")
                
        self.fileprefix = QtGui.QLineEdit(self.fileprefixText)
        self.fileprefix.setFixedWidth(120)
        self.fileprefix.textChanged[str].connect(self.fileprefixChanged)
        
        resetPrefixButton = QtGui.QPushButton(QtGui.QIcon(iconPath("edit-paste.svg")), "")
        resetPrefixButton.setStatusTip("Set prefix to input file")
        resetPrefixButton.setToolTip("Set prefix to input file")
        resetPrefixButton.clicked.connect(self.resetPrefix)
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.fileprefix)
        rowLayout.addWidget(resetPrefixButton)
        
        mainLayout.addWidget(row)
        
        group = QtGui.QGroupBox("Numbering")
        group.setAlignment(QtCore.Qt.AlignHCenter)
        
        groupLayout = QtGui.QVBoxLayout(group)
        groupLayout.setContentsMargins(0, 0, 0, 0)
        groupLayout.setSpacing(0)
        
        # numbering format
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
#        label = QtGui.QLabel("Number format")
        self.numberFormatCombo = QtGui.QComboBox()
        self.numberFormatCombo.addItem("%04d")
        self.numberFormatCombo.addItem("%d")
        self.numberFormatCombo.currentIndexChanged[str].connect(self.numberFormatChanged)
        
#        rowLayout.addWidget(label)
        rowLayout.addWidget(self.numberFormatCombo)
        
        groupLayout.addWidget(row)
        
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.minIndexSpinBox = QtGui.QSpinBox()
        self.minIndexSpinBox.setMinimum(0)
        self.minIndexSpinBox.setMaximum(99999)
        self.minIndexSpinBox.setValue(self.minIndex)
        self.minIndexSpinBox.valueChanged[int].connect(self.minIndexChanged)
        
        label = QtGui.QLabel("to")
        
        self.maxIndexSpinBox = QtGui.QSpinBox()
        self.maxIndexSpinBox.setMinimum(1)
        self.maxIndexSpinBox.setMaximum(99999)
        self.maxIndexSpinBox.setValue(self.maxIndex)
        self.maxIndexSpinBox.valueChanged[int].connect(self.maxIndexChanged)
        
        label2 = QtGui.QLabel("by")
        
        self.intervalSpinBox = QtGui.QSpinBox()
        self.intervalSpinBox.setMinimum(1)
        self.intervalSpinBox.setMaximum(99999)
        self.intervalSpinBox.setValue(self.interval)
        self.intervalSpinBox.valueChanged[int].connect(self.intervalChanged)
        
        rowLayout.addWidget(self.minIndexSpinBox)
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.maxIndexSpinBox)
        rowLayout.addWidget(label2)
        rowLayout.addWidget(self.intervalSpinBox)
        
        groupLayout.addWidget(row)
        
        mainLayout.addWidget(group)
        
        # first file
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        label = QtGui.QLabel("First file:")
        
        self.firstFileLabel = QtGui.QLabel("")
        self.setFirstFileLabel()
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.firstFileLabel)
        
        mainLayout.addWidget(row)
        
        # overwrite check box
#         row = QtGui.QWidget(self)
#         rowLayout = QtGui.QHBoxLayout(row)
# #        rowLayout.setSpacing(0)
#         rowLayout.setContentsMargins(0, 0, 0, 0)
#         rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
#         self.overwriteCheck = QtGui.QCheckBox("Overwrite")
#         self.overwriteCheck.stateChanged[int].connect(self.overwriteCheckChanged)
#         rowLayout.addWidget(self.overwriteCheck)
#         mainLayout.addWidget(row)
        
        # eliminate flicker check
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        self.flickerCheck = QtGui.QCheckBox("Eliminate flicker")
        self.flickerCheck.stateChanged[int].connect(self.flickerCheckChanged)
        rowLayout.addWidget(self.flickerCheck)
        mainLayout.addWidget(row)
        
        # rotate at end
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        self.rotateAfterCheck = QtGui.QCheckBox("Rotate at end")
        self.rotateAfterCheck.stateChanged[int].connect(self.rotateAfterCheckChanged)
        rowLayout.addWidget(self.rotateAfterCheck)
        mainLayout.addWidget(row)
        
        # create movie box
        self.createMovieBox = CreateMovieBox(self)
        mainLayout.addWidget(self.createMovieBox)
        
        # start button
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        startSequencerButton = QtGui.QPushButton(QtGui.QIcon(iconPath("loadandsave-icon.svg")), "START")
        startSequencerButton.setStatusTip("Start sequencer")
        startSequencerButton.setToolTip("Start sequencer")
        startSequencerButton.clicked.connect(self.startSequencer)
        
        rowLayout.addWidget(startSequencerButton)
        
        mainLayout.addWidget(row)
    
    def rotateAfterCheckChanged(self, state):
        """
        Rotate after sequencer changed
        
        """
        if state == QtCore.Qt.Unchecked:
            self.rotateAfter = False
        
        else:
            self.rotateAfter = True
    
    def resetPrefix(self):
        """
        Reset the prefix to the one from 
        the input page
        
        """
        pp = self.rendererWindow.getCurrentPipelinePage()
        
        if pp is None:
            filename = ""
        
        else:
            filename = pp.filename
        
        count = 0
        lim = None
        for i in xrange(len(filename)):
            if filename[i] == ".":
                break
            
            try:
                int(filename[i])
                
                if lim is None:
                    lim = count
            
            except ValueError:
                lim = None
            
            count += 1
        
        self.fileprefix.setText(filename[:lim])
    
    def startSequencer(self):
        """
        Start the sequencer
        
        """
        self.runSequencer()
        
    def runSequencer(self):
        """
        Run the sequencer
        
        """
        if self.parent.renderType == "POV":
            settings = self.mainWindow.preferences.povrayForm
            povray = utilities.checkForExe(settings.pathToPovray)
            if not povray:
                utilities.warnExeNotFound(self, "%s (POV-Ray)" % (settings.pathToPovray,))
                return
        
        else:
            povray = ""
        
        self.setFirstFileLabel()
        
        # get pipeline page
        pipelinePage = self.rendererWindow.getCurrentPipelinePage()
        
        # formatted string
        fileText = "%s%s.%s" % (str(self.fileprefix.text()), self.numberFormat, pipelinePage.extension)
        
        # check abspath (for sftp)
        abspath = pipelinePage.abspath
        sftpBrowser = None
        if ":" in abspath:
            self.logger.debug("Sequencing SFTP file")
            sftpHost, sftpFile = abspath.split(":")
            self.logger.debug("Host: '%s'; path: '%s'", sftpHost, sftpFile)
            
            sysDiag = self.mainWindow.systemsDialog
            sftpDlg = sysDiag.load_system_form.sftp_browser
            match = False
            for i in xrange(sftpDlg.stackedWidget.count()):
                w = sftpDlg.stackedWidget.widget(i)
                if w.connectionID == sftpHost:
                    match = True
                    break
            
            if not match:
                self.logger.error("Could not find SFTP browser for '%s'", sftpHost)
                return
            
            # browser
            sftpBrowser = w
        
        # check first file exists
        if sftpBrowser is None:
            firstFileExists = utilities.checkForFile(str(self.firstFileLabel.text()))
        else:
            rp = os.path.join(os.path.dirname(sftpFile), str(self.firstFileLabel.text()))
            self.logger.debug("Checking first file exists (SFTP): '%s'", rp)
            firstFileExists = bool(sftpBrowser.checkPathExists(rp)) or bool(sftpBrowser.checkPathExists(rp+".gz")) or bool(sftpBrowser.checkPathExists(rp+".bz2"))
        
        if not firstFileExists:
            self.warnFileNotPresent(str(self.firstFileLabel.text()), tag="first")
            return
        
        # check last file exists
        lastFile = fileText % self.maxIndex
        if sftpBrowser is None:
            lastFileExists = utilities.checkForFile(lastFile)
        else:
            rp = os.path.join(os.path.dirname(sftpFile), lastFile)
            self.logger.debug("Checking last file exists (SFTP): '%s'", rp)
            lastFileExists = bool(sftpBrowser.checkPathExists(rp)) or bool(sftpBrowser.checkPathExists(rp+".gz")) or bool(sftpBrowser.checkPathExists(rp+".bz2"))
        
        if not lastFileExists:
            self.warnFileNotPresent(lastFile, tag="last")
            return
        
        self.logger.info("Running sequencer")
        
        # store current input state
        origInput = copy.deepcopy(self.rendererWindow.getCurrentInputState())
        
        # pipeline index
        pipelineIndex = self.rendererWindow.currentPipelineIndex
        
        # stack index from systems dialog
        ida, idb = pipelinePage.inputStackIndex
        
        if ida != 0:
            self.logger.error("Cannot sequence a generated lattice")
            return
        
        self.logger.debug("  Input stack index: %d, %d", ida, idb)
        
        systemsDialog = self.mainWindow.systemsDialog
        iniStackIndexa = systemsDialog.new_system_stack.currentIndex()
        
        systemsDialog.new_system_stack.setCurrentIndex(ida)
        in_page = systemsDialog.new_system_stack.currentWidget()
        iniStackIndexb = in_page.stackedWidget.currentIndex()
        
        in_page.stackedWidget.setCurrentIndex(idb)
        readerForm = in_page.stackedWidget.currentWidget()
        reader = readerForm.latticeReader
        
        # back to original
        in_page.stackedWidget.setCurrentIndex(iniStackIndexb)
        systemsDialog.new_system_stack.setCurrentIndex(iniStackIndexa)
        
        self.logger.debug("  Reader: %s %s", str(readerForm), str(reader))
        
        # directory
        saveDir = str(self.outputFolder.text())
        saveDir += "-%s" % datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        if os.path.exists(saveDir):
            if self.overwrite:
                shutil.rmtree(saveDir)
            
            else:
                count = 0
                while os.path.exists(saveDir):
                    count += 1
                    saveDir = "%s.%d" % (str(self.outputFolder.text()), count)
        
        os.mkdir(saveDir)
        
        saveText = os.path.join(saveDir, "%s%s" % (str(self.fileprefix.text()), self.numberFormat))
        
        # progress dialog
        NSteps = int((self.maxIndex - self.minIndex) / self.interval) + 1
        progDialog = QtGui.QProgressDialog("Running sequencer...", "Cancel", self.minIndex, NSteps)
        progDialog.setWindowModality(QtCore.Qt.WindowModal)
        progDialog.setWindowTitle("Progress")
        progDialog.setValue(self.minIndex)
        progDialog.show()
        
        QtGui.QApplication.processEvents()
        
        # loop over files
        status = 0
        previousPos = None
        try:
            count = 0
            for i in xrange(self.minIndex, self.maxIndex + self.interval, self.interval):
                if sftpBrowser is None:
                    currentFile = fileText % i
                    self.logger.info("Current file: '%s'", currentFile)
                
                else:
                    # we have to copy current file locally and use that, then delete it afterwards
                    basename = fileText % i
                    remoteFile = os.path.join(os.path.dirname(sftpFile), basename)
                    currentFile = os.path.join(self.mainWindow.tmpDirectory, basename)
                    
                    # check exists
                    remoteFileTest = remoteFile
                    fileExists = bool(sftpBrowser.checkPathExists(remoteFileTest))
                    if not fileExists:
                        # check gzip
                        remoteFileTest = remoteFile + ".gz"
                        fileExists = bool(sftpBrowser.checkPathExists(remoteFileTest))
                        if fileExists:
                            currentFile += ".gz"
                        else:
                            # check bzip
                            remoteFileTest = remoteFile + ".bz2"
                            fileExists = bool(sftpBrowser.checkPathExists(remoteFileTest))
                            if fileExists:
                                currentFile += ".bz2"
                            else:
                                self.logger.error("SFTP sequencer file does not exist: '%s'", remoteFile)
                                return
                    
                    remoteFile = remoteFileTest
                    
                    # copy locally
                    self.logger.debug("Copying file for sequencer: '%s' to '%s'", remoteFile, currentFile)
                    # copy file and roulette if exists..,
                    sftpBrowser.copySystem(remoteFile, currentFile)
                
                # read in state
                if reader.requiresRef:
                    status, state = reader.readFile(currentFile, readerForm.currentRefState, rouletteIndex=i-1)
                
                else:
                    status, state = reader.readFile(currentFile, rouletteIndex=i-1)
                
                if status:
                    self.logger.error("Sequencer read file failed with status: %d" % status)
                    break
                
                # eliminate flicker across PBCs
                if self.flickerFlag:
                    self.eliminateFlicker(state, previousPos, pipelinePage)
                    previousPos = copy.deepcopy(state.pos)
                
                # set input state on current pipeline
                pipelinePage.inputState = state
                
                # exit if cancelled
                if progDialog.wasCanceled():
                    if sftpBrowser is not None:
                        os.unlink(currentFile)
                    return
                
                # now apply all filters
                pipelinePage.runAllFilterLists(sequencer=True)
                
                # exit if cancelled
                if progDialog.wasCanceled():
                    if sftpBrowser is not None:
                        os.unlink(currentFile)
                    return
                
                saveName = saveText % (count,)
                self.logger.info("  Saving image: '%s'", saveName)
                
                # now save image
                filename = self.rendererWindow.renderer.saveImage(self.parent.renderType, self.parent.imageFormat, saveName, 1, povray=povray)
                
                count += 1
                
                # exit if cancelled
                if progDialog.wasCanceled():
                    if sftpBrowser is not None:
                        os.unlink(currentFile)
                    return
                
                # delete local copy of file (SFTP)
                if sftpBrowser is not None:
                    os.unlink(currentFile)
                
                # update progress
                progDialog.setValue(count)
                
                QtGui.QApplication.processEvents()
            
            # rotate?
            if self.rotateAfter:
                self.logger.debug("Running rotator after sequencer...")
                self.parent.imageRotateTab.startRotator()
        
        finally:
            self.logger.debug("Reloading original input")
            
            # reload original input
            pipelinePage.inputState = origInput
            pipelinePage.postInputLoaded()
            
            # run filter list if didn't auto run
            if origInput.NAtoms > self.mainWindow.preferences.renderingForm.maxAtomsAutoRun:
                pipelinePage.runAllFilterLists()
            
            # close progress dialog
            progDialog.close()
        
        # create movie
        if not status and self.createMovieBox.isChecked():
            # show wait cursor
            QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
            
            try:
                self.parent.createMovie(saveDir, saveText, self.createMovieBox)
            
            finally:
                # set cursor to normal
                QtGui.QApplication.restoreOverrideCursor()
    
    def eliminateFlicker(self, state, previousPos, pipelinePage):
        """
        Attempt to eliminate flicker across PBCs
        
        """
        if previousPos is None or len(previousPos) != len(state.pos):
            return
        
        pbc = pipelinePage.PBC
        if not pbc[0] and not pbc[1] and not pbc[2]:
            return
        
        logger = self.logger
        logger.debug("Attempting to eliminate PBC flicker")
        
        count = vectors_c.eliminatePBCFlicker(state.NAtoms, state.pos, previousPos, state.cellDims, pbc)
        
        logger.debug("Modified: %d", count)
    
    def warnFileNotPresent(self, filename, tag="first"):
        """
        Warn the first file is not present.
        
        """
#         QtGui.QMessageBox.warning(self, "Warning", "Could not locate %s file in sequence: %s" % (tag, filename))
        
        message = "Could not locate %s file in sequence: %s" % (tag, filename)
        
        msgBox = QtGui.QMessageBox(self)
        msgBox.setText(message)
        msgBox.setWindowFlags(msgBox.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
        msgBox.setIcon(QtGui.QMessageBox.Warning)
        msgBox.exec_()
    
    def flickerCheckChanged(self, state):
        """
        Flicker check changed
        
        """
        if state == QtCore.Qt.Unchecked:
            self.flickerFlag = False
        
        else:
            self.flickerFlag = True
    
    def overwriteCheckChanged(self, val):
        """
        Overwrite check changed
        
        """
        if self.overwriteCheck.isChecked():
            self.overwrite = 1
        
        else:
            self.overwrite = 0
    
    def fileprefixChanged(self, text):
        """
        File prefix has changed
        
        """
        self.fileprefixText = str(text)
        
        self.setFirstFileLabel()
    
    def setFirstFileLabel(self):
        """
        Set the first file label
        
        """
        pp = self.rendererWindow.getCurrentPipelinePage()
        
        if pp is None:
            ext = ""
        
        else:
            ext = pp.extension
        
        text = "%s%s.%s" % (self.fileprefix.text(), self.numberFormat, ext)
        self.firstFileLabel.setText(text % (self.minIndex,))
    
    def minIndexChanged(self, val):
        """
        Minimum index changed
        
        """
        self.minIndex = val
        
        self.setFirstFileLabel()
    
    def maxIndexChanged(self, val):
        """
        Maximum index changed
        
        """
        self.maxIndex = val
    
    def intervalChanged(self, val):
        """
        Interval changed
        
        """
        self.interval = val
    
    def numberFormatChanged(self, text):
        """
        Change number format
        
        """
        self.numberFormat = str(text)
        
        self.setFirstFileLabel()


################################################################################
class ImageRotateTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(ImageRotateTab, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.width = width
        self.rendererWindow = self.parent.rendererWindow
        
        # initial values
        self.fileprefixText = "rotate"
        self.overwrite = 0
        self.degreesPerRotation = 5.0
        
        self.logger = logging.getLogger(__name__+".ImageRotateTab")
        
        # layout
        mainLayout = QtGui.QVBoxLayout(self)
#        mainLayout.setContentsMargins(0, 0, 0, 0)
#        mainLayout.setSpacing(0)
        mainLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # output directory
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        label = QtGui.QLabel("Output folder")
        self.outputFolder = QtGui.QLineEdit("rotate")
        self.outputFolder.setFixedWidth(120)
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.outputFolder)
        
        mainLayout.addWidget(row)
        
        # file prefix
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        label = QtGui.QLabel("File prefix")
                
        self.fileprefix = QtGui.QLineEdit(self.fileprefixText)
        self.fileprefix.setFixedWidth(120)
        self.fileprefix.textChanged[str].connect(self.fileprefixChanged)
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.fileprefix)
        
        mainLayout.addWidget(row)
        
        # degrees per rotation
        label = QtGui.QLabel("Degrees per rotation")
        
        degPerRotSpinBox = QtGui.QSpinBox(self)
        degPerRotSpinBox.setMinimum(1)
        degPerRotSpinBox.setMaximum(360)
        degPerRotSpinBox.setValue(self.degreesPerRotation)
        degPerRotSpinBox.valueChanged.connect(self.degPerRotChanged)
        
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.addWidget(label)
        rowLayout.addWidget(degPerRotSpinBox)
        
        mainLayout.addWidget(row)
                
        # overwrite check box
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.overwriteCheck = QtGui.QCheckBox("Overwrite")
        self.overwriteCheck.stateChanged[int].connect(self.overwriteCheckChanged)
        
        rowLayout.addWidget(self.overwriteCheck)
        
        mainLayout.addWidget(row)
        
        # create movie box
        self.createMovieBox = CreateMovieBox(self)
        mainLayout.addWidget(self.createMovieBox)
        
        # start button
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        startRotatorButton = QtGui.QPushButton(QtGui.QIcon(iconPath("loadandsave-icon.svg")), "START")
        startRotatorButton.setToolTip("Start sequencer")
        startRotatorButton.clicked.connect(self.startRotator)
        
        rowLayout.addWidget(startRotatorButton)
        
        mainLayout.addWidget(row)
    
    def startRotator(self):
        """
        Start the rotator.
        
        """
        if self.parent.renderType == "POV":
            settings = self.mainWindow.preferences.povrayForm
            povray = utilities.checkForExe(settings.pathToPovray)
            if not povray:
                utilities.warnExeNotFound(self, "%s (POV-Ray)" % (settings.pathToPovray,))
                return
        
        else:
            povray = ""
        
        self.logger.debug("Running rotator")
        
        # directory
        saveDir = str(self.outputFolder.text())
        saveDir += "-%s" % datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        if os.path.exists(saveDir):
            if self.overwrite:
                shutil.rmtree(saveDir)
            
            else:
                count = 0
                while os.path.exists(saveDir):
                    count += 1
                    saveDir = "%s.%d" % (str(self.outputFolder.text()), count)
        
        os.mkdir(saveDir)
        
        # file name prefix
        fileprefix = os.path.join(saveDir, str(self.fileprefix.text()))
        
        # send to renderer
        status = self.rendererWindow.renderer.rotateAndSaveImage(self.parent.renderType, self.parent.imageFormat, fileprefix, 
                                                                 1, self.degreesPerRotation, povray=povray)
        
        # movie?
        if status:
            print "ERROR: rotate failed"
        
        else:
            # create movie
            if self.createMovieBox.isChecked():
                # show wait cursor
                QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
                
                
                try:
                    saveText = os.path.join(saveDir, "%s%s" % (str(self.fileprefix.text()), "%d"))
                    self.parent.createMovie(saveDir, saveText, self.createMovieBox)
                
                finally:
                    # set cursor to normal
                    QtGui.QApplication.restoreOverrideCursor()
    
    def degPerRotChanged(self, val):
        """
        Degrees per rotation changed.
        
        """
        self.degreesPerRotation = val
    
    def overwriteCheckChanged(self, val):
        """
        Overwrite check changed
        
        """
        if self.overwriteCheck.isChecked():
            self.overwrite = 1
        
        else:
            self.overwrite = 0

    def fileprefixChanged(self, text):
        """
        File prefix has changed
        
        """
        self.fileprefixText = str(text)
