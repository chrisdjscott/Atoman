# -*- coding: utf-8 -*-

"""
The output tab for the main toolbar

@author: Chris Scott

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
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
from PIL import Image

from ..visutils import utilities
from ..visutils import threading_vis
from ..visutils.utilities import iconPath
from . import genericForm
from ..plotting import rdf
from ..algebra import _vectors as vectors_c
from ..plotting import plotDialog
from . import utils
import six
from six.moves import range


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
        
        self.logger = logging.getLogger(__name__ + ".ScalarsHistogramOptionsForm")
    
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
        for scalarsID in list(self.currentPlots.keys()):
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
            for scalarsName, scalarsArray in six.iteritems(filterList.filterer.scalarsDict):
                # make unique id
                scalarsID = "%s (%s)" % (scalarsName, filterListID)
                
                # add
                self.addScalarPlotOptions(scalarsID, scalarsName, scalarsArray)
            
            # loop over scalars in latticeScalarsDict on filterer
            latticeScalarKeys = list(filterList.pipelinePage.inputState.scalarsDict.keys())
            for key in latticeScalarKeys:
                if key in filterList.filterer.latticeScalarsDict:
                    scalarsArray = filterList.filterer.latticeScalarsDict[key]
                    self.logger.debug("Using Filterer scalars for '%s'", key)
                
                else:
                    scalarsArray = filterList.pipelinePage.inputState.scalarsDict[key]
                    self.logger.debug("Using Lattice scalars for '%s'", key)
            
                # make unique id
                scalarsName = key
                scalarsID = "%s (%s)" % (scalarsName, filterListID)
                
                # add
                self.addScalarPlotOptions(scalarsID, scalarsName, scalarsArray)
            
            # add cluster size/volume distributions too
            if len(filterList.filterer.clusterList):
                clusterSizes = []
                clusterVolumes = []
                haveVolumes = True
                for c in filterList.filterer.clusterList:
                    # cluster sizes
                    clusterSizes.append(len(c))
                    
                    # cluster volumes
                    vol = c.getVolume()
                    if vol is not None:
                        clusterVolumes.append(vol)
                    
                    else:
                        haveVolumes = False
                
                # plot cluster size
                scalarsID = "Cluster size (%s)" % filterListID
                self.addScalarPlotOptions(scalarsID, "Cluster size", np.asarray(clusterSizes, dtype=np.float64))
                
                if haveVolumes:
                    # plot volumes
                    scalarsID = "Cluster volume (%s)" % filterListID
                    self.addScalarPlotOptions(scalarsID, "Cluster volume", np.asarray(clusterVolumes, dtype=np.float64))
        
        # hide if no plots, otherwise show
        if self.numScalarsPlots > 0:
            self.show()
        else:
            self.hide()


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
        self.logger = logging.getLogger(__name__ + ".PlotTab")
    
    def newRow(self):
        """
        New row
        
        """
        row = genericForm.FormRow()
        self.layout.addWidget(row)
        
        return row


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
        self.logger = logging.getLogger(__name__ + ".GenericHistogramPlotForm")
        
        # scalar stats
        self.scalarMin = np.min(scalarsArray)
        self.scalarMax = np.max(scalarsArray)
        self.scalarMean = np.mean(scalarsArray)
        self.scalarSTD = np.std(scalarsArray)
        self.scalarSE = self.scalarSTD / math.sqrt(len(scalarsArray))
        
        # default
        self.useNumBins = True
        self.numBins = 10
        self.binWidth = 1.0
        self.showAsFraction = False
        
        # stats labels
        row = self.newRow()
        row.addWidget(QtGui.QLabel("Min: %f" % self.scalarMin))
        row = self.newRow()
        row.addWidget(QtGui.QLabel("Max: %f" % self.scalarMax))
        row = self.newRow()
        row.addWidget(QtGui.QLabel("Mean: %f" % self.scalarMean))
        row = self.newRow()
        row.addWidget(QtGui.QLabel("STD: %f; SE: %f" % (self.scalarSTD, self.scalarSE)))
        
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
        
        binCombo.setCurrentIndex(1)
        
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
        plotButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/office-chart-bar.png")), "Plot")
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
        
        if self.scalarMax == self.scalarMin:
            self.logger.error("Max val == min val; not plotting histogram")
            return
        
        scalars = self.scalarsArray
        minVal = self.scalarMin
        maxVal = self.scalarMax
        
        # number of bins
        if self.useNumBins:
            numBins = self.numBins
        else:
            binWidth = self.binWidth
            
            # min
            tmp = math.floor(minVal / binWidth)
            assert tmp * binWidth <= minVal and (tmp + 1) * binWidth > minVal
            minVal = tmp * binWidth
            
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
                                        (binEdges[:-1], fracHist), {"width": binWidth}, settingsDict=settingsDict)
        
        else:
            # y label
            settingsDict["ylabel"] = "Number"
            
            # histogram plot
            dlg = plotDialog.PlotDialog(self, self.parent.mainWindow, "%s histogram" % self.scalarsID, "hist",
                                        (scalars, numBins), {"range": (minVal, maxVal)}, settingsDict=settingsDict)
        
        # show dialog
        dlg.show()


class RDFForm(genericForm.GenericForm):
    """
    RDF output form.
    
    """
    def __init__(self, parent, mainWindow):
        super(RDFForm, self).__init__(parent, 0, "RDF plot options")
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.rendererWindow = self.parent.rendererWindow
        self.logger = logging.getLogger(__name__ + ".RDFForm")
        
        # defaults
        self.spec1 = "ALL"
        self.spec2 = "ALL"
        self.binMin = 2.0
        self.binMax = 10.0
        self.binWidth = 0.1
        
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
        binMinSpin.setSingleStep(1.0)
        binMinSpin.setValue(self.binMin)
        binMinSpin.valueChanged.connect(self.binMinChanged)
        row.addWidget(binMinSpin)
        
        label = QtGui.QLabel(" - ")
        row.addWidget(label)
        
        binMaxSpin = QtGui.QDoubleSpinBox()
        binMaxSpin.setMinimum(0.0)
        binMaxSpin.setMaximum(500.0)
        binMaxSpin.setSingleStep(1.0)
        binMaxSpin.setValue(self.binMax)
        binMaxSpin.valueChanged.connect(self.binMaxChanged)
        row.addWidget(binMaxSpin)
        
        # num bins
        label = QtGui.QLabel("Bin width:")
        row = self.newRow()
        row.addWidget(label)
        
        binWidthSpin = QtGui.QDoubleSpinBox()
        binWidthSpin.setMinimum(0.01)
        binWidthSpin.setMaximum(1.00)
        binWidthSpin.setSingleStep(0.1)
        binWidthSpin.setValue(self.binWidth)
        binWidthSpin.valueChanged.connect(self.binWidthChanged)
        row.addWidget(binWidthSpin)
        
        # plot button
        plotButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/office-chart-bar.png")), "Plot")
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
            msg = "The maximum radius you have requested is greater than half the box length"
            msg += " in the %s direction(s)!" % ", ".join(warnDims)
            self.mainWindow.displayError(msg)
            return
        
        # first gather vis atoms
        visibleAtoms = self.rendererWindow.gatherVisibleAtoms()
        if not len(visibleAtoms):
            self.mainWindow.displayWarning("No visible atoms: cannot calculate RDF")
            return
        
        # then determine species
        if self.spec1 == "ALL":
            spec1Index = -1
        else:
            spec1Index = inputLattice.getSpecieIndex(self.spec1)
        
        if self.spec2 == "ALL":
            spec2Index = -1
        else:
            spec2Index = inputLattice.getSpecieIndex(self.spec2)
        
        # rdf calulator
        rdfCalculator = rdf.RDFCalculator()
        
        # show progress dialog
        progDiag = utils.showProgressDialog("Calculating RDF", "Calculating RDF...", self)
        try:
            # then calculate
            xn, rdfArray = rdfCalculator.calculateRDF(visibleAtoms, inputLattice, self.binMin, self.binMax,
                                                      self.binWidth, spec1Index, spec2Index)
        
        finally:
            utils.cancelProgressDialog(progDiag)
        
        # prepare to plot
        settingsDict = {}
        settingsDict["title"] = "Radial distribution function"
        settingsDict["xlabel"] = "Bond length (Angstroms)"
        settingsDict["ylabel"] = "g(r) (%s - %s)" % (self.spec1, self.spec2)
        
        # show plot dialog
        dialog = plotDialog.PlotDialog(self, self.mainWindow, "Radial distribution function ",
                                       "plot", (xn, rdfArray), {"linewidth": 2, "label": None},
                                       settingsDict=settingsDict)
        dialog.show()
    
    def binWidthChanged(self, val):
        """
        Num bins changed.
        
        """
        self.binWidth = val
    
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
        saveFileButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/document-save.png")), "")
        saveFileButton.setToolTip("Save to file")
        saveFileButton.clicked.connect(self.saveToFile)
        
        row.addWidget(label)
        row.addWidget(self.outputFileName)
        row.addWidget(saveFileButton)
        
        # dialog
        row = fileNameGroup.newRow()
        
        saveFileDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('oxygen/document-open.png')), "Save to file")
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
        
        # gather vis atoms if required
        if self.writeFullLattice:
            visibleAtoms = None
        else:
            visibleAtoms = self.rendererWindow.gatherVisibleAtoms()
        
        # write Lattice
        lattice.writeLattice(filename, visibleAtoms=visibleAtoms)
    
    def saveToFileDialog(self):
        """
        Open dialog.
        
        """
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save File', '.')[0]
        
        if len(filename):
            self.outputFileName.setText(str(filename))
            self.saveToFile()
    
    def outputTypeChanged(self, fileType):
        """
        Output type changed.
        
        """
        self.outputFileType = str(fileType)


class ImageTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(ImageTab, self).__init__(parent)
        
        self.logger = logging.getLogger(__name__ + ".ImageTab")
        
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
        
        self.POVButton = QtGui.QPushButton(QtGui.QIcon(iconPath("other/pov-icon.png")), "POV-Ray")
        self.POVButton.setCheckable(1)
        self.POVButton.setChecked(0)
        
        self.VTKButton = QtGui.QPushButton(QtGui.QIcon(iconPath("other/vtk-icon.png")), "VTK")
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
        logger = logging.getLogger(__name__ + ".MovieGenerator")
        method = getattr(logger, level, None)
        if method is not None:
            method(message)
    
    def createMovie(self, saveDir, inputText, createMovieBox, prefix=None):
        """
        Create movie.
        
        """
        settings = self.mainWindow.preferences.ffmpegForm
        ffmpeg = utilities.checkForExe(settings.pathToFFmpeg)
        if not ffmpeg:
            utilities.warnExeNotFound(self, "%s (FFmpeg)" % (settings.pathToFFmpeg,))
            return 2
        
        # settings
        settings = self.mainWindow.preferences.ffmpegForm
        framerate = createMovieBox.framerate
        bitrate = settings.bitrate
        if prefix is None:
            outputprefix = createMovieBox.prefix
        else:
            outputprefix = prefix
        outputprefix = os.path.join(saveDir, outputprefix)
        outputsuffix = createMovieBox.suffix
        
        self.logger.info("Creating movie file: %s.%s", outputprefix, outputsuffix)
        
        # movie generator object
        generator = MovieGenerator()
        generator.log.connect(self.createMovieLogger)
        generator.allDone.connect(generator.deleteLater)
         
        # runnable for sending to thread pool
        runnable = threading_vis.GenericRunnable(generator, args=(ffmpeg, framerate, inputText, self.imageFormat,
                                                                  bitrate, outputprefix, outputsuffix))
        runnable.setAutoDelete(False)
         
        # add to thread pool
        QtCore.QThreadPool.globalInstance().start(runnable)
        
#         generator.run(ffmpeg, framerate, inputText, self.imageFormat, bitrate, outputprefix, outputsuffix)


class MovieGenerator(QtCore.QObject):
    """
    Call ffmpeg to generate a movie
    
    """
    log = QtCore.Signal(str, str)
    allDone = QtCore.Signal()
    
    def __init__(self):
        super(MovieGenerator, self).__init__()
    
    def run(self, ffmpeg, framerate, saveText, imageFormat, bitrate, outputPrefix, outputSuffix):
        """
        Create movie
        
        """
        ffmpegTime = time.time()
        try:
            if outputSuffix == "mp4":
                # determine image size
                firstFile = "%s.%s" % (saveText, imageFormat)
                firstFile = firstFile % 0
                self.log.emit("debug", "Checking first file size: '%s'" % firstFile)
                
                im = Image.open(firstFile)
                width, height = im.size
                self.log.emit("debug", "Image size: %s x %s" % (width, height))
                
                # h264 requires width and height be divisible by 2
                newWidth = width - 1 if width % 2 else width
                newHeight = height - 1 if height % 2 else height
                if newWidth != width:
                    self.log.emit("debug", "Resizing image width: %d -> %d" % (width, newWidth))
                if newHeight != height:
                    self.log.emit("debug", "Resizing image height: %d -> %d" % (height, newHeight))
                
                # construct command; scale if required
                if newWidth == width and newHeight == height:
                    # no scaling required
                    command = "'%s' -r %d -y -i %s.%s -c:v h264 -r %d -b:v %dk '%s.%s'" % (ffmpeg, framerate, saveText,
                                                                                           imageFormat, 25, bitrate,
                                                                                           outputPrefix, outputSuffix)
                
                else:
                    # scaling required
                    command = "'%s' -r %d -y -i %s.%s -vf scale=%d:%d -c:v h264 -r %d -b:v %dk '%s.%s'" % (ffmpeg, framerate, saveText,
                                                                                                           imageFormat, newWidth, newHeight,
                                                                                                           25, bitrate, outputPrefix,
                                                                                                           outputSuffix)
                
                # run command
                self.log.emit("debug", 'Command: "%s"' % command)
                process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, stderr = process.communicate()
                status = process.poll()
            
            else:
                command = "'%s' -r %d -y -i %s.%s -r %d -b:v %dk '%s.%s'" % (ffmpeg, framerate, saveText,
                                                                             imageFormat, 25, bitrate,
                                                                             outputPrefix, outputSuffix)
                
                self.log.emit("debug", 'Command: "%s"' % command)
                
                process = subprocess.Popen(command, shell=True, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 
                output, stderr = process.communicate()
                status = process.poll()
            
            if status:
                self.log.emit("error", "FFmpeg failed (%d)" % status)
                self.log.emit("error", output.decode('utf-8'))
                self.log.emit("error", stderr.decode('utf-8'))
        
        finally:
            ffmpegTime = time.time() - ffmpegTime
            self.log.emit("debug", "FFmpeg time taken: %f s" % ffmpegTime)
            self.allDone.emit()


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
        saveImageButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/document-save.png")), "")
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
        
        saveImageDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('oxygen/document-open.png')), "Save image")
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
            print("SAVE IMAGE FAILED")
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
        self.suffix = "mp4"
        
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
        containerCombo.addItem("mp4")
        containerCombo.addItem("flv")
        containerCombo.addItem("mpg")
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


class ImageSequenceTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(ImageSequenceTab, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.width = width
        self.rendererWindow = self.parent.rendererWindow
        
        self.logger = logging.getLogger(__name__ + ".ImageSequenceTab")
        
        # initial values
        self.numberFormats = ["%04d", "%d"]
        self.numberFormat = self.numberFormats[0]
        self.minIndex = 0
        self.maxIndex = -1
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
        
        resetPrefixButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/edit-find.png")), "")
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
        self.numberFormatCombo.addItems(self.numberFormats)
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
        self.maxIndexSpinBox.setMinimum(-1)
        self.maxIndexSpinBox.setMaximum(99999)
        self.maxIndexSpinBox.setValue(self.maxIndex)
        self.maxIndexSpinBox.valueChanged[int].connect(self.maxIndexChanged)
        self.maxIndexSpinBox.setToolTip("The max index (inclusive; if less than min index do all we can find)")
        
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
#         mainLayout.addWidget(row)
        
        # rotate at end
#         row = QtGui.QWidget(self)
#         rowLayout = QtGui.QHBoxLayout(row)
#         rowLayout.setContentsMargins(0, 0, 0, 0)
#         rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        rowLayout.addStretch()
        self.rotateAfterCheck = QtGui.QCheckBox("Rotate at end")
        self.rotateAfterCheck.stateChanged[int].connect(self.rotateAfterCheckChanged)
        rowLayout.addWidget(self.rotateAfterCheck)
        mainLayout.addWidget(row)
        
        # link to other renderer combo
        self.linkedRenderWindowIndex = None
        self.linkedRendererCombo = QtGui.QComboBox()
        self.linkedRendererCombo.currentIndexChanged[str].connect(self.linkedRendererChanged)
        row = QtGui.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setAlignment(QtCore.Qt.AlignHCenter)
        row.addWidget(QtGui.QLabel("Linked render window:"))
        row.addWidget(self.linkedRendererCombo)
        mainLayout.addLayout(row)
        
        # populate
        self.linkedRendererCombo.addItem("<Off>")
        myrwi = self.rendererWindow.rendererIndex
        rws = [str(rw.rendererIndex) for rw in self.mainWindow.rendererWindows if rw.rendererIndex != myrwi]
        self.linkedRendererCombo.addItems(rws)
        
        # create movie box
        self.createMovieBox = CreateMovieBox(self)
        mainLayout.addWidget(self.createMovieBox)
        
        # start button
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        startSequencerButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/go-last.png")), "START")
        startSequencerButton.setStatusTip("Start sequencer")
        startSequencerButton.setToolTip("Start sequencer")
        startSequencerButton.clicked.connect(self.startSequencer)
        
        rowLayout.addWidget(startSequencerButton)
        
        mainLayout.addWidget(row)
    
    def refreshLinkedRenderers(self):
        """
        Refresh the linked renderers combo
        
        """
        self.logger.debug("Refreshing linked renderer options")
        
        # clear
        self.linkedRendererCombo.clear()
        self.linkedRenderWindowIndex = None
        
        # populate
        self.linkedRendererCombo.addItem("<Off>")
        
        myrwi = self.rendererWindow.rendererIndex
        rws = [str(rw.rendererIndex) for rw in self.mainWindow.rendererWindows if rw.rendererIndex != myrwi]
        assert len(self.mainWindow.rendererWindows) == len(rws) + 1
        
        self.linkedRendererCombo.addItems(rws)
    
    def linkedRendererChanged(self, currentText):
        """
        Linked renderer changed
        
        """
        if self.linkedRendererCombo.currentIndex() > 0:
            index = int(currentText)
            
            rw2 = None
            for rwIndex, rw in enumerate(self.mainWindow.rendererWindows):
                if rw.rendererIndex == index:
                    rw2 = rw
                    break
            
            if rw2 is None:
                self.logger.error("Cannot find linked render window (%d)", index)
                self.linkedRenderWindowIndex = None
                return
            
            # do some checks
            if rw2.currentPipelineIndex == self.rendererWindow.currentPipelineIndex:
                if rw2.vtkRenWinInteract.size().height() == self.rendererWindow.vtkRenWinInteract.size().height():
                    self.linkedRenderWindowIndex = rwIndex
                    return rw2
                
                else:
                    self.logger.error("Cannot select linked render window %d; heights do not match", index)
                    self.linkedRendererCombo.setCurrentIndex(0)
            
            else:
                self.logger.error("Cannote select linked render window %d; cannot handle different pipelines yet (ask me...)", index)
                self.linkedRendererCombo.setCurrentIndex(0)
        
        else:
            self.linkedRenderWindowIndex = None
    
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
        
        guess = self.guessFilePrefix(filename)
        
        self.fileprefix.setText(guess)
    
    def guessFilePrefix(self, filename):
        """
        Guess the file prefix
        
        """
        count = 0
        lim = None
        for i in range(len(filename)):
            if filename[i] == ".":
                break
            
            try:
                int(filename[i])
                
                if lim is None:
                    lim = count
            
            except ValueError:
                lim = None
            
            count += 1
        
        if lim is None:
            array = os.path.splitext(filename)
            
            if array[1] == '.gz' or array[1] == '.bz2':
                array = os.path.splitext(array[0])
            
            filename = array[0]
        
        else:
            filename = filename[:lim]
        
        return filename
    
    def startSequencer(self):
        """
        Start the sequencer
        
        """
        self.runSequencer()
        
    def runSequencer(self):
        """
        Run the sequencer
        
        """
        self.logger.info("Running sequencer")
        
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
        
        # check this is not a generated system
        if pipelinePage.fileFormat is None:
            self.logger.error("Cannot sequence a generated file")
            self.mainWindow.displayError("Cannot sequence a generated file")
            return
        
        # formatted string
        fileText = "%s%s%s" % (str(self.fileprefix.text()), self.numberFormat, pipelinePage.extension)
        
        # check abspath (for sftp)
        abspath = pipelinePage.abspath
        sftpBrowser = None
        if pipelinePage.fromSFTP:
            self.logger.debug("Sequencing SFTP file: '%s'", abspath)
            array = abspath.split(":")
            sftpHost = array[0]
            # handle case where ":"'s are in the file path
            sftpFile = ":".join(array[1:])
            self.logger.debug("Host: '%s'; path: '%s'", sftpHost, sftpFile)
            
            sysDiag = self.mainWindow.systemsDialog
            sftpDlg = sysDiag.load_system_form.sftp_browser
            match = False
            for i in range(sftpDlg.stackedWidget.count()):
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
        if self.maxIndex > self.minIndex:
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
            
            maxIndex = self.maxIndex
        
        else:
            # find greatest file
            self.logger.info("Auto-detecting last sequencer file")
            
            lastIndex = self.minIndex
            lastFile = fileText % lastIndex
            
            if sftpBrowser is None:
                def _checkForLastFile(fn):
                    return utilities.checkForFile(fn)
            
            else:
                def _checkForLastFile(fn):
                    rp = os.path.join(os.path.dirname(sftpFile), lastFile)
                    return bool(sftpBrowser.checkPathExists(rp)) or bool(sftpBrowser.checkPathExists(rp+".gz")) or bool(sftpBrowser.checkPathExists(rp+".bz2"))
            
            while _checkForLastFile(lastFile):
                lastIndex += 1
                lastFile = fileText % lastIndex
            
            lastIndex -= 1
            lastFile = fileText % lastIndex
            maxIndex = lastIndex
            
            self.logger.info("Last file detected as: '%s'", lastFile)
        
        # store current input state
        origInput = copy.deepcopy(self.rendererWindow.getCurrentInputState())
        
        # pipeline index
        pipelineIndex = self.rendererWindow.currentPipelineIndex
        
        # systems dialog
        systemsDialog = self.mainWindow.systemsDialog
        loadPage = systemsDialog.load_system_form
        
        # reader 
        readerForm = loadPage.readerForm
        reader = readerForm.latticeReader
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
        
        # check if linked
        rw2 = None
        if self.linkedRenderWindowIndex is not None:
            # make sure still ok to use this index
            rw2 = self.linkedRendererChanged(self.linkedRendererCombo.currentText())
        
        if rw2 is not None:
            saveText2 = saveText + "_2"
        
        # progress dialog
        NSteps = int((maxIndex - self.minIndex) / self.interval) + 1
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
            for i in range(self.minIndex, maxIndex + self.interval, self.interval):
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
                status, state = reader.readFile(currentFile, pipelinePage.fileFormat, rouletteIndex=i-1, linkedLattice=pipelinePage.linkedLattice)
                if status:
                    self.logger.error("Sequencer read file failed with status: %d" % status)
                    break
                
                # eliminate flicker across PBCs
                if self.flickerFlag:
                    self.eliminateFlicker(state, previousPos, pipelinePage)
                    previousPos = copy.deepcopy(state.pos)
                
                # set PBCs the same
                state.PBC[:] = origInput.PBC[:]
                
                # attempt to read any scalars/vectors files
                for vectorsName, vectorsFile in six.iteritems(origInput.vectorsFiles):
                    self.logger.debug("Sequencer checking vectors file: '%s'", vectorsFile)
                    
                    vdn, vbn = os.path.split(vectorsFile)
                    
                    # guess prefix
                    guessvfn = self.guessFilePrefix(vbn)
                    
                    if guessvfn != vbn:
                        ext = "." + vbn.split(".")[-1]
                        if ext == vbn:
                            ext = ""
                        
                        vfn = "%s%s%s" % (guessvfn, self.numberFormat, ext)
                        if len(vdn):
                            vfn = os.path.join(vdn, vfn)
                        
                        vfn = vfn % i
                        
                        self.logger.debug("Looking for vectors file: '%s' (%s)", vfn, os.path.exists(vfn))
                        if os.path.exists(vfn):
                            # read vectors file
                            ok = True
                            with open(vfn) as f:
                                vectors = []
                                try:
                                    for line in f:
                                        array = line.split()
                                        array[0] = float(array[0])
                                        array[1] = float(array[1])
                                        array[2] = float(array[2])
                                        
                                        vectors.append(array)
                                
                                except:
                                    self.logger.error("Error reading vector file")
                                    ok = False
                            
                            if ok and len(vectors) != state.NAtoms:
                                self.logger.error("The vector data is the wrong length")
                                ok = False
                            
                            if ok:
                                # convert to numpy array
                                vectors = np.asarray(vectors, dtype=np.float64)
                                assert vectors.shape[0] == state.NAtoms and vectors.shape[1] == 3
                                
                                state.vectorsDict[vectorsName] = vectors
                                state.vectorsFiles[vectorsName] = vfn
                                
                                self.logger.debug("Added vectors data (%s) to sequencer lattice", vectorsName)
                
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
                
                saveName = saveText % count
                self.logger.info("  Saving image: '%s'", saveName)
                
                # now save image
                filename = self.rendererWindow.renderer.saveImage(self.parent.renderType, self.parent.imageFormat, saveName, 1, povray=povray)
                
                # linked image
                if rw2 is not None:
                    saveName2 = saveText2 % count
                    self.logger.info("  Saving linked image: '%s'", saveName2)
                    
                    filename2 = rw2.renderer.saveImage(self.parent.renderType, self.parent.imageFormat, saveName2, 1, povray=povray)
                    
                    # merge the files
                    mergeFn = os.path.join(saveDir, "merge%d.%s" % (i, self.parent.imageFormat))
                    self.logger.debug("Merging the files together: '%s'", mergeFn)
                    
                    # read images
                    im1 = Image.open(filename)
                    im2 = Image.open(filename2)
                    
                    assert im1.size[1] == im2.size[1], "Image sizes do not match: %r != %r" % (im1.size, im2.size)
                    
                    # new empty image
                    newSize = (im1.size[0] + im2.size[0], im1.size[1])
                    newIm = Image.new('RGB', newSize)
                    
                    # paste images
                    newIm.paste(im1, (0, 0))
                    newIm.paste(im2, (im1.size[0], 0))
                    
                    # save
                    newIm.save(mergeFn)
                
                # increment output counter
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
            
            # create movie
            if not status and self.createMovieBox.isChecked():
                # show wait cursor
#                 QtGui.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
                
                try:
                    self.parent.createMovie(saveDir, saveText, self.createMovieBox)
                    
                    if rw2 is not None:
                        self.parent.createMovie(saveDir, os.path.join(saveDir, "merge%d"), self.createMovieBox, prefix="merged")
                
                finally:
                    # set cursor to normal
#                     QtGui.QApplication.restoreOverrideCursor()
                    pass
            
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
        
        text = "%s%s%s" % (self.fileprefix.text(), self.numberFormat, ext)
        
        foundFormat = False
        testfn = text % self.minIndex
        if not (os.path.isfile(testfn) or os.path.isfile(testfn+'.gz') or os.path.isfile(testfn+'.bz2')):
            self.logger.debug("First file does not exist; checking other number formats")
            for i, nfmt in enumerate(self.numberFormats):
                if nfmt == self.numberFormat:
                    continue
                
                testText = "%s%s%s" % (self.fileprefix.text(), nfmt, ext)
                testfn = testText % self.minIndex
                if os.path.isfile(testfn) or os.path.isfile(testfn+'.gz') or os.path.isfile(testfn+'.bz2'):
                    foundFormat = True
                    break
            
            if foundFormat:
                self.logger.debug("Found suitable number format: '%s'", nfmt)
                self.numberFormatCombo.setCurrentIndex(i)
        
        if not foundFormat:
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
        
        startRotatorButton = QtGui.QPushButton(QtGui.QIcon(iconPath("oxygen/go-last.png")), "START")
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
            print("ERROR: rotate failed")
        
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
