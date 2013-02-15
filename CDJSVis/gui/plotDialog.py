
"""
Plot dialog.

@author: Chris Scott

"""
import os
import sys
import traceback

from PySide import QtGui, QtCore
import numpy as np

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rc
import pylab

from ..visutils.utilities import iconPath

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class PlotDialog(QtGui.QDialog):
    """
    Dialog for displaying a plot.
    
    """
    def __init__(self, parent, mainWindow, dlgTitle, plotType, plotArgs, plotKwargs, settingsDict={}):
        super(PlotDialog, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Plotter - %s" % dlgTitle)
        self.setWindowIcon(QtGui.QIcon(iconPath("Plotter.png")))
        
        # settings
        settings = self.mainWindow.preferences.matplotlibForm
        
        figWidth = settings.figWidth
        figHeight = settings.figHeight
        figDpi = settings.figDpi
        showGrid = settings.showGrid
        fontsize = settings.fontsize
        tickFontsize = settings.tickFontsize
        legendFontsize = settings.legendFontsize
        
        # set dimension of dialog
        self.dlgWidth = figWidth * figDpi + 20
        self.dlgHeight = figHeight * figDpi + 80
        self.resize(self.dlgWidth, self.dlgHeight)
        
        # make size fixed
        self.setMinimumSize(self.dlgWidth, self.dlgHeight)
        self.setMaximumSize(self.dlgWidth, self.dlgHeight)
        
        # plot widget
        self.mainWidget = QtGui.QWidget(self)
        
        # setup figure
        self.fig = Figure((figWidth, figHeight), dpi=figDpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.mainWidget)
        
        # axes
        self.axes = self.fig.add_subplot(111)
        
        # toolbar
        self.mplToolbar = NavigationToolbar(self.canvas, self.mainWidget)
        
        # get plot method
        if hasattr(self.axes, plotType):
            plotMethod = getattr(self.axes, plotType)
        
            try:
                # plot
                plotMethod(*plotArgs, **plotKwargs)
            
            except Exception as e:
                self.mainWindow.displayError("Matplotlib plot failed with following error:\n\n%s" % "".join(traceback.format_exception(*sys.exc_info())))
                self.close()
        
        else:
            self.mainWindow.displayError("Unrecognised matplotlib plot method:\n\n%s" % plotType)
        
        # show grid
        if showGrid:
            self.axes.grid(True)
        
        # text size
        for tick in self.axes.xaxis.get_major_ticks():
            tick.label1.set_fontsize(tickFontsize)
        
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label1.set_fontsize(tickFontsize)
        
        # axis labels (if specified!)
        if "xlabel" in settingsDict:
            self.axes.set_xlabel(settingsDict["xlabel"], fontsize=fontsize)
        
        if "ylabel" in settingsDict:
            self.axes.set_ylabel(settingsDict["ylabel"], fontsize=fontsize)
        
        if "title" in settingsDict:
            self.axes.set_title(settingsDict["title"], fontsize=fontsize)
        
        # draw canvas
        self.canvas.draw()
        
        # layout
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mplToolbar)
        
        self.mainWidget.setLayout(vbox)
        
    def closeEvent(self, event):
        """
        Override close event.
        
        """
        self.done(0)

