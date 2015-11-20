
"""
Plot dialog.

@author: Chris Scott

"""
import sys
import traceback
import logging

from PySide import QtGui, QtCore
import matplotlib
matplotlib.use("Qt4Agg")
matplotlib.rcParams["backend.qt4"] = "PySide"
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rc

from ..visutils.utilities import iconPath


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
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/office-chart-bar.png")))
        
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
        self.dlgHeight = figHeight * figDpi + 100
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
                
                # store plot args for later use
                self.plotArgs = plotArgs
            
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
        
        # tight layout
        self.fig.tight_layout()
        
        # draw canvas
        self.canvas.draw()
        
        # write to file button
        writeDataButton = QtGui.QPushButton("Write csv")
        writeDataButton.setAutoDefault(False)
        writeDataButton.setDefault(False)
        writeDataButton.clicked.connect(self.writeData)
        writeDataButton.setToolTip("Write csv file containing plot data")
        
        # close button
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.accept)
        
        # button box
        buttonBox = QtGui.QDialogButtonBox()
        buttonBox.addButton(writeDataButton, QtGui.QDialogButtonBox.ActionRole)
        buttonBox.addButton(closeButton, QtGui.QDialogButtonBox.AcceptRole)
        
        # layout
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mplToolbar)
        vbox.addWidget(buttonBox)
        
        self.mainWidget.setLayout(vbox)
    
    def writeData(self):
        """
        Write data to csv file
        
        """
        logger = logging.getLogger(__name__)
        
        if hasattr(self, "plotArgs"):
            showError = False
            plotArgs = list(self.plotArgs)
            if len(plotArgs) == 2:
                try:
                    l0 = len(plotArgs[0]) 
                    l1 = len(plotArgs[1])
                
                except TypeError:
                    showError = True
                
                else:
                    if l0 == l1:
                        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save File', '.')[0]
                        
                        if len(filename):
                            logger.debug("Writing data to csv file: '%s'", filename)
                            
                            #TODO: use numpy method?
                            
                            f = open(filename, "w")
                            for x, y in zip(plotArgs[0], plotArgs[1]):
                                f.write("%r, %r\n" % (x, y))
                            f.close()
                    
                    else:
                        showError = True
            
            else:
                showError = True
            
            if showError:
                self.mainWindow.displayError("Write data not implemented for this type of plot!\n\nFor histograms try selecting 'show as fraction'")
    
    def closeEvent(self, event):
        """
        Override close event.
        
        """
        self.done(0)
