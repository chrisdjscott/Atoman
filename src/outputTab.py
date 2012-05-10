
"""
The output tab for the main toolbar

@author: Chris Scott

"""

import os
import sys
import shutil
import subprocess

from PyQt4 import QtGui, QtCore

import utilities
from utilities import iconPath
from genericForm import GenericForm
import resources
import globals



################################################################################
class OutputTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(OutputTab, self).__init__(parent)
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.width = width
        
        # layout
        outputTabLayout = QtGui.QVBoxLayout(self)
        outputTabLayout.setContentsMargins(0, 0, 0, 0)
        outputTabLayout.setSpacing(0)
        outputTabLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # add tab bar
        self.outputTypeTabBar = QtGui.QTabWidget(self)
        self.outputTypeTabBar.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.connect(self.outputTypeTabBar, QtCore.SIGNAL('currentChanged(int)'), self.outputTypeTabBarChanged)
        
        # add tabs to tab bar
        imageTabWidget = QtGui.QWidget()
        imageTabLayout = QtGui.QVBoxLayout(imageTabWidget)
        imageTabLayout.setContentsMargins(0, 0, 0, 0)
        
        self.imageTab = ImageTab(self, self.mainWindow, self.width)
        imageTabLayout.addWidget(self.imageTab)
        
        self.outputTypeTabBar.addTab(imageTabWidget, "Image")
        
        # add tab bar to layout
        outputTabLayout.addWidget(self.outputTypeTabBar)
        
        
    def outputTypeTabBarChanged(self):
        pass


################################################################################
class ImageTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(ImageTab, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.width = width
        
        # initial values
        self.renderType = "VTK"
        self.imageFormat = "jpg"
        
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
        
        self.connect(renderTypeButtonGroup, QtCore.SIGNAL('buttonClicked(int)'), self.setRenderType)
        
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
        
        self.connect(imageFormatButtonGroup, QtCore.SIGNAL('buttonClicked(int)'), self.setImageFormat)
        
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
        
        imageTabLayout.addWidget(group)
        
        # tab bar for different types of image output
        self.imageTabBar = QtGui.QTabWidget(self)
        self.imageTabBar.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.connect(self.imageTabBar, QtCore.SIGNAL('currentChanged(int)'), self.imageTabBarChanged)
        
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
        
        imageTabLayout.addWidget(self.imageTabBar)
        
        # check ffmpeg/povray installed
        self.ffmpeg = utilities.checkForExe("ffmpeg")
        self.povray = utilities.checkForExe("povray")
        
        if self.ffmpeg:
            self.mainWindow.console.write("'ffmpeg' executable located at: %s" % (self.ffmpeg,))
        
        if self.povray:
            self.mainWindow.console.write("'povray' executable located at: %s" % (self.povray,))
    
    def imageTabBarChanged(self, val):
        """
        
        
        """
        pass
    
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
            if not self.povray:
                self.POVButton.setChecked(0)
                self.VTKButton.setChecked(1)
                utilities.warnExeNotFound(self, "povray")
            
            else:
                self.renderType = "POV"
                self.imageFormat = "png"
                self.PNGCheck.setChecked(1)
        
        elif self.VTKButton.isChecked():
            self.renderType = "VTK"
            self.imageFormat = "jpg"
            self.JPEGCheck.setChecked(1)
        

################################################################################
class SingleImageTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(SingleImageTab, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.width = width
        
        # initial values
        self.overwriteImage = 0
        self.openImage = 0
        
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
        saveImageButton.setStatusTip("Save image")
        self.connect(saveImageButton, QtCore.SIGNAL('clicked()'), self.saveSingleImage)
        
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
        saveImageDialogButton.setStatusTip("Save image")
        saveImageDialogButton.setCheckable(0)
        saveImageDialogButton.setFixedWidth(150)
        self.connect(saveImageDialogButton, QtCore.SIGNAL('clicked()'), self.saveSingleImageDialog)
        
        rowLayout.addWidget(saveImageDialogButton)
        
        mainLayout.addWidget(row)
        
        # options
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.overwriteCheck = QtGui.QCheckBox("Overwrite")
        self.connect(self.overwriteCheck, QtCore.SIGNAL('stateChanged(int)'), self.overwriteCheckChanged)
        
        self.openImageCheck = QtGui.QCheckBox("Open image")
        self.connect(self.openImageCheck, QtCore.SIGNAL('stateChanged(int)'), self.openImageCheckChanged)
        
        rowLayout.addWidget(self.overwriteCheck)
        rowLayout.addWidget(self.openImageCheck)
        
        mainLayout.addWidget(row)
        
    def saveSingleImageDialog(self):
        """
        Open dialog to get save file name
        
        """
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save File', '.')
        
        if len(filename):
            self.imageFileName.setText(str(filename))
            self.saveSingleImage()
    
    def saveSingleImage(self):
        """
        Screen capture.
        
        """
        filename = self.imageFileName.text()
        
        if not len(filename):
            return
        
        filename = self.mainWindow.renderer.saveImage(self.parent.renderType, self.parent.imageFormat, 
                                                      filename, self.overwriteImage)
    
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
class ImageSequenceTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(ImageSequenceTab, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.width = width
        
        # initial values
        self.numberFormat = "%04d"
        self.minIndex = 0
        self.maxIndex = 10
        self.interval = 1
        self.fileprefixText = "guess"
        self.overwrite = 0
        self.createMovie = 0
        
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
        self.connect(self.fileprefix, QtCore.SIGNAL('textChanged(QString)'), self.fileprefixChanged)
        
        resetPrefixButton = QtGui.QPushButton(QtGui.QIcon(iconPath("edit-paste.svg")), "")
        resetPrefixButton.setStatusTip("Set prefix to input file")
        self.connect(resetPrefixButton, QtCore.SIGNAL("clicked()"), self.resetPrefix)
        
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
        self.connect(self.numberFormatCombo, QtCore.SIGNAL("currentIndexChanged(QString)"), self.numberFormatChanged)
        
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
        self.connect(self.minIndexSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.minIndexChanged)
        
        label = QtGui.QLabel("to")
        
        self.maxIndexSpinBox = QtGui.QSpinBox()
        self.maxIndexSpinBox.setMinimum(1)
        self.maxIndexSpinBox.setMaximum(99999)
        self.maxIndexSpinBox.setValue(self.maxIndex)
        self.connect(self.maxIndexSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.maxIndexChanged)
        
        label2 = QtGui.QLabel("by")
        
        self.intervalSpinBox = QtGui.QSpinBox()
        self.intervalSpinBox.setMinimum(1)
        self.intervalSpinBox.setMaximum(99999)
        self.intervalSpinBox.setValue(self.interval)
        self.connect(self.intervalSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.intervalChanged)
        
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
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.overwriteCheck = QtGui.QCheckBox("Overwrite")
        self.connect(self.overwriteCheck, QtCore.SIGNAL('stateChanged(int)'), self.overwriteCheckChanged)
        
        rowLayout.addWidget(self.overwriteCheck)
        
        mainLayout.addWidget(row)
        
        # create movie check box
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        self.createMovieCheck = QtGui.QCheckBox("Create movie")
        self.connect(self.createMovieCheck, QtCore.SIGNAL('stateChanged(int)'), self.createMovieCheckChanged)
        
        rowLayout.addWidget(self.createMovieCheck)
        
        mainLayout.addWidget(row)
        
        # start button
        row = QtGui.QWidget(self)
        rowLayout = QtGui.QHBoxLayout(row)
#        rowLayout.setSpacing(0)
        rowLayout.setContentsMargins(0, 0, 0, 0)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)
        
        startSequencerButton = QtGui.QPushButton(QtGui.QIcon(iconPath("loadandsave-icon.svg")), "START")
        self.connect(startSequencerButton, QtCore.SIGNAL('clicked()'), self.startSequencer)
        
        rowLayout.addWidget(startSequencerButton)
        
        mainLayout.addWidget(row)
        
    def createMovieCheckChanged(self, val):
        """
        Create movie?
        
        """
        if self.createMovieCheck.isChecked():
            if not self.parent.ffmpeg:
                utilities.warnExeNotFound(self.parent, "ffmpeg")
                self.createMovieCheck.setCheckState(0)
                return
            
            self.createMovie = 1
        
        else:
            self.createMovie = 0
    
    def resetPrefix(self):
        """
        Reset the prefix to the one from 
        the input page
        
        """
        filename = self.mainWindow.inputFile
        
        count = 0
        for i in xrange(len(filename)):
            if filename[i] == ".":
                break
            
            error = 0
            try:
                int(filename[i])
            except ValueError:
                error = 1
            
            if not error:
                break
            
            count += 1
        
        self.fileprefix.setText(filename[:count])
    
    def startSequencer(self):
        """
        Start the sequencer
        
        """
        self.runSequencer()
        
    def runSequencer(self):
        """
        Run the sequencer
        
        """
        self.setFirstFileLabel()
        
        # check first file exists
        firstFileExists = utilities.checkForFile(str(self.firstFileLabel.text()))
        if not firstFileExists:
            self.warnFirstFileNotPresent(str(self.firstFileLabel.text()))
            return
        
        # formatted string
        fileText = "%s%s.%s" % (str(self.fileprefix.text()), self.numberFormat, self.mainWindow.fileExtension)
        
        log = self.mainWindow.console.write
        log("Running sequencer", 0, 0)
        
        # directory
        saveDir = str(self.outputFolder.text())
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
        
        # loop over files
        for i in xrange(self.minIndex, self.maxIndex, self.interval):
            currentFile = fileText % (i,)
            log("Current file: %s" % (currentFile,), 0, 1)
            
            # first open the file
            tmpname = self.mainWindow.openFile(currentFile, "input")
            
            if tmpname is None:
                print "ERROR"
                return
            
            # now apply all filters
            self.mainWindow.mainToolbar.filterPage.runAllFilterLists()
            
            saveName = saveText % (i,)
            log("Saving image: %s" % (saveName,), 0, 2)
            
            # now save image
            filename = self.mainWindow.renderer.saveImage(self.parent.renderType, self.parent.imageFormat, 
                                                          saveName, 1)
        
        # create movie
        if self.createMovie:
            CWD = os.getcwd()
            os.chdir(saveDir)
            
            # temporary (should be optional)
            framerate = 10
            bitrate = 10000000
            outputprefix = "movie"
            outputsuffix = "mpg"
            
            saveText = os.path.basename(saveText)
            
            command = "%s -r %d -y -i %s.%s -r %d -b %d %s.%s" % (self.parent.ffmpeg, framerate, saveText, 
                                                                  self.parent.imageFormat, 25, bitrate, 
                                                                  outputprefix, outputsuffix)
            
            log("Creating movie file: %s.%s" % (outputprefix, outputsuffix))
            
            # change to QProcess
            process = subprocess.Popen(command, shell=True, executable="/bin/bash", stdin=subprocess.PIPE, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, stderr = process.communicate()
            status = process.poll()
            if status:
                log("FFMPEG FAILED")
                print stderr
            
            os.chdir(CWD)
    
    def warnFirstFileNotPresent(self, filename):
        """
        Warn the first file is not present.
        
        """
        QtGui.QMessageBox.warning(self, "Warning", "Could not locate first file in sequence: %s" % (filename,))
    
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
        text = "%s%s.%s" % (self.fileprefix.text(), self.numberFormat, self.mainWindow.fileExtension)
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

    