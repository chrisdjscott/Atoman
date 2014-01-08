
"""
Lattice reader forms for the inputTab.

@author: Chris Scott

"""
import os
import sys
import platform
import logging

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from .genericForm import GenericForm
from .. import latticeReaders

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)




################################################################################

class GenericReaderForm(GenericForm):
    """
    Generic reader widget.
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, name):
        super(GenericReaderForm, self).__init__(parent, None, "%s READER" % name)
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.toolbarWidth = None
        self.widgetTitle = "%s READER" % name
        
        self.logger = logging.getLogger(__name__)
        
        self.stateType = "ref"
        
        self.fileFormatString = None
        
        self.fileExtension = None
        
        self.stackIndex = parent.readerFormsKeys.index(name)
        self.logger.debug("Setting up readerForm '%s': stack index is %d", name, self.stackIndex)
        
        # always show widget
        self.show()
        
    def openFile(self, label=None, filename=None, rouletteIndex=None):
        """
        This should be sub-classed to load the selected file.
        
        Should return the name of the loaded file.
        
        """
        self.mainWindow.displayError("GenericReaderWidget: openFile has not been overriden on:\n%s" % str(self))
        return 1
    
    def getFileName(self):
        pass
    
    def updateFileLabel(self, filename):
        pass
    
    def postOpenFile(self, stateType, state, filename):
        """
        Should always be called at the end of openFile.
        
        """
        self.updateFileLabel(filename)
        
        self.parent.fileLoaded(stateType, state, filename, self.fileExtension, self.stackIndex)
    
    def openFileDialog(self):
        """
        Open a file dialog to select a file.
        
        """
        if self.fileFormatString is None:
            self.mainWindow.displayError("GenericReaderWidget: fileFormatString not set on:\n%s" % str(self))
            return None
        
        fdiag = QtGui.QFileDialog()
        
        filesString = str(self.fileFormatString)
        
        if platform.system() == "Darwin":
            filenames = fdiag.getOpenFileNames(self.parent.parent, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString, options=QtGui.QFileDialog.DontUseNativeDialog)[0]
        
        else:
            filenames = fdiag.getOpenFileNames(self.parent.parent, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString)[0]
        
        filenames = [str(fn) for fn in filenames]
        
        if not len(filenames):
            return None
        
        for filename in filenames:
            nwd, filename = os.path.split(filename)        
            
            # change to new working directory
            if nwd != os.getcwd():
                self.mainWindow.console.write("Changing to dir "+nwd)
                os.chdir(nwd)
                self.mainWindow.updateCWD()
            
            # remove zip extensions
            if filename[-3:] == ".gz":
                filename = filename[:-3]
                
            elif filename[-4:] == ".bz2":
                filename = filename[:-4]
            
            # open file
            result = self.openFile(filename=filename)
            
            if result:
                break
        
        return result

################################################################################

class AutoDetectReaderForm(GenericReaderForm):
    """
    Auto detect form
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, name):
        super(AutoDetectReaderForm, self).__init__(parent, mainToolbar, mainWindow, name)
        
        self.loadSystemForm = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        
        self.tmpLocation = self.mainWindow.tmpDirectory
        
        # acceptable formats string
        self.fileFormatString = "Lattice files (*.dat *.dat.bz2 *.dat.gz *.xyz *.xyz.bz2 *.xyz.gz)"
        
        self.show()
        
        # file name line
        row = self.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        if os.path.exists("ref.dat"):
            ininame = "ref.dat"
        elif os.path.exists("animation-reference.xyz"):
            ininame = "animation-reference.xyz"
        elif os.path.exists("launch.dat"):
            ininame = "launch.dat"
        elif os.path.exists("KMC0.dat"):
            ininame = "KMC0.dat"
        elif os.path.exists("lattice0.dat"):
            ininame = "lattice0.dat"
        else:
            ininame = "lattice.dat"
        
        self.latticeLabel = QtGui.QLineEdit(ininame)
        self.latticeLabel.setFixedWidth(150)
        row.addWidget(self.latticeLabel)
        
        self.loadLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadLatticeButton.setToolTip("Load file")
        self.loadLatticeButton.clicked.connect(self.openFile)
        row.addWidget(self.loadLatticeButton)
        
        # open dialog
        row = self.newRow()
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open file")
        self.openLatticeDialogButton.setToolTip("Open file")
        self.openLatticeDialogButton.setCheckable(0)
        self.openLatticeDialogButton.clicked.connect(self.openFileDialog)
        row.addWidget(self.openLatticeDialogButton)
        
    def updateFileLabel(self, filename):
        """
        Update file label.
        
        """
        self.latticeLabel.setText(filename)
    
    def getFileName(self):
        """
        Returns file name from label.
        
        """
        return str(self.latticeLabel.text())
    
    def openFile(self, filename=None, rouletteIndex=None):
        """
        Open file.
        
        """
        if filename is None:
            filename = self.getFileName()
        
        filename = str(filename)
        
        logger = self.logger
        logger.info("Opening file using autodetect method: '%s'", filename)
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        # unzip if required
        filepath, zipFlag = self.checkForZipped(filename)
        if zipFlag == -1:
            self.displayWarning("Could not find file: "+filename)
            self.logger.warning("Could not find file: '%s'", filename)
            return -1, None
        
        # determine format and type of reader
        formatIdentifier = self.determineFileFormat(filepath)
        
        # clean unzipped
        self.cleanUnzipped(filepath, zipFlag)
        
        if formatIdentifier is None:
            self.logger.error("Could not auto detect format of file")
            self.mainWindow.displayError("ERROR: could not auto detect format of file: '%s'\n\nPlease send me the file." % filename)
            return 1
        
        # get reader from LoadSystemForm
        # first get list of readerForms
        readerForms = self.loadSystemForm.readerForms.values()
        
        # now get list of readers
        readers = [form.latticeReader for form in readerForms if (hasattr(form, "latticeReader") and form.latticeReader is not None)]
        self.logger.debug("Readers: %s", str([type(r) for r in readers]))
        
        # check if matches
        selectedReaderForm = None
        for readerForm in readerForms:
            if hasattr(readerForm, "latticeReader") and readerForm.latticeReader is not None:
                for fmt in readerForm.latticeReader.formatIdentifiers:
                    if fmt == formatIdentifier:
                        selectedReaderForm = readerForm
                        break
                
                if selectedReaderForm is not None:
                    break
        
        if selectedReaderForm is None:
            self.logger.error("Could not find matching format in available readers (%s)", formatIdentifier)
            self.mainWindow.displayError("ERROR: could not find matching format in available readers.\n\nNeed to implement new reader.")
            return 2
        
        self.logger.info("Selected reader: '%s'", selectedReaderForm.widgetTitle)
        
        # read file
        status = selectedReaderForm.openFile(filename=filename)
        
        return status
    
    def determineFileFormat(self, filename):
        """
        Determine file format
        
        """
        self.logger.debug("Attempting to determine file format")
        
        f = open(filename)
        
        # if can't determine within limit stop
        maxLines = 20
        
        # stop when this number of lines are repeating (won't work for FAILSAFE!!!)
        repeatThreshold = 3
        
        lineArrayCountList = []
        
        count = 0
        success = False
        repeatCount = 0
        repeatVal = None
        repeatFirstIndexLen = None
        while True and count < maxLines:
            line = f.readline().strip()
            
            # ignore blank lines
            if not len(line):
                continue
            
            # split line
            array = line.split()
            num = len(array)
            
            lineArrayCountList.append(num)
            
            if num == repeatVal:
                repeatCount += 1
            
            else:
                repeatCount = 0
                repeatVal = num
                repeatFirstIndexLen = len(lineArrayCountList)
            
            if repeatCount == repeatThreshold:
                self.logger.debug("  Repeat threshold reached (%d): exiting detect loop", repeatCount)
                success = True
                break
            
            count += 1
        
        if not success:
            self.logger.debug("Failed to detect file format")
            return None
        
        format_ident = lineArrayCountList[:repeatFirstIndexLen]
        self.logger.debug("Format identifier: '%s'", str(format_ident))
        
        return format_ident
    
    def checkForZipped(self, filename):
        """
        Check if file exists (unzip if required)
        
        """
        if os.path.exists(filename):
            fileLocation = '.'
            zipFlag = 0
        
        else:
            if os.path.exists(filename + '.bz2'):
                command = 'bzcat -k "%s.bz2" > ' % (filename)
            
            elif os.path.exists(filename + '.gz'):
                command = 'gzip -dc "%s.gz" > ' % (filename)
                
            else:
                return None, -1
                
            fileLocation = self.tmpLocation
            command = command + os.path.join(fileLocation, filename)
            os.system(command)
            zipFlag = 1
            
        filepath = os.path.join(fileLocation, filename)
        if not os.path.exists(filepath):
            return None, -1
            
        return filepath, zipFlag
    
    def cleanUnzipped(self, filepath, zipFlag):
        """
        Clean up unzipped file.
        
        """
        if zipFlag:
            os.unlink(filepath)

################################################################################

class LbomdDatReaderForm(GenericReaderForm):
    """
    LBOMD DAT input widget.
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, name):
        super(LbomdDatReaderForm, self).__init__(parent, mainToolbar, mainWindow, name)
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        
        self.fileExtension = "dat"
        
        # acceptable formats string
        self.fileFormatString = "Lattice files (*.dat *.dat.bz2 *.dat.gz)"
        
        # reader
        self.latticeReader = latticeReaders.LbomdDatReader(self.mainWindow.tmpDirectory, self.mainWindow.console.write, self.mainWindow.displayWarning, 
                                                           self.mainWindow.displayError)
        
        self.show()
        
        # file name line
        row = self.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        if os.path.exists("ref.dat"):
            ininame = "ref.dat"
        elif os.path.exists("launch.dat"):
            ininame = "launch.dat"
        else:
            ininame = "lattice.dat"
        
        self.latticeLabel = QtGui.QLineEdit(ininame)
        self.latticeLabel.setFixedWidth(150)
        row.addWidget(self.latticeLabel)
        
        self.loadLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadLatticeButton.setToolTip("Load file")
        self.loadLatticeButton.clicked.connect(self.openFile)
        row.addWidget(self.loadLatticeButton)
        
        # open dialog
        row = self.newRow()
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open file")
        self.openLatticeDialogButton.setToolTip("Open file")
        self.openLatticeDialogButton.setCheckable(0)
        self.openLatticeDialogButton.clicked.connect(self.openFileDialog)
        row.addWidget(self.openLatticeDialogButton)
    
    def updateFileLabel(self, filename):
        """
        Update file label.
        
        """
        self.latticeLabel.setText(filename)
    
    def getFileName(self):
        """
        Returns file name from label.
        
        """
        return str(self.latticeLabel.text())
    
    def openFile(self, filename=None, rouletteIndex=None):
        """
        Open file.
        
        """
        if filename is None:
            filename = self.getFileName()
        
        filename = str(filename)
        
#        if not len(filename):
#            return None
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        status, state = self.latticeReader.readFile(filename, rouletteIndex=rouletteIndex)
        
        if not status:
            GenericReaderForm.postOpenFile(self, self.stateType, state, filename)
        
        return status
        
################################################################################

class LbomdRefReaderForm(GenericReaderForm):
    """
    LBOMD REF input widget.
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, name):
        super(LbomdRefReaderForm, self).__init__(parent, mainToolbar, mainWindow, name)
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        
        self.fileExtension = "xyz"
        
        self.fileFormatString = "REF files (*.xyz *.xyz.bz2 *.xyz.gz)"
        
        self.latticeReader = latticeReaders.LbomdRefReader(self.mainWindow.tmpDirectory, self.mainWindow.console.write, self.mainWindow.displayWarning, 
                                                           self.mainWindow.displayError)
        
        self.show()
        
        # file name line
        row = self.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        self.latticeLabel = QtGui.QLineEdit("animation-reference.xyz")
        self.latticeLabel.setFixedWidth(150)
        row.addWidget(self.latticeLabel)
        
        self.loadLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadLatticeButton.setToolTip("Load file")
        self.connect(self.loadLatticeButton, QtCore.SIGNAL('clicked()'), self.openFile)
        row.addWidget(self.loadLatticeButton)
        
        # open dialog
        row = self.newRow()
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open file")
        self.openLatticeDialogButton.setToolTip("Open file")
        self.openLatticeDialogButton.setCheckable(0)
        self.connect(self.openLatticeDialogButton, QtCore.SIGNAL('clicked()'), self.openFileDialog)
        row.addWidget(self.openLatticeDialogButton)
    
    def updateFileLabel(self, filename):
        """
        Update file label.
        
        """
        self.latticeLabel.setText(filename)
    
    def getFileName(self):
        """
        Returns file name from label.
        
        """
        return str(self.latticeLabel.text())
    
    def openFile(self, filename=None, rouletteIndex=None):
        """
        Open file.
        
        """
        if filename is None:
            filename = self.getFileName()
        
        filename = str(filename)
        
#        if not len(filename):
#            return None
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        status, state = self.latticeReader.readFile(filename, rouletteIndex=rouletteIndex)
        
        if not status:
            GenericReaderForm.postOpenFile(self, self.stateType, state, filename)
            
            # set xyz input form to use this as ref
            self.logger.debug("Setting as ref on xyz reader")
            self.parent.readerForms["LBOMD XYZ"].setRefState(state, filename)
        
        return status

################################################################################

class LbomdXYZReaderForm(GenericReaderForm):
    """
    LBOMD XYZ reader.
    
    """
    def __init__(self, parent, mainToolbar, mainWindow, name):
        super(LbomdXYZReaderForm, self).__init__(parent, mainToolbar, mainWindow, name)
        
        self.inputTab = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        
        self.refLoaded = False
        
        self.fileExtension = "xyz"
        
        self.fileFormatString = "XYZ files (*.xyz *.xyz.bz2 *.xyz.gz)"
        self.fileFormatStringRef = "XYZ files (*.xyz *.xyz.bz2 *.xyz.gz)"
        
        self.latticeReader = latticeReaders.LbomdXYZReader(self.mainWindow.tmpDirectory, self.mainWindow.console.write, self.mainWindow.displayWarning, 
                                                           self.mainWindow.displayError)
        self.refReader = latticeReaders.LbomdRefReader(self.mainWindow.tmpDirectory, self.mainWindow.console.write, self.mainWindow.displayWarning, 
                                                       self.mainWindow.displayError)
        
        self.show()
        
        # ref file
        row = self.newRow()
        label = QtGui.QLabel("Ref name")
        row.addWidget(label)
        
        self.refLabel = QtGui.QLineEdit("animation-reference.xyz")
        self.refLabel.setFixedWidth(150)
        row.addWidget(self.refLabel)
        
        self.loadRefButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadRefButton.setToolTip("Load ref")
        self.connect(self.loadRefButton, QtCore.SIGNAL('clicked()'), lambda isRef=True: self.openFile(isRef=isRef))
        row.addWidget(self.loadRefButton)
        
        # open dialog
        row = self.newRow()
        self.openRefDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open reference")
        self.openRefDialogButton.setToolTip("Open ref")
        self.openRefDialogButton.setCheckable(0)
        self.connect(self.openRefDialogButton, QtCore.SIGNAL('clicked()'), lambda isRef=True: self.openFileDialog(isRef))
        row.addWidget(self.openRefDialogButton)
        
        # input file name line
        row = self.newRow()
        label = QtGui.QLabel("File name")
        row.addWidget(label)
        
        self.latticeLabel = QtGui.QLineEdit("PuGaH0000.xyz")
        self.latticeLabel.setFixedWidth(150)
        row.addWidget(self.latticeLabel)
        
        self.loadLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath("go-jump.svg")), '')
        self.loadLatticeButton.setToolTip("Load file")
        self.connect(self.loadLatticeButton, QtCore.SIGNAL('clicked()'), lambda isRef=False: self.openFile(isRef=isRef))
        row.addWidget(self.loadLatticeButton)
        
        # open dialog
        row = self.newRow()
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "Open file")
        self.openLatticeDialogButton.setToolTip("Open file")
        self.openLatticeDialogButton.setCheckable(0)
        self.connect(self.openLatticeDialogButton, QtCore.SIGNAL('clicked()'), lambda isRef=False: self.openFileDialog(isRef))
        row.addWidget(self.openLatticeDialogButton)
        
        # help icon
        row = self.newRow()
        row.RowLayout.addStretch(1)
        
        helpButton = QtGui.QPushButton(QtGui.QIcon(iconPath("Help-icon.png")), "")
        helpButton.setFixedWidth(20)
        helpButton.setFixedHeight(20)
        helpButton.setToolTip("""<p>XYZ files must be linked with a REF file!</p>
                                 <p>If you have loaded a REF already it will automatically be linked to the XYZ files you load.</p>
                                 <p>Otherwise, you will need to load a REF before loading XYZs.</p>""")
        
        row.addWidget(helpButton)
    
    def getFileName(self, isRef):
        """
        Get filename
        
        """
        if isRef:
            fn = str(self.refLabel.text())
        else:
            fn = str(self.latticeLabel.text())
        
        return fn
    
    def updateFileLabelCustom(self, filename, isRef):
        """
        Update file label
        
        """
        if isRef:
            self.refLabel.setText(filename)
        else:
            self.latticeLabel.setText(filename)
    
    def openFileDialog(self, isRef):
        """
        Open a file dialog to select a file.
        
        """
        if self.fileFormatString is None:
            self.mainWindow.displayError("GenericReaderWidget: fileFormatString not set on:\n%s" % str(self))
            return None
        
        if not isRef and not self.refLoaded:
            self.logger.warning("Must load corresponding reference first")
            self.mainWindow.displayWarning("Must load corresponding reference first!\n\nXYZ files MUST always be linked ta a reference.")
            return None
        
        if isRef and self.refLoaded:
            reply = QtGui.QMessageBox.question(self, "Message", 
                                               "Ref is already loaded: do you want to overwrite it?",
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
            if reply != QtGui.QMessageBox.Yes:
                return None
        
        fdiag = QtGui.QFileDialog()
        
        if isRef:
            filesString = str(self.fileFormatStringRef)
        else:
            filesString = str(self.fileFormatString)
        
        if isRef:
            if platform.system() == "Darwin":
                filename = fdiag.getOpenFileName(self, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString, options=QtGui.QFileDialog.DontUseNativeDialog)[0]
            
            else:
                filename = fdiag.getOpenFileName(self, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString)[0]
            
            filename = str(filename)
            
            if not len(filename):
                return None
            
            (nwd, filename) = os.path.split(filename)        
            
            # change to new working directory
            if nwd != os.getcwd():
                self.mainWindow.console.write("Changing to dir "+nwd)
                os.chdir(nwd)
                self.mainWindow.updateCWD()
            
            # remove zip extensions
            if filename[-3:] == ".gz":
                filename = filename[:-3]
            elif filename[-4:] == ".bz2":
                filename = filename[:-4]
            
            # open file
            result = self.openFile(filename=filename, isRef=isRef)
        
        else:
            if platform.system() == "Darwin":
                filenames = fdiag.getOpenFileNames(self, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString, options=QtGui.QFileDialog.DontUseNativeDialog)[0]
            
            else:
                filenames = fdiag.getOpenFileNames(self, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString)[0]
            
            filenames = [str(fn) for fn in filenames]
            
            if not len(filenames):
                return None
            
            for filename in filenames:
                nwd, filename = os.path.split(filename)        
                
                # change to new working directory
                if nwd != os.getcwd():
                    self.mainWindow.console.write("Changing to dir "+nwd)
                    os.chdir(nwd)
                    self.mainWindow.updateCWD()
                
                # remove zip extensions
                if filename[-3:] == ".gz":
                    filename = filename[:-3]
                elif filename[-4:] == ".bz2":
                    filename = filename[:-4]
                
                # open file
                result = self.openFile(filename=filename, isRef=isRef)
                
                if result:
                    break
        
        return result
    
    def setRefState(self, state, filename):
        """
        Set the ref state.
        
        """
        self.currentRefState = state
        self.refLabel.setText(filename)
        self.refLoaded = True
    
    def openFile(self, filename=None, isRef=False, rouletteIndex=None):
        """
        Open file.
        
        """
        if not isRef and not self.refLoaded:
            self.logger.warning("Must load corresponding reference first")
            self.mainWindow.displayWarning("Must load corresponding reference first!\n\nXYZ files MUST always be linked to a reference.")
            return 2
        
        if isRef and self.refLoaded:
            reply = QtGui.QMessageBox.question(self, "Message", 
                                               "Ref is already loaded: do you want to overwrite it?",
                                               QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)
        
            if reply != QtGui.QMessageBox.Yes:
                return 3
        
        if filename is None:
            filename = self.getFileName(isRef)
        
        filename = str(filename)
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        if isRef:
            status, state = self.refReader.readFile(filename)
        else:
            status, state = self.latticeReader.readFile(filename, self.currentRefState, rouletteIndex=rouletteIndex)
        
        if not status:
            if isRef:
                self.setRefState(state, filename)
                self.parent.lbomdXyzWidget.setRefState(state, filename)
            else:
                GenericReaderForm.postOpenFile(self, self.stateType, state, filename)
            
            self.updateFileLabelCustom(filename, isRef)
        
        return status
