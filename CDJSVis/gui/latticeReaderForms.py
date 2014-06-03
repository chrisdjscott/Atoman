
"""
File input is handled by the lattice reader forms on the systems dialog.
The file type should be selected from the drop down menu.
Multiple files can be loaded at the same time by shift/cmd clicking them.

Basically you should always leave this as 'AUTO DETECT'. The available formats are listed below. 

"""
import os
import sys
import platform
import logging
import glob

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath
from .genericForm import GenericForm
from ..state import latticeReaders

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
        
    def openFile(self, label=None, filename=None, rouletteIndex=None, sftpPath=None):
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
    
    def postOpenFile(self, stateType, state, filename, sftpPath):
        """
        Should always be called at the end of openFile.
        
        """
        self.updateFileLabel(filename)
        
        self.parent.fileLoaded(stateType, state, filename, self.fileExtension, self.stackIndex, sftpPath)
    
    def openFileDialog(self):
        """
        Open a file dialog to select a file.
        
        """
        if self.fileFormatString is None:
            self.mainWindow.displayError("GenericReaderWidget: fileFormatString not set on:\n%s" % str(self))
            return None
        
        fdiag = QtGui.QFileDialog()
         
        filesString = str(self.fileFormatString)
        
        # temporarily remove stays on top hint on systems dialog
        sd = self.parent.parent
        sd.tmpHide()
        
        filenames = fdiag.getOpenFileNames(self, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString)[0]
        
        filenames = [str(fn) for fn in filenames]
        
        sd.showAgain()
        
        if not len(filenames):
            return None
        
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        
        try:
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
        
        finally:
            QtGui.QApplication.restoreOverrideCursor()
        
        return result

################################################################################

class AutoDetectReaderForm(GenericReaderForm):
    """
    This method should usually always be selected. It will read the first few lines of a file and decide which reader 
    should be used (datReader, xyzReader, refReader, ...).  If this doesn't work then probably there is something 
    wrong with the file or it is in a format not recognised yet (let me know).
    
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
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "File dialog")
        self.openLatticeDialogButton.setToolTip("Open file dialog")
        self.openLatticeDialogButton.setCheckable(0)
        self.openLatticeDialogButton.clicked.connect(self.openFileDialog)
        row.addWidget(self.openLatticeDialogButton)
        
        # sftp browser
        if hasattr(self.parent, "sftp_browser"):
            openSFTPBrowserButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "SFTP browser")
            openSFTPBrowserButton.setToolTip("Open SFTP browser")
            openSFTPBrowserButton.setCheckable(0)
            openSFTPBrowserButton.clicked.connect(self.openSFTPBrowser)
            row = self.newRow()
            row.addWidget(openSFTPBrowserButton)
    
    def openSFTPBrowser(self):
        """
        Open SFTP browser
        
        """
        self.logger.debug("Opening SFTP browser (AUTO DETECT)")
        
        ok = self.parent.sftp_browser.exec_()
        if ok and len(self.parent.sftp_browser.filename_remote):
            remotefns = self.parent.sftp_browser.filename_remote
            localfns = self.parent.sftp_browser.filename_local
            sftpPaths = self.parent.sftp_browser.sftpPath
            
            for remotefn, localfn, sftpPath in zip(remotefns, localfns, sftpPaths):
                self.logger.info("Opening remote file (%s) on local machine", remotefn)
                self.logger.debug("Local filename: '%s'", localfn)
                self.logger.debug("SFTP path: '%s'", sftpPath)
                
                # read file
                status = self.openFile(filename=localfn, sftpPath=sftpPath)
                
                # remove local copy
                self.cleanUnzipped(localfn, True)
            
            # remove Roulettes if exists
            rfns = glob.glob(os.path.join(self.mainWindow.tmpDirectory, "Roulette*.OUT"))
            for rfn in rfns:
                os.unlink(rfn)
    
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
    
    def openFile(self, filename=None, rouletteIndex=None, sftpPath=None):
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
            self.mainWindow.displayWarning("Could not find file: "+filename)
            self.logger.warning("Could not find file: '%s'", filename)
            return -1, None
        
        # determine format and type of reader
        formatIdentifier = self.determineFileFormat(filepath)
        
        # clean unzipped
        self.cleanUnzipped(filepath, zipFlag)
        
        if formatIdentifier is None:
            self.logger.error("Could not auto detect format of file")
            self.mainWindow.displayError("ERROR: could not auto detect format of file: '%s'\n\nIf the lattice is small (<5 atoms) you may have to select the right reader manually; otherwise please send me the file and I'll see what the problem is." % filename)
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
        status = selectedReaderForm.openFile(filename=filename, sftpPath=sftpPath)
        
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
        
        success = False
        repeatCount = 0
        repeatVal = None
        repeatFirstIndexLen = None
        for count, line in enumerate(f):
            if count == maxLines:
                self.logger.debug("  Max lines reached; stopping")
                break
            
            line = line.strip()
            
            # ignore blank lines
            if not len(line):
                continue
            
            # split line
            array = line.split()
            num = len(array)
            
            self.logger.debug("  Line %d; num %d (%s)", count, num, line)
            
            # store num items in line
            lineArrayCountList.append(num)
            
            if num == repeatVal:
                repeatCount += 1
            else:
                repeatCount = 0
                repeatVal = num
                repeatFirstIndexLen = len(lineArrayCountList)
            
            self.logger.debug("  Repeat: cnt %d; val %d; first %d", repeatCount, repeatVal, repeatFirstIndexLen)
            
            if repeatCount == repeatThreshold:
                self.logger.debug("  Repeat threshold reached (%d): exiting detect loop", repeatCount)
                success = True
                break
        
        if not success:
            if count < 10:
                self.logger.warning("Trying to determine format of small file; this might not work")
            else:    
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
    Read LBOMD lattice format files.  They should be in the format that LBOMD requires for 'lattice.dat':

    #) First line should be the number of atoms
    #) Second line should be the cell dimensions (x, y, z)
    #) Then one line per atom containing (separated by whitespace): 
      
       * symbol
       * x position
       * y position
       * z position
       * charge
    
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
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "File dialog")
        self.openLatticeDialogButton.setToolTip("Open file dialog")
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
    
    def openFile(self, filename=None, rouletteIndex=None, sftpPath=None):
        """
        Open file.
        
        """
        if filename is None:
            filename = self.getFileName()
        
        filename = str(filename)
        
        # remove zip extensions
        if filename[-3:] == ".gz":
            filename = filename[:-3]
        elif filename[-4:] == ".bz2":
            filename = filename[:-4]
        
        status, state = self.latticeReader.readFile(filename, rouletteIndex=rouletteIndex)
        
        if not status:
            GenericReaderForm.postOpenFile(self, self.stateType, state, filename, sftpPath)
        
        return status
        
################################################################################

class LbomdRefReaderForm(GenericReaderForm):
    """
    Read LBOMD animation-reference format files. They should be in the following format

    #) First line should be the number of atoms
    #) Second line should be the cell dimensions (x, y, z)
    #) Then one line per atom containing (separated by whitespace): 
      
       * symbol
       * atom index (starting from 1 to NATOMS)
       * x position
       * y position
       * z position
       * kinetic energy at time 0
       * potential energy at time 0
       * x force at time 0
       * y force at time 0
       * z force at time 0
       * charge
    
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
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "File dialog")
        self.openLatticeDialogButton.setToolTip("Open file dialog")
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
    
    def openFile(self, filename=None, rouletteIndex=None, sftpPath=None):
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
            GenericReaderForm.postOpenFile(self, self.stateType, state, filename, sftpPath)
            
            # set xyz input form to use this as ref
            self.logger.debug("Setting as ref on xyz reader")
            self.parent.readerForms["LBOMD XYZ"].setRefState(state, filename)
        
        return status

################################################################################

class LbomdXYZReaderForm(GenericReaderForm):
    """
    Read LBOMD XYZ format files.  XYZ files must be linked to an :ref:`LBOMD_REF` file (i.e. you must read one 
    of those files first).  It does not make sense to have an XYZ file without an animation-reference file 
    because the atom symbols are only stored in the reference. The number of atoms must be the same in the 
    reference you are using to link with the XYZs.  When you load an :ref:`LBOMD_REF` file it will automatically 
    be linked to any subsequently loaded XYZ files.

    Different formats of XYZ files are supported (more can be added...)

    #)  Positions and energies

        *   First line is number of atoms
        *   Second line is simulation time in fs
        *   Then one line per atom containing (separated by whitespace)
    
            *   atom index
            *   x position
            *   y position
            *   z position
            *   kinetic energy
            *   potential energy

    #)  Positions, energies and charges
    
        *   First line is number of atoms
        *   Second line is simulation time in fs
        *   Then one line per atom containing (separated by whitespace)
        
            *   atom index
            *   x position
            *   y position
            *   z position
            *   kinetic energy
            *   potential energy
            *   charge
    
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
        self.openLatticeDialogButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "File dialog")
        self.openLatticeDialogButton.setToolTip("Open file dialog")
        self.openLatticeDialogButton.setCheckable(0)
        self.connect(self.openLatticeDialogButton, QtCore.SIGNAL('clicked()'), lambda isRef=False: self.openFileDialog(isRef))
        row.addWidget(self.openLatticeDialogButton)
    
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
        
        # temporarily remove stays on top hint on systems dialog
        sd = self.parent.parent
        sd.tmpHide()
        
        if isRef:
            filename = fdiag.getOpenFileName(self, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString)[0]
            
            sd.showAgain()
            
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
            filenames = fdiag.getOpenFileNames(self, "%s - Open file" % (self.widgetTitle,), os.getcwd(), filesString)[0]
            
            sd.showAgain()
            
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
    
    def openFile(self, filename=None, isRef=False, rouletteIndex=None, sftpPath=None):
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
            else:
                GenericReaderForm.postOpenFile(self, self.stateType, state, filename, sftpPath)
            
            self.updateFileLabelCustom(filename, isRef)
        
        return status
