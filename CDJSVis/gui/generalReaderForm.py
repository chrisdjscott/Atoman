
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
import shutil

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath, resourcePath
from ..state import latticeReaderGeneric

try:
    from .. import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)


################################################################################

class GeneralLatticeReaderForm(QtGui.QWidget):
    """
    Lattice reader form
    
    """
    def __init__(self, parent, mainToolbar, mainWindow):
        super(GeneralLatticeReaderForm, self).__init__(parent)
        
        self.loadSystemForm = parent
        self.systemsDialog = parent.systemsDialog
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        self.tmpLocation = self.mainWindow.tmpDirectory
        self.currentFile = None
        
        self.logger = logging.getLogger(__name__)
        
        vbox = QtGui.QVBoxLayout()
        
        # lattice reader
        self.latticeReader = latticeReaderGeneric.LatticeReaderGeneric(self.tmpLocation)
        
        # open dialog
        self.openLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "File dialog")
        self.openLatticeButton.setToolTip("Open file dialog")
        self.openLatticeButton.setCheckable(0)
        self.openLatticeButton.clicked.connect(self.openFileDialog)
        hbox = QtGui.QHBoxLayout()
        hbox.setAlignment(QtCore.Qt.AlignHCenter)
        hbox.addWidget(self.openLatticeButton)
        vbox.addLayout(hbox)
        
        # sftp browser
        if hasattr(self.loadSystemForm, "sftp_browser"):
            openSFTPBrowserButton = QtGui.QPushButton(QtGui.QIcon(iconPath('document-open.svg')), "SFTP browser")
            openSFTPBrowserButton.setToolTip("Open SFTP browser")
            openSFTPBrowserButton.setCheckable(0)
            openSFTPBrowserButton.clicked.connect(self.openSFTPBrowser)
            hbox = QtGui.QHBoxLayout()
            hbox.setAlignment(QtCore.Qt.AlignHCenter)
            hbox.addWidget(openSFTPBrowserButton)
            vbox.addLayout(hbox)
        
        self.setLayout(vbox)
        
        # file formats
        self.fileFormats = latticeReaderGeneric.FileFormats()
        
        # check if user dir exists
        userdir = os.path.join(os.path.expanduser("~"), ".cdjsvis")
        fn = os.path.join(userdir, "file_formats.IN")
        if not os.path.isfile(fn):
            if not os.path.exists(userdir):
                os.mkdir(userdir)
            # copy from data dir
            defaultPath = resourcePath("file_formats.IN")
            shutil.copy(defaultPath, fn)
        
        # load file formats
        self.fileFormats.read(filename=fn)
    
    def openSFTPBrowser(self):
        """
        Open SFTP browser
        
        """
        if self.parent.sftp_browser is None:
            self.mainWindow.displayError("Paramiko must be installed to use the SFTP browser")
            return
        
        self.logger.debug("Opening SFTP browser")
        
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
                status = self.openFile(localfn, sftpPath=sftpPath)
                
                # remove local copy
                self.cleanUnzipped(localfn, True)
            
            # remove Roulettes if exists
            rfns = glob.glob(os.path.join(self.mainWindow.tmpDirectory, "Roulette*.OUT"))
            for rfn in rfns:
                os.unlink(rfn)
    
    def openFileDialog(self):
        """
        Open a file dialog to select a file.
        
        """
        # if no file formats, pop up a dialog telling the user to make some/reset to the default formats
        if not len(self.fileFormats):
            self.mainWindow.displayError("No file formats have been defined")
            self.logger.error("No file formats have been defined")
            return
        
        fdiag = QtGui.QFileDialog()
         
        # temporarily remove stays on top hint on systems dialog (Mac only)
        if platform.system() == "Darwin":
            sd = self.systemsDialog
            sd.tmpHide()
        
        # open the dialog
        filenames = fdiag.getOpenFileNames(self, "Open file", os.getcwd())[0]
        filenames = [str(fn) for fn in filenames]
        
        if platform.system() == "Darwin":
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
                
                # open file
                result = self.openFile(filename)
                
                if result:
                    break
        
        finally:
            QtGui.QApplication.restoreOverrideCursor()
        
        return result
    
    def openFile(self, filename, rouletteIndex=None, sftpPath=None):
        """
        Open file.
        
        """
        self.logger.debug("Open file: '%s'", filename)
        if rouletteIndex is not None:
            self.logger.debug("Roulette index is: %d", rouletteIndex)
        if sftpPath is not None:
            self.logger.debug("SFTP path is: '%s'", sftpPath)
        
        # first we attempt to detect a format
        # if there are multiple possibilities we pop up a dialog to ask the user
        # (database of user responses, possibly including extension, to help order the options)
        # if there is a linked lattice type, check one is loaded, if not ask user to load one first,
        #   if 1 is loaded use it, if >1 then pop up a dialog with most recent at the top
        
        # status
        self.currentFile = os.path.basename(filename)
#         self.mainWindow.setStatus("Reading '%s'" % self.currentFile)
        self.mainWindow.updateProgress(0, 0, "Reading '%s'" % self.currentFile)
        
        # unzip if required
        filepath, zipFlag = self.latticeReader.checkForZipped(filename)
        
        try:
            # file format
            fileFormat = self.determineFileFormat(filepath, filename)
            if fileFormat is None:
                return 1
            
            # linked lattice
            linkedLattice = None
            if fileFormat.linkedName is not None:
                linkedLattice = self.getLinkedLattice(fileFormat, filename)
                if linkedLattice is None:
                    return 2

            # open file
            status, state = self.latticeReader.readFile(filepath, fileFormat, rouletteIndex=rouletteIndex, linkedLattice=linkedLattice, callback=None)
        
        finally:
            # delete unzipped file if required
            self.latticeReader.cleanUnzipped(filepath, zipFlag)
            self.mainWindow.hideProgressBar()
            self.currentFile = None
        
        if not status:
            self.postOpenFile(state, filename, fileFormat, sftpPath)
    
    def updateProgress(self, n, nmax):
        """
        Update progress
        
        """
        print "Hello, world!", n, nmax
        
        return 0
    
    def getLinkedLattice(self, fileFormat, properName):
        """
        Get linked lattice.
        
        If one option, select it automatically,
        otherwise ask user.
        
        """
        linkedType = fileFormat.linkedName
        self.logger.debug("Getting linked lattice (type: '%s')", linkedType)
        
        # list of lattices with required format
        availableSystems = self.systemsDialog.getLatticesByFormat(linkedType)
        
        # nothing available
        if not len(availableSystems):
            self.logger.error("No files of linked type")
            self.mainWindow.displayError("Cannot open '%s'; it depends on a '%s' file being loaded first!" % (properName, linkedType))
            lattice = None
        
        # one available
        elif len(availableSystems) == 1:
            latticeDisplayName, lattice = availableSystems[0]
            self.logger.info("Found 1 possible linked lattice: '%s'", latticeDisplayName)
        
        # multiple available
        else:
            self.logger.debug("Found %d possible linked lattices", len(availableSystems))
            
            # items list
            items = [item[0] for item in availableSystems]
            
            # open dialog
            dlg = SelectLinkedLatticeDialog(properName, items, parent=self)
            status = dlg.exec_()
            
            if status == QtGui.QDialog.Accepted:
                index = dlg.combo.currentIndex()
                latticeDisplayName, lattice = availableSystems[index]
                self.logger.debug("User selected linked lattice %d: '%s'", index, latticeDisplayName)
            
            else:
                lattice = None
        
        return lattice
    
    def postOpenFile(self, state, filename, fileFormat, sftpPath):
        """
        Should always be called at the end of openFile.
        
        """
        self.loadSystemForm.fileLoaded(state, filename, fileFormat, sftpPath)
    
    def determineFileFormat(self, filename, properName):
        """
        Attempt to automatically detect the file format.
        
        If more that one format open a dialog to ask the user.
        
        """
        self.logger.debug("Attempting to determine file format")
        
        # max identifier length
        maxIdLen = self.fileFormats.getMaxIdentifierLength()
        self.logger.debug("Max identifier length: %d", maxIdLen)
        
        # read required lines
        lines = []
        with open(filename) as f:
            #TODO: handle case where less than maxIdLen lines...
            for _ in xrange(maxIdLen):
                lines.append(f.readline())
        
        # loop over formats
        matchedFormats = []
        for fileFormat in self.fileFormats:
            self.logger.debug("Checking for match with format: '%s'", fileFormat.name)
            
            # format identifier
            identifier = fileFormat.getIdentifier()
            
            # delimiter
            delim = fileFormat.delimiter
            # handle whitespace properly
            if delim == " ":
                delim = None
            
            # check for match
            match = True
            for i in xrange(len(identifier)):
                lineLenFormat = identifier[i]
                lineLenInput = len(lines[i].split(delim))
                self.logger.debug("Line %d: %d <-> %d", i, lineLenInput, lineLenFormat)
                if lineLenFormat != lineLenInput:
                    match = False
                    break
            
            if match:
                self.logger.debug("Found possible file format: '%s'", fileFormat.name)
                matchedFormats.append(fileFormat)
        
        if len(matchedFormats) == 1:
            fileFormat = matchedFormats[0]
            self.logger.debug("Found 1 possible file format: '%s'", fileFormat.name)
        
        elif len(matchedFormats) > 1:
            self.logger.debug("Found %d possible file formats", len(matchedFormats))
            
            # open dialog
            items = [fmt.name for fmt in matchedFormats]
            name, ok = QtGui.QInputDialog.getItem(self, "Select file format", "File '%s'" % properName, items,
                                                  editable=False)
            
            if ok:
                self.logger.debug("User selected format '%s'", name)
                fileFormat = self.fileFormats.getFormat(name)
            
            else:
                fileFormat = None
        
        else:
            self.logger.error("Found 0 possible file formats")
            fileFormat = None
            self.mainWindow.displayError("Unrecognised file format for: '%s'" % properName)
        
        return fileFormat

################################################################################

class SelectLinkedLatticeDialog(QtGui.QDialog):
    """
    Select linked lattice from a list
    
    """
    def __init__(self, filename, items, parent=None):
        super(SelectLinkedLatticeDialog, self).__init__(parent)
        
        self.setWindowTitle("Select linked Lattice")
        
        # layout
        layout = QtGui.QFormLayout()
        self.setLayout(layout)
        
        # combo box
        self.combo = QtGui.QComboBox()
        self.combo.addItems(items)
        self.combo.setCurrentIndex(self.combo.count() - 1)
        layout.addRow("File '{0}'".format(filename), self.combo)
        
        # buttons
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)
