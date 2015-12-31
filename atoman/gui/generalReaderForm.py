
"""
File input
==========

File input is handled by the *Load new system* box on the *Input* tab of the main toolbar.
Initially the file format will try to be determined automatically, however in the case of
ambiguity a dialog will popup asking the user for input.
File formats are defined in the *file_formats.IN* file (more on this and a link...).
Multiple files can be loaded at the same time by shift/cmd clicking them.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
import logging
import glob

from PySide import QtGui, QtCore

from ..visutils.utilities import iconPath, resourcePath
from ..system import latticeReaderGeneric
from six.moves import range
from six.moves import zip


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
        
        self.logger = logging.getLogger(__name__)
        
        vbox = QtGui.QVBoxLayout()
        
        # lattice reader
        self.latticeReader = latticeReaderGeneric.LatticeReaderGeneric(tmpLocation=self.tmpLocation, updateProgress=self.mainWindow.updateProgress, 
                                                                       hideProgress=self.mainWindow.hideProgressBar)
        
        # open dialog
        self.openLatticeButton = QtGui.QPushButton(QtGui.QIcon(iconPath('oxygen/document-open.png')), "File dialog")
        self.openLatticeButton.setToolTip("Open file dialog")
        self.openLatticeButton.setCheckable(0)
        self.openLatticeButton.clicked.connect(self.openFileDialog)
        hbox = QtGui.QHBoxLayout()
        hbox.setAlignment(QtCore.Qt.AlignHCenter)
        hbox.addWidget(self.openLatticeButton)
        
        # sftp browser
        if hasattr(self.loadSystemForm, "sftp_browser"):
            openSFTPBrowserButton = QtGui.QPushButton(QtGui.QIcon(iconPath('oxygen/document-open-remote.png')), "SFTP browser")
            openSFTPBrowserButton.setToolTip("Open SFTP browser")
            openSFTPBrowserButton.setCheckable(0)
            openSFTPBrowserButton.clicked.connect(self.openSFTPBrowser)
            hbox.addWidget(openSFTPBrowserButton)
        
        vbox.addLayout(hbox)
        
        self.setLayout(vbox)
        
        # file formats
        self.fileFormats = latticeReaderGeneric.FileFormats()
        
        # default file formats
        self.fileFormats.read()
    
    def openSFTPBrowser(self):
        """
        Open SFTP browser
        
        """
        if self.loadSystemForm.sftp_browser is None:
            self.mainWindow.displayError("Paramiko must be installed to use the SFTP browser")
            return
        
        self.logger.debug("Opening SFTP browser")
        
        ok = self.loadSystemForm.sftp_browser.exec_()
        if ok and len(self.loadSystemForm.sftp_browser.filename_remote):
            remotefns = self.loadSystemForm.sftp_browser.filename_remote
            localfns = self.loadSystemForm.sftp_browser.filename_local
            sftpPaths = self.loadSystemForm.sftp_browser.sftpPath
            
            for remotefn, localfn, sftpPath in zip(remotefns, localfns, sftpPaths):
                self.logger.info("Opening remote file (%s) on local machine", remotefn)
                self.logger.debug("Local filename: '%s'", localfn)
                self.logger.debug("SFTP path: '%s'", sftpPath)
                
                # read file
                status = self.openFile(localfn, sftpPath=sftpPath)
                
                # remove local copy
                self.latticeReader.cleanUnzipped(localfn, True)
            
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
        
        # open the dialog
        filenames = fdiag.getOpenFileNames(parent=self, caption="Open file", dir=os.getcwd(), options=QtGui.QFileDialog.DontResolveSymlinks)[0]
        filenames = [str(fn) for fn in filenames]
        
        if not len(filenames):
            return None
        
        QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for filename in filenames:
                nwd, filename = os.path.split(filename)        
                
                # change to new working directory
                if nwd != os.getcwd():
                    self.logger.info("Changing to directory: '{0}'".format(nwd))
                    os.chdir(nwd)
                    self.mainWindow.updateCWD()
                
                # open file
                result = self.openFile(filename)
                
                if result:
                    break
        
        finally:
            QtGui.QApplication.restoreOverrideCursor()
        
        return result
    
    def openFile(self, filename, rouletteIndex=None, sftpPath=None, linkedLattice=None):
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
        self.mainWindow.setStatus("Opening '%s'" % os.path.basename(filename))
        
        # unzip if required
        filepath, zipFlag = self.latticeReader.checkForZipped(filename)
        
        try:
            # file format
            fileFormat = self.determineFileFormat(filepath, filename)
            if fileFormat is None:
                return 1
            
            # linked lattice
            if fileFormat.linkedName is not None and linkedLattice is None:
                linkedLattice = self.getLinkedLattice(fileFormat, filename)
                if linkedLattice is None:
                    return 2

            # open file
            status, state = self.latticeReader.readFile(filepath, fileFormat, rouletteIndex=rouletteIndex, linkedLattice=linkedLattice)
        
        except:
            exctype, value = sys.exc_info()[:2]
            self.logger.exception("Lattice reader failed!")
            self.mainWindow.displayError("Lattice reader failed!\n\n%s: %s" % (exctype, value))
            status = 255
        
        finally:
            # delete unzipped file if required
            self.latticeReader.cleanUnzipped(filepath, zipFlag)
        
        if not status:
            self.postOpenFile(state, filename, fileFormat, sftpPath, linked=linkedLattice)
    
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
            QtGui.QApplication.restoreOverrideCursor()
            QtGui.QApplication.processEvents()
            dlg = SelectLinkedLatticeDialog(properName, items, parent=self)
            status = dlg.exec_()
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            QtGui.QApplication.processEvents()
            
            if status == QtGui.QDialog.Accepted:
                index = dlg.combo.currentIndex()
                latticeDisplayName, lattice = availableSystems[index]
                self.logger.debug("User selected linked lattice %d: '%s'", index, latticeDisplayName)
            
            else:
                lattice = None
        
        return lattice
    
    def postOpenFile(self, state, filename, fileFormat, sftpPath, linked=None):
        """
        Should always be called at the end of openFile.
        
        """
        self.loadSystemForm.fileLoaded(state, filename, fileFormat, sftpPath, linked)
    
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
            for count, line in enumerate(f):
                if count == maxIdLen:
                    break
                lines.append(line)
        
        if len(lines) < maxIdLen:
            self.logger.debug("Attempting to auto-detect short file; may be less accurate")
        
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
            n = min(len(identifier), len(lines))
            match = True
            for i in range(n):
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
            QtGui.QApplication.restoreOverrideCursor()
            QtGui.QApplication.processEvents()
            items = [fmt.name for fmt in matchedFormats]
            name, ok = QtGui.QInputDialog.getItem(self, "Select file format", "File '%s'" % properName, items, editable=False)
            QtGui.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            QtGui.QApplication.processEvents()
            
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
