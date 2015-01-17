
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
        
        self.systemsDialog = parent
        self.mainToolbar = mainToolbar
        self.mainWindow = mainWindow
        
        self.logger = logging.getLogger(__name__)
        
        vbox = QtGui.QVBoxLayout()
        
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
        if hasattr(self.systemsDialog, "sftp_browser"):
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
        if self.fileFormatString is None:
            self.mainWindow.displayError("GenericReaderWidget: fileFormatString not set on:\n%s" % str(self))
            return None
        
        # if no file formats, pop up a dialog telling the user to make some/reset to the default formats
        
        
        fdiag = QtGui.QFileDialog()
         
        # temporarily remove stays on top hint on systems dialog
        sd = self.parent.parent
        sd.tmpHide()
        
        filenames = fdiag.getOpenFileNames(self, "Open file", os.getcwd())[0]
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
        print "OPEN FILE NOT IMPLEMENTED"
        
        # first we attempt to detect a format
        # if there are multiple possibilites we pop up a dialog to ask the user
        # (database of user responses, possibly including extension, to help order the options)
        # if there is a linked lattice type, check one is loaded, if not ask user to load one first,
        #   if 1 is loaded use it, if >1 then pop up a dialog with most recent at the top
        
        
        
        
    
    def determineFileFormat(self, filename):
        """
        Determine file format
        
        * loop over formats, matching to each one
        * if one, use it
        * if more, show dialog
        
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
