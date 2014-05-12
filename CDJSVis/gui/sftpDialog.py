
"""
The SFTP browser can be used to load files on a remote system that is running an SFTP
server.  To add a connection you must specify the hostname.  If you don't specify a
username your current username will be used.  You don't need to specify a password if you
have public/private keys set up.

Once a connection is established you can navigate through the browser by double clicking
or pressing "Open".  Opening a file will copy the file onto the local machine and attempt
to load that file.  It will also copy a Roulette file if one exists, so KMC information
can be visualised.

**Note**: when loading a file via SFTP the current working directory does not change.
Therefore, if you save an image or run the sequencer the output will be stored whatever
directory you are in.  You can change the current working directory using one of the
buttons on the toolbar at the top of the main window.

"""
import os
import sys
import stat
import getpass
import logging
import re
import errno

from PySide import QtGui
import paramiko
logging.getLogger("paramiko").setLevel(logging.WARNING)

from . import genericForm
from .. import resources
from ..visutils.utilities import iconPath

################################################################################

class SFTPConnectionDialog(QtGui.QDialog):
    """
    Dialog to get settings for connection
    
    """
    def __init__(self, parent=None):
        super(SFTPConnectionDialog, self).__init__(parent)
        
        self.setModal(1)
        self.setWindowTitle("SFTP connection settings")
        
        self.username = getpass.getuser()
        self.password = None
        self.hostname = None
        
        # layout
        layout = QtGui.QVBoxLayout(self)
        
        #TODO: access previous hosts in settings and add those as options
        
        # user name
        usernameLineEdit = QtGui.QLineEdit(self.username)
        usernameLineEdit.textEdited.connect(self.usernameEdited)
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Username"))
        row.addWidget(usernameLineEdit)
        layout.addLayout(row)
        
        # host name
        hostnameLineEdit = QtGui.QLineEdit()
        hostnameLineEdit.textEdited.connect(self.hostnameEdited)
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Hostname"))
        row.addWidget(hostnameLineEdit)
        layout.addLayout(row)
        
        # help label
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Leave password blank if using public key"))
        layout.addLayout(row)
        
        # password (can be blank)
        passwordLineEdit = QtGui.QLineEdit()
        passwordLineEdit.textEdited.connect(self.passwordEdited)
        row = QtGui.QHBoxLayout()
        row.addWidget(QtGui.QLabel("Password"))
        row.addWidget(passwordLineEdit)
        layout.addLayout(row)
        
        # buttons
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)
    
    def usernameEdited(self, text):
        """
        Username edited
        
        """
        text = str(text)
        if not len(text):
            self.username = None
        else:
            self.username = text
    
    def hostnameEdited(self, text):
        """
        Hostname edited
        
        """
        self.hostname = str(text)
    
    def passwordEdited(self, text):
        """
        Password edited
        
        """
        text = str(text)
        if not len(text):
            self.password = None
        else:
            self.password = text

################################################################################

class SFTPBrowserDialog(QtGui.QDialog):
    """
    SFTP browser dialog
    
    """
    def __init__(self, mainWindow, parent=None):
        super(SFTPBrowserDialog, self).__init__(parent)
        
        self.mainWindow = mainWindow
        self.logger = logging.getLogger(__name__)
        
        self.setModal(1)
        self.setWindowTitle("SFTP Browser")
#         self.setWindowIcon(QtGui.QIcon(iconPath("console-icon.png")))
        self.resize(800,600)
        
        # layout
        layout = QtGui.QVBoxLayout(self)
#         layout.setContentsMargins(0,0,0,0)
#         layout.setSpacing(0)
        
        # add connection
        self.connectionsCombo = QtGui.QComboBox()
        self.connectionsCombo.currentIndexChanged[int].connect(self.connectionChanged)
        addConnectionButton = QtGui.QPushButton(QtGui.QIcon(iconPath("list-add.svg")), "")
        addConnectionButton.setToolTip("Add new connection")
        addConnectionButton.setFixedWidth(35)
        addConnectionButton.clicked.connect(self.addNewConnection)
        helpButton = QtGui.QPushButton(QtGui.QIcon(iconPath("Help-icon.png")), "")
        helpButton.setFixedWidth(20)
        helpButton.setFixedHeight(20)
        helpButton.setToolTip("Show help page")
        helpButton.clicked.connect(self.loadHelpPage)
        row = QtGui.QHBoxLayout()
        row.addWidget(self.connectionsCombo)
        row.addWidget(addConnectionButton)
        row.addWidget(helpButton)
        layout.addLayout(row)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget()
        layout.addWidget(self.stackedWidget)
        
        # list of open connections
        self.connectionsList = []
        
        # buttons
        buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Open | QtGui.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        self.openButton = buttonBox.button(QtGui.QDialogButtonBox.Open)
        self.openButton.setEnabled(False)
        layout.addWidget(buttonBox)
        row = QtGui.QHBoxLayout()
        row.addStretch()
        row.addWidget(buttonBox)
        layout.addLayout(row)
    
    def loadHelpPage(self):
        """
        Load help page
        
        """
        self.mainWindow.helpWindow.loadPage("usage/input/sftp_browser.html")
        self.mainWindow.showHelp()
    
    def exec_(self, *args, **kwargs):
        """
        exec_ override
        
        """
        self.filename_remote = None
        self.filename_local = None
#         self.roulette_remote = None
#         self.roulette_local = None
        self.sftpPath = None
        
        return super(SFTPBrowserDialog, self).exec_(*args, **kwargs)
    
    def accept(self, *args, **kwargs):
        """
        Accept override
        
        """
        if self.stackedWidget.count() > 0:
            # check if file is selected
            browser = self.stackedWidget.currentWidget()
            item = browser.listWidget.currentItem()
            fn = str(item.text())
            if item is not None and not item.is_dir:
                self.logger.debug("Selecting item: '%s'", item.text())
                self.filename_remote = str(browser.sftp.normalize(fn))
                self.filename_source = None
                self.filename_local = os.path.join(self.mainWindow.tmpDirectory, "%s" % item.text())
                self.sftpPath = "%s:%s" % (browser.connectionID, self.filename_remote)
                
                # we also need to copy the file locally and store that path
                browser.copySystem(self.filename_remote, self.filename_local)
        
        return super(SFTPBrowserDialog, self).accept(*args, **kwargs)
    
    def addNewConnection(self):
        """
        Add new connection
        
        """
        self.logger.debug("Adding new connection")
        
        # first we pop up a dialog to get host info
        dlg = SFTPConnectionDialog(self)
        
        if dlg.exec_():
            # verify options
            if dlg.hostname is None or not len(dlg.hostname):
                self.logger.error("Must specify hostname for SFTP connection")
                return
            
            # identifier
            if dlg.username is None:
                connectionID = dlg.hostname
            else:
                connectionID = "%s@%s" % (dlg.username, dlg.hostname)
            
            if connectionID in self.connectionsList:
                self.logger.warning("Connection already open for '%s': not opening a new one", connectionID)
                return
            
            # test connection
            try:
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(dlg.hostname, username=dlg.username, password=dlg.password, look_for_keys=True, timeout=5)
                self.logger.debug("Test connection ok: '%s'", connectionID)
            
            except:
                self.logger.error("Test connection failed (%s): %r", connectionID, sys.exc_info())
                return
            
            # add widget
            w = SFTPBrowser(dlg.hostname, dlg.username, dlg.password, connectionID, parent=self)
            self.stackedWidget.addWidget(w)
            self.connectionsCombo.addItem(connectionID)
            self.connectionsList.append(connectionID)
            
            # change to this widget
            self.connectionsCombo.setCurrentIndex(self.connectionsCombo.count() - 1)
    
    def connectionChanged(self, index):
        """
        Connection changed
        
        """
        self.logger.debug("Connection changed: %d", index)
        self.stackedWidget.setCurrentIndex(index)

################################################################################

class SFTPBrowserListWidgetItem(QtGui.QListWidgetItem):
    """
    List widget item for SFTP Browser
    
    """
    def __init__(self, icon, text, is_dir):
        super(SFTPBrowserListWidgetItem, self).__init__(icon, text)
        self.is_dir = is_dir

################################################################################

class SFTPBrowser(genericForm.GenericForm):
    """
    Basic SFTP file browser
    
    """
    def __init__(self, hostname, username, password, connectionID, parent=None):
        self.hostname = hostname
        self.connectionID = connectionID
        self.username = username
        if self.username is None:
            self.username = getpass.getuser()

        super(SFTPBrowser, self).__init__(parent, None, "%s@%s" % (self.username, self.hostname))
        
        self.logger = logging.getLogger(__name__)
        
        # defaults
        self.connected = False
        self.currentFilter = 0
        
        # SETTINGS
        self.showHidden = False
        self.filters = [
            ("LBOMD files", ["*.dat", "*.dat.gz", "*.dat.bz2", "*.xyz", "*.xyz.gz", "*.xyz.bz2"]),
            ("All files", ["*"]),
        ]
        
        # regular expression for finding Roulette index
        self.intRegex = re.compile(r'[0-9]+')
        
        # current path label
        self.currentPathLabel = QtGui.QLabel("CWD: ''")
        row = self.newRow()
        row.addWidget(self.currentPathLabel)
        
        # list widget
        self.listWidget = QtGui.QListWidget(self)
        self.listWidget.itemDoubleClicked.connect(self.itemDoubleClicked)
        self.listWidget.currentRowChanged.connect(self.currentRowChanged)
        row = self.newRow()
        row.addWidget(self.listWidget)
        
        # filters
        filtersCombo = QtGui.QComboBox()
        filtersCombo.currentIndexChanged[int].connect(self.filtersComboChanged)
        for tup in self.filters:
            filtersCombo.addItem("%s (%s)" % (tup[0], " ".join(tup[1])))
        row = self.newRow()
        row.addWidget(filtersCombo)
        
        self.connect(password)
    
    def checkPathExists(self, path):
        """
        Check the given path exists (and normalize it)
        
        """
        self.logger.debug("Checking path exists: '%s'", path)
        
        try:
            pathn = self.sftp.normalize(path)
        
        except IOError, e:
            if e.errno == errno.ENOENT:
                self.logger.debug("File does not exist")
                return None
        
        else:
            self.logger.debug("File does exist: '%s'", pathn)
            return pathn
    
    def lookForRoulette(self, fn):
        """
        Look for linked Roulette file
        
        """
        rouletteFile = None
        localName = None
        
        # file name
        basename = os.path.basename(fn)
        self.logger.debug("Looking for Roulette file for: '%s'", basename)
        
        # look for integers in the name
        result = self.intRegex.findall(basename)
        if len(result):
            rouletteIndex = int(result[0]) - 1
            self.logger.debug("Found integer in filename: %d", rouletteIndex)
            
            # possible roulette file paths
            roulette_paths = [
                "Roulette%d.OUT" % rouletteIndex,
                "../Step%d/Roulette.OUT" % rouletteIndex,
            ]
            
            # check if one exists
            while rouletteFile is None and len(roulette_paths):
                test = roulette_paths.pop(0)
                rouletteFile = self.checkPathExists(test)
        
        if rouletteFile is not None:
            self.logger.debug("Found Roulette file: '%s'", rouletteFile)
            
            # local name must be "Roulette%d.OUT" % rouletteIndex for it to be picked up
            localName = "Roulette%d.OUT" % rouletteIndex
        
        return rouletteFile, localName
    
    def filtersComboChanged(self, index):
        """
        Filters combo changed
        
        """
        self.logger.debug("Filters combo changed: %d", index)
        self.currentFilter = index
        self.applyFilterToItems()
    
    def applyFilterToItems(self):
        """
        Apply filter to items
        
        """
        self.logger.debug("Applying filter to view")
        
        filters = self.filters[self.currentFilter][1]
        filters = [f[1:] for f in filters]
        self.logger.debug("Filter: %r", filters)
        
        # all visible
        if not len(filters):
            for i in xrange(1, self.listWidget.count()):
                item = self.listWidget.item(i)
                item.setHidden(False)
        
        # need to check
        else:
            for i in xrange(1, self.listWidget.count()):
                item = self.listWidget.item(i)
                if item.is_dir:
                    continue
            
                name = str(item.text())
            
                hidden = True
                for flt in filters:
                    if name.endswith(flt):
                        hidden = False
                        break
            
                item.setHidden(hidden)
    
    def connect(self, password):
        """
        Connect to server
        
        """
        # make ssh connection
        self.logger.debug("Opening connection: '%s@%s'", self.username, self.hostname)
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(self.hostname, username=self.username, password=password, look_for_keys=True)

        # home directory
        self.get_home()
        
        # open sftp session
        self.sftp = self.ssh.open_sftp()
        
        # change to home dir
        self.chdir(self.home)
        
        self.connected = True
    
    def disconnect(self):
        """
        Close session
        
        """
        self.logger.debug("Closing SFTP/SSH connection")
        self.sftp.close()
        self.ssh.close()
        
        self.connected = False
    
    def chdir(self, dirname):
        """
        Change directory
        
        """
        self.logger.debug("Changing to directory: '%s'", dirname)
        
        # change dir
        self.sftp.chdir(dirname)
        
        # set current path label
        self.currentPathLabel.setText("CWD: '%s'" % self.sftp.getcwd())
        
        # list directory
        self.list_dir()
    
    def list_dir(self):
        """
        List the working directory
        
        """
        self.logger.debug("Listing directory")
        
        # first clear the list widget
        self.listWidget.clear()
        
        # add ".." to top of listing
        item = SFTPBrowserListWidgetItem(QtGui.QIcon(iconPath("undo_64.png")), "..", True)
        self.listWidget.addItem(item)
        
        # add files and directories
        listdir = self.sftp.listdir()
        listdir.sort()
        for name in listdir:
            if name.startswith(".") and not self.showHidden:
                continue
            
            # file or directory
            statout = self.sftp.stat(name)
            if stat.S_ISDIR(statout.st_mode):
                # list widget item
                item = SFTPBrowserListWidgetItem(QtGui.QIcon(iconPath("folder.svg")), name, True)
            
            elif stat.S_ISREG(statout.st_mode):
                # list widget item
                item = SFTPBrowserListWidgetItem(QtGui.QIcon(iconPath("x-office-document.svg")), name, False)
            
            else:
                logging.warning("Item in directory listing is neither file nor directory: '%s (%s)'", name, statout.st_mode)
                continue
            
            # add to list widget
            self.listWidget.addItem(item)
        
        # apply selected filter
        self.applyFilterToItems()
    
    def itemDoubleClicked(self):
        """
        Item double clicked
        
        """
        item = self.listWidget.currentItem()
        if item is None:
            return
        
        if item.is_dir:
            self.logger.debug("Changing to directory: '%s'", item.text())
            self.chdir(item.text())
        
        else:
            self.parent.accept()
    
    def currentRowChanged(self, row):
        """
        Current row changed
        
        """
        item = self.listWidget.currentItem()
        if row < 0 or item.is_dir:
            self.parent.openButton.setEnabled(False)
        else:
            self.parent.openButton.setEnabled(True)
    
    def get_home(self):
        """
        Get home dir
    
        """
        # home directory
        stdin, stdout, stderr = self.ssh.exec_command("echo $HOME")
        self.home = stdout.readline().strip()
        stdin.close()
        stdout.close()
        stderr.close()
        self.logger.debug("HOME DIR: '%s'", self.home)
    
    def refresh(self):
        """
        Refresh view
        
        """
        self.logger.debug("Refreshing...")
    
    def copySystem(self, remotePath, localPath):
        """
        Copy the system onto the local machine.
        Also copy Roulette if that exists
        
        """
        self.logger.debug("Copying file: '%s' to '%s'", remotePath, localPath)
        self.sftp.get(remotePath, localPath)
        
        fn = os.path.basename(remotePath)
        
        # we also need to look for Roulette file
        if fn.endswith(".dat") or fn.endswith(".dat.gz") or fn.endswith(".dat.bz2"):
            self.logger.debug("Looking for Roulette file too")
            
            # roulette
            roulette_remote, roulette_local_bn = self.lookForRoulette(fn)
            if roulette_remote is not None:
                roulette_local = os.path.join(os.path.dirname(localPath), roulette_local_bn)
                self.logger.debug("Copying file: '%s' to '%s'", roulette_remote, roulette_local)
                self.sftp.get(roulette_remote, roulette_local)
    
    def closeEvent(self, event):
        """
        Close event
        
        """
        self.disconnect()
        super(SFTPBrowser, self).closeEvent(event)
