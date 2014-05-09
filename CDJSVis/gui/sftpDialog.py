
"""
SFTP Browser dialog

@author: Chris Scott

"""
import os
import sys
import stat
import getpass
import logging

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
        self.resize(800,400)
        
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
        row = QtGui.QHBoxLayout()
        row.addWidget(self.connectionsCombo)
        row.addWidget(addConnectionButton)
        layout.addLayout(row)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget()
        layout.addWidget(self.stackedWidget)
        
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
    
    def exec_(self, *args, **kwargs):
        """
        exec_ override
        
        """
        self.filename_remote = None
        self.filename_local = None
        
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
                
                # we also need to copy the file locally and store that path
                browser.sftp.get(fn, self.filename_local)
                
                # we also need to look for Roulette file
                if fn.endswith(".dat") or fn.endswith(".dat.gz") or fn.endswith(".dat.bz2"):
                    self.logger.debug("Looking for Roulette file too")
                    
                    
        
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
            w = SFTPBrowser(dlg.hostname, dlg.username, dlg.password, parent=self)
            self.stackedWidget.addWidget(w)
            self.connectionsCombo.addItem(connectionID)
            
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
    def __init__(self, hostname, username, password, parent=None):
        self.hostname = hostname
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
        
        # layout
#         layout = QtGui.QVBoxLayout(self)
#         layout.setContentsMargins(0,0,0,0)
#         layout.setSpacing(0)
        
        # connection label
        self.connectedToLabel = QtGui.QLabel("Connected to: ''")
        row = self.newRow()
        row.addWidget(self.connectedToLabel)
        
        # current path label
        self.currentPathLabel = QtGui.QLabel("Current directory: ''")
        row = self.newRow()
        row.addWidget(self.currentPathLabel)
        
        # list widget
        self.listWidget = QtGui.QListWidget(self)
        self.listWidget.itemDoubleClicked.connect(self.itemDoubleClicked)
        self.listWidget.itemSelectionChanged.connect(self.itemSelectionChanged)
        row = self.newRow()
        row.addWidget(self.listWidget)
        
        # filters
        label = QtGui.QLabel("Filters:")
        filtersCombo = QtGui.QComboBox()
        filtersCombo.currentIndexChanged[int].connect(self.filtersComboChanged)
        for tup in self.filters:
            filtersCombo.addItem("%s (%s)" % (tup[0], " ".join(tup[1])))
        row = self.newRow()
        row.addWidget(label)
        row.addWidget(filtersCombo)
        
        # buttons (open, close?, ...)
#         openButton = QtGui.QPushButton("Open")
#         openButton.setAutoDefault(1)
#         openButton.clicked.connect(self.openButtonClicked)
#         buttonWidget = QtGui.QWidget()
#         buttonLayout = QtGui.QHBoxLayout(buttonWidget)
#         buttonLayout.addStretch()
#         buttonLayout.addWidget(openButton)
#         layout.addWidget(buttonWidget)
        
        self.connect(password)
    
    def itemSelectionChanged(self):
        """
        Item selection changed
        
        """
        item = self.listWidget.currentItem()
        print "ITEM CHANGED", item, item.text()
    
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
        
        # update connected to label
        self.connectedToLabel.setText("Connected to: '%s@%s'" % (self.username, self.hostname))
        
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
        self.currentPathLabel.setText("Current directory: '%s'" % self.sftp.getcwd())
        
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
    
    def openButtonClicked(self):
        """
        Open button clicked
        
        """
        item = self.listWidget.currentItem()
        if item is None:
            return
        
        self.logger.debug("Open button clicked: %s (dir=%r)", item.text(), item.is_dir)
    
    def itemDoubleClicked(self):
        """
        Item double clicked
        
        """
        item = self.listWidget.currentItem()
        if item is None:
            return
        
        self.logger.debug("Item double clicked: %s (dir=%r)", item.text(), item.is_dir)
        
        if item.is_dir:
            self.logger.debug("Changing to directory: '%s'", item.text())
            self.chdir(item.text())
    
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
    
    def closeEvent(self, event):
        """
        Close event
        
        """
        self.disconnect()
        super(SFTPBrowser, self).closeEvent(event)
