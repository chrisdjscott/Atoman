
"""
Utility methods

@author: Chris Scott

"""
import os
import sys
import random
import string
import glob
import subprocess
import tempfile
import logging

from PySide import QtGui, QtCore

from .appdirs import appdirs

# set where to look for executables when os.getenv("PATH")
# returns something less than expected (eg when running
# pyinstaller executable)
EXTENDED_PATH = [
    "/opt/local/bin",
    "/usr/local/bin",
]

################################################################################

class TextEditHandler(logging.Handler):
    """
    Logging handler than outputs to a QTextEdit
    
    """
    def __init__(self, textEdit):
        super(TextEditHandler, self).__init__()
        
        # text edit
        self.textEdit = textEdit
    
    def emit(self, record):
        """
        Write log message
         
        """
        # format the record
        record.message = self.format(record)
        
        # choose colour based on level
        level = record.levelno
        
        if level >= 50:
            colour = QtCore.Qt.cyan
        elif level >= 40:
            colour = QtCore.Qt.red
        elif level >= 30:
            colour = QtCore.Qt.magenta
        elif level >= 20:
            colour = QtCore.Qt.black
        elif level >= 10:
            colour = QtCore.Qt.blue
        else:
            colour = QtCore.Qt.gray
        
        # set colour
        self.textEdit.setTextColor(colour)
        
        # add message
        self.textEdit.append("> %s" % record.message)

################################################################################

def log_error(message):
    """
    Write an error message to stderr
    
    """
    sys.stderr.write(message)
    sys.stderr.flush()

################################################################################
def resourcePath(relative, dirname="data"):
    """
    Find path to given resource regardless of when running from within
    PyInstaller bundle or from command line.
    
    """
    # first look in pyinstaller bundle
    if hasattr(sys, "_MEIPASS"):
        path = os.path.join(sys._MEIPASS, dirname)
    
    else:
        # then look in py2app bundle
        path = os.environ.get("RESOURCEPATH", None)
        if path is None:
            # then look in source code directory
            path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), dirname)
    
    path = os.path.join(path, relative)
    
    return path

################################################################################
# data storage directory
_dataDir = appdirs.user_data_dir(appname="Atoman", appauthor="chrisdjscott")

################################################################################
def dataPath(relative):
    """Path within data directory."""
    return os.path.join(_dataDir, relative)

################################################################################
def createDataFile(relative, text=""):
    """Create the given path within the data directory with the given text."""
    # create the directory if it doesn't exist
    if not os.path.exists(_dataDir):
        os.makedirs(_dataDir, 0755)
    
    # check if there are subdirectories within relative
    head, tail = os.path.split(relative)
    
    # directory file is going in
    dirname = os.path.join(_dataDir, head)
    if not os.path.exists(dirname):
        os.makedirs(dirname, 0755)
    
    # write file
    with open(os.path.join(dirname, tail), "w") as fh:
        fh.write(text)

################################################################################
def imagePath(image):
    """
    Return full path to given image.
    
    """
    return os.path.join(":/images", image)

################################################################################
def iconPath(icon):
    """
    Return full path to given icon.
    
    """
    return resourcePath(icon, dirname="icons")

################################################################################
def helpPath(page):
    """
    Return full path to given help page.
    
    """
    return resourcePath(page, dirname="doc")

################################################################################
def idGenerator(size=16, chars=string.digits + string.ascii_letters + string.digits):
    """
    Generate random string of size "size" (defaults to 16)
    
    """
    return ''.join(random.choice(chars) for _ in xrange(size))

################################################################################
def createTmpDirectory():
    """
    Create temporary directory
    
    """
    # force /tmp as POV-Ray seems to require it
    if os.path.isdir("/tmp"):
        dirname = "/tmp"
    else:
        dirname = None
    
    tmpDir = tempfile.mkdtemp(prefix="atoman-", dir=dirname)
    
    return tmpDir

################################################################################
def checkForFile(filename):
    
    found = 0
    if os.path.exists(filename):
        found = 1
    
    else:
        if os.path.exists(filename + '.bz2'):
            found = 1
        
        elif os.path.exists(filename + '.gz'):
            found = 1
            
    return found


################################################################################
def warnExeNotFound(parent, exe):
    """
    Warn that an executable was not located.
    
    """
#     QtGui.QMessageBox.warning(parent, "Warning", "Could not locate '%s' executable!" % (exe,))
    
    message = "Could not locate '%s' executable!" % exe
    
    msgBox = QtGui.QMessageBox(parent)
    msgBox.setText(message)
    msgBox.setWindowFlags(msgBox.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
    msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
    msgBox.setIcon(QtGui.QMessageBox.Warning)
    msgBox.exec_()

################################################################################
def checkForExe(exe):
    """
    Check if executable can be located 
    
    """
    exepath = None
    
    # first check if we've been given an absolute path
    if len(os.path.split(exe)[0]):
#         print "CHECK FOR EXE ABS PATH", exe
        
        if os.path.exists(exe):
            exepath = exe
        
        else:
            # basename
            exe = os.path.basename(exe)
#             print "SEARCHING FOR BASENAME IN SYS PATH", exe
    
    if exepath is None:
        # check if exe programme located
        syspath = os.getenv("PATH", "")
        syspatharray = syspath.split(":")
        found = 0
        for syspath in syspatharray:
            if os.path.exists(os.path.join(syspath, exe)):
                found = 1
                break
    
        if found:
            exepath = exe
    
        else:
            for syspath in EXTENDED_PATH:
                if os.path.exists(os.path.join(syspath, exe)):
                    found = 1
                    break
        
            if found:
                exepath = os.path.join(syspath, exe)
        
            else:
                exepath = 0
    
    return exepath


################################################################################
def checkForExeGlob(exe):
    """
    Check if executable can be located 
    
    """
    # check if exe programme located
    syspath = os.getenv("PATH", "")
    syspatharray = syspath.split(":")
    found = 0
    for syspath in syspatharray:
        matches = glob.glob(os.path.join(syspath, exe))
        if len(matches):
            found = 1
            break
    
    if found:
        exepath = matches[0]
    
    else:
        for syspath in EXTENDED_PATH:
            matches = glob.glob(os.path.join(syspath, exe))
            if len(matches):
                found = 1
                break
        
        if found:
            exepath = matches[0]
        
        else:
            exepath = 0
    
    return exepath


################################################################################
def runSubProcess(command, verbose=0):
    """
    Run command using subprocess module.
    Return tuple containing STDOUT, STDERR, STATUS
    Caller can decide what to do if status is true
    
    """
    if verbose:
        print command
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, stderr = process.communicate()
    status = process.poll()
    
    return output, stderr, status

################################################################################
def runSubprocessInThread(command, resultQ, verbose=False):
    """
    Run subprocess in separate thread
    
    """
    result = runSubProcess(command, verbose=verbose)
    resultQ.put(result)

################################################################################
def simulationTimeLine(simTimeInFs):
    """
    Scales simulation time and returns line including units.
    
    """
    if simTimeInFs > 1.0E27:
        # years
        simTime = "%.3f years" % ((simTimeInFs / 1.0E15) / 31557600.0,)
    
    elif simTimeInFs > 1.0E24:
        # days
        simTime = "%.3f days" % ((simTimeInFs / 1.0E15) / 86400.0,)
    
    elif simTimeInFs > 1.0E21:
        # hours
        simTime = "%.3f hours" % ((simTimeInFs / 1.0E15) / 3600.0,)
    
    elif simTimeInFs > 1.0E18:
        # minutes
        simTime = "%.3f mins" % ((simTimeInFs / 1.0E15) / 60.0,)
    
    elif simTimeInFs > 1.0E15:
        simTime = "%.3f s" % (simTimeInFs / 1.0E15,)
    
    elif simTimeInFs > 1.0E12:
        simTime = "%.3f ms" % (simTimeInFs / 1.0E12,)
    
    elif simTimeInFs > 1.0E9:
        simTime = "%.3f us" % (simTimeInFs / 1.0E9,)
    
    elif simTimeInFs > 1.0E6:
        simTime = "%.3f ns" % (simTimeInFs / 1.0E6,)
    
    elif simTimeInFs > 1.0E3:
        simTime = "%.3f ps" % (simTimeInFs / 1.0E3,)
    
    else:
        simTime = "%.3f fs" % (simTimeInFs,)
    
    return simTime

################################################################################
def getTimeFromRoulette(rouletteIndex, testpath=None):
    """
    Attempt to get time from KMC Roulette file.
    
    """
    fn = None
    if os.path.exists("Roulette%d.OUT" % rouletteIndex):
        fn = "Roulette%d.OUT" % rouletteIndex
    
    elif os.path.exists("../Step%d/Roulette.OUT" % rouletteIndex):
        fn = "../Step%d/Roulette.OUT" % rouletteIndex
    
    elif testpath is not None:
        testfn = os.path.join(testpath, "Roulette%d.OUT" % rouletteIndex)
        if os.path.exists(testfn):
            fn = testfn
    
    timeInFs = None
    if fn is not None:
        f = open(fn)
        
        for line in f:
            if line[:15] == "Simulation time":
                array = line.split()
                timeStr = array[2]
                timeInS = float(timeStr)
                timeInFs = timeInS * 1e15
    
    return timeInFs
    
################################################################################
def getBarrierFromRoulette(rouletteIndex, testpath=None):
    """
    Attempt to get barrier from roulette.
    
    """
    fn = None
    if os.path.exists("Roulette%d.OUT" % rouletteIndex):
        fn = "Roulette%d.OUT" % rouletteIndex
    
    elif os.path.exists("../Step%d/Roulette.OUT" % rouletteIndex):
        fn = "../Step%d/Roulette.OUT" % rouletteIndex
    
    elif testpath is not None:
        testfn = os.path.join(testpath, "Roulette%d.OUT" % rouletteIndex)
        if os.path.exists(testfn):
            fn = testfn
    
    barrier = None
    if fn is not None:
        f = open(fn)
        
        for line in f:
            line = line.strip()
            if line[:7] == "Barrier":
                array = line.split()
                barrier = float(array[1])
                break
    
    return barrier


