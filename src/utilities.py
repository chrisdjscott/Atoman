
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

from PyQt4 import QtGui

import globalsModule


################################################################################
def resourcePath(relative):
    """
    Find path to given resource regardless of when running from within
    PyInstaller bundle or from command line.
    
    """
    return os.path.join(os.environ.get("_MEIPASS2", sys.path[0]), relative)


################################################################################
def iconPath(icon):
    """
    Return full path to given icon.
    
    """
    return os.path.join(":/icons", icon)


################################################################################
def helpPath(page):
    """
    Return full path to given help page.
    
    """
    return os.path.join(":/help", page)


################################################################################
def idGenerator(size=16, chars=string.digits + string.ascii_letters + string.digits):
    """
    Generate random string of size "size" (defaults to 16)
    
    """
    return ''.join(random.choice(chars) for x in range(size))


################################################################################
def createTmpDirectory():
    """
    Create temporary directory
    
    """
    name = "CDJSVis-" + idGenerator(size=8)
    try:
        tmpDir = os.path.join("/tmp", name)
        while os.path.exists(tmpDir):
            name = "CDJSVis-" + idGenerator(size=8)
            tmpDir = os.path.join("/tmp", name)
        os.mkdir(tmpDir)
    except:
        tmpDir = os.path.join(os.getcwd(), name)
        while os.path.exists(tmpDir):
            name = "CDJSVis-" + idGenerator(size=8)
            tmpDir = os.path.join(os.getcwd(), name)
    
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
    QtGui.QMessageBox.warning(parent, "Warning", "Could not locate '%s' executable!" % (exe,))


################################################################################
def checkForExe(exe):
    """
    Check if executable can be located 
    
    """
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
        for syspath in globalsModule.PATH:
            if os.path.join(syspath, exe):
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
        for syspath in globalsModule.PATH:
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
    
    return (output, stderr, status)

