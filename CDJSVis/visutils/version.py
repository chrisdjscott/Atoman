
"""
Module will return the current version of the code.

@author: Chris Scott

"""
import os
import sys
import subprocess


def getVersion():
    """
    Attempt to determine LKMC version.
    
    """
    # get current version
    if hasattr(sys, "_MEIPASS"):
        # if frozen get version from version_freeze.py
        try:
            import version_freeze
            version = version_freeze.__version__
        except:
            version = "vUNKNOWN"
    
    else:
        # otherwise get version from "git describe"
        CWD = os.getcwd()
        
        dn = os.path.dirname(__file__)
        if dn:
            os.chdir(dn)
        
        try:
            command = "git describe"
            proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            
            status = proc.poll()
            if status:
                version = "vUNKNOWN"
            else:
                version = stdout.strip()
        
        finally:
            os.chdir(CWD)
    
    return str(version)

if __name__ == "__main__":
    version = getVersion()
    if version is not None:
        print "CDJSVis %s" % (getVersion(),)
