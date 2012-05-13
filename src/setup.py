
"""
Setup script for CDJSVis.

Currently compiles resource file.

Eventually should compile C libraries too.

@author: Chris Scott

"""
import os
import glob


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
def main():
    
    print "IN SETUP SCRIPT"
    
    



################################################################################
if __name__ == "__main__":
    main()

