
import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, "..", "visclibs"))
    
    # config
    config = Configuration("rendering", parent_package, top_path)
    
    config.add_extension("_rendering", 
                         ["rendering.c"],
                         include_dirs=[incdir],
                         libraries=["array_utils"])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
