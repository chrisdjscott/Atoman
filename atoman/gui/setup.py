
import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, "..", "visclibs"))
    
    # config
    config = Configuration("gui", parent_package, top_path)
    
    # add subpackages
    config.add_subpackage("dialogs")
    config.add_subpackage("filterListOptions")
    
    # add extensions
    config.add_extension("picker", 
                         ["picker.c", "../visclibs/utilities.c",
                          "../visclibs/boxeslib.c", "../visclibs/array_utils.c"],
                         include_dirs=[incdir])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
