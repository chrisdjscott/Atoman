
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
    
    boxesdeps = [os.path.join("..", "visclibs", "boxeslib.c"), 
                 os.path.join("..", "visclibs", "boxeslib.h")]
    utildeps = [os.path.join("..", "visclibs", "utilities.c"),
                os.path.join("..", "visclibs", "utilities.h")]
    arraydeps = [os.path.join("..", "visclibs", "array_utils.c"),
                 os.path.join("..", "visclibs", "array_utils.h")]
    
    # add extensions
    config.add_extension("picker", 
                         ["picker.c"],
                         include_dirs=[incdir],
                         depends=boxesdeps+utildeps+arraydeps,
                         libraries=["boxeslib", "utilities", "array_utils"])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
