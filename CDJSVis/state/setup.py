
import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, "..", "visclibs"))
    
    # config
    config = Configuration("state", parent_package, top_path)
    
    config.add_extension("_input", 
                         ["input.c", "../visclibs/array_utils.c"],
                         include_dirs=[incdir])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
