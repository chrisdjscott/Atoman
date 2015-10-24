
import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, "..", "..", "visclibs"))
    
    # config
    config = Configuration("filters", parent_package, top_path)
    
    config.add_extension("_acna", 
                         ["acna.c"],
                         include_dirs=[incdir, ".."],
                         libraries=["boxeslib", "utilities", "neb_list", "array_utils"])
    
    config.add_extension("_bond_order", 
                         ["bond_order.c"],
                         include_dirs=[incdir],
                         libraries=["boxeslib", "utilities", "neb_list", "array_utils"])
     
    config.add_extension("_defects", 
                         ["defects.c"],
                          include_dirs=[incdir, ".."],
                          libraries=["boxeslib", "utilities", "neb_list", "array_utils"])
     
    config.add_extension("_filtering", 
                         ["filtering.c"],
                          include_dirs=[incdir],
                          libraries=["boxeslib", "utilities", "array_utils"])
    
    config.add_extension("_bubbles", 
                         ["bubbles.c"],
                          include_dirs=[incdir, ".."],
                          libraries=["boxeslib", "utilities", "neb_list", "array_utils"])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
