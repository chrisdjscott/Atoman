
import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, "..", "..", "visclibs"))
    
    # config
    config = Configuration("filters", parent_package, top_path)
    
    config.add_extension("_acna", 
                         ["acna.c", "../../visclibs/utilities.c",
                          "../../visclibs/boxeslib.c", "../../visclibs/neb_list.c",
                          "../../visclibs/array_utils.c"],
                         libraries=["gsl", "gslcblas"],
                         include_dirs=[incdir, ".."])
    
    config.add_extension("_bond_order", 
                         ["bond_order.c", "../../visclibs/utilities.c",
                          "../../visclibs/boxeslib.c", "../../visclibs/neb_list.c",
                          "../../visclibs/array_utils.c"],
                         libraries=["gsl", "gslcblas"],
                         include_dirs=[incdir])
     
    config.add_extension("_defects", 
                         ["defects.c", "../../visclibs/utilities.c",
                          "../../visclibs/boxeslib.c", "../../visclibs/neb_list.c",
                          "../../visclibs/array_utils.c"],
                          libraries=["gsl", "gslcblas"],
                          include_dirs=[incdir, ".."])
     
    config.add_extension("_filtering", 
                         ["filtering.c", "../../visclibs/utilities.c",
                          "../../visclibs/boxeslib.c", "../../visclibs/array_utils.c"],
                          include_dirs=[incdir])
    
    config.add_extension("_bubbles", 
                         ["bubbles.c", "../../visclibs/neb_list.c",
                          "../../visclibs/array_utils.c", "../../visclibs/boxeslib.c",
                          "../../visclibs/utilities.c"],
                          libraries=[],
                          extra_compile_args=[],
                          include_dirs=[incdir, ".."])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
