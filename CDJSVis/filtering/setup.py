
import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, "..", "visclibs"))
    
    # config
    config = Configuration("filtering", parent_package, top_path)
    
    config.add_subpackage("filters")
    
    config.add_extension("bonds", 
                         ["bonds.c", "../visclibs/utilities.c",
                          "../visclibs/boxeslib.c", "../visclibs/array_utils.c"],
                          include_dirs=[incdir])
    
    config.add_extension("_clusters", 
                         ["clusters.c", "../visclibs/utilities.c",
                          "../visclibs/boxeslib.c", "../visclibs/array_utils.c"],
                          include_dirs=[incdir])
    
    config.add_extension("_defects", 
                         ["defects.c", "../visclibs/utilities.c",
                          "../visclibs/boxeslib.c", "../visclibs/neb_list.c",
                          "../visclibs/array_utils.c"],
                          libraries=["gsl", "gslcblas", "gomp"],
                          extra_compile_args=["-fopenmp"],
                          include_dirs=[incdir])
    
    config.add_extension("_voronoi", 
                         ["voronoi.c", "voro_iface.cpp", 
                          "voro++/src/voro++.cc",
                          "../visclibs/array_utils.c"],
                          libraries=[],
                          include_dirs=[incdir, "voro++/src"])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
