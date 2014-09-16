
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration("visclibs", parent_package, top_path)
    config.add_extension("acna", 
                         ["acna.c", "utilities.c",
                          "boxeslib.c", "neb_list.c",
                          "array_utils.c"],
                         libraries=["gsl", "gslcblas"],
                         include_dirs=["/opt/local/include"])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"