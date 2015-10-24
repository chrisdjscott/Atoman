

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration("visclibs", parent_package, top_path)
    config.add_library("boxeslib", ["boxeslib.c"], depends=["boxeslib.h"])
    config.add_library("utilities", ["utilities.c"], depends=["utilities.h"])
    config.add_library("neb_list", ["neb_list.c", "boxeslib.c", "utilities.c"],
                       depends=["neb_list.h", "boxeslib.h", "utilities.h"])
    config.add_library("array_utils", ["array_utils.c"], depends=["array_utils.h"])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
