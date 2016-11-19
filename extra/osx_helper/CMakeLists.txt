cmake_minimum_required(VERSION 3.1)
project(osx_helper)

find_library(LIB_COCOA cocoa)

add_library(osx_helper SHARED osx_helper.mm osx_helper.cxx)
target_link_libraries(osx_helper ${LIB_COCOA})
