cmake_minimum_required(VERSION 2.8.3)
project(maplab_features)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

##########
# PYTHON #
##########

catkin_python_setup()

##########
# EXPORT #
##########
cs_install()
cs_export()
