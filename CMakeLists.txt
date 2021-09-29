cmake_minimum_required(VERSION 2.8.3)
project(maplab_features)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

##########
# EXPORT #
##########
install(DIRECTORY src config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
cs_export()
