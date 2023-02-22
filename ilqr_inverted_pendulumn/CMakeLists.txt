cmake_minimum_required(VERSION 3.0.2)
project(ilqr_inverted_pendulumn)
add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE "Release")

set(pinocchio_DIR /opt/openrobots/lib/cmake/pinocchio)
set(CASADI_DIR /usr/local/lib/cmake/casadi)
set(PYTHON_INCLUDE_DIRS /usr/include/python3.8)
set(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython3.8.so)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  pinocchio REQUIRED
  CASADI REQUIRED
#  PythonLibs REQUIRED
)

catkin_package(
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PINOCCHIO_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PYTHON_INCLUDE_DIRS}
)

add_library(loss_function src/loss_function.cpp)

add_executable(SLQ src/SLQ.cpp)
target_link_libraries(SLQ loss_function casadi ${PINOCCHIO_LIBRARIES} ${PYTHON_LIBRARIES})

add_executable(cartpole_test src/cartpole_test.cpp)
target_link_libraries(cartpole_test ${PYTHON_LIBRARIES} ${catkin_LIBRARIES} ${PINOCCHIO_LIBRARIES})
