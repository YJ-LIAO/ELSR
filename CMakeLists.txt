cmake_minimum_required(VERSION 2.6)
project(ELSR)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)

link_libraries(pthread)

# 寻找OpenCV包
##-----------------------------------------------------------------------------
## OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

##-----------------------------------------------------------------------------

# 寻找Boost库
## Boost
find_package(Boost REQUIRED COMPONENTS serialization filesystem system thread)
include_directories(${Boost_INCLUDE_DIRS})

##-----------------------------------------------------------------------------

# 添加额外库
## stlplus
SET(Stlplus3_SOURCES stlplus3/file_system.cpp stlplus3/portability_fixes.cpp stlplus3/wildcard.cpp)
ADD_LIBRARY(stlplus ${Stlplus3_SOURCES})

##-----------------------------------------------------------------------------

##-----------------------------------------------------------------------------
## Eigen
set(EIGEN3_ROOT_DIR "" CACHE PATH "include (root) directory of the Eigen lib")

set(EIGEN3_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3.3.4)

set(Eigen_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3.3.4)

set(CERES_EIGEN_VERSION 3.3.4)
set(Eigen3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3.3.4/share/eigen3/cmake)
set(CERES_EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(EIGEN3_VERSION 3.3.4)
set(Eigen_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3.3.4/share/eigen3/cmake)

if ("${EIGEN3_ROOT_DIR}" STREQUAL "")
  # Eigen directory not set
  message("Eigen3 include dir not set... trying to find it...")
  # FIND_PACKAGE(Eigen REQUIRED)
  find_package(Eigen3)
else ("${EIGEN3_ROOT_DIR}" STREQUAL "")
  message("Eigen3 include dir set to: " ${EIGEN3_ROOT_DIR})
  SET( EIGEN3_INCLUDE_DIR "${EIGEN3_ROOT_DIR}" )
endif ("${EIGEN3_ROOT_DIR}" STREQUAL "")
message("EIGEN3_INCLUDE_DIR dir set to: " ${EIGEN3_INCLUDE_DIR})
set(EIGEN3_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen3.3.4/include/eigen3)

# set(EIGEN_INCLUDE_DIR_HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
# find_package(Eigen QUIET)
# message("********EIGEN_INCLUDE_DIR: " ${EIGEN_INCLUDE_DIR})
# set(EIGEN3_INCLUDE_DIR ${EIGEN_INCLUDE_DIR})
# set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})
# include_directories(${EIGEN_INCLUDE_DIR})
# add_definitions(${EIGEN_DEFINITIONS} -D_USE_EIGEN)
# message("EIGEN3_INCLUDE_DIR dir set to: " ${EIGEN3_INCLUDE_DIR})
set (EXTRA_INC_DIRS
    ${EXTRA_INC_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}/unsupported
    ${EIGEN3_INCLUDE_DIR}/eigen3
    ${EIGEN3_INCLUDE_DIR}/Eigen
)

##-----------------------------------------------------------------------------


# 添加你的源文件
set(SOURCE_FILES
    ELSR_visualSfM.cpp BasicMath.cpp Epipolar.cpp fastatan2.cpp FindHomography.cpp IO.cpp LSD.cpp PairMatch.cpp PairWiseLines.cpp ReadSfM.cpp ScoreRecorder.cpp SingleImage.cpp SpatialSimilarity.cpp
)

# 创建可执行文件
add_executable(ELSR ${SOURCE_FILES})

# 链接库
target_link_libraries(ELSR ${Boost_LIBRARIES} ${OpenCV_LIBS} stlplus)

