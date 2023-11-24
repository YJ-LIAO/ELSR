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
  # include_directories(third_party/eigen3.3.4/include/eigen3)
  find_package(Eigen3 REQUIRED)

  ##-----------------------------------------------------------------------------


  # 添加你的源文件
  set(SOURCE_FILES
          ELSR_visualSfM.cpp BasicMath.cpp Epipolar.cpp fastatan2.cpp FindHomography.cpp IO.cpp LSD.cpp PairMatch.cpp PairWiseLines.cpp ReadSfM.cpp ScoreRecorder.cpp SingleImage.cpp SpatialSimilarity.cpp
  )

  # 创建可执行文件
  add_executable(ELSR ${SOURCE_FILES})

  # 链接库
  target_include_directories(ELSR PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(ELSR ${Boost_LIBRARIES} ${OpenCV_LIBS} stlplus)

