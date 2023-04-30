cmake_minimum_required(VERSION 2.8.3)
project(mm_loam)

SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
  	     message_generation
  	     geometry_msgs
  	     nav_msgs
  	     sensor_msgs
  	     roscpp
  	     rospy
  	     std_msgs
  	     tf
  	     tf_conversions
  	     rosbag
  	     livox_ros_driver
  	     eigen_conversions
  	     pcl_conversions
  	     pcl_ros
  	     message_filters
  	     std_srvs)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)

add_message_files(
	DIRECTORY msg
	FILES
	union_cloud.msg
)

include_directories(
		include
		${catkin_INCLUDE_DIRS}
		${EIGEN3_INCLUDE_DIR}
		${PCL_INCLUDE_DIRS}
		${CERES_INCLUDE_DIRS}
		${OpenCV_INCLUDE_DIRS}
		${SUITESPARSE_INCLUDE_DIRS}
	)

generate_messages(DEPENDENCIES std_msgs nav_msgs sensor_msgs livox_ros_driver)
catkin_package(CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime livox_ros_driver
  	       pcl_conversions pcl_ros message_filters std_srvs tf_conversions tf
	       eigen_conversions DEPENDS PCL OpenCV INCLUDE_DIRS include)

add_executable(mm_lidarsAligner src/unionLidarsAligner.cpp)
target_link_libraries(mm_lidarsAligner ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable(mm_scanRegistration src/unionFeatureExtract.cpp)
target_link_libraries(mm_scanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})

add_executable (mm_PoseEstimation
	src/unionPoseEstimation.cpp
	src/lio/Estimator.cpp
	src/lio/IMUIntegrator.cpp
	src/lio/ceresfunc.cpp
	src/lio/Map_Manager.cpp)
target_link_libraries(mm_PoseEstimation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

