cmake_minimum_required(VERSION 2.8.3)
project(union_cloud)


find_package(catkin REQUIRED COMPONENTS
  message_generation
  geometry_msgs
  std_msgs
  sensor_msgs
  nav_msgs
  livox_ros_driver
)

add_message_files(
  DIRECTORY msg
  FILES
  union_cloud.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
  livox_ros_driver
)


catkin_package(
  CATKIN_DEPENDS 
  message_runtime 
  message_generation
  geometry_msgs 
  std_msgs
  nav_msgs
  sensor_msgs
  livox_ros_driver
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
