Header header

# Velo Point cloud messages
sensor_msgs/PointCloud2 velo_time_aligned
sensor_msgs/PointCloud2 velo_corner
sensor_msgs/PointCloud2 velo_surface
sensor_msgs/PointCloud2 velo_combine
int64 velo_corner_num
int64 velo_surf_num


# LivoxPoint cloud messages
livox_ros_driver/CustomMsg livox_time_aligned
sensor_msgs/PointCloud2 livox_corner
sensor_msgs/PointCloud2 livox_surface
sensor_msgs/PointCloud2 livox_combine
int64 livox_corner_num
int64 livox_surf_num


#
bool livo_degenerated_env