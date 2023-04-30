// The key objective of this code is to align the spinning lidar
// and livox code both on position and time side.
// and publish to a new topic, which the timestamp
// are same and under the same coordinate as livox_frame;
//  Qingqing Li @uTU

#include <time.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include "livox_ros_driver/CustomMsg.h"
#include "lidars_extrinsic_cali.h"
#include "union_om/union_cloud.h"

#include <fstream>
#include <chrono>
#include <string>
#include <Eigen/Dense>
#include <mutex>
#include <stdint.h>

#include "sophus/so3.hpp"

typedef uint64_t uint64;
typedef pcl::PointXYZINormal PointNormal;

class LidarsParamEstimator{
    private:
        ros::NodeHandle nh;
        // subscribe raw data
        ros::Subscriber sub_hori;
        ros::Subscriber sub_velo;
        ros::Subscriber sub_imu;

        // publish points shared FOV with Horizon
        ros::Publisher pub_velo_inFOV;

        // pub extrinsic aligned data
        ros::Publisher pub_hori;
        ros::Publisher pub_velo;
        ros::Publisher pub_hori_livoxmsg;
        // pub time_offset compensated data
        ros::Publisher pub_time_hori;
        ros::Publisher pub_time_velo;
        ros::Publisher pub_time_hori_livoxmsg;
        ros::Publisher pub_time_union_cloud;

        ros::Publisher pub_merged_cloud;

        // Hori TF
        int                         _hori_itegrate_frames = 5;
        pcl::PointCloud<PointType>  _hori_igcloud;
        Eigen::Matrix4f             _velo_hori_tf_matrix;
        bool                        _hori_tf_initd         = false;
        bool                        _first_velo_reveived= false;
        int                         _cut_raw_message_pieces = 2;

        // real time angular yaw speed
        double                      _yaw_velocity;

        pcl::PointCloud<PointType> _velo_new_cloud ;

        // raw message queue for time_offset
        // std::queue< sensor_msgs::PointCloud2 >          _velo_queue;
        std::vector<sensor_msgs::PointCloud2 >          _velo_queue;
        std::queue< pcl::PointCloud<pcl::PointXYZI> >   _velo_fov_queue;
        std::queue< livox_ros_driver::CustomMsg >       _hori_queue;
        std::vector<float>                              _hori_msg_yaw_vec;
        std::vector<sensor_msgs::Imu::ConstPtr>         _imu_vec;
        std::mutex _mutexIMUVec;
        std::mutex _mutexHoriQueue;
        std::mutex _mutexVeloQueue;

        uint64                                      _hori_start_stamp ;
        bool                                        _first_hori = true;
        std::mutex _mutexHoriPointsQueue;
        // std::queue<livox_ros_driver::CustomPoint>   _hori_points_queue;
        // std::queue<uint64>                          _hori_points_stamp_queue;
        std::vector<livox_ros_driver::CustomPoint>   _hori_points_queue;
        std::vector<uint64>                          _hori_points_stamp_queue;


        bool        _time_offset_initd    = false;
        double      _time_esti_error_th   = 400.0;
        uint64      _time_offset          = 0; //secs (velo_stamp - hori_stamp)


        // Parameter
        bool    en_timeoffset_esti         = true;   // enable time_offset estimation
        bool    en_extrinsic_esti          = true;   // enable extrinsic estimation
        bool    en_timestamp_align         = true;
        bool    _use_given_extrinsic_lidars  = false;  // using given extrinsic parameter
        bool    _use_given_timeoffset       = true;   // using given timeoffset estimation
        float   _time_start_yaw_velocity   = 0.5;    // the angular speed when time offset estimation triggered
        int     _offset_search_resolution    = 30;     // points
        int     _offset_search_sliced_points = 12000;     // points
        float   given_timeoffset          = 0;      // the given time-offset value

        // Distortion
        double _last_imu_time = 0.0;
        Eigen::Quaterniond _delta_q;

        uint64 _last_search_stamp = 0;

        double time_cost = 0;
        int processed_cnt = 0;

    public:
        LidarsParamEstimator()
        {
            sub_velo = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1000, &LidarsParamEstimator::velo_cloud_handler, this);
            sub_hori = nh.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar", 100, &LidarsParamEstimator::hori_cloud_handler, this);
            sub_imu  = nh.subscribe("/livox/imu", 2000,  &LidarsParamEstimator::imu_handler, this);
            pub_hori        = nh.advertise<sensor_msgs::PointCloud2>("/a_horizon", 1);
            pub_hori_livoxmsg = nh.advertise<livox_ros_driver::CustomMsg>("/a_horizon_livoxmsg", 1);
            pub_velo        = nh.advertise<sensor_msgs::PointCloud2>("/a_velo", 1);
            pub_time_hori   = nh.advertise<sensor_msgs::PointCloud2>("/a_time_hori", 1);
            pub_time_velo   = nh.advertise<sensor_msgs::PointCloud2>("/a_time_velo", 1);
            pub_velo_inFOV  = nh.advertise<sensor_msgs::PointCloud2>("/velo_fov_cloud", 1);
            pub_merged_cloud = nh.advertise<sensor_msgs::PointCloud2>("/merged_cloud", 1);
            pub_time_hori_livoxmsg = nh.advertise<livox_ros_driver::CustomMsg>("/a_time_hori_livoxmsg", 1);

            pub_time_union_cloud = nh.advertise<union_om::union_cloud>("/a_time_union_cloud", 1);

            // Get parameters
            ros::NodeHandle private_nh_("~");
            if (!private_nh_.getParam("enable_extrinsic_estimation",  en_extrinsic_esti))       en_extrinsic_esti = true;
            if (!private_nh_.getParam("enable_timeoffset_estimation", en_timeoffset_esti))      en_timeoffset_esti = true;
            if (!private_nh_.getParam("extri_esti_hori_integ_frames", _hori_itegrate_frames))    _hori_itegrate_frames = 1;
            if (!private_nh_.getParam("time_esti_error_threshold",    _time_esti_error_th))     _time_esti_error_th = 500.0;
            if (!private_nh_.getParam("give_extrinsic_Velo_to_Hori",  _use_given_extrinsic_lidars))   _use_given_extrinsic_lidars = false;
            if (!private_nh_.getParam("time_esti_start_yaw_velocity", _time_start_yaw_velocity))     _time_start_yaw_velocity = 0.5;
            if (!private_nh_.getParam("give_timeoffset_Velo_to_Hori", _use_given_timeoffset))        _use_given_timeoffset = false;
            if (!private_nh_.getParam("timeoffset_Velo_to_Hori",      given_timeoffset))            given_timeoffset = 0.0;
            if (!private_nh_.getParam("timeoffset_search_resolution", _offset_search_resolution))    _offset_search_resolution = 30;
            if (!private_nh_.getParam("timeoffset_search_sliced_points", _offset_search_sliced_points)) _offset_search_sliced_points = 12000;

            if (!private_nh_.getParam("cut_raw_Hori_message_pieces", _cut_raw_message_pieces)) _cut_raw_message_pieces = 1;

            ROS_INFO_STREAM( "enable_timeoffset_estimation    : " << en_timeoffset_esti );
            ROS_INFO_STREAM( "enable_extrinsic_estimation     : " << en_extrinsic_esti );
            ROS_INFO_STREAM( "extri_esti_hori_integ_frames    : " << _hori_itegrate_frames );
            ROS_INFO_STREAM( "time_esti_error_threshold       : " << _time_esti_error_th );
            ROS_INFO_STREAM( "give_extrinsic_Velo_to_Hori     : " << _use_given_extrinsic_lidars );
            ROS_INFO_STREAM( "time_esti_start_yaw_velocity    : " <<  _time_start_yaw_velocity );
            ROS_INFO_STREAM( "give_timeoffset_Velo_to_Hori    : " <<  _use_given_timeoffset );
            ROS_INFO_STREAM( "timeoffset_Velo_to_Hori         : " <<  given_timeoffset );
            ROS_INFO_STREAM( "timeoffset_search_resolution    : " <<  _offset_search_resolution );
            ROS_INFO_STREAM( "timeoffset_search_sliced_points : " <<  _offset_search_sliced_points );
            ROS_INFO_STREAM( "cut_raw_Hori_message_pieces     : " <<  _cut_raw_message_pieces );

            std::vector<double> vecVeloHoriExtri;
            if ( _use_given_extrinsic_lidars && private_nh_.getParam("Extrinsic_Velohori", vecVeloHoriExtri )){
                _velo_hori_tf_matrix <<    vecVeloHoriExtri[0], vecVeloHoriExtri[1], vecVeloHoriExtri[2], vecVeloHoriExtri[3],
                                    vecVeloHoriExtri[4], vecVeloHoriExtri[5], vecVeloHoriExtri[6], vecVeloHoriExtri[7],
                                    vecVeloHoriExtri[8], vecVeloHoriExtri[9], vecVeloHoriExtri[10], vecVeloHoriExtri[11],
                                    vecVeloHoriExtri[12], vecVeloHoriExtri[13], vecVeloHoriExtri[14], vecVeloHoriExtri[15];
                _hori_tf_initd = true;
                ROS_INFO_STREAM("Reveived transformation_matrix Velo-> Hori: \n" << _velo_hori_tf_matrix );
            }

            if(_use_given_timeoffset)
            {
                ROS_INFO_STREAM("Given time offset " << given_timeoffset);
                ros::Time tmp_stamp;
                _time_offset = tmp_stamp.fromSec(given_timeoffset).toNSec();
                _time_offset_initd = true;
            }

        };

        ~LidarsParamEstimator(){};

        /**
         * @brief subscribe raw pointcloud message from Livox lidar and process the data.
         * - save the first timestamp of first message to init the timestamp
         * - Undistort pointcloud based on rotation from IMU
         * - If TF is not initlized,  Push the current undistorted message and yaw to queue;
         * - If TF is not intilized,  align two pointclouds with ICP after integrating enough frames
         * - If TF has been initized, publish aligned cloud in Horizon frame-id
         */
        void hori_cloud_handler(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in)
        {
            auto tick = std::chrono::high_resolution_clock::now();
            if(!_first_velo_reveived) return; // to make sure we have velo cloud to match

            if(_first_hori){
                _hori_start_stamp = livox_msg_in->timebase; // global hori message time_base
                ROS_INFO_STREAM("Update _hori_start_stamp :" << _hori_start_stamp);
                _first_hori = false;
            }

            livox_ros_driver::CustomMsg livox_msg_in_distort(*livox_msg_in);
            // RemoveLidarDistortion( livox_msg_in, livox_msg_in_distort);

            //#### push to queue to aligh timestamp
            std::unique_lock<std::mutex> lock_hori(_mutexHoriQueue);
            _hori_queue.push(livox_msg_in_distort);
            _hori_msg_yaw_vec.push_back(_yaw_velocity);
            // if( _hori_queue.size() > 5 ) _hori_queue.pop();
            lock_hori.unlock();
            // if(_yaw_velocity > 0.6)
            //     std::cout << "Current yaw:  "<< _yaw_velocity << std::endl;

            // ###$ integrate more msgs to get extrinsic transform
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointCloudIn(new  pcl::PointCloud<pcl::PointXYZI>);
            livoxToPCLCloud(livox_msg_in_distort, *pointCloudIn, _cut_raw_message_pieces);
            if(_hori_itegrate_frames > 0 && !_hori_tf_initd )
            {
                _hori_igcloud += *pointCloudIn;
                _hori_itegrate_frames--;
                ROS_INFO_STREAM("hori cloud integrating: " << _hori_itegrate_frames);
                return;
            }
            else
            {
                // Calibrate the Lidar first
                if(!_hori_tf_initd && en_extrinsic_esti){
                    Eigen::AngleAxisf init_rot_x( 0.0 , Eigen::Vector3f::UnitX());
                    Eigen::AngleAxisf init_rot_y( 0.0 , Eigen::Vector3f::UnitY());
                    Eigen::AngleAxisf init_rot_z( 0.0 , Eigen::Vector3f::UnitZ());

                    Eigen::Translation3f init_trans(0.0,0.0,0.0);
                    Eigen::Matrix4f init_tf = (init_trans * init_rot_z * init_rot_y * init_rot_x).matrix();
                    Eigen::Matrix4f hori_tf_matrix;
                    // pcl::transformPointCloud (full_cloud , cloud_out, transformation_matrix);
                    ROS_INFO("\n\n\n  Calibrate Horizon ...");
                    calibratePCLICP(_hori_igcloud.makeShared(), _velo_new_cloud.makeShared(), hori_tf_matrix, true);
                    // Eigen::Matrix3f rot_matrix = hori_tf_matrix.block(0,0,3,3);
                    // Eigen::Vector3f trans_vector = hori_tf_matrix.block(0,3,3,1);

                    std::cout << "transformation_matrix Hori-> Velo: \n"<<hori_tf_matrix << std::endl;
                    Eigen::Matrix3f rot_matrix = hori_tf_matrix.block(0,0,3,3);
                    Eigen::Vector3f trans_vector = hori_tf_matrix.block(0,3,3,1);
                    _velo_hori_tf_matrix.block(0,0,3,3) = rot_matrix.transpose();
                    _velo_hori_tf_matrix.block(0,3,3,1) =  hori_tf_matrix.block(0,3,3,1) * -1;
                    _velo_hori_tf_matrix.block(3,0,1,4) = hori_tf_matrix.block(3,0,1,4);
                    std::cout << "transformation_matrix Velo-> Hori: \n"<<_velo_hori_tf_matrix << std::endl;

                    // std::cout << "hori -> base_link " << trans_vector.transpose()
                    //     << " " << rot_matrix.eulerAngles(2,1,0).transpose() << " /" << "hori_frame"
                    //     << " /" << "livox_frame" << " 10" << std::endl;

                    // publish result
                    pcl::PointCloud<PointType>  out_cloud;
                    out_cloud += _hori_igcloud;

                    sensor_msgs::PointCloud2 hori_msg;
                    pcl::toROSMsg(out_cloud, hori_msg);
                    hori_msg.header.stamp = ros::Time::now();
                    hori_msg.header.frame_id = "lio_world";
                    pub_hori.publish(hori_msg);

                    _hori_tf_initd = true;
                }else
                {  //publish pointcloud
                    pcl::PointCloud<pcl::PointXYZI>  out_cloud;
                    out_cloud += *pointCloudIn;

                    sensor_msgs::PointCloud2 hori_msg;
                    pcl::toROSMsg(out_cloud, hori_msg);
                    double stamp_sec = livox_msg_in->header.stamp.toSec();
                    ros::Time stamp(stamp_sec);
                    hori_msg.header.stamp = stamp ;
                    hori_msg.header.frame_id = "lio_world";
                    pub_hori.publish(hori_msg);

                    // livox_msg_in_distort
                    // livox_ros_driver::CustomMsg livox_msg_pieces;
                    // livox_msg_pieces.header = livox_msg_in_distort.header;
                    // livox_msg_pieces.header.frame_id = "lio_world";
                    // livox_msg_pieces.lidar_id = 0;
                    // livox_msg_pieces.point_num = livox_msg_pieces.points.size();
                    // livox_msg_pieces.timebase = interpo_stamp.toNSec();

                    // for (int i = 1; i < _cut_raw_message_pieces; i++ )
                    // {
                    //     int raw_points_num =  livox_msg_in_distort.point_num;

                    //     // new message
                    //     livox_ros_driver::CustomMsg         livox_msg_pieces;
                    //     livox_msg_pieces.point_num = raw_points_num / _cut_raw_message_pieces;
                    //     livox_msg_pieces.timebase  = livox_msg_in_distort.timebase +
                    //                     livox_msg_in_distort.points[raw_points_num * (i-1) / _cut_raw_message_pieces].offset_time;
                    //     for (  int j = 0;
                    //             j < raw_points_num * i / _cut_raw_message_pieces  &&
                    //             j > raw_points_num * (i-1) / _cut_raw_message_pieces;
                    //             ++j)
                    //     {
                    //         livox_ros_driver::CustomPoint pt;
                    //         pt.x = livox_msg_in_distort.points[j].x;
                    //         pt.y = livox_msg_in_distort.points[j].y;
                    //         pt.z = livox_msg_in_distort.points[j].z;
                    //         pt.offset_time = livox_msg_in_distort.points[j].offset_time - livox_msg_in_distort.points[j];
                    //         pt.reflectivity =  livox_msg_in_distort.points[j].reflectivity;
                    //         // pt.intensity = livox_msg_in.timebase + livox_msg_in.points[i].offset_time;
                    //         out_cloud.push_back(pt);
                    //     }
                    //     pub_hori_livoxmsg;
                    // }

                    int raw_pts_num   =  livox_msg_in_distort.point_num;
                    int piece_pts_num = raw_pts_num / _cut_raw_message_pieces;
                    // std::cout<<  "Cut pieces: Input number," << raw_pts_num
                    //             << " | pieces: " << _cut_raw_message_pieces
                    //             << " | piece point number: " <<  piece_pts_num << std::endl;

                    livox_ros_driver::CustomMsg         livox_msg_pieces;
                    livox_msg_pieces.header = livox_msg_in_distort.header;
                    livox_msg_pieces.header.frame_id = "lio_world";
                    livox_msg_pieces.lidar_id = 0;
                    livox_msg_pieces.point_num = piece_pts_num;

                    // livox_msg_pieces.timebase = interpo_stamp.toNSec();
                    int pieces_cnt = 0;
                    // std::cout <<  raw_pts_num << " |  " << piece_pts_num << std::endl;
                    if(_cut_raw_message_pieces == 1 && _velo_queue.size() > 2){
                        // livox_msg_pieces = livox_msg_in_distort;
                        // Align the begining timestamp of each frame
                        ROS_INFO_STREAM( " _velo_queue size : " << _velo_queue.size() << " | Hori points queue size: " << _hori_points_queue.size() );

                        uint64 hori_pts_front_stamp = _hori_start_stamp + _hori_points_stamp_queue.front();
                        while(_velo_queue.begin()->header.stamp.toNSec() < hori_pts_front_stamp ){
                            ROS_WARN("Pop front");
                            _velo_queue.erase(_velo_queue.begin());
                        }
                        auto velo_start_stamp  = _velo_queue[0].header.stamp.toNSec();
                        auto velo_end_stamp  = _velo_queue[1].header.stamp.toNSec();
                        livox_ros_driver::CustomMsg livox_aligned_msg;
                        if(pub_horipoints_given_stamp( velo_start_stamp, velo_end_stamp, livox_aligned_msg)){

                            ros::Time v_stamp;
                            v_stamp.fromNSec(_velo_queue.front().header.stamp.toNSec());
                            pcl::PointCloud<PointType>  v_cloud;
                            pcl::fromROSMsg(_velo_queue.front(), v_cloud);
                            pcl::transformPointCloud (v_cloud , v_cloud, _velo_hori_tf_matrix);

                            sensor_msgs::PointCloud2 velo_msg;
                            pcl::toROSMsg(v_cloud, velo_msg);
                            velo_msg.header.stamp =  v_stamp;
                            velo_msg.header.frame_id = "lio_world";
                            pub_time_velo.publish(velo_msg);


                            union_om::union_cloud a_time_union;
                            a_time_union.livox_time_aligned = livox_aligned_msg;
                            a_time_union.velo_time_aligned = velo_msg;
                            pub_time_union_cloud.publish(a_time_union);

                            processed_cnt++;
                            auto t1 = std::chrono::high_resolution_clock::now();
                            double dt = 1.e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1-tick).count();
                            time_cost = dt + time_cost ;
                            ROS_WARN_STREAM("[LidarAligner]Time cost : " << time_cost / processed_cnt);

                            ROS_INFO_STREAM("Pub union aligned cloud spatio & time at " << std::setprecision(15) << a_time_union.livox_time_aligned.header.stamp.toSec() << " - " <<
                                        a_time_union.velo_time_aligned.header.stamp.toSec());
                            if( abs(a_time_union.livox_time_aligned.header.stamp.toSec() - a_time_union.velo_time_aligned.header.stamp.toSec()) > 0.01 )
                                ROS_WARN("Current union cloud stamp doesn't match");

                            _velo_queue.erase(_velo_queue.begin());
                        }
                            // int i = 1;

                        // pub_hori_livoxmsg.publish(livox_msg_in_distort);
                        // ROS_INFO_STREAM("Publish Hori cloud at stamp : " << velo_start_stamp);
                    }else if(_cut_raw_message_pieces>1){
                        for (  int i = 0 ; i < raw_pts_num; ++i)
                        {
                            if ( pieces_cnt ==  (i / piece_pts_num))
                            {
                                livox_ros_driver::CustomPoint pt;
                                pt.x = livox_msg_in_distort.points[i].x;
                                pt.y = livox_msg_in_distort.points[i].y;
                                pt.z = livox_msg_in_distort.points[i].z;
                                pt.offset_time = livox_msg_in_distort.points[i].offset_time
                                            - livox_msg_in_distort.points[pieces_cnt * piece_pts_num].offset_time;
                                pt.reflectivity =  livox_msg_in_distort.points[i].reflectivity;
                                // pt.intensity = livox_msg_in.timebase + livox_msg_in.points[i].offset_time;
                                livox_msg_pieces.points.push_back(pt);
                            }else{
                                // std::cout << " |||  " << std::endl <<  i << "|" << i / piece_pts_num << "|>  ";
                                livox_msg_pieces.timebase = livox_msg_in_distort.timebase +
                                                            livox_msg_in_distort.points[pieces_cnt * piece_pts_num].offset_time;
                                ros::Time pieces_stamp = ros::Time().fromNSec(livox_msg_pieces.timebase);
                                livox_msg_pieces.header.stamp = pieces_stamp;
                                pub_hori_livoxmsg.publish(livox_msg_pieces);
                                std::cout<< "Publish points ->  piece" << pieces_cnt << " | number: " << livox_msg_pieces.points.size()
                                        << " | stamp: " << livox_msg_pieces.timebase << std::endl;
                                pieces_cnt++;
                                livox_msg_pieces.points.clear();

                            }
                        }
                    }else{
                        ROS_WARN_STREAM("Wrong cut pieces number : " << _cut_raw_message_pieces);
                    }

                }
            }
        }

        /**
         * @brief Subscribe pointcloud from Velodyne
         * - save the first timestamp of first message to init the timestamp
         * - undistort pointcloud based on rotation from IMU
         * - select Velo points in same FOV with Hori
         * -
         * @param pointCloudIn
         */
        void velo_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn)
        {

            if(!_first_velo_reveived) _first_velo_reveived = true;

            pcl::PointCloud<PointType>  full_cloud_in;
            pcl::fromROSMsg(*pointCloudIn, full_cloud_in);


            // ################ select Velo points in same FOV with Hori #################
            int cloudSize = full_cloud_in.points.size();
            pcl::PointCloud<pcl::PointXYZI> velo_fovs_cloud;
            float startOri = -atan2(full_cloud_in.points[0].y, full_cloud_in.points[0].x);
            float endOri = -atan2(full_cloud_in.points[cloudSize - 1].y,
                    full_cloud_in.points[cloudSize - 1].x) +
                    2 * M_PI;

            if (endOri - startOri > 3 * M_PI)
                endOri -= 2 * M_PI;
            else if (endOri - startOri < M_PI)
                endOri += 2 * M_PI;

            // ROS_INFO_STREAM("Velodyne Lidar start angle: " <<  startOri << " | end angle : " << endOri << " | range: " << endOri - startOri);

            pcl::PointCloud<PointType> undistort_cloud;
            pcl::PointXYZI  point;
            bool halfPassed = false;
            for (int i = 0; i < cloudSize; i++)
            {
                point.x = full_cloud_in.points[i].x;
                point.y = full_cloud_in.points[i].y;
                point.z = full_cloud_in.points[i].z;

                float ori = -atan2(point.y, point.x);
                if (!halfPassed)
                {
                    if (ori < startOri - M_PI / 2)
                        ori += 2 * M_PI;
                    else if (ori > startOri + M_PI * 3 / 2)
                        ori -= 2 * M_PI;

                    if (ori - startOri > M_PI)
                        halfPassed = true;
                }
                else {
                    ori += 2 * M_PI;
                    if (ori < endOri - M_PI * 3 / 2)
                        ori += 2 * M_PI;
                    else if (ori > endOri + M_PI / 2)
                        ori -= 2 * M_PI;
                }

                float relTime = (ori - startOri) / (endOri - startOri);
                point.intensity = relTime;

                if( ( ori > -0.7608 && ori < 0.7158 ) ||
                    ori > -0.7608+ 2*M_PI && ori < 0.7158 +2*M_PI)
                {
                    // velo_fovs_cloud.push_back(point);
                    undistort_cloud.push_back(point);
                    // if(undistort_cloud.size() == 1)
                    //     ROS_INFO_STREAM("First Points in FOV, start angle : " << ori << "  | relTime" << relTime);
                }
            }

            // // ************ UnDistortion ****************
            // for (int i = 0; i < undistort_cloud.size(); i++) {
            //     Eigen::Vector3d po;
            //     float s = undistort_cloud.points[i].intensity; // relTime;
            //     Eigen::Quaterniond delta_qlc = Eigen::Quaterniond::Identity().slerp(s, _delta_q).normalized();
            //     po = delta_qlc.conjugate() * Eigen::Vector3d(undistort_cloud.points[i].x,undistort_cloud.points[i].y,undistort_cloud.points[i].z);

            //     // create a point
            //     livox_ros_driver::CustomPoint pt;
            //     undistort_cloud.points[i].x = po.x();
            //     undistort_cloud.points[i].y = po.y();
            //     undistort_cloud.points[i].z = po.z();
            // }

            // ****************************************
            velo_fovs_cloud += undistort_cloud;

            std::unique_lock<std::mutex> lock_velo(_mutexVeloQueue);
            _velo_queue.push_back(*pointCloudIn);
            _velo_fov_queue.push(velo_fovs_cloud);
            lock_velo.unlock();
            if( (_hori_tf_initd || _use_given_extrinsic_lidars) && _time_offset_initd && en_timeoffset_esti)
            {
                //#### substtract time-offset to get the livox based timestamp
                // std::cout<< "\n\n ==>> Velo time: "  << std::setprecision(15) << _velo_queue.front().header.stamp.toSec()  << " | Looking for Time "
                //             << _velo_queue.front().header.stamp.toSec() -  ros::Time().fromNSec(_time_offset).toSec() << std::endl;

                livox_ros_driver::CustomMsg livox_aligned_msg;
                if(pub_horipoints_given_stamp( _velo_queue.front().header.stamp.toNSec() - _time_offset, livox_aligned_msg))
                {
                    // Publish velodyne restamped message
                    std::unique_lock<std::mutex> lock_velo(_mutexVeloQueue);
                    ros::Time v_stamp;
                    v_stamp.fromNSec(_velo_queue.front().header.stamp.toNSec() - _time_offset);
                    pcl::PointCloud<PointType>  v_cloud;
                    pcl::fromROSMsg(_velo_queue.front(), v_cloud);
                    pcl::transformPointCloud (v_cloud , v_cloud, _velo_hori_tf_matrix);

                    sensor_msgs::PointCloud2 velo_msg;
                    pcl::toROSMsg(v_cloud, velo_msg);
                    velo_msg.header.stamp =  v_stamp;
                    velo_msg.header.frame_id = "lio_world";
                    pub_time_velo.publish(velo_msg);

                    union_om::union_cloud a_time_union;
                    a_time_union.livox_time_aligned = livox_aligned_msg;
                    a_time_union.velo_time_aligned = velo_msg;
                    pub_time_union_cloud.publish(a_time_union);



                    _velo_queue.erase(_velo_queue.begin());
                    _velo_fov_queue.pop();
                    lock_velo.unlock();

                }else{
                    ROS_WARN_STREAM("Publishing aligned horipoints failed! ");
                }
                return;
            }

            if(!_hori_tf_initd && en_extrinsic_esti){
                //  std::cout << "OS0 -> base_link " << trans_vector.transpose()
                //     << " " << rot_matrix.eulerAngles(2,1,0).transpose() << " /" << "os0_sensor"
                //     << " /" << "livox_frame" << " 10" << std::endl;

                Eigen::AngleAxisf init_rot_x( 0.0 , Eigen::Vector3f::UnitX());
                Eigen::AngleAxisf init_rot_y( 0.0 , Eigen::Vector3f::UnitY());
                Eigen::AngleAxisf init_rot_z( 0.0 , Eigen::Vector3f::UnitZ());
                Eigen::Translation3f init_trans(0.0,0.0,0.0);
                Eigen::Matrix4f init_tf = (init_trans * init_rot_z * init_rot_y * init_rot_x).matrix();

                Eigen::Matrix3f rot_matrix = init_tf.block(0,0,3,3);
                Eigen::Vector3f trans_vector = init_tf.block(0,3,3,1);

                pcl::PointCloud<PointType>  out_cloud;
                pcl::transformPointCloud (full_cloud_in , full_cloud_in, init_tf);

                _velo_new_cloud.clear();
                _velo_new_cloud += full_cloud_in;


                // sensor_msgs::PointCloud2 velo_msg;
                // pcl::toROSMsg(_velo_new_cloud, velo_msg);
                // // velo_msg.header.stamp = ros::Time::now();
                // velo_msg.header.stamp = pointCloudIn->header.stamp;
                // velo_msg.header.frame_id = "lio_world";
                // pub_velo.publish(velo_msg);
            }

            if(_hori_tf_initd){

                pcl::PointCloud<PointType>  out_cloud;
                pcl::transformPointCloud (full_cloud_in , out_cloud, _velo_hori_tf_matrix);

                _velo_new_cloud.clear();
                _velo_new_cloud += full_cloud_in;

                sensor_msgs::PointCloud2 velo_msg;
                pcl::toROSMsg(out_cloud, velo_msg);
                // velo_msg.header.stamp = ros::Time::now();
                double stamp_sec = pointCloudIn->header.stamp.toSec() - 0.05;
                ros::Time stamp(stamp_sec);
                velo_msg.header.stamp =  stamp;

                // velo_msg.header.stamp = pointCloudIn->header.stamp;
                velo_msg.header.frame_id = "lio_world";
                pub_velo.publish(velo_msg);


                // publish the velo in same FOV
                pcl::PointCloud<PointType>  out_cloud2;
                pcl::transformPointCloud (velo_fovs_cloud , out_cloud2, _velo_hori_tf_matrix);
                sensor_msgs::PointCloud2 velo_fovs_cloud_msg;
                pcl::toROSMsg(out_cloud2, velo_fovs_cloud_msg);
                velo_fovs_cloud_msg.header.stamp = pointCloudIn->header.stamp;
                velo_fovs_cloud_msg.header.frame_id = "lio_world";
                pub_velo_inFOV.publish(velo_fovs_cloud_msg);

            }
        }

        /**
         * @brief Subscribe imu message, check current IMU message
         * * if tf inited, and time offset is not inited, enable timeoffset estimation
         *
         */
        void imu_handler(const sensor_msgs::ImuConstPtr &imu_msg){
            _yaw_velocity = imu_msg->angular_velocity.z;

            std::unique_lock<std::mutex> lock_velo(_mutexIMUVec);
            _imu_vec.push_back(imu_msg);
            lock_velo.unlock();

            if(_time_offset_initd || en_timestamp_align){
                transform_hori_timestamp();
                return;
            }

            // time_offset estimate
            if( _hori_tf_initd && !_time_offset_initd && en_timeoffset_esti)
            {
                // check last a perio
                float last_yaw_speed = _hori_msg_yaw_vec[ _hori_msg_yaw_vec.size() - 3];
                if(last_yaw_speed > abs( _time_start_yaw_velocity ))
                {
                    ROS_INFO_STREAM(" Yaw speed: " << last_yaw_speed);
                    // get the data from queue to vector
                    std::vector<livox_ros_driver::CustomMsg> hori_vec;
                    // std::vector<float>                      yaw_vec;
                    std::vector<sensor_msgs::PointCloud2>        velo_vec;
                    std::vector<pcl::PointCloud<pcl::PointXYZI>> velo_fov_vec;

                    std::unique_lock<std::mutex> lock_hori(_mutexHoriQueue);
                    if( _hori_queue.size() > 3){
                        for(int i=_hori_queue.size(); i > 0   ; i--){
                            hori_vec.push_back(_hori_queue.front());
                            // yaw_vec.push_back(_hori_points_stamp_queue.front());
                            // hori_vec.erase(hori_vec.begin());
                            _hori_queue.pop();
                            // _hori_points_stamp_queue.erase( _hori_points_stamp_queue.begin());
                        }
                        // _hori_msg_yaw_vec.clear();
                    }
                    lock_hori.unlock();

                    std::unique_lock<std::mutex> lock_velo(_mutexVeloQueue);
                    if( _velo_queue.size() > 3){
                        for(int i=_velo_queue.size(); i > 0   ; i--){
                            assert(_velo_queue.size() == _velo_fov_queue.size());
                            velo_vec.push_back(_velo_queue.front());
                            velo_fov_vec.push_back(_velo_fov_queue.front());
                            // velo_vec.erase(velo_vec.begin());
                             _velo_queue.erase(_velo_queue.begin());
                            _velo_fov_queue.pop();
                        }
                    }
                    lock_velo.unlock();

                    if(!_time_offset_initd && en_timeoffset_esti){
                        ROS_INFO_STREAM( " hori_vec size: " << hori_vec.size()
                                        << " | velo_vec size: "  << velo_vec.size());
                        estimate_timeoffset(hori_vec, velo_vec, velo_fov_vec);
                    }
                }
            }

        }

        void RemoveLidarDistortion(const livox_ros_driver::CustomMsgConstPtr& cloudIn,
                                          livox_ros_driver::CustomMsg& cloudOut)
        {
            int PointsNum = cloudIn->point_num;
            Eigen::Quaterniond dq;

            // std::cout << "cloudOut.point_num  : "<< cloudOut.point_num << std::endl;
            get_delta_rotation(dq);
            Eigen::Quaterniond qlc = dq.normalized();
            _delta_q = dq.normalized();
            for (int i = 0; i < PointsNum; i++) {
                Eigen::Vector3d po;
                float s = ros::Time().fromNSec(cloudIn->points[i].offset_time).toSec();
                Eigen::Quaterniond delta_qlc = Eigen::Quaterniond::Identity().slerp(s, qlc).normalized();
                po = delta_qlc.conjugate() * Eigen::Vector3d(cloudIn->points[i].x,cloudIn->points[i].y,cloudIn->points[i].z);

                // create a point
                livox_ros_driver::CustomPoint pt;
                cloudOut.points[i].x = po.x();
                cloudOut.points[i].y = po.y();
                cloudOut.points[i].z = po.z();
                // pt.line = cloudIn->points[i].line;
                // pt.offset_time = cloudIn->points[i].offset_time;
                // pt.reflectivity   = cloudIn->points[i].reflectivity;
                // pt.tag  = cloudIn->points[i].tag;
                // cloudOut.points[i].(pt);
            }

        }

        void get_delta_rotation(Eigen::Quaterniond &dq){
            dq.setIdentity();
            double current_time = _last_imu_time;
            std::unique_lock<std::mutex> lock_velo(_mutexIMUVec);
            for(auto & imu : _imu_vec){
                Eigen::Vector3d gyr;
                gyr << imu->angular_velocity.x,
                        imu->angular_velocity.y,
                        imu->angular_velocity.z;
                double dt = imu->header.stamp.toSec() - current_time;
                ROS_ASSERT(dt >= 0);
                Eigen::Matrix3d dR = Sophus::SO3d::exp(gyr*dt).matrix();
                Eigen::Quaterniond qr(dq*dR);
                if (qr.w()<0)
                    qr.coeffs() *= -1;
                dq = qr.normalized(); // Delta Quaternion
                current_time = imu->header.stamp.toSec();
                _last_imu_time = current_time;
            }
            _imu_vec.clear();
            lock_velo.unlock();
        }

        // transform each points in Horizon pointcloud offset time to first received timebase
        // maintain points queue(_hori_points_queue) and points stamp queue(_hori_points_stamp_queue)
        void transform_hori_timestamp()
        {
            //  make points in order by offset
            std::unique_lock<std::mutex> lock_hori(_mutexHoriQueue);
            std::vector<livox_ros_driver::CustomMsg> hori_msg_vec;

            while(_hori_queue.size() > 0){
                hori_msg_vec.push_back( _hori_queue.front() );
                _hori_queue.pop();
            }
            lock_hori.unlock();

            std::unique_lock<std::mutex> lock_horiPoints(_mutexHoriPointsQueue);
            for(int i = 0; i < hori_msg_vec.size(); i++)
            {
                uint64 delta_t =  hori_msg_vec[i].timebase - _hori_start_stamp;
                std::cout << " i:" << i << " | delta_t: " << delta_t << " -> " << ros::Time().fromNSec( delta_t).toSec()
                            << " | _hori_start_stamp + delta_t:"<< _hori_start_stamp+ delta_t << std::endl;
                for(int j = 0; j < hori_msg_vec[i].points.size(); j++)
                {
                    uint64 stamp = ros::Time().fromNSec( delta_t + hori_msg_vec[i].points[j].offset_time).toNSec();

                    _hori_points_stamp_queue.push_back(stamp);
                    _hori_points_queue.push_back(hori_msg_vec[i].points[j]);
                }
            }
            lock_horiPoints.unlock();
        }


        bool pub_horipoints_given_stamp( uint64 velo_start_stamp, uint64 velo_end_stamp, livox_ros_driver::CustomMsg &livox_aligned_msg)
        {

            if(_hori_points_stamp_queue.empty())
            {
                ROS_WARN_STREAM( "_hori_points_stamp_queue.size() " << _hori_points_stamp_queue.size());
                return false;
            }


            std::unique_lock<std::mutex> lock_horiPoints(_mutexHoriPointsQueue);
            uint64 hori_pts_front_stamp = _hori_start_stamp + _hori_points_stamp_queue.front();
            uint64 hori_pts_back_stamp  = _hori_start_stamp + _hori_points_stamp_queue.back();

            ROS_INFO_STREAM( " VELO asked Start stamp: " << std::setprecision(15) <<  ros::Time().fromNSec(velo_start_stamp).toSec()
                << " VELO asked End stamp: " <<  ros::Time().fromNSec(velo_end_stamp).toSec()
                << "|| Hori Front stamp "  << ros::Time().fromNSec(hori_pts_front_stamp).toSec()
                << " | Hori hori_pts_back_stamp: " << ros::Time().fromNSec(hori_pts_back_stamp).toSec()
                << " |  _hori_points_queue size: " <<  _hori_points_queue.size());


            int idx = 0;
            // if( velo_start_stamp > hori_pts_front_stamp &&  hori_pts_back_stamp >  velo_end_stamp ){
            while( hori_pts_front_stamp < velo_start_stamp ){
                // std::cout << idx << " | " ;
                if( _hori_points_queue.size() > 1 && idx < _hori_points_stamp_queue.size()){
                    // _hori_points_queue.erase(_hori_points_queue.begin() );
                    // _hori_points_stamp_queue.erase( _hori_points_stamp_queue.begin());
                    hori_pts_front_stamp = _hori_start_stamp + _hori_points_stamp_queue[idx];
                    idx ++;
                }else{
                    ROS_WARN(" _hori_points_queue are empty");
                    return false;
                }
            }
            // }

            ROS_INFO_STREAM("Stamp Difference : " << std::setprecision(15) << ros::Time().fromNSec( hori_pts_front_stamp - velo_start_stamp).toSec() << " | "
                <<ros::Time().fromNSec( hori_pts_front_stamp).toSec() << " | "  << ros::Time().fromNSec(velo_start_stamp).toSec());


            pcl::PointCloud<pcl::PointXYZI>     aligned_cloud;
            // livox_ros_driver::CustomMsg         livox_aligned_msg;

            // while(  velo_start_stamp > hori_pts_front_stamp &&  hori_pts_front_stamp < velo_end_stamp && idx < _hori_points_queue.size())
            while(   hori_pts_front_stamp < velo_end_stamp && idx < _hori_points_queue.size())
            {

                ros::Time stamp = ros::Time().fromNSec(_hori_points_stamp_queue[idx] + _hori_start_stamp - velo_start_stamp);
                // std::cout << " | " << std::setprecision(15) << stamp.toSec() ;
                // _hori_points_stamp_queue[idx] = _hori_points_stamp_queue[idx] + _hori_start_stamp - start_stamp;

                livox_ros_driver::CustomPoint pt;
                pt.x = _hori_points_queue[idx].x;
                pt.y = _hori_points_queue[idx].y;
                pt.z = _hori_points_queue[idx].z;
                // pt.offset_time = _hori_points_queue[idx].offset_time;
                pt.offset_time = stamp.toNSec();
                pt.line = _hori_points_queue[idx].line;
                pt.tag = _hori_points_queue[idx].tag;
                pt.reflectivity = _hori_points_queue[idx].reflectivity;
                livox_aligned_msg.points.push_back(pt);

                pcl::PointXYZI point;
                point.x = _hori_points_queue[idx].x;
                point.y = _hori_points_queue[idx].y;
                point.z = _hori_points_queue[idx].z;
                aligned_cloud.push_back(point);

                idx++;

                hori_pts_front_stamp = _hori_start_stamp + _hori_points_stamp_queue[idx];
            }

            ROS_INFO_STREAM("Publish hori cloud with " << aligned_cloud.size() << " | at: " << std::setprecision(15) <<  ros::Time().fromNSec(velo_start_stamp).toSec());

            if(livox_aligned_msg.points.front().offset_time > 100000)
                ROS_WARN_STREAM("Timestamp jump detected" << livox_aligned_msg.points.front().offset_time);

            ros::Time msg_stamp;
            livox_aligned_msg.header.stamp = msg_stamp.fromNSec(velo_start_stamp);
            livox_aligned_msg.header.frame_id = "lio_world";
            livox_aligned_msg.lidar_id = 0;
            livox_aligned_msg.point_num = livox_aligned_msg.points.size();
            livox_aligned_msg.timebase = msg_stamp.toNSec();
            pub_time_hori_livoxmsg.publish(livox_aligned_msg);


            // Publish pointcloud
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(aligned_cloud, cloud_msg);
            cloud_msg.header.stamp = msg_stamp.fromNSec(velo_start_stamp);
            cloud_msg.header.frame_id = "lio_world";
            pub_time_hori.publish(cloud_msg);

            _last_search_stamp = velo_start_stamp;
            _hori_points_queue.erase(_hori_points_queue.begin(), _hori_points_queue.begin() + idx -100);
            _hori_points_stamp_queue.erase( _hori_points_stamp_queue.begin(), _hori_points_stamp_queue.begin() + idx -100);

            // ROS_INFO_STREAM( "_hori_points_queue SIZE : " << _hori_points_queue.size() << " \n\n " );
            return true;

        }


        // TODO: pub points instead of removing them
        bool pub_horipoints_given_stamp( uint64 start_stamp, livox_ros_driver::CustomMsg &livox_aligned_msg)
        {
            if(_hori_points_stamp_queue.empty())
            {
                ROS_WARN_STREAM( "_hori_points_stamp_queue.size() " << _hori_points_stamp_queue.size());
                return false;
            }

            std::unique_lock<std::mutex> lock_horiPoints(_mutexHoriPointsQueue);
            uint64 hori_pts_front_stamp = _hori_start_stamp + _hori_points_stamp_queue.front();
            uint64 hori_pts_back_stamp  = _hori_start_stamp + _hori_points_stamp_queue.back();
            ROS_INFO_STREAM( " VELO asked stamp: " << std::setprecision(15) <<  ros::Time().fromNSec(start_stamp).toSec() << "|| Hori Front stamp "  << ros::Time().fromNSec(hori_pts_front_stamp).toSec()
                        << " | Hori hori_pts_back_stamp: " << ros::Time().fromNSec(hori_pts_back_stamp).toSec() << " |  _hori_points_queue size: " <<  _hori_points_queue.size());
            // Remove points earlier than given stamp

            int ccnt = 0;
            livox_ros_driver::CustomMsg         livox_interpolate_msg;
            ros::Time interpo_stamp = ros::Time().fromNSec(_hori_points_stamp_queue.front() + _hori_start_stamp);

            if(hori_pts_front_stamp > start_stamp){
                // _velo_queue.pop();
                ROS_WARN_STREAM("Velo Given stamp smaller start Hori stamp, pop up velo_queue "  <<  _velo_queue.size());
            }

            while( hori_pts_front_stamp < start_stamp ){
                if( _hori_points_queue.size() > 1){
                    ros::Time stamp = ros::Time().fromNSec(_hori_points_stamp_queue.front() + _hori_start_stamp - interpo_stamp.toNSec());
                    livox_ros_driver::CustomPoint pt;
                    pt.x = _hori_points_queue.front().x;
                    pt.y = _hori_points_queue.front().y;
                    pt.z = _hori_points_queue.front().z;
                    // pt.offset_time = _hori_points_queue.front().offset_time;
                    pt.offset_time = stamp.toNSec();
                    pt.line = _hori_points_queue.front().line;
                    pt.tag = _hori_points_queue.front().tag;
                    pt.reflectivity = _hori_points_queue.front().reflectivity;
                    livox_interpolate_msg.points.push_back(pt);

                    _hori_points_queue.erase(_hori_points_queue.begin() );
                    _hori_points_stamp_queue.erase( _hori_points_stamp_queue.begin());
                    hori_pts_front_stamp = _hori_start_stamp + _hori_points_stamp_queue.front();
                    ccnt ++;
                }else{
                    ROS_WARN(" _hori_points_queue are empty");
                    return false;
                }
            }
            ROS_INFO_STREAM("Stamp Difference : " << std::setprecision(15) << ros::Time().fromNSec( hori_pts_front_stamp - start_stamp).toSec() << " | "
                            <<ros::Time().fromNSec( hori_pts_front_stamp).toSec() << " | "  << ros::Time().fromNSec(start_stamp).toSec());
            // if(livox_interpolate_msg.points.size() > 0)
            // {
            //     livox_interpolate_msg.header.stamp = interpo_stamp;
            //     livox_interpolate_msg.header.frame_id = "livox_frame";
            //     livox_interpolate_msg.lidar_id = 0;
            //     livox_interpolate_msg.point_num = livox_interpolate_msg.points.size();
            //     livox_interpolate_msg.timebase = interpo_stamp.toNSec();
            //     pub_time_hori_livoxmsg.publish(livox_interpolate_msg);
            //     ROS_WARN_STREAM("Publish points ealier than given stamp : "  << std::setprecision(15)
            //         <<  ros::Time().fromNSec(start_stamp).toSec() << " | points amount " << livox_interpolate_msg.point_num);
            // }

            // std::cout << "===>>> Removed pts " << ccnt++ <<" |  hori_pts_front_stamp < start_stamp: "
            //             <<  hori_pts_front_stamp <<  " - " << start_stamp  << std::endl;

            // Add points to a msg;
            // if( _hori_points_queue.size() < _offset_search_sliced_points){ //24000/12000
            //     return false;
            // }

            pcl::PointCloud<pcl::PointXYZI>     cloud;
            // livox_ros_driver::CustomMsg         livox_aligned_msg;
            // ros::Time select_stamp;
            // select_stamp.fromSec(0.1/2); //
            // std::cout << "\n\n Points: " <<  (hori_pts_front_stamp - start_stamp) << std::endl;
            int points_num = -1;
            if (_offset_search_sliced_points < _hori_points_queue.size())
                points_num = _offset_search_sliced_points;
            else
                points_num = _hori_points_queue.size();

            for(int i=0; i< points_num; i++)
            {

                ros::Time stamp = ros::Time().fromNSec(_hori_points_stamp_queue.front() + _hori_start_stamp - start_stamp);
                // std::cout << " | " << std::setprecision(15) << stamp.toSec() ;
                // _hori_points_stamp_queue.front() = _hori_points_stamp_queue.front() + _hori_start_stamp - start_stamp;

                livox_ros_driver::CustomPoint pt;
                pt.x = _hori_points_queue.front().x;
                pt.y = _hori_points_queue.front().y;
                pt.z = _hori_points_queue.front().z;
                // pt.offset_time = _hori_points_queue.front().offset_time;
                pt.offset_time = stamp.toNSec();
                pt.line = _hori_points_queue.front().line;
                pt.tag = _hori_points_queue.front().tag;
                pt.reflectivity = _hori_points_queue.front().reflectivity;
                livox_aligned_msg.points.push_back(pt);

                pcl::PointXYZI point;
                point.x = _hori_points_queue.front().x;
                point.y = _hori_points_queue.front().y;
                point.z = _hori_points_queue.front().z;
                cloud.push_back(point);

                // std::cout<< "Pop points: " <<  cloud.size() << " | _hori_points_queue : "<< _hori_points_queue.size() << std::endl;
                _hori_points_queue.erase(_hori_points_queue.begin() );
                _hori_points_stamp_queue.erase( _hori_points_stamp_queue.begin());

            }
            lock_horiPoints.unlock();
            ROS_INFO_STREAM("Publish hori cloud with " << cloud.size() << " | at: " << std::setprecision(15) <<  ros::Time().fromNSec(start_stamp).toSec());

            if(livox_aligned_msg.points.front().offset_time > 100000)
                ROS_WARN_STREAM("Timestamp jump detected" << livox_aligned_msg.points.front().offset_time);

            ros::Time msg_stamp;
            livox_aligned_msg.header.stamp = msg_stamp.fromNSec(start_stamp);
            livox_aligned_msg.header.frame_id = "lio_world";
            livox_aligned_msg.lidar_id = 0;
            livox_aligned_msg.point_num = livox_aligned_msg.points.size();
            livox_aligned_msg.timebase = msg_stamp.toNSec();
            pub_time_hori_livoxmsg.publish(livox_aligned_msg);


            // Publish pointcloud
            sensor_msgs::PointCloud2 cloud_msg;
            pcl::toROSMsg(cloud, cloud_msg);
            cloud_msg.header.stamp = msg_stamp.fromNSec(start_stamp);
            cloud_msg.header.frame_id = "lio_world";
            pub_time_hori.publish(cloud_msg);

            _last_search_stamp = start_stamp;

            _velo_queue.erase(_velo_queue.begin());

            std::cout << " \n\n " << std::endl;
            return true;
        }


        // estimate the timeoffset; this function is triggered by an fast rotation movement
        /**
         * @brief Estimate the time offset when a fast rotation movement is detected
         * - Caculate the distance based on the K nearest neighbor with pointcloud in same FOV
         * @param hori_vec
         * @param velo_vec
         * @param velo_fov_vec
         * WARN this timeoffset is not so accurate
         */
        void estimate_timeoffset(   std::vector<livox_ros_driver::CustomMsg>&       hori_vec,
                                    std::vector<sensor_msgs::PointCloud2>&          velo_vec,
                                    std::vector<pcl::PointCloud<pcl::PointXYZI>>&   velo_fov_vec )
        {
            //######################### Prepare data 1 Velo & 5 hori msgs #############################
            ros::Time velo_stamp = velo_vec.back().header.stamp;
            ros::Time hori_stamp = hori_vec.back().header.stamp;

            // double time_received_diff = abs( hori_stamp.toSec() - velo_stamp.toSec() );
            ROS_INFO_STREAM( "Time stamp diff:  velo back."  << std::setprecision(15)<< velo_stamp.toSec() << " | hori back:" << hori_stamp.toSec()
                            << " -> "<< hori_stamp.toSec() - velo_stamp.toSec());

            // velo is faster than livox, get the difference
            while( velo_stamp.toSec() > hori_stamp.toSec() || abs( velo_stamp.toSec() - hori_stamp.toSec() ) < 0.2)
            {
                // velo_vec.erase(velo_vec.end());
                velo_vec.pop_back();
                velo_fov_vec.pop_back();
                velo_stamp = velo_vec.back().header.stamp;
                hori_stamp = hori_vec.back().header.stamp;
                std::cout<< "POP FIRST:  velo."  << std::setprecision(15)<< velo_stamp.toSec() << ", hori." << hori_stamp.toSec()
                        << " -> "<< hori_stamp.toSec() - velo_stamp.toSec() << std::endl;
            }
            ROS_INFO_STREAM("Selcted Velo stamp:  " << std::setprecision(15) <<  velo_stamp.toSec());
            ROS_INFO_STREAM("Selcted Hori end stamp ->" << std::setprecision(15) <<  hori_stamp.toSec()
                                                        << " | start stamp-> " << hori_vec.front().header.stamp.toSec(););
            ROS_INFO_STREAM(" hori_vec.size() : "<< hori_vec.size());

            int msg_size = hori_vec.size();
            livox_ros_driver::CustomMsg livox_msg;
            livox_msg.header = hori_vec[msg_size - 8].header;
            livox_msg.lidar_id = hori_vec[msg_size - 8].lidar_id;
            livox_msg.timebase = hori_vec[msg_size - 8].timebase;
            std::vector<uint64> livox_msg_offset_vec;

            // Merge multiple livox cloud
            std::cout<< " VELO time stamp, start: " << std::setprecision(15) << velo_vec.back().header.stamp.toSec() << std::endl;
            std::cout<< " HORI time stamp, start: " << ros::Time().fromNSec(livox_msg.timebase).toSec() << " -> "
                        << ros::Time().fromNSec(hori_vec[msg_size - 1].timebase).toSec() << std::endl;
            for( int i = 7; i >= 0; i--)
            {
                uint64_t delta_t =  hori_vec[msg_size - i - 1].timebase - livox_msg.timebase;
                int pts_size = hori_vec[msg_size - i - 1].points.size();
                for(int ii = 0; ii < pts_size; ii++)
                {
                    // hori_vec[msg_size - i - 1].points[ii].offset_time
                    //         = hori_vec[msg_size - i - 1].points[ii].offset_time + delta_t;
                    livox_msg_offset_vec.push_back(hori_vec[msg_size - i - 1].points[ii].offset_time + delta_t);
                    livox_msg.points.push_back( hori_vec[msg_size - i - 1].points[ii] );
                }
                std::cout<< "-> "<< i << " |  Get Hori points " << livox_msg.points.size() << std::endl;
                // livox_msg.points.insert( livox_msg.points.end(), hori_vec[msg_size - i].points.begin(),
                //                                                  hori_vec[msg_size - i].points.end() );
            }


            //######################### Pre-Caculate the distance to nearest neighbor (hori->Velo) #############################
            //Caculate the error
            std::vector<float> dis_errors ;
            pcl::KdTreeFLANN<pcl::PointXYZI> velo_tree;
            pcl::PointCloud<PointType>  full_cloud_in;
            // pcl::fromROSMsg( velo_vec.back(), full_cloud_in);
            full_cloud_in += velo_fov_vec.back();
            pcl::PointCloud<PointType>  out_cloud;
            pcl::transformPointCloud (full_cloud_in , out_cloud, _velo_hori_tf_matrix);

            velo_tree.setInputCloud(out_cloud.makeShared());
            pcl::PointXYZI searchPoint;
            for(int i = 0; i < livox_msg.points.size(); i++)
            {
                searchPoint.x = livox_msg.points[i].x;
                searchPoint.y = livox_msg.points[i].y;
                searchPoint.z = livox_msg.points[i].z;
                int K = 1;
                std::vector<int> pointIdxNKNSearch(K);
                std::vector<float> pointNKNSquaredDistance(K);

                if ( velo_tree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
                {
                    for (std::size_t t = 0; t < pointIdxNKNSearch.size (); ++t)
                    {
                        dis_errors .push_back(pointNKNSquaredDistance[t]);
                    }
                }
            }

            //######################### Find the highest possibility window #############################
            ros::Rate r(10);
            int cnt = 0;
            ros::Time optim_start_time;
            double lowest_error = 1000000.0;
            int search_res = _offset_search_resolution;  // given search resolution
            int sliced_pts = _offset_search_sliced_points;
            while( cnt * search_res + sliced_pts < livox_msg.points.size() && ros::ok())
            {
                double sum_error = 0;
                // publish Velo cloud as reference
                sensor_msgs::PointCloud2 velo_msg;
                pcl::toROSMsg(out_cloud, velo_msg);
                velo_msg.header.frame_id = "lio_world";
                velo_msg.header.stamp = velo_vec.back().header.stamp;
                // pub_time_velo.publish(velo_msg);

                // publish Hori cloud
                pcl::PointCloud<pcl::PointXYZI> cloud;
                for(int i = cnt * search_res ; i < cnt * search_res + sliced_pts ; i++)
                {
                    pcl::PointXYZI pt;
                    pt.x = livox_msg.points[i].x;
                    pt.y = livox_msg.points[i].y;
                    pt.z = livox_msg.points[i].z;
                    // pt.intensity = livox_msg.points[i].reflectivity;

                    sum_error += dis_errors [i] + 0.2* sqrt(pt.x * pt.x + pt.y * pt.y );
                    cloud.push_back(pt);
                }
                sensor_msgs::PointCloud2 cloud_msg;
                pcl::toROSMsg(cloud, cloud_msg);
                cloud_msg.header.stamp =  velo_vec.back().header.stamp;
                cloud_msg.header.frame_id = "lio_world";
                // pub_time_hori.publish(cloud_msg);


                if(sum_error < lowest_error ){
                    // optim_start_time.fromNSec( livox_msg.timebase +  livox_msg.points[cnt * search_res ].offset_time);
                    optim_start_time.fromNSec( livox_msg.timebase +  livox_msg_offset_vec[cnt * search_res ]);
                    std::cout   << "-> Publishing : " << cnt  << " | Velo Stamp: " << velo_stamp.toSec()
                                << " | Hori Stamp: " << optim_start_time.toSec() << " | size: " << livox_msg.points.size()
                                << " | error: " << sum_error << std::endl;
                    lowest_error = sum_error;
                    pub_time_velo.publish(velo_msg);
                    pub_time_hori.publish(cloud_msg);
                    // r.sleep();

                }
                cnt++;
            }

            if( lowest_error < _time_esti_error_th){
                _time_offset_initd = true;
                _time_offset = velo_stamp.toNSec() - optim_start_time.toNSec();
                ROS_INFO_STREAM("Time offset init successful! | Time offset (velo->hori)" <<
                      ros::Time().fromNSec(_time_offset).toSec() << "  s| ");
                // break;
            }
        }

};

int
main(int argc, char **argv)
{
    ros::init(argc, argv, "Lidar_Calibrate");

    LidarsParamEstimator swo;
    ROS_INFO("\033[1;32m---->\033[0m Lidar Calibrate Started.");

    ros::Rate r(10);
    while(ros::ok()){

        ros::spinOnce();
        r.sleep();

    }


    return 0;
}
