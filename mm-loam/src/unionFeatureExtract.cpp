// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <vector>
#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/filter.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <ros/ros.h>
#include "livox_ros_driver/CustomMsg.h"
#include "lidars_extrinsic_cali.h"
#include <message_filters/time_synchronizer.h>
#include <chrono>
#include <sensor_msgs/PointCloud2.h>
#include <eigen3/Eigen/Core>
#include <Eigen/Dense>

#include "union_cloud/union_cloud.h"


struct ExtractedFeatures{
    pcl::PointCloud<pcl::PointXYZI>::Ptr plane_features;
    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_features;
};


bool icp_ext_matching(pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_src,
                  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_tgt,
                  pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud_aligned,
                  Eigen::Matrix4f &icp_mtx,
                  bool en_viewer)
{
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZINormal, pcl::PointXYZINormal> icp;
    icp.setInputSource(cloud_src);
    icp.setInputTarget(cloud_tgt);

    // Set the maximum number of iterations and the transformation epsilon.
    icp.setMaximumIterations(10);
    icp.setTransformationEpsilon(1e-6);

    // Perform the ICP alignment.
    icp.align(*cloud_aligned);

    // Check if the ICP alignment was successful.
    if (icp.hasConverged())
    {
        // Print the ICP fitness score.
        std::cout << "ICP has converged, score is " << icp.getFitnessScore() << std::endl;
        icp_mtx = icp.getFinalTransformation();

        if (en_viewer)
        {
            // Create a PCL visualizer.
            pcl::visualization::PCLVisualizer viewer("ICP Results");

            // Add the two point clouds to the visualizer.
            viewer.addPointCloud<pcl::PointXYZINormal>(cloud_tgt, "cloud_out");
            viewer.addPointCloud<pcl::PointXYZINormal>(cloud_aligned, "cloud_aligned");

            // Color the aligned cloud in green.
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "cloud_aligned");

            // Spin until the visualizer is closed.
            while (!viewer.wasStopped() && ros::ok())
            {
                viewer.spinOnce();
            }
        }
        return true;
    }
    else
    {
        ROS_WARN("ICP Failed!");
        return false;
    }
}



// For Hesai lidar
struct PointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    _Float64 timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (_Float64, timestamp, timestamp)
)


class feature_extraction{
private:
    ros::NodeHandle nh;

    ros::Publisher pubHoriCloud;
    ros::Publisher pubHoriSharp;
    ros::Publisher pubHoriFlat;
    ros::Publisher pubHoriCloudNormal;
    ros::Publisher pubVeloCloud;
    ros::Publisher pubVeloSharp;
    ros::Publisher pubVeloFlat;
    ros::Publisher pubVeloCloudNormal;
    ros::Publisher pubMergedFeature;
    ros::Publisher pubMergedCloud;
    ros::Publisher pubUnionFeatureCloud;

    // Subscribe pointcloud
    ros::Subscriber subVeloCloud;
    ros::Subscriber subHoriCloud;
    ros::Subscriber subUnionCloud;
    ros::Subscriber subHesaCloud; // sub hesai



    // Msg buffer
    std::vector<sensor_msgs::PointCloud2ConstPtr>   _veloMsgBuffer;
    std::vector<livox_ros_driver::CustomMsg>        _horiMsgBuffer;
    std::mutex                                      _mtxVeloBuffer;
    std::mutex                                      _mtxHoriBuffer;

    // Low feature environment
    bool  _hori_lowfeature_env = true;
    bool  _velo_lowfeature_env = true;
    int  _hori_lowfeature_th ;
    int  _velo_lowfeature_th ;
    int  _velo_frames_cnt = 0;
    int  _hori_frames_cnt = 0;
    int  _hesai_frames_cnt= 0;
    int  _velo_skip_frames = 1;
    int  _hori_skip_frames = 1;
    int  _hesai_skip_frames;
    bool _pub_feature_points = false;
    float _near_points_th = 1.0;
    float _far_points_th  = 50.0;
    // Feature extraction parameter
    int scanID;
    int CloudFeatureFlag[32000];

    int HORI_N_SCANS = 6;
    int VELO_N_SCANS = 16;
    std_msgs::Header cloud_header;
    int ds_rate = 2;
    double ds_v = 0.6;
    float cloudCurvature[400000];

    int _extrin_recali_times = 30;
    int _extrin_cnt = 0;
    union_cloud::union_cloud union_msg;
    Eigen::Matrix4f extri_mtx = Eigen::Matrix4f::Identity();
    // Preprocess p_pre;

    double time_cost = 0;
    int processed_cnt = 0;

public:
    // LidarsCalibrater lidarCali; // No need calibrate now;
    feature_extraction(){

        // subHoriCloud = nh.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar_msg", 100, &feature_extraction::horiCloudHandler, this);
        // subVeloCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, &feature_extraction::veloCloudHandler, this);
        subHoriCloud = nh.subscribe<livox_ros_driver::CustomMsg>("/a_horizon_livoxmsg", 100, &feature_extraction::horiCloudHandler, this);
        subVeloCloud = nh.subscribe<sensor_msgs::PointCloud2>("/a_velo", 100, &feature_extraction::veloCloudHandler, this);
        // subHesaCloud = nh.subscribe<sensor_msgs::PointCloud2>("/hesai/pandar", 100, &feature_extraction::hesaiCloudHandler, this);
        subUnionCloud = nh.subscribe<union_cloud::union_cloud>("/a_time_union_cloud", 2, &feature_extraction::unionCloudHandler, this);

        pubHoriCloud = nh.advertise<sensor_msgs::PointCloud2> ("/hori_cloud", 20);
        pubHoriSharp = nh.advertise<sensor_msgs::PointCloud2> ("/hori_cloud_sharp", 20);
        pubHoriFlat = nh.advertise<sensor_msgs::PointCloud2> ("/hori_cloud_flat", 20);
        pubHoriCloudNormal = nh.advertise<sensor_msgs::PointCloud2> ("/hori_cloud_normal", 20);

        pubVeloCloud = nh.advertise<sensor_msgs::PointCloud2> ("/velo_cloud", 20);
        pubVeloCloudNormal = nh.advertise<sensor_msgs::PointCloud2> ("/velo_cloud_normal", 20);
        pubVeloSharp = nh.advertise<sensor_msgs::PointCloud2> ("/velo_cloud_sharp", 20);
        pubVeloFlat = nh.advertise<sensor_msgs::PointCloud2> ("/velo_cloud_flat", 20);

        pubMergedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/merged_cloud", 20);
        pubMergedFeature = nh.advertise<sensor_msgs::PointCloud2> ("/merged_feature_cloud", 20);

        pubUnionFeatureCloud = nh.advertise<union_cloud::union_cloud> ("/union_feature_cloud", 20);
        // Get parameters
        ros::NodeHandle private_nh_("~");
        if (!private_nh_.getParam("hori_feature_num_threshold",   _hori_lowfeature_th))  _hori_lowfeature_th = 200;
        if (!private_nh_.getParam("velo_feature_num_threshold",  _velo_lowfeature_th ))  _velo_lowfeature_th = 200;
        if (!private_nh_.getParam("velo_skip_frames",            _velo_skip_frames ))    _velo_skip_frames = 2;
        if (!private_nh_.getParam("hesai_skip_frames",            _hesai_skip_frames ))  _hesai_skip_frames = 2;
        if (!private_nh_.getParam("near_points_threshold",        _near_points_th ))     _near_points_th = 1.0;
        if (!private_nh_.getParam("far_points_threshold",        _far_points_th ))       _far_points_th = 50.0;
        if (!private_nh_.getParam("extrin_recali_times", _extrin_recali_times))          _extrin_recali_times = 30;
        if (!private_nh_.getParam("pub_feature_points", _pub_feature_points))            _pub_feature_points = false;

        ROS_INFO_STREAM( "hori_feature_num_threshold  : " <<  _hori_lowfeature_th );
        ROS_INFO_STREAM( "velo_feature_num_threshold  : " <<  _velo_lowfeature_th );
        ROS_INFO_STREAM( "velo_skip_frames            : " <<  _velo_skip_frames );
        ROS_INFO_STREAM( "hori_skip_frames            : " <<  _hori_skip_frames );
        ROS_INFO_STREAM( "hesai_skip_frames           : " <<  _hesai_skip_frames );
        ROS_INFO_STREAM( "near_points_threshold       : " <<  _near_points_th );
        ROS_INFO_STREAM( "far_points_threshold       : " <<  _far_points_th );
        ROS_INFO_STREAM( "extrin_recali_times        : " <<  _extrin_recali_times );
        ROS_INFO_STREAM( "pub_feature_points         : " <<  _pub_feature_points );

        // p_pre =  Preprocess() ;
        // p_pre.blind = 4;
        // p_pre.N_SCANS = 16;
        // p_pre.lidar_type = 2;
        // p_pre.SCAN_RATE = 10;
        // p_pre.feature_enabled = true;
        // p_pre.point_filter_num = 4;


    }

    ~feature_extraction(){}

    void unionCloudHandler(const union_cloud::union_cloudConstPtr & msg ){
        // livox_ros_driver::CustomMsg livo_msg = msg->livox_time_aligned;
        // sensor_msgs::PointCloud2    velo_msg = msg->velo_time_aligned;

        auto tick = std::chrono::high_resolution_clock::now();

        union_msg.header = msg->header;

        sensor_msgs::PointCloud2  livo_CloudMsg;
        sensor_msgs::PointCloud2  livo_SurfCloudMsg;
        sensor_msgs::PointCloud2  livo_CornerCloudMsg;
        livox_ros_driver::CustomMsgConstPtr livox_msg( new livox_ros_driver::CustomMsg( msg->livox_time_aligned ) );
        getHoriFeature( livox_msg, std::ref(livo_CloudMsg), livo_SurfCloudMsg,  livo_CornerCloudMsg );

        // velo points processing
        sensor_msgs::PointCloud2 velo_pts_normal;
        sensor_msgs::PointCloud2 velo_pts_corner;
        sensor_msgs::PointCloud2 velo_pts_surf;
        getVeloFeature( msg->velo_time_aligned, velo_pts_normal, velo_pts_corner, velo_pts_surf);


        union_msg.livox_corner  =  livo_CornerCloudMsg;
        union_msg.livox_surface =  livo_SurfCloudMsg;
        union_msg.livox_combine =  livo_CloudMsg;

        union_msg.velo_corner   =  velo_pts_corner;
        union_msg.velo_surface  =  velo_pts_surf;
        union_msg.velo_combine  =  velo_pts_normal;

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr livoSurfPtr (new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::fromROSMsg(livo_SurfCloudMsg , *livoSurfPtr);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr livoCombinePtr (new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::fromROSMsg(livo_CloudMsg , *livoCombinePtr);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr veloSurfPtr (new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::fromROSMsg(velo_pts_surf , *veloSurfPtr);

        if( union_msg.livox_corner_num > 100)
        {
            // cloud matching
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_aligned(new pcl::PointCloud<pcl::PointXYZINormal>);

            if( _extrin_cnt % _extrin_recali_times == 0){
                icp_ext_matching(livoSurfPtr, veloSurfPtr, cloud_aligned, extri_mtx, false);
            }
            // icp_ext_matching(laserCloudFullHoriRes, laserCloudFullVeloRes, cloud_aligned, icp_mtx, false))

            pcl::transformPointCloud(*livoCombinePtr, *livoCombinePtr, extri_mtx);

            sensor_msgs::PointCloud2 tfHoriCloudMsg;
            pcl::toROSMsg(*livoCombinePtr, tfHoriCloudMsg);
            tfHoriCloudMsg.header = livo_CloudMsg.header;

            union_msg.livox_combine =  tfHoriCloudMsg;
        }

        pubUnionFeatureCloud.publish(union_msg);

        processed_cnt++;
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = 1.e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1-tick).count();

        time_cost = dt + time_cost ;
        ROS_WARN_STREAM("[FeatureExtraction]Time cost : " << time_cost / processed_cnt);

        ROS_INFO_STREAM("Pub union aligned cloud spatio & time at " << std::setprecision(15)
                                << union_msg.livox_corner.header.stamp.toSec() << " - " << livo_CornerCloudMsg.width << " | "
                                << union_msg.livox_surface.header.stamp.toSec() << " - " << livo_SurfCloudMsg.width << " | "
                                << union_msg.livox_combine.header.stamp.toSec() << " - " << livo_CloudMsg.width << " | "
                                << union_msg.velo_corner.header.stamp.toSec() << " - " << velo_pts_corner.width << " | "
                                << union_msg.velo_combine.header.stamp.toSec() << " - " << velo_pts_surf.width << " | "
                                << union_msg.velo_combine.header.stamp.toSec() << " - " << velo_pts_normal.width);
    }



    void detectFeaturePoints(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud,
                                                    std::vector<int>& pointsLessSharp,
                                                    std::vector<int>& pointsLessFlat)
    {
    int CloudFeatureFlag[20000];
    float cloudCurvature[20000];
    float cloudDepth[20000];
    int cloudSortInd[20000];
    float cloudReflect[20000];
    int reflectSortInd[20000];
    int cloudAngle[20000];

    int thNumCurvSize = 2;
    float thDistanceFaraway = 50.0;
    int thNumFlat = 1; // maxium number of flat points in each part
    int thPartNum = 50;
    float thFlatThreshold = 0.02;
    float thLidarNearestDis = 1.0;
    float thBreakCornerDis = 1;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr& laserCloudIn = cloud;

    int cloudSize = laserCloudIn->points.size();

    pcl::PointXYZINormal point;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr _laserCloud(new pcl::PointCloud<pcl::PointXYZINormal>());
    _laserCloud->reserve(cloudSize);

    for (int i = 0; i < cloudSize; i++) {
        point.x = laserCloudIn->points[i].x;
        point.y = laserCloudIn->points[i].y;
        point.z = laserCloudIn->points[i].z;
    #ifdef UNDISTORT
        point.normal_x = laserCloudIn.points[i].normal_x;
    #else
        point.normal_x = 1.0;
    #endif
        point.intensity = laserCloudIn->points[i].intensity;

        if (!pcl_isfinite(point.x) ||
            !pcl_isfinite(point.y) ||
            !pcl_isfinite(point.z)) {
        continue;
        }

        _laserCloud->push_back(point);
        CloudFeatureFlag[i] = 0;
    }

    cloudSize = _laserCloud->size();

    int debugnum1 = 0;
    int debugnum2 = 0;
    int debugnum3 = 0;
    int debugnum4 = 0;
    int debugnum5 = 0;

    int count_num = 1;
    bool left_surf_flag = false;
    bool right_surf_flag = false;

    int scanStartInd = 5;
    int scanEndInd = cloudSize - 6;

    int thDistanceFaraway_fea = 0;

    for (int i = 5; i < cloudSize - 5; i ++ ) {

        float diffX = 0;
        float diffY = 0;
        float diffZ = 0;

        float dis = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                        _laserCloud->points[i].y * _laserCloud->points[i].y +
                        _laserCloud->points[i].z * _laserCloud->points[i].z);

        Eigen::Vector3d pt_last(_laserCloud->points[i-1].x, _laserCloud->points[i-1].y, _laserCloud->points[i-1].z);
        Eigen::Vector3d pt_cur(_laserCloud->points[i].x, _laserCloud->points[i].y, _laserCloud->points[i].z);
        Eigen::Vector3d pt_next(_laserCloud->points[i+1].x, _laserCloud->points[i+1].y, _laserCloud->points[i+1].z);

        double angle_last = (pt_last-pt_cur).dot(pt_cur) / ((pt_last-pt_cur).norm()*pt_cur.norm());
        double angle_next = (pt_next-pt_cur).dot(pt_cur) / ((pt_next-pt_cur).norm()*pt_cur.norm());

        if (dis > thDistanceFaraway || (fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966)) {
        thNumCurvSize = 2;
        } else {
        thNumCurvSize = 3;
        }

        if(fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966) {
        cloudAngle[i] = 1;
        }

        float diffR = -2 * thNumCurvSize * _laserCloud->points[i].intensity;
        for (int j = 1; j <= thNumCurvSize; ++j) {
        diffX += _laserCloud->points[i - j].x + _laserCloud->points[i + j].x;
        diffY += _laserCloud->points[i - j].y + _laserCloud->points[i + j].y;
        diffZ += _laserCloud->points[i - j].z + _laserCloud->points[i + j].z;
        diffR += _laserCloud->points[i - j].intensity + _laserCloud->points[i + j].intensity;
        }
        diffX -= 2 * thNumCurvSize * _laserCloud->points[i].x;
        diffY -= 2 * thNumCurvSize * _laserCloud->points[i].y;
        diffZ -= 2 * thNumCurvSize * _laserCloud->points[i].z;

        cloudDepth[i] = dis;
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;// / (2 * thNumCurvSize * dis + 1e-3);
        cloudSortInd[i] = i;
        cloudReflect[i] = diffR;
        reflectSortInd[i] = i;

    }

    for (int j = 0; j < thPartNum; j++) {
        int sp = scanStartInd + (scanEndInd - scanStartInd) * j / thPartNum;
        int ep = scanStartInd + (scanEndInd - scanStartInd) * (j + 1) / thPartNum - 1;

        // sort the curvatures from small to large
        for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
            if (cloudCurvature[cloudSortInd[l]] <
                cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
            }
        }
        }

        // sort the reflectivity from small to large
        for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
            if (cloudReflect[reflectSortInd[l]] <
                cloudReflect[reflectSortInd[l - 1]]) {
            int temp = reflectSortInd[l - 1];
            reflectSortInd[l - 1] = reflectSortInd[l];
            reflectSortInd[l] = temp;
            }
        }
        }

        int smallestPickedNum = 1;
        int sharpestPickedNum = 1;
        for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];

        if (CloudFeatureFlag[ind] != 0) continue;

        if (cloudCurvature[ind] < thFlatThreshold * cloudDepth[ind] * thFlatThreshold * cloudDepth[ind]) {

            CloudFeatureFlag[ind] = 3;

            for (int l = 1; l <= thNumCurvSize; l++) {
            float diffX = _laserCloud->points[ind + l].x -
                            _laserCloud->points[ind + l - 1].x;
            float diffY = _laserCloud->points[ind + l].y -
                            _laserCloud->points[ind + l - 1].y;
            float diffZ = _laserCloud->points[ind + l].z -
                            _laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
                break;
            }

            CloudFeatureFlag[ind + l] = 1;
            }
            for (int l = -1; l >= -thNumCurvSize; l--) {
            float diffX = _laserCloud->points[ind + l].x -
                            _laserCloud->points[ind + l + 1].x;
            float diffY = _laserCloud->points[ind + l].y -
                            _laserCloud->points[ind + l + 1].y;
            float diffZ = _laserCloud->points[ind + l].z -
                            _laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
                break;
            }

            CloudFeatureFlag[ind + l] = 1;
            }
        }
        }

        for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];
        if(((CloudFeatureFlag[ind] == 3) && (smallestPickedNum <= thNumFlat)) ||
            ((CloudFeatureFlag[ind] == 3) && (cloudDepth[ind] > thDistanceFaraway)) ||
            cloudAngle[ind] == 1){
            smallestPickedNum ++;
            CloudFeatureFlag[ind] = 2;
            if(cloudDepth[ind] > thDistanceFaraway) {
            thDistanceFaraway_fea++;
            }
        }

        int idx = reflectSortInd[k];
        if(cloudCurvature[idx] < 0.7 * thFlatThreshold * cloudDepth[idx] * thFlatThreshold * cloudDepth[idx]
            && sharpestPickedNum <= 3 && cloudReflect[idx] > 20.0){
            sharpestPickedNum ++;
            CloudFeatureFlag[idx] = 300;
        }
        }

    }

    for (int i = 5; i < cloudSize - 5; i += count_num ) {
        float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                        _laserCloud->points[i].y * _laserCloud->points[i].y +
                        _laserCloud->points[i].z * _laserCloud->points[i].z);
        //left curvature
        float ldiffX =
                _laserCloud->points[i - 4].x + _laserCloud->points[i - 3].x
                - 4 * _laserCloud->points[i - 2].x
                + _laserCloud->points[i - 1].x + _laserCloud->points[i].x;

        float ldiffY =
                _laserCloud->points[i - 4].y + _laserCloud->points[i - 3].y
                - 4 * _laserCloud->points[i - 2].y
                + _laserCloud->points[i - 1].y + _laserCloud->points[i].y;

        float ldiffZ =
                _laserCloud->points[i - 4].z + _laserCloud->points[i - 3].z
                - 4 * _laserCloud->points[i - 2].z
                + _laserCloud->points[i - 1].z + _laserCloud->points[i].z;

        float left_curvature = ldiffX * ldiffX + ldiffY * ldiffY + ldiffZ * ldiffZ;

        if(left_curvature < thFlatThreshold * depth){

            std::vector<pcl::PointXYZINormal> left_list;

            for(int j = -4; j < 0; j++){
                left_list.push_back(_laserCloud->points[i + j]);
            }

            left_surf_flag = true;
        }
        else{
        left_surf_flag = false;
        }

        //right curvature
        float rdiffX =
                _laserCloud->points[i + 4].x + _laserCloud->points[i + 3].x
                - 4 * _laserCloud->points[i + 2].x
                + _laserCloud->points[i + 1].x + _laserCloud->points[i].x;

        float rdiffY =
                _laserCloud->points[i + 4].y + _laserCloud->points[i + 3].y
                - 4 * _laserCloud->points[i + 2].y
                + _laserCloud->points[i + 1].y + _laserCloud->points[i].y;

        float rdiffZ =
                _laserCloud->points[i + 4].z + _laserCloud->points[i + 3].z
                - 4 * _laserCloud->points[i + 2].z
                + _laserCloud->points[i + 1].z + _laserCloud->points[i].z;

        float right_curvature = rdiffX * rdiffX + rdiffY * rdiffY + rdiffZ * rdiffZ;

        if(right_curvature < thFlatThreshold * depth){
        std::vector<pcl::PointXYZINormal> right_list;

        for(int j = 1; j < 5; j++){
            right_list.push_back(_laserCloud->points[i + j]);
        }
        count_num = 4;
        right_surf_flag = true;
        }
        else{
        count_num = 1;
        right_surf_flag = false;
        }

        //calculate the included angle
        if(left_surf_flag && right_surf_flag){
        debugnum4 ++;

        Eigen::Vector3d norm_left(0,0,0);
        Eigen::Vector3d norm_right(0,0,0);
        for(int k = 1;k<5;k++){
            Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                                _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                                _laserCloud->points[i - k].z - _laserCloud->points[i].z);
            tmp.normalize();
            norm_left += (k/10.0)* tmp;
        }
        for(int k = 1;k<5;k++){
            Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                                _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                                _laserCloud->points[i + k].z - _laserCloud->points[i].z);
            tmp.normalize();
            norm_right += (k/10.0)* tmp;
        }

        //calculate the angle between this group and the previous group
        double cc = fabs( norm_left.dot(norm_right) / (norm_left.norm()*norm_right.norm()) );
        //calculate the maximum distance, the distance cannot be too small
        Eigen::Vector3d last_tmp = Eigen::Vector3d(_laserCloud->points[i - 4].x - _laserCloud->points[i].x,
                                                    _laserCloud->points[i - 4].y - _laserCloud->points[i].y,
                                                    _laserCloud->points[i - 4].z - _laserCloud->points[i].z);
        Eigen::Vector3d current_tmp = Eigen::Vector3d(_laserCloud->points[i + 4].x - _laserCloud->points[i].x,
                                                        _laserCloud->points[i + 4].y - _laserCloud->points[i].y,
                                                        _laserCloud->points[i + 4].z - _laserCloud->points[i].z);
        double last_dis = last_tmp.norm();
        double current_dis = current_tmp.norm();

        if(cc < 0.5 && last_dis > 0.05 && current_dis > 0.05 ){ //
            debugnum5 ++;
            CloudFeatureFlag[i] = 150;
        }
        }

    }
    for(int i = 5; i < cloudSize - 5; i ++){
        float diff_left[2];
        float diff_right[2];
        float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                        _laserCloud->points[i].y * _laserCloud->points[i].y +
                        _laserCloud->points[i].z * _laserCloud->points[i].z);

        for(int count = 1; count < 3; count++ ){
        float diffX1 = _laserCloud->points[i + count].x - _laserCloud->points[i].x;
        float diffY1 = _laserCloud->points[i + count].y - _laserCloud->points[i].y;
        float diffZ1 = _laserCloud->points[i + count].z - _laserCloud->points[i].z;
        diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);

        float diffX2 = _laserCloud->points[i - count].x - _laserCloud->points[i].x;
        float diffY2 = _laserCloud->points[i - count].y - _laserCloud->points[i].y;
        float diffZ2 = _laserCloud->points[i - count].z - _laserCloud->points[i].z;
        diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
        }

        float depth_right = sqrt(_laserCloud->points[i + 1].x * _laserCloud->points[i + 1].x +
                                _laserCloud->points[i + 1].y * _laserCloud->points[i + 1].y +
                                _laserCloud->points[i + 1].z * _laserCloud->points[i + 1].z);
        float depth_left = sqrt(_laserCloud->points[i - 1].x * _laserCloud->points[i - 1].x +
                                _laserCloud->points[i - 1].y * _laserCloud->points[i - 1].y +
                                _laserCloud->points[i - 1].z * _laserCloud->points[i - 1].z);

        if(fabs(diff_right[0] - diff_left[0]) > thBreakCornerDis){
        if(diff_right[0] > diff_left[0]){

            Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i - 1].x - _laserCloud->points[i].x,
                                                        _laserCloud->points[i - 1].y - _laserCloud->points[i].y,
                                                        _laserCloud->points[i - 1].z - _laserCloud->points[i].z);
            Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                        _laserCloud->points[i].y,
                                                        _laserCloud->points[i].z);
            double left_surf_dis = surf_vector.norm();
            //calculate the angle between the laser direction and the surface
            double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

            std::vector<pcl::PointXYZINormal> left_list;
            double min_dis = 10000;
            double max_dis = 0;
            for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support
            left_list.push_back(_laserCloud->points[i - j]);
            Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i - j].x - _laserCloud->points[i - j - 1].x,
                                                            _laserCloud->points[i - j].y - _laserCloud->points[i - j - 1].y,
                                                            _laserCloud->points[i - j].z - _laserCloud->points[i - j - 1].z);

            if(j == 3) break;
            double temp_dis = temp_vector.norm();
            if(temp_dis < min_dis) min_dis = temp_dis;
            if(temp_dis > max_dis) max_dis = temp_dis;
            }
            // bool left_is_plane = plane_judge(left_list,100);

            if( cc < 0.95 ){//(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
            if(depth_right > depth_left){
                CloudFeatureFlag[i] = 100;
            }
            else{
                if(depth_right == 0) CloudFeatureFlag[i] = 100;
            }
            }
        }
        else{

            Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i + 1].x - _laserCloud->points[i].x,
                                                        _laserCloud->points[i + 1].y - _laserCloud->points[i].y,
                                                        _laserCloud->points[i + 1].z - _laserCloud->points[i].z);
            Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                        _laserCloud->points[i].y,
                                                        _laserCloud->points[i].z);
            double right_surf_dis = surf_vector.norm();
            //calculate the angle between the laser direction and the surface
            double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

            std::vector<pcl::PointXYZINormal> right_list;
            double min_dis = 10000;
            double max_dis = 0;
            for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
            right_list.push_back(_laserCloud->points[i - j]);
            Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i + j].x - _laserCloud->points[i + j + 1].x,
                                                            _laserCloud->points[i + j].y - _laserCloud->points[i + j + 1].y,
                                                            _laserCloud->points[i + j].z - _laserCloud->points[i + j + 1].z);

            if(j == 3) break;
            double temp_dis = temp_vector.norm();
            if(temp_dis < min_dis) min_dis = temp_dis;
            if(temp_dis > max_dis) max_dis = temp_dis;
            }
            // bool right_is_plane = plane_judge(right_list,100);

            if( cc < 0.95){ //right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

            if(depth_right < depth_left){
                CloudFeatureFlag[i] = 100;
            }
            else{
                if(depth_left == 0) CloudFeatureFlag[i] = 100;
            }
            }
        }
        }

        // break points select
        if(CloudFeatureFlag[i] == 100){
        debugnum2++;
        std::vector<Eigen::Vector3d> front_norms;
        Eigen::Vector3d norm_front(0,0,0);
        Eigen::Vector3d norm_back(0,0,0);

        for(int k = 1;k<4;k++){

            float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                            _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                            _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

            if(temp_depth < 1){
            continue;
            }

            Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                                _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                                _laserCloud->points[i - k].z - _laserCloud->points[i].z);
            tmp.normalize();
            front_norms.push_back(tmp);
            norm_front += (k/6.0)* tmp;
        }
        std::vector<Eigen::Vector3d> back_norms;
        for(int k = 1;k<4;k++){

            float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                            _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                            _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

            if(temp_depth < 1){
            continue;
            }

            Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                                _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                                _laserCloud->points[i + k].z - _laserCloud->points[i].z);
            tmp.normalize();
            back_norms.push_back(tmp);
            norm_back += (k/6.0)* tmp;
        }
        double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
        if(cc < 0.95){
            debugnum3++;
        }else{
            CloudFeatureFlag[i] = 101;
        }

        }

    }

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloudCorner(new pcl::PointCloud<pcl::PointXYZINormal>());
    pcl::PointCloud<pcl::PointXYZINormal> cornerPointsSharp;

    std::vector<int> pointsLessSharp_ori;

    int num_surf = 0;
    int num_corner = 0;

    //push_back feature

    for(int i = 5; i < cloudSize - 5; i ++){

        float dis = _laserCloud->points[i].x * _laserCloud->points[i].x
                + _laserCloud->points[i].y * _laserCloud->points[i].y
                + _laserCloud->points[i].z * _laserCloud->points[i].z;

        if(dis < thLidarNearestDis*thLidarNearestDis) continue;

        if(CloudFeatureFlag[i] == 2){
        pointsLessFlat.push_back(i);
        num_surf++;
        continue;
        }

        if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 150){ //
        pointsLessSharp_ori.push_back(i);
        laserCloudCorner->push_back(_laserCloud->points[i]);
        }

    }

    for(int i = 0; i < laserCloudCorner->points.size();i++){
        pointsLessSharp.push_back(pointsLessSharp_ori[i]);
        num_corner++;
    }

    }

    sensor_msgs::PointCloud2 livoxToSensorMsg(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in)
    {
        pcl::PointCloud<pcl::PointXYZI> pcl_in;
        for (unsigned int i = 0; i < livox_msg_in->point_num; ++i) {
            pcl::PointXYZI pt;
            pt.x = livox_msg_in->points[i].x;
            pt.y = livox_msg_in->points[i].y;
            pt.z = livox_msg_in->points[i].z;
            pt.intensity =  livox_msg_in->points[i].reflectivity;
            pcl_in.push_back(pt);
        }
        ros::Time timestamp(livox_msg_in->header.stamp.toSec());
        sensor_msgs::PointCloud2 pcl_ros_msg;
        pcl::toROSMsg(pcl_in, pcl_ros_msg);
        pcl_ros_msg.header.stamp = timestamp;
        pcl_ros_msg.header.frame_id = livox_msg_in->header.frame_id;

        return pcl_ros_msg;
    }

    void horiCloudHandler(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in)
    {
        // std::unique_lock<std::mutex> lock_hori(_mtxHoriBuffer);
        // _horiMsgBuffer.push_back(*livox_msg_in);
        // lock_hori.unlock();

        // std::unique_lock<std::mutex> lock_hori(_mtxHoriBuffer);
        // getHoriFeature(_horiMsgBuffer[0]);
        // _horiMsgBuffer.erase(_horiMsgBuffer.begin());
        // lock_hori.unlock();

        _hori_frames_cnt++;
        // ROS_INFO_STREAM("Horizon Cloud Handler,  msg  " << _hori_frames_cnt);

        if(_hori_frames_cnt%_hori_skip_frames == 0)
        {
            sensor_msgs::PointCloud2  livo_CloudMsg;
            sensor_msgs::PointCloud2  livo_SurfCloudMsg;
            sensor_msgs::PointCloud2  livo_CornerCloudMsg;
            getHoriFeature( livox_msg_in, livo_CloudMsg, livo_SurfCloudMsg, livo_CornerCloudMsg );
        }
        //  getHoriFeature(livox_msg_in);

    }

    void getHoriFeature(const livox_ros_driver::CustomMsgConstPtr&   msg,
                        sensor_msgs::PointCloud2 &laserCloudMsg,
                        sensor_msgs::PointCloud2 &laserSurfCloudMsg,
                        sensor_msgs::PointCloud2 &laserCornerCloudMsg
    )
    {
        // hori points processing
        pcl::PointCloud< pcl::PointXYZINormal>::Ptr horiLaserCloud;
        pcl::PointCloud< pcl::PointXYZINormal>::Ptr laserConerCloud;
        pcl::PointCloud< pcl::PointXYZINormal>::Ptr laserSurfCloud;

        horiLaserCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
        laserConerCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);
        laserSurfCloud.reset(new pcl::PointCloud<pcl::PointXYZINormal>);

        int N_SCANS = 6;
        getHoriFeatureExtract(msg, horiLaserCloud, laserConerCloud, laserSurfCloud,N_SCANS);

        double stamp_sec = msg->header.stamp.toSec();
        ros::Time stamp(stamp_sec);


        // ROS_INFO_STREAM("Punlishing Horizon features, points:  " << horiLaserCloud->points.size());
        pcl::PointCloud<pcl::PointXYZINormal> combine_cuttedCloud;
        // removeNearPointCloud(*horiLaserCloud, combine_cuttedCloud, _near_points_th);
        removeNearFarPoints(*horiLaserCloud, combine_cuttedCloud, _near_points_th, _far_points_th);
        // sensor_msgs::PointCloud2 horiLaserCloudMsg;
        pcl::toROSMsg(combine_cuttedCloud, laserCloudMsg);
        laserCloudMsg.header.frame_id = "velodyne";
        // laserCloudMsg.header.stamp.fromNSec(msg->timebase+msg->points.back().offset_time);
        laserCloudMsg.header.stamp = stamp ;

        pcl::PointCloud<pcl::PointXYZINormal>  surf_cuttedCloud;
        // sensor_msgs::PointCloud2 laserSurfCloudMsg;
        removeNearPointCloud(*laserSurfCloud, surf_cuttedCloud, _near_points_th);
        pcl::toROSMsg(surf_cuttedCloud, laserSurfCloudMsg);
        laserSurfCloudMsg.header.frame_id = "velodyne";
        // laserSurfCloudMsg.header.stamp.fromNSec(msg->timebase+msg->points.back().offset_time);
        laserCloudMsg.header.stamp = stamp ;

        pcl::PointCloud<pcl::PointXYZINormal>  corner_cuttedCloud;
        // sensor_msgs::PointCloud2 laserCornerCloudMsg;
        removeNearPointCloud(*laserConerCloud, corner_cuttedCloud, _near_points_th);
        pcl::toROSMsg(corner_cuttedCloud, laserCornerCloudMsg);
        laserCornerCloudMsg.header.frame_id = "velodyne";
        // laserCornerCloudMsg.header.stamp.fromNSec(msg->timebase+msg->points.back().offset_time);
        laserCloudMsg.header.stamp = stamp ;

        union_msg.livox_corner_num = corner_cuttedCloud.points.size();
        union_msg.livox_surf_num   = surf_cuttedCloud.points.size();

        if(_pub_feature_points)
        {
            pubHoriCloudNormal.publish(laserCloudMsg);
            pubHoriFlat.publish(laserSurfCloudMsg);
            pubHoriSharp.publish(laserCornerCloudMsg);
        }

        return;
    }

    void getHoriFeatureExtract(const livox_ros_driver::CustomMsgConstPtr &msg,
                        pcl::PointCloud<pcl::PointXYZINormal>::Ptr& laserCloud,
                        pcl::PointCloud<pcl::PointXYZINormal>::Ptr& laserConerFeature,
                        pcl::PointCloud<pcl::PointXYZINormal>::Ptr& laserSurfFeature,
                        const int Used_Line)
    {
        // ROS_INFO_STREAM("Extracting Horizon features " << _hori_frames_cnt);
        int N_SCANS = 6;
        std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> vlines;
        std::vector<std::vector<int>> vcorner;
        std::vector<std::vector<int>> vsurf;

        laserCloud->clear();
        laserConerFeature->clear();
        laserSurfFeature->clear();
        laserCloud->reserve(15000*N_SCANS);

        vlines.resize(N_SCANS);
        for(auto & ptr : vlines){
            ptr.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        }
        vcorner.resize(N_SCANS);
        vsurf.resize(N_SCANS);

        // for(auto & ptr : vlines){
        //     ptr->clear();
        // }
        // for(auto & v : vcorner){
        //     v.clear();
        // }
        // for(auto & v : vsurf){
        //     v.clear();
        // }
        double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
        pcl::PointXYZINormal point;
        for(const auto& p : msg->points){
            int line_num = (int)p.line;
            if(line_num > Used_Line-1) continue;
            if(p.x < 0.01) continue;
            point.x = p.x;
            point.y = p.y;
            point.z = p.z;
            point.intensity = p.reflectivity;
            point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
            point.normal_y = line_num * 1.0;
            laserCloud->push_back(point);
        }
        std::size_t cloud_num = laserCloud->size();

        for(std::size_t i=0; i<cloud_num; ++i){
            int line_idx = int(laserCloud->points[i].normal_y);
            laserCloud->points[i].normal_z = i * 1.0;
            vlines[line_idx]->push_back(laserCloud->points[i]);
            laserCloud->points[i].normal_z = 0;
        }

        std::thread threads[N_SCANS];
        for(int i=0; i<N_SCANS; ++i){
            threads[i] = std::thread(&feature_extraction::detectFeaturePoints, this, std::ref(vlines[i]),
                            std::ref(vcorner[i]), std::ref(vsurf[i]));
        }
        for(int i=0; i<N_SCANS; ++i){
            threads[i].join();
        }
        for(int i=0; i<N_SCANS; ++i){
            for(int j=0; j<vcorner[i].size(); ++j){
                laserCloud->points[int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0;
            }
            for(int j=0; j<vsurf[i].size(); ++j){
                laserCloud->points[int(vlines[i]->points[vsurf[i][j]].normal_z)].normal_z = 2.0;
            }
        }

        for(const auto& p : laserCloud->points){
            if(std::fabs(p.normal_z - 1.0) < 1e-5)
            laserConerFeature->push_back(p);
        }
        for(const auto& p : laserCloud->points){
            if(std::fabs(p.normal_z - 2.0) < 1e-5)
            laserSurfFeature->push_back(p);
        }


    }

    bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]);}

    bool get_is_lowFeatureEnv(){
        return  _hori_lowfeature_env;
    }

    bool get_is_lowFeatureEnv(const livox_ros_driver::CustomMsgConstPtr& livoxcloud, int lowfeature_th){
         _hori_lowfeature_th = lowfeature_th;
        horiCloudHandler(livoxcloud);
        return  _hori_lowfeature_env;
    }

    void veloCloudHandler(const sensor_msgs::PointCloud2::ConstPtr & laserCloudMsg)
    {   _velo_frames_cnt++;
        if(_velo_frames_cnt%_velo_skip_frames == 0)
        {
            sensor_msgs::PointCloud2 velo_pts_normal;
            sensor_msgs::PointCloud2 velo_pts_corner;
            sensor_msgs::PointCloud2 velo_pts_surf;
            getVeloFeature( *laserCloudMsg, velo_pts_normal, velo_pts_corner, velo_pts_surf);
            // getVeloFeature(*laserCloudMsg);

            // velo points processing
            std::cout << " msgs  Velo published " << std::endl;
            union_msg.velo_corner   =  velo_pts_corner;
            union_msg.velo_surface  =  velo_pts_surf;
            union_msg.velo_combine  =  velo_pts_normal;
            pubUnionFeatureCloud.publish(union_msg);

            pcl::PointCloud<pcl::PointXYZINormal>::Ptr  ptr(new pcl::PointCloud<pcl::PointXYZINormal>());
            // p_pre.process(laserCloudMsg, ptr);
            // pcl::PointCloud<pcl::PointXYZINormal> surfFeatureCloud = p_pre.pl_surf;
            // pcl::PointCloud<pcl::PointXYZINormal> cornFeatureCloud = p_pre.pl_corn;
            // ROS_INFO_STREAM("Feature points " << surfFeatureCloud.points.size() << " | "
            //     << cornFeatureCloud.points.size());

            // double stamp_sec = laserCloudMsg->header.stamp.toSec();
            // ros::Time stamp(stamp_sec);

            // pubVeloCloud.publish(*laserCloudMsg);
            // ROS_WARN_STREAM("FeatureExtraction -> Velo ...corner_cnt:" << corner_cnt << " , surf_cnt:" << surf_cnt);
            // sensor_msgs::PointCloud2 points_normal;
            // pcl::PointCloud<pcl::PointXYZINormal> cuttedCloud;
            // removeNearFarPoints(*laserCloud, cuttedCloud, _near_points_th, _far_points_th);
            // pcl::toROSMsg(cuttedCloud, points_normal);
            // points_normal.header.stamp = stamp;
            // points_normal.header.frame_id = "velodyne";
            // pubVeloCloudNormal.publish(points_normal);

            // sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
            // pcl::toROSMsg(cornFeatureCloud, cornerPointsLessSharpMsg);
            // cornerPointsLessSharpMsg.header.stamp = stamp;
            // cornerPointsLessSharpMsg.header.frame_id = "velodyne";
            // pubVeloSharp.publish(cornerPointsLessSharpMsg);

            // sensor_msgs::PointCloud2 surfPointsLessFlat2;
            // pcl::toROSMsg(surfFeatureCloud, surfPointsLessFlat2);
            // surfPointsLessFlat2.header.stamp = stamp;
            // surfPointsLessFlat2.header.frame_id = "velodyne";
            // pubVeloFlat.publish(surfPointsLessFlat2);
        }
        // std::unique_lock<std::mutex> lock_velo(_mtxVeloBuffer);
        // _veloMsgBuffer.push_back(laserCloudMsg);
        // lock_velo.unlock();

        // /******** process features ********/
        // std::unique_lock<std::mutex> lock_velo(_mtxVeloBuffer);
        // getVeloFeature(*_veloMsgBuffer[0]);
        // _veloMsgBuffer.erase(_veloMsgBuffer.begin());
        // lock_velo.unlock();


        // bufferProcessing();
    }


    void getVeloFeature(    sensor_msgs::PointCloud2 laserCloudMsg,
                            sensor_msgs::PointCloud2 &points_normal,
                            sensor_msgs::PointCloud2 &cornerPointsLessSharpMsg,
                            sensor_msgs::PointCloud2 &surfPointsLessFlat2
        )
    {
        // // if(! _hori_lowfeature_env)
        // //     return;

        // int cloudSortInd[400000];
        // int cloudNeighborPicked[400000];
        // int cloudLabel[400000];

        // std::vector<int> scanStartInd(VELO_N_SCANS, 0);
        // std::vector<int> scanEndInd(VELO_N_SCANS, 0);

        pcl::PointCloud< pcl::PointXYZI> lidar_cloud_in;
        pcl::fromROSMsg(laserCloudMsg, lidar_cloud_in);
        std::vector<int> indices;

        pcl::removeNaNFromPointCloud(lidar_cloud_in, lidar_cloud_in, indices);

        int cloudSize = lidar_cloud_in.points.size();
        float startOri = -atan2(lidar_cloud_in.points[0].y, lidar_cloud_in.points[0].x);
        float endOri = -atan2(lidar_cloud_in.points[cloudSize - 1].y,
                lidar_cloud_in.points[cloudSize - 1].x) +
                2 * M_PI;

        if (endOri - startOri > 3 * M_PI)
            endOri -= 2 * M_PI;

        else if (endOri - startOri < M_PI)
            endOri += 2 * M_PI;


        bool halfPassed = false;
        int count = cloudSize;
        pcl::PointXYZINormal point;
        pcl::PointXYZINormal point_undis;
        std::vector<pcl::PointCloud<pcl::PointXYZINormal>> laserCloudScans(VELO_N_SCANS);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserCloud(new pcl::PointCloud<pcl::PointXYZINormal>());
        for (int i = 0; i < cloudSize; i++) {
            point.x = lidar_cloud_in.points[i].x;
            point.y = lidar_cloud_in.points[i].y;
            point.z = lidar_cloud_in.points[i].z;

            float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            int scanID = 0;

            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (VELO_N_SCANS - 1) || scanID < 0) {
                count--;
                continue;
            }

            float ori = -atan2(point.y, point.x);
            if (!halfPassed) {
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
            point.normal_x =  relTime;
            point.normal_y =  scanID ;
            // std::cout<< point.normal_y  << " | ";
            point.normal_z = 0;
            point.intensity = lidar_cloud_in.points[i].intensity;

            laserCloud->push_back(point);
            laserCloudScans[scanID].push_back(point);
        }

        std::size_t cloud_num = laserCloud->size();
        std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> vlines;
        std::vector<std::vector<int>> vcorner;
        std::vector<std::vector<int>> vsurf;

        vlines.resize(VELO_N_SCANS);
        for(auto & ptr : vlines){
            ptr.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        }
        vcorner.resize(VELO_N_SCANS);
        vsurf.resize(VELO_N_SCANS);

        for(std::size_t i=0; i<cloud_num; ++i){
            int line_idx = int(laserCloud->points[i].normal_y);
            if(line_idx >= 0 && line_idx < VELO_N_SCANS){
                laserCloud->points[i].normal_z = i * 1.0;
                vlines[line_idx]->push_back(laserCloud->points[i]);
                laserCloud->points[i].normal_z = 0;
            }else{
                ROS_INFO_STREAM("Invalid scan line_idx estimation : " << line_idx << " | " << laserCloud->points[i].normal_y );
            }
        }

        // std::thread threads[VELO_N_SCANS];
        // for(int i=0; i<VELO_N_SCANS; ++i){
        //     threads[i] = std::thread(&feature_extraction::detectFeaturePoints, this, std::ref(vlines[i]),
        //                     std::ref(vcorner[i]), std::ref(vsurf[i]));
        // }
        // for(int i=0; i<VELO_N_SCANS; ++i){
        //     threads[i].join();
        // }
        for(int i=0; i<VELO_N_SCANS; ++i){
            detectFeaturePoints( std::ref(vlines[i]), std::ref(vcorner[i]), std::ref(vsurf[i]));
        }


        for(int i=0; i<VELO_N_SCANS; ++i){
            for(int j=0; j<vcorner[i].size(); ++j){
                laserCloud->points[int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0;
            }
            for(int j=0; j<vsurf[i].size(); ++j){
                laserCloud->points[int(vlines[i]->points[vsurf[i][j]].normal_z)].normal_z = 2.0;
            }
        }

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserConerFeature (new pcl::PointCloud<pcl::PointXYZINormal>);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr laserSurfFeature (new pcl::PointCloud<pcl::PointXYZINormal>);
        for(const auto& p : laserCloud->points){
            if(std::fabs(p.normal_z - 1.0) < 1e-5){
                laserConerFeature->push_back(p);
            }
        }
        for(const auto& p : laserCloud->points){
            if(std::fabs(p.normal_z - 2.0) < 1e-5)
            laserSurfFeature->push_back(p);
        }

        for( auto  & p : laserCloud->points){
            p.intensity = 0; // invisiable velodyne points
        }
        // int corner_cnt = 0;
        // int surf_cnt = 0;
        // for (int i = 5; i < cloudSize - 5; i++) {
        //     if( cloudLabel[i] == 1){
        //         laserCloud->points[i].normal_z = 1.0;
        //         velo_corner_pts.push_back(laserCloud->points[i]);
        //         corner_cnt++;
        //     }
        //     if ( cloudLabel[i] == -1){
        //         laserCloud->points[i].normal_z = 2.0;
        //         velo_surf_pts.push_back(laserCloud->points[i]);
        //         surf_cnt++;
        //     }
        // }

        double stamp_sec = laserCloudMsg.header.stamp.toSec();
        ros::Time stamp(stamp_sec);

        // pubVeloCloud.publish(*laserCloudMsg);
        // ROS_WARN_STREAM("FeatureExtraction -> Velo ...corner_cnt:" << corner_cnt << " , surf_cnt:" << surf_cnt);
        // sensor_msgs::PointCloud2 points_normal;
        pcl::PointCloud<pcl::PointXYZINormal> cuttedCloud;
        removeNearFarPoints(*laserCloud, cuttedCloud, _near_points_th, _far_points_th);
        pcl::toROSMsg(cuttedCloud, points_normal);
        points_normal.header.stamp = stamp;
        points_normal.header.frame_id = "velodyne";


        // sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
        pcl::PointCloud<pcl::PointXYZINormal> cuttedSharpCloud;
        removeNearFarPoints(*laserConerFeature, cuttedSharpCloud, _near_points_th, _far_points_th);
        pcl::toROSMsg(cuttedSharpCloud, cornerPointsLessSharpMsg);
        cornerPointsLessSharpMsg.header.stamp = stamp;
        cornerPointsLessSharpMsg.header.frame_id = "velodyne";

        // sensor_msgs::PointCloud2 surfPointsLessFlat2;
        pcl::PointCloud<pcl::PointXYZINormal> cuttedSurfCloud;
        removeNearFarPoints(*laserSurfFeature, cuttedSurfCloud, _near_points_th, _far_points_th);
        pcl::toROSMsg(cuttedSurfCloud, surfPointsLessFlat2);
        surfPointsLessFlat2.header.stamp = stamp;
        surfPointsLessFlat2.header.frame_id = "velodyne";

        union_msg.velo_corner_num = cuttedSharpCloud.points.size();
        union_msg.velo_surf_num   = cuttedSurfCloud.points.size();

        if(_pub_feature_points)
        {
            pubVeloCloudNormal.publish(points_normal);
            pubVeloSharp.publish(cornerPointsLessSharpMsg);
            pubVeloFlat.publish(surfPointsLessFlat2);


            // union_msg.velo_corner =  cornerPointsLessSharpMsg;
            // union_msg.velo_surface = surfPointsLessFlat2;
            // union_msg.velo_combine = points_normal;

            std::cout << std::setprecision(15) << stamp << " Velo: Normal ," << cuttedCloud.points.size ()
                                                        << " corner, " << cuttedSharpCloud.points.size ()
                                                        << " surf , " << cuttedSurfCloud.points.size () << std::endl;
        }
    }


};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "UnionFeatureExtraction");
    ROS_INFO("\033[1;32m---->\033[0m LIO UnionFeatureExtraction Started.");
    feature_extraction fe;
    ros::spin();

    return 0;
}
