// BSD 3-Clause License

// Copyright (c) 2021, LIVOX
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.

// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.

// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "Estimator/Estimator.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include "union_om/union_cloud.h"
#include <chrono>
#include <opencv/cv.h>


typedef pcl::PointXYZINormal PointType;

int color_array[20][3] = {
    {128,0, 0}, {255,215,0}, {128,128,0}, {154,205,50}, {0,100,0},
    {0,250,154}, {0,255,255},{175,238,238}, {0,191,255}, {135,206,250},
    {0,0,205}, {138,43,226}, {123,104,238}, {148,0,211}, {216,191,216},
    {199,21,133}, {255,20,147}, {250,235,215}, {255,255,224}, {210,105,30}
};

int hori_frame_cnt = 0;
int velo_frame_cnt = 0;
int frame_cnt = 0;

int WINDOWSIZE;
bool LidarIMUInited = false;
bool VeloIMUInited = false;
bool IMU_MAP_init = true;
bool USE_STEREO_ODOM = true;
bool velo_only_mode  = false;
bool real_time_mode  = false;
boost::shared_ptr<std::list<Estimator::LidarFrame>> lidarFrameList;
boost::shared_ptr<std::list<Estimator::LidarFrame>> veloFrameList;
pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
pcl::PointCloud<PointType>::Ptr laserCloudFullHoriRes;
pcl::PointCloud<PointType>::Ptr laserCloudFullVeloRes;
pcl::PointCloud<PointType>::Ptr veloSurfCloudPtr ;
pcl::PointCloud<PointType>::Ptr horiSurfCloudPtr ;
pcl::PointCloud<PointType>::Ptr cornerCloudPtr;
pcl::PointCloud<PointType>::Ptr surfCloudPtr;
Estimator* estimator;
Estimator* veloEstimator;


double time_imuqueue_start = -1;
double time_imuqueue_end = -1;

ros::Publisher pubLaserOdometry;
ros::Publisher pubLaserOdometryPath;
ros::Publisher pubVeloUndistortCloud;
ros::Publisher pubHoriUndistortCloud;
ros::Publisher pubFullLaserCloud;
ros::Publisher pubVeloFullLaserCloud;
ros::Publisher pubSegmentedRGB;

ros::Publisher pubVeloSurfMap;
ros::Publisher pubHoriSurfMap;
ros::Publisher pubMergedSurfMap;
ros::Publisher pubVeloCornerMap;
ros::Publisher pubHoriCornerMap;
ros::Publisher pubMergedCornerMap;

ros::Publisher pubHoriCornerMapFiltered;


tf::StampedTransform laserOdometryTrans;
tf::TransformBroadcaster* tfBroadcaster;

bool newHoriFullCloud = false;
bool newVeloFullCloud = false;

Eigen::Matrix4d transformAftMapped = Eigen::Matrix4d::Identity();
Eigen::Matrix4d veloTransformAftMapped = Eigen::Matrix4d::Identity();

std::mutex _mutexHoriLidarQueue;
std::mutex _mutexVeloLidarQueue;
std::mutex _mutexStereoOdomQueue;

std::queue<sensor_msgs::PointCloud2ConstPtr> _horiLidarMsgQueue;
std::queue<sensor_msgs::PointCloud2ConstPtr> _veloLidarMsgQueue;
std::queue<union_om::union_cloudConstPtr>    _unionLidarMsgQueue;
std::mutex _mutexIMUQueue;
// std::queue<sensor_msgs::ImuConstPtr> _imuMsgQueue;
std::vector<sensor_msgs::ImuConstPtr> _imuMsgVector;
Eigen::Matrix4d exTlb;
Eigen::Matrix3d exRlb, exRbl;
Eigen::Vector3d exPlb, exPbl;
Eigen::Vector3d GravityVector;
float filter_parameter_corner = 0.2;
float filter_parameter_surf = 0.2;
double velo_rotate_th = 1.5;
double hori_rotate_th = 0.3;
int IMU_Mode = 2;
int Hori_IMU_Mode = 2;
int pushCount = 0;
int veloPushCount = 0;
double startTime = 0;
double veloStartTime = 0;
int extrin_recali_times = 50;

double _time_last_lidar = -1;
double _time_last_hori = -1;
double _time_last_velo = -1;
// Eigen::Matrix4f tfVeloHori; // Transformation matrix from velodyne to Horizon

nav_msgs::Path laserOdomPath;

double dt_last = 0; //time counter
int processed_cnt = 0;
std::vector<int> feature_num = {0,0,0,0,0,0}; // velo, e, p, hori, e, p

pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn,
                            Eigen::Quaterniond quaternion, Eigen::Vector3d transition)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    // Eigen::Quaterniond quaternion(PointInfoIn->qw,
    //                                 PointInfoIn->qx,
    //                                 PointInfoIn->qy,
    //                                 PointInfoIn->qz);
    // Eigen::Vector3d transition(PointInfoIn->x,
    //                             PointInfoIn->y,
    //                             PointInfoIn->z);

    int numPts = cloudIn->points.size();
    cloudOut->resize(numPts);

    for (int i = 0; i < numPts; ++i) {
        Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;

        PointType pt;
        pt.x = ptOut.x();
        pt.y = ptOut.y();
        pt.z = ptOut.z();
        pt.intensity = cloudIn->points[i].intensity;

        cloudOut->points[i] = pt;
    }

    return cloudOut;
}

void pointAssociateToMap(PointType const * const pi,
                                      PointType * const po,
                                      const Eigen::Matrix4d& _transformTobeMapped)
{
    Eigen::Vector3d pin, pout;
    pin.x() = pi->x;
    pin.y() = pi->y;
    pin.z() = pi->z;
    pout = _transformTobeMapped.topLeftCorner(3,3) * pin + _transformTobeMapped.topRightCorner(3,1);
    po->x = pout.x();
    po->y = pout.y();
    po->z = pout.z();
    po->intensity = pi->intensity;
    po->normal_z = pi->normal_z;
}


/** \brief publish odometry infomation
  * \param[in] newPose: pose to be published
  * \param[in] timefullCloud: time stamp
  */
void pubOdometry(const Eigen::Matrix4d& newPose, double& timefullCloud){
    // std::cout << "Pub matrix:\n" << newPose  << "\n" << std::setprecision(15)<< timefullCloud<< std::endl;

    nav_msgs::Odometry laserOdometry;

    Eigen::Matrix3d Rcurr = newPose.topLeftCorner(3, 3);
    Eigen::Quaterniond newQuat(Rcurr);
    Eigen::Vector3d newPosition = newPose.topRightCorner(3, 1);
    laserOdometry.header.frame_id = "/lio_world";
    laserOdometry.child_frame_id = "/livox_frame";
    laserOdometry.header.stamp = ros::Time().fromSec(timefullCloud);
    laserOdometry.pose.pose.orientation.x = newQuat.x();
    laserOdometry.pose.pose.orientation.y = newQuat.y();
    laserOdometry.pose.pose.orientation.z = newQuat.z();
    laserOdometry.pose.pose.orientation.w = newQuat.w();
    laserOdometry.pose.pose.position.x = newPosition.x();
    laserOdometry.pose.pose.position.y = newPosition.y();
    laserOdometry.pose.pose.position.z = newPosition.z();
    pubLaserOdometry.publish(laserOdometry);



    geometry_msgs::PoseStamped laserPose;
    laserPose.header = laserOdometry.header;
    laserPose.pose = laserOdometry.pose.pose;
    laserOdomPath.header.stamp = laserOdometry.header.stamp;
    laserOdomPath.poses.push_back(laserPose);
    laserOdomPath.header.frame_id = "/lio_world";
    pubLaserOdometryPath.publish(laserOdomPath);

    laserOdometryTrans.frame_id_ = "/lio_world";
    laserOdometryTrans.child_frame_id_ = "/livox_frame";
    laserOdometryTrans.stamp_ = ros::Time().fromSec(timefullCloud);
    laserOdometryTrans.setRotation(tf::Quaternion(newQuat.x(), newQuat.y(), newQuat.z(), newQuat.w()));
    laserOdometryTrans.setOrigin(tf::Vector3(newPosition.x(), newPosition.y(), newPosition.z()));
    tfBroadcaster->sendTransform(laserOdometryTrans);

}

void horiFullCallBack(const sensor_msgs::PointCloud2ConstPtr &msg){
    // push lidar msg to queue
    // ROS_INFO_STREAM("  ================ Hori " <<  frame_cnt++ << " ===================");
    // ROS_WARN_STREAM("[PoseEstimation::Fullcallback] Hori Frame : "<< frame_cnt++ << " | _horiLidarMsgQueue size() " << _horiLidarMsgQueue.size());
	std::unique_lock<std::mutex> lock(_mutexHoriLidarQueue);
    _horiLidarMsgQueue.push(msg);
    // _veloLidarMsgQueue.push(msg);
    // // TODO: Check the segmentation:
    // pcl::PointCloud<pcl::PointXYZ>::Ptr laser_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::fromROSMsg(*msg , *laser_cloud);
    // planeDetection(laser_cloud);

}

void veloFullCallBack(const sensor_msgs::PointCloud2ConstPtr &msg){
    // push lidar msg to queue
    // ROS_INFO_STREAM("  ================ Velo " <<  velo_frame_cnt++ << " ===================");
    // ROS_INFO_STREAM("[PoseEstimation::Fullcallback] Velo Frame : "<< velo_frame_cnt++);
	std::unique_lock<std::mutex> lock(_mutexVeloLidarQueue);
    while(! _veloLidarMsgQueue.empty()) _veloLidarMsgQueue.pop();
    _veloLidarMsgQueue.push(msg);

    // _horiLidarMsgQueue.push(msg);


}

void unionCloudHandler(const union_om::union_cloudConstPtr & msg ){
    std::unique_lock<std::mutex> lock(_mutexVeloLidarQueue);
    while( _unionLidarMsgQueue.size() > 2 && real_time_mode) _unionLidarMsgQueue.pop();
    _unionLidarMsgQueue.push(msg);
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg){
    // push IMU msg to queue
    std::unique_lock<std::mutex> lock(_mutexIMUQueue);
    // _imuMsgQueue.push(imu_msg);
    _imuMsgVector.push_back(imu_msg);

    time_imuqueue_start =  _imuMsgVector.front()->header.stamp.toSec();
    time_imuqueue_end = _imuMsgVector.back()->header.stamp.toSec() ;
}

/** \brief get IMU messages in a certain time interval
  * \param[in] startTime: left boundary of time interval
  * \param[in] endTime: right boundary of time interval
  * \param[in] vimuMsg: store IMU messages
  */
bool fetchImuMsgs(double startTime, double endTime, std::vector<sensor_msgs::ImuConstPtr> &vimuMsg)
{
    if(_imuMsgVector.empty()){
            ROS_WARN_STREAM("[FetchImuMsg]: fetch vimuMsg _imuMsgVector.empty():"<< _imuMsgVector.empty());
            return false;
    }

    std::unique_lock<std::mutex> lock(_mutexIMUQueue);
    double current_time = 0;
    vimuMsg.clear();

    ROS_INFO_STREAM("[FetchImuMsg]: ask vimuMsg " << std::setprecision(15) << "startTime: " << startTime << ",  endTime: " << endTime );
    ROS_INFO_STREAM("[FetchImuMsg]: fetch vimuMsg " << std::setprecision(15)<< " IMU queue startTime: "<< _imuMsgVector.front()->header.stamp.toSec()
                    << " IMU queue endTime: "<< _imuMsgVector.back()->header.stamp.toSec() );

    int imu_idx = 0;
    while(ros::ok())
    {
        if(endTime < startTime) break;

        if(_imuMsgVector.empty()){
            ROS_WARN_STREAM("[FetchImuMsg]: fetch vimuMsg _imuMsgVector.empty():"<< _imuMsgVector.empty());
            break;
        }

        if(_imuMsgVector.back()->header.stamp.toSec()<endTime){
            ROS_WARN_STREAM("[FetchImuMsg]: fetch vimuMsg " << std::setprecision(15)<< _imuMsgVector.back()->header.stamp.toSec()<< " < " << endTime );
            break;
        }

        // make sure, the imu msgs contain all lidar poses
        if( _imuMsgVector.front()->header.stamp.toSec()>=endTime){
            ROS_WARN_STREAM("[FetchImuMsg]: fetch vimuMsg " << std::setprecision(15)<< _imuMsgVector.front()->header.stamp.toSec()<< " >= " << endTime );
            break;
        }

        // sensor_msgs::ImuConstPtr& tmpimumsg = _imuMsgQueue.front();
        auto tmpimumsg = _imuMsgVector[imu_idx];
        double time = tmpimumsg->header.stamp.toSec();

        // ROS_INFO_STREAM("[PoseEstimation] Adding message " << std::setprecision(15)<< " request time: " << time << " ( s: "<< startTime << " , e: "<< endTime << " )" );
        if(time<=endTime && time>startTime)
        {
            vimuMsg.push_back(tmpimumsg);
            current_time = time;
            // _imuMsgQueue.pop();
            if(time == endTime) break;
        }else{
            if(time<=startTime){
                // _imuMsgQueue.pop();
            } else{

                double dt_1 = endTime - current_time;
                double dt_2 = time - endTime;
                // std::cout << 7 << " : " << std::setprecision(15)<< current_time <<" " << time  << " " << endTime << " " << dt_1 <<" " << dt_2 << std::endl;
                ROS_ASSERT(dt_1 >= 0);
                ROS_ASSERT(dt_2 >= 0);
                ROS_ASSERT(dt_1 + dt_2 > 0);
                double w1 = dt_2 / (dt_1 + dt_2);
                double w2 = dt_1 / (dt_1 + dt_2);
                sensor_msgs::ImuPtr theLastIMU(new sensor_msgs::Imu);
                theLastIMU->linear_acceleration.x = w1 * vimuMsg.back()->linear_acceleration.x + w2 * tmpimumsg->linear_acceleration.x;
                theLastIMU->linear_acceleration.y = w1 * vimuMsg.back()->linear_acceleration.y + w2 * tmpimumsg->linear_acceleration.y;
                theLastIMU->linear_acceleration.z = w1 * vimuMsg.back()->linear_acceleration.z + w2 * tmpimumsg->linear_acceleration.z;
                theLastIMU->angular_velocity.x = w1 * vimuMsg.back()->angular_velocity.x + w2 * tmpimumsg->angular_velocity.x;
                theLastIMU->angular_velocity.y = w1 * vimuMsg.back()->angular_velocity.y + w2 * tmpimumsg->angular_velocity.y;
                theLastIMU->angular_velocity.z = w1 * vimuMsg.back()->angular_velocity.z + w2 * tmpimumsg->angular_velocity.z;
                theLastIMU->header.stamp.fromSec(endTime);
                vimuMsg.emplace_back(theLastIMU);
                break;
            }
        }
        imu_idx++;
    }

    // Keep a imu sequence where timestamp smaller than both
    if(!vimuMsg.empty() && _imuMsgVector.size() > 200){
        double time_imu_front = _imuMsgVector.front()->header.stamp.toSec();

        while(_imuMsgVector.size() > 200 && _time_last_hori > time_imu_front && _time_last_velo > time_imu_front)
        {
            _imuMsgVector.erase( _imuMsgVector.begin() );
            time_imu_front = _imuMsgVector.front()->header.stamp.toSec();
        }
    }
    ROS_INFO_STREAM("[FetchImuMsg] _imuMsgVector size : " << _imuMsgVector.size());

    return !vimuMsg.empty();
}

/** \brief Remove Lidar Distortion
  * \param[in] cloud: lidar cloud need to be undistorted
  * \param[in] dRlc: delta rotation
  * \param[in] dtlc: delta displacement
  */
void RemoveLidarDistortion(pcl::PointCloud<PointType>::Ptr& cloud,
                           const Eigen::Matrix3d& dRlc, const Eigen::Vector3d& dtlc)
{
    // 插值矫正位姿
    int PointsNum = cloud->points.size();
    for (int i = 0; i < PointsNum; i++) {
        Eigen::Vector3d startP;
        float s = cloud->points[i].normal_x;
        Eigen::Quaterniond qlc = Eigen::Quaterniond(dRlc).normalized();
        Eigen::Quaterniond delta_qlc = Eigen::Quaterniond::Identity().slerp(s, qlc).normalized();
        const Eigen::Vector3d delta_Plc = s * dtlc;
        startP = delta_qlc * Eigen::Vector3d(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z) + delta_Plc;
        Eigen::Vector3d _po = dRlc.transpose() * (startP - dtlc);

        cloud->points[i].x = _po(0);
        cloud->points[i].y = _po(1);
        cloud->points[i].z = _po(2);
        cloud->points[i].normal_x = 1.0;
    }
}


// Estimate IMU biases, velocities, and the gravity direction
bool TryMAPInitialization(boost::shared_ptr<std::list<Estimator::LidarFrame>> frameList)
{
    // 静止状态估计 IMU 的重力方向，
    Eigen::Vector3d average_acc = -frameList->begin()->imuIntegrator.GetAverageAcc();
    // ROS_INFO_STREAM("[TryMAPInitialization] average_acc 1: " << average_acc.x()
    //                             << " " << average_acc.y() << " " << average_acc.z());
    double info_g = std::fabs(9.805 - average_acc.norm());
    average_acc = average_acc * 9.805 / average_acc.norm();
    ROS_INFO_STREAM("[TryMAPInitialization] average_acc 2: " << average_acc.x()
                                << " " << average_acc.y() << " " << average_acc.z());

    // calculate the initial gravity direction
    double para_quat[4];
    para_quat[0] = 1;
    para_quat[1] = 0;
    para_quat[2] = 0;
    para_quat[3] = 0;

    //############# world to gravity vector transform matrix;
    ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
    ceres::Problem problem_quat;

    problem_quat.AddParameterBlock(para_quat, 4, quatParam);

    problem_quat.AddResidualBlock(Cost_Initial_G::Create(average_acc),
                                    nullptr,
                                    para_quat);

    ceres::Solver::Options options_quat;
    ceres::Solver::Summary summary_quat;
    ceres::Solve(options_quat, &problem_quat, &summary_quat);
    // std::cout<< summary_quat.FullReport() << std::endl;

    Eigen::Quaterniond q_wg(para_quat[0], para_quat[1], para_quat[2], para_quat[3]);
    ROS_INFO_STREAM("[TryMAPInitialization] q_wg: " << para_quat[0] << " " << para_quat[1]
                                << " " << para_quat[2] << " " << para_quat[3]);

    //build prior factor of LIO initialization
    Eigen::Vector3d prior_r = Eigen::Vector3d::Zero();
    Eigen::Vector3d prior_ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d prior_bg = Eigen::Vector3d::Zero(); // bg: angular bias
    std::vector<Eigen::Vector3d> prior_v;

    int v_size = frameList->size();
    for(int i = 0; i < v_size; i++) {
        prior_v.push_back(Eigen::Vector3d::Zero());
    }
    Sophus::SO3d SO3_R_wg(q_wg.toRotationMatrix()); // To Lie group SO3 matrix
    prior_r = SO3_R_wg.log();                       //(rotation-vector).
    /// Logarithmic map
    /// Computes the logarithm, the inverse of the group exponential which maps
    /// element of the group (rotation matrices) to elements of the tangent space
    /// (rotation-vector).
    // To tangent space

    for (int i = 1; i < v_size; i++){
        auto iter = frameList->begin();
        auto iter_next = frameList->begin();
        std::advance(iter, i-1);
        std::advance(iter_next, i);

        // get the imu speed (dp / dt)
        Eigen::Vector3d velo_imu = (iter_next->P - iter->P + iter_next->Q*exPlb - iter->Q*exPlb) / (iter_next->timeStamp - iter->timeStamp);
        // std::cout<< "-> I: " << i << " | velo_imu: " <<velo_imu << std::endl;
        prior_v[i] = velo_imu;
    }
    prior_v[0] = prior_v[1];

    double para_v[v_size][3]; // imu speed bias;
    double para_r[3];
    double para_ba[3];
    double para_bg[3];

    for(int i = 0; i < 3; i++) {
        para_r[i] = 0;
        para_ba[i] = 0;
        para_bg[i] = 0;
    }

    for(int i = 0; i < v_size; i++) {
        for(int j = 0; j < 3; j++) {
            para_v[i][j] = prior_v[i][j];
        }
    }

    Eigen::Matrix<double, 3, 3> sqrt_information_r = 2000.0 * Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 3> sqrt_information_ba = 1000.0 * Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 3> sqrt_information_bg = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 3> sqrt_information_v = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();

    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_r, 3); // 旋转参数 lie group rotation vector
    problem.AddParameterBlock(para_ba, 3); // acc bias
    problem.AddParameterBlock(para_bg, 3); // gravity bias
    for(int i = 0; i < v_size; i++) {
        problem.AddParameterBlock(para_v[i], 3);
    }

    // add CostFunction
    problem.AddResidualBlock(Cost_Initialization_Prior_R::Create(prior_r, sqrt_information_r),
                            nullptr,
                            para_r);

    problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_ba, sqrt_information_ba),
                            nullptr,
                            para_ba);
    problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_bg, sqrt_information_bg),
                            nullptr,
                            para_bg);

    for(int i = 0; i < v_size; i++) {
        problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_v[i], sqrt_information_v),
                                nullptr,
                                para_v[i]);
    }

    for(int i = 1; i < v_size; i++) {
        auto iter = frameList->begin();
        auto iter_next = frameList->begin();
        std::advance(iter, i-1);
        std::advance(iter_next, i);

        Eigen::Vector3d pi = iter->P + iter->Q*exPlb; // position at base_link coordinate
        Sophus::SO3d SO3_Ri(iter->Q*exRlb);
        Eigen::Vector3d ri = SO3_Ri.log(); // Lie group rotation vector
        Eigen::Vector3d pj = iter_next->P + iter_next->Q*exPlb;
        Sophus::SO3d SO3_Rj(iter_next->Q*exRlb);
        Eigen::Vector3d rj = SO3_Rj.log();

        problem.AddResidualBlock(Cost_Initialization_IMU::Create(iter_next->imuIntegrator,
                                    ri,
                                    rj,
                                    pj-pi,
                                    Eigen::LLT<Eigen::Matrix<double, 9, 9>>
                                    (iter_next->imuIntegrator.GetCovariance().block<9,9>(0,0).inverse())
                                    .matrixL().transpose()),
                                nullptr,
                                para_r,
                                para_v[i-1],
                                para_v[i],
                                para_ba,
                                para_bg);
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 6;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector3d r_wg(para_r[0], para_r[1], para_r[2]);
    GravityVector = Sophus::SO3d::exp(r_wg) * Eigen::Vector3d(0, 0, -9.805);

    Eigen::Vector3d ba_vec(para_ba[0], para_ba[1], para_ba[2]);
    Eigen::Vector3d bg_vec(para_bg[0], para_bg[1], para_bg[2]);

    if(ba_vec.norm() > 0.5 || bg_vec.norm() > 0.5) {
        ROS_WARN_STREAM("Too Large Biases! Initialization Failed! -  ba_vec.norm(): "
                            <<  ba_vec.norm() << " | bg_vec.norm():"<< bg_vec.norm() );
        return false;
    }

    for(int i = 0; i < v_size; i++) {
        auto iter = frameList->begin();
        std::advance(iter, i);
        iter->ba = ba_vec;
        iter->bg = bg_vec;
        Eigen::Vector3d bv_vec(para_v[i][0], para_v[i][1], para_v[i][2]);
        if((bv_vec - prior_v[i]).norm() > 2.0) {
            ROS_WARN("Too Large Velocity! Initialization Failed!");
            std::cout<<"delta v norm: "<<(bv_vec - prior_v[i]).norm()<<std::endl;
            return false;
        }
        iter->V = bv_vec;
    }

    for(size_t i = 0; i < v_size - 1; i++){
        auto laser_trans_i = frameList->begin();
        auto laser_trans_j = frameList->begin();
        std::advance(laser_trans_i, i);
        std::advance(laser_trans_j, i+1);
        laser_trans_j->imuIntegrator.PreIntegration(laser_trans_i->timeStamp, laser_trans_i->bg, laser_trans_i->ba);
    }


    // if IMU success initialized
    WINDOWSIZE = Estimator::SLIDEWINDOWSIZE;
    while(frameList->size() > WINDOWSIZE){
        frameList->pop_front();
    }
    Eigen::Vector3d Pwl = frameList->back().P;
    Eigen::Quaterniond Qwl = frameList->back().Q;
    frameList->back().P = Pwl + Qwl*exPlb;
    frameList->back().Q = Qwl * exRlb;

    std::cout << "\n=============================\n| Initialization Successful |"<<"\n=============================\n" << std::endl;

    return true;
}


/** \brief Mapping main thread
  */
void process(){
    double time_curr_lidar = -1;
    double time_curr_velo = -1;
    double time_curr_hori = -1;
    double time_imuqueue_start = -1;
    double time_imuqueue_end = -1;
    Eigen::Matrix3d delta_Rl = Eigen::Matrix3d::Identity();
    Eigen::Vector3d delta_tl = Eigen::Vector3d::Zero();
	Eigen::Matrix3d delta_Rb = Eigen::Matrix3d::Identity();
	Eigen::Vector3d delta_tb = Eigen::Vector3d::Zero();

    Eigen::Matrix3d velo_delta_Rl = Eigen::Matrix3d::Identity();
    Eigen::Vector3d velo_delta_tl = Eigen::Vector3d::Zero();
	Eigen::Matrix3d velo_delta_Rb = Eigen::Matrix3d::Identity();
	Eigen::Vector3d velo_delta_tb = Eigen::Vector3d::Zero();

    std::vector<sensor_msgs::ImuConstPtr> vimuMsg;

    int lidarMode = 0;  // 1: hori; 2:velo

    ros::Rate r(20);

    bool first_velo_frame = true;
    bool first_hori_frame = true;

    int extrin_cnt = 0;
    Eigen::Matrix4f extri_mtx = Eigen::Matrix4f::Identity();
    while(ros::ok()){
        newHoriFullCloud = false;
        newVeloFullCloud = false;

        if( _unionLidarMsgQueue.size() > 1 && !_imuMsgVector.empty() ){

            auto tick = std::chrono::high_resolution_clock::now();

            laserCloudFullRes.reset(new pcl::PointCloud<PointType>());
            laserCloudFullHoriRes.reset(new pcl::PointCloud<PointType>);
            laserCloudFullVeloRes.reset(new pcl::PointCloud<PointType>);
            veloSurfCloudPtr.reset(new pcl::PointCloud<PointType>);
            horiSurfCloudPtr.reset(new pcl::PointCloud<PointType>);
            ROS_WARN_STREAM(" \n\n ====================== Union Cloud processing ========================== ");
            ROS_INFO_STREAM( std::setprecision(15) <<  _unionLidarMsgQueue.size() << " livox c/s: "
                            << _unionLidarMsgQueue.front()->livox_corner_num << "/" << _unionLidarMsgQueue.front()->livox_surf_num <<
                            " velo c/s: " << _unionLidarMsgQueue.front()->velo_corner_num << "/" << _unionLidarMsgQueue.front()->velo_surf_num
                             );

            time_curr_hori = _unionLidarMsgQueue.front()->livox_combine.header.stamp.toSec();
            time_curr_velo = _unionLidarMsgQueue.front()->velo_combine.header.stamp.toSec();

            sensor_msgs::PointCloud2 velo_full = _unionLidarMsgQueue.front()->velo_combine;
            pcl::fromROSMsg(velo_full, *laserCloudFullVeloRes);
            sensor_msgs::PointCloud2 velo_suf = _unionLidarMsgQueue.front()->velo_surface;
            pcl::fromROSMsg(velo_suf, *veloSurfCloudPtr);
            newVeloFullCloud = true;

            sensor_msgs::PointCloud2 hori_msg = _unionLidarMsgQueue.front()->livox_combine;
            sensor_msgs::PointCloud2 hori_surf = _unionLidarMsgQueue.front()->livox_surface;
            pcl::fromROSMsg(hori_surf, *horiSurfCloudPtr);
            pcl::fromROSMsg(hori_msg, *laserCloudFullHoriRes);
            newHoriFullCloud = false;
            processed_cnt ++;
            if( feature_num[0] == 0){
                feature_num[0] = laserCloudFullVeloRes->points.size();feature_num[1] =  _unionLidarMsgQueue.front()->velo_corner_num;
                feature_num[2] =  _unionLidarMsgQueue.front()->velo_surf_num;
                feature_num[3] = laserCloudFullHoriRes->points.size();feature_num[4] =  _unionLidarMsgQueue.front()->livox_corner_num;
                feature_num[5] =  _unionLidarMsgQueue.front()->livox_surf_num;
            }else{
                feature_num[0] = (laserCloudFullVeloRes->points.size() + feature_num[0] );
                feature_num[1] = ( _unionLidarMsgQueue.front()->velo_corner_num + feature_num[1] );
                feature_num[2] = ( _unionLidarMsgQueue.front()->velo_surf_num + feature_num[2] );
                feature_num[3] = (laserCloudFullHoriRes->points.size() + feature_num[3] );
                feature_num[4] = ( _unionLidarMsgQueue.front()->livox_corner_num + feature_num[4] );
                feature_num[5] = ( _unionLidarMsgQueue.front()->livox_surf_num + feature_num[5] );
                for(int i =0; i<6; i++)
                    ROS_WARN_STREAM("Feature avg:" << i << " - " << feature_num[i]/processed_cnt);
            }

            if(newVeloFullCloud)
            {
                lidarMode = 2;
                ROS_INFO_STREAM("[VeloProcessing] New Frame : \t velo: " << laserCloudFullVeloRes->points.size()
                                                     << " \t Hori: " << laserCloudFullHoriRes->points.size());

                if(IMU_Mode > 0 ){
                    // get IMU msg int the Specified time interval
                    vimuMsg.clear();
                    int countFail = 0;
                    // TODO check the timestamp here (Merged points!?)
                    ROS_INFO_STREAM("[VeloProcessing] Fectch imu message " << std::setprecision(15)<< _time_last_velo << " -> " << time_curr_velo );
                    while (!fetchImuMsgs(_time_last_velo, time_curr_velo, vimuMsg)) {
                        countFail++;
                        if (countFail > 4){
                            break;
                        }
                        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
                    }
                    if (countFail > 3){
                        ROS_WARN_STREAM("[VeloProcessing]: get vimuMsg, countFail "<< countFail);
                        _unionLidarMsgQueue.pop();
                        continue;
                    }
                }

                // sensor_msgs::PointCloud2 laserCloudMsg;
                bool detected_fast_rotation = false;

                Estimator::LidarFrame veloFrame;
                veloFrame.laserCloud = laserCloudFullVeloRes;
                veloFrame.timeStamp = time_curr_velo;

                boost::shared_ptr<std::list<Estimator::LidarFrame>> velo_list;
                if(!vimuMsg.empty() && !first_velo_frame)
                {
                    // only not fast rotation included
                    std::cout << "velo_only_mode: "<< velo_only_mode << std::endl;
                    // if ( !velo_only_mode && (abs( vimuMsg.front()->angular_velocity.z) < hori_rotate_th || abs( vimuMsg.back()->angular_velocity.z)) < hori_rotate_th){
                    if ( !velo_only_mode  ){
                        if(abs( vimuMsg.front()->angular_velocity.z) < hori_rotate_th || abs( vimuMsg.back()->angular_velocity.z) < hori_rotate_th){
                            if( _unionLidarMsgQueue.front()->livox_corner_num > 100 )
                            {
                                // if( extrin_cnt % extrin_recali_times == 0){
                                //     icp_ext_matching(horiSurfCloudPtr, veloSurfCloudPtr, cloud_aligned, extri_mtx, false);
                                // }
                                // // icp_ext_matching(laserCloudFullHoriRes, laserCloudFullVeloRes, cloud_aligned, icp_mtx, false))

                                // pcl::transformPointCloud(*laserCloudFullHoriRes, *laserCloudFullHoriRes, extri_mtx);
                                *laserCloudFullVeloRes = *laserCloudFullVeloRes + *laserCloudFullHoriRes;
                                veloFrame.laserCloud = laserCloudFullVeloRes;
                                ROS_INFO("Merged points ");
                                extrin_cnt++;
                            }else{
                                ROS_WARN_STREAM("Insufficient feature points : " << _unionLidarMsgQueue.front()->livox_corner_num);
                            }
                        }else{
                            ROS_WARN_STREAM( "Fast rotation detected : " << vimuMsg.front()->angular_velocity.z << "<" <<  hori_rotate_th  << "||"
                                        << abs( vimuMsg.back()->angular_velocity.z)  << "<" << hori_rotate_th);
                        }
                    }else if(velo_only_mode){
                        ROS_INFO("Velo Only Mode! ");
                    }

                    if ( abs( vimuMsg.front()->angular_velocity.z) >  velo_rotate_th || abs( vimuMsg.back()->angular_velocity.z) > velo_rotate_th){
                        detected_fast_rotation = true;
                    }
                    if(!LidarIMUInited) {
                        ROS_WARN_STREAM("[VELO]: LidarIMUInited : " << LidarIMUInited);
                        // if get IMU msg successfully, use gyro integration to update delta_Rl
                        veloFrame.imuIntegrator.PushIMUMsg(vimuMsg);
                        veloFrame.imuIntegrator.GyroIntegration(_time_last_velo);
                        velo_delta_Rb = veloFrame.imuIntegrator.GetDeltaQ().toRotationMatrix();
                        velo_delta_Rl = exTlb.topLeftCorner(3, 3) * velo_delta_Rb * exTlb.topLeftCorner(3, 3).transpose();

                        veloFrame.P = veloTransformAftMapped.topLeftCorner(3,3) * velo_delta_tb
                                    + veloTransformAftMapped.topRightCorner(3,1);
                        Eigen::Matrix3d m3d = veloTransformAftMapped.topLeftCorner(3,3) * velo_delta_Rb;
                        veloFrame.Q = m3d;
                        // std::cout<< "pose: " << veloFrame.P << std::endl;
                        // std::cout<< "quat: " << veloFrame.Q.x() << " " << veloFrame.Q.y() << " " <<
                        //                 veloFrame.Q.z() << " " << veloFrame.Q.w() << std::endl;

                        veloFrame.lidarType = 2;

                        velo_list.reset(new std::list<Estimator::LidarFrame>);
                        velo_list->push_back(veloFrame);
                    }else{
                        // std::cout<< " vimuMsg.empty() : " <<  vimuMsg.empty() << std::endl;
                        ROS_INFO_STREAM("[VELO]: LidarIMUInited : " << LidarIMUInited);
                        // if get IMU msg successfully, use pre-integration to update delta lidar pose
                        veloFrame.imuIntegrator.PushIMUMsg(vimuMsg);
                        veloFrame.imuIntegrator.PreIntegration(veloFrameList->back().timeStamp, veloFrameList->back().bg, veloFrameList->back().ba);

                        const Eigen::Vector3d& Pwbpre = veloFrameList->back().P;
                        const Eigen::Quaterniond& Qwbpre = veloFrameList->back().Q;
                        const Eigen::Vector3d& Vwbpre = veloFrameList->back().V;

                        const Eigen::Quaterniond& dQ =  veloFrame.imuIntegrator.GetDeltaQ();
                        const Eigen::Vector3d& dP = veloFrame.imuIntegrator.GetDeltaP();
                        const Eigen::Vector3d& dV = veloFrame.imuIntegrator.GetDeltaV();
                        double dt = veloFrame.imuIntegrator.GetDeltaTime();

                        veloFrame.Q = Qwbpre * dQ;
                        // veloFrame.P = Pwbpre + Vwbpre*dt ; //+ 0.5*GravityVector*dt*dt
                        veloFrame.P = Pwbpre + Qwbpre*(dP);
                        // veloFrame.V = Vwbpre + GravityVector*dt + Qwbpre*(dV);
                        veloFrame.V = Vwbpre + Qwbpre*(dV);
                        veloFrame.bg = veloFrameList->back().bg;
                        veloFrame.ba = veloFrameList->back().ba;
                        veloFrame.lidarType = 2;  // set the lidar type

                        Eigen::Quaterniond Qwlpre = Qwbpre * Eigen::Quaterniond(exRbl); // external Transform baselink to lidar
                        Eigen::Vector3d Pwlpre = Qwbpre * exPbl + Pwbpre;               // Lidar pose

                        Eigen::Quaterniond Qwl = veloFrame.Q * Eigen::Quaterniond(exRbl);
                        Eigen::Vector3d Pwl = veloFrame.Q * exPbl + veloFrame.P;

                        velo_delta_Rl = Qwlpre.conjugate() * Qwl;
                        velo_delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
                        velo_delta_Rb = dQ.toRotationMatrix();
                        velo_delta_tb = dP;

                        veloFrameList->push_back( veloFrame);
                        if(veloFrameList->size() > WINDOWSIZE)
                            veloFrameList->pop_front();
                        velo_list = veloFrameList;


                        ROS_INFO_STREAM(" [VeloProcessing] Update time : " << std::setprecision(15) << _time_last_velo  );
                    }
                }else{
                    first_velo_frame = false;

                    if(LidarIMUInited)
                        break;
                    else
                    {
                        ROS_WARN_STREAM("[Velo] IMU Info unavaliable" );
                        // predict current lidar pose based on position
                        veloFrame.P = veloTransformAftMapped.topLeftCorner(3,3) * velo_delta_tb
                                    + veloTransformAftMapped.topRightCorner(3,1);
                        ROS_INFO_STREAM("PoseEstimation::process " << "[Before Optim] veloFrame.P : " <<  veloFrame.P);

                        Eigen::Matrix3d m3d = veloTransformAftMapped.topLeftCorner(3,3) * velo_delta_Rb;
                        veloFrame.Q = m3d;

                        velo_list.reset(new std::list<Estimator::LidarFrame>);
                        velo_list->push_back(veloFrame);
                    }
                }

                ROS_INFO_STREAM(" [VeloProcessing] velo_list->size() : " <<  velo_list->size() );

                // remove lidar distortion
                RemoveLidarDistortion(laserCloudFullVeloRes, velo_delta_Rl, velo_delta_tl);
                sensor_msgs::PointCloud2 laserCloudMsg;
                pcl::toROSMsg(*laserCloudFullVeloRes, laserCloudMsg);
                laserCloudMsg.header.frame_id = "/lio_world";
                laserCloudMsg.header.stamp = ros::Time().fromSec(time_curr_velo);
                pubVeloUndistortCloud.publish(laserCloudMsg);

                // optimize current lidar pose with IMU
                // TODO: failure detection, update with stereo info; map updated inside
                // veloEstimator->EstimateLidarPose(*velo_list, exTlb, GravityVector, lidarMode);
                estimator->EstimateLidarPose(*velo_list, exTlb, GravityVector, lidarMode);

                pcl::PointCloud<PointType>::Ptr veloCloudCornerMap(new pcl::PointCloud<PointType>());
                pcl::PointCloud<PointType>::Ptr veloCloudSurfMap(new pcl::PointCloud<PointType>());
                Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();                  // after mapped odometry

                // No failure detected, using the estimator pose
                transformTobeMapped.topLeftCorner(3,3) = velo_list->front().Q * exRbl;
                transformTobeMapped.topRightCorner(3,1) = velo_list->front().Q * exPbl + velo_list->front().P;

                // update delta transformation
                velo_delta_Rb = veloTransformAftMapped.topLeftCorner(3, 3).transpose() * velo_list->front().Q.toRotationMatrix();
                velo_delta_tb = veloTransformAftMapped.topLeftCorner(3, 3).transpose() * (velo_list->front().P - veloTransformAftMapped.topRightCorner(3, 1));
                Eigen::Matrix3d Rwlpre = veloTransformAftMapped.topLeftCorner(3, 3) * exRbl;
                Eigen::Vector3d Pwlpre = veloTransformAftMapped.topLeftCorner(3, 3) * exPbl + veloTransformAftMapped.topRightCorner(3, 1);
                velo_delta_Rl = Rwlpre.transpose() * transformTobeMapped.topLeftCorner(3,3);
                velo_delta_tl = Rwlpre.transpose() * (transformTobeMapped.topRightCorner(3,1) - Pwlpre);
                veloTransformAftMapped.topLeftCorner(3,3) = velo_list->front().Q.toRotationMatrix();
                veloTransformAftMapped.topRightCorner(3,1) = velo_list->front().P;

                // publish odometry rostopic
                pubOdometry(transformTobeMapped, velo_list->front().timeStamp);

                // // publish lidar points
                int laserCloudFullResNum = velo_list->front().laserCloud->points.size();
                pcl::PointCloud<PointType>::Ptr laserCloudAfterEstimate(new pcl::PointCloud<PointType>());
                laserCloudAfterEstimate->reserve(laserCloudFullResNum);
                for (int i = 0; i < laserCloudFullResNum; i++) {
                    PointType temp_point;
                    pointAssociateToMap(&velo_list->front().laserCloud->points[i], &temp_point, transformTobeMapped);
                    laserCloudAfterEstimate->push_back(temp_point);
                }


                if(!estimator->failureDetected() && !detected_fast_rotation){
                    pcl::toROSMsg(*laserCloudAfterEstimate, laserCloudMsg);
                    laserCloudMsg.header.frame_id = "/lio_world";
                    laserCloudMsg.header.stamp.fromSec(velo_list->front().timeStamp);
                    pubVeloFullLaserCloud.publish(laserCloudMsg);
                    ROS_INFO_STREAM("[VELO ]: Publish result point cloud \n\n");
                }else{
                    ROS_WARN_STREAM("[Velo] : Failure Detected ! Quit publishing points" <<
                                "->  estimator->failureDetected(): " << estimator->failureDetected() <<
                                " || -> detected_fast_rotation :" << detected_fast_rotation << " th_ " << velo_rotate_th <<
                                " -( " << vimuMsg.front()->angular_velocity.z <<
                                "," << vimuMsg.back()->angular_velocity.z << " )");
                }

                _time_last_velo = time_curr_velo;

                // surfCloudPtr = veloEstimator->get_surf_map();
                surfCloudPtr = estimator->get_surf_map();
                pcl::toROSMsg(*surfCloudPtr, laserCloudMsg);
                laserCloudMsg.header.frame_id = "/lio_world";
                laserCloudMsg.header.stamp.fromSec(velo_list->front().timeStamp);
                pubVeloSurfMap.publish(laserCloudMsg);

                cornerCloudPtr = estimator->get_corner_map();
                pcl::toROSMsg(*cornerCloudPtr, laserCloudMsg);
                laserCloudMsg.header.frame_id = "/lio_world";
                laserCloudMsg.header.stamp.fromSec(velo_list->front().timeStamp);
                pubVeloCornerMap.publish(laserCloudMsg);
                // if(!surfCloudPtr->empty())
                //     pcl::io::savePCDFileASCII ("/home/qing/velo_surf.pcd", *surfCloudPtr);
                ROS_INFO_STREAM("[VELO ]: Publish Surf & Corner Map ");

                // if tightly coupled IMU message, start IMU initialization
                if(IMU_Mode > 1 && !LidarIMUInited){
                    // update lidar frame pose
                    veloFrame.P = transformTobeMapped.topRightCorner(3,1);
                    Eigen::Matrix3d m3d = transformTobeMapped.topLeftCorner(3,3);
                    veloFrame.Q = m3d;

                    ROS_WARN_STREAM ("WINDOWSIZE :  " << WINDOWSIZE  << " | veloFrameList->size(): " << veloFrameList->size() );
                    // static int pushCount = 0;
                    if(veloPushCount == 0){
                        veloFrameList->push_back(veloFrame);
                        veloFrameList->back().imuIntegrator.Reset();
                        if(veloFrameList->size() > WINDOWSIZE)
                            veloFrameList->pop_front();
                    }else{
                        veloFrameList->back().laserCloud = veloFrame.laserCloud;
                        veloFrameList->back().imuIntegrator.PushIMUMsg(vimuMsg);
                        veloFrameList->back().timeStamp = veloFrame.timeStamp;
                        veloFrameList->back().P = veloFrame.P;
                        veloFrameList->back().Q = veloFrame.Q;
                    }
                    veloPushCount++;
                    if (veloPushCount >= 3){
                        veloPushCount = 0;
                        if(veloFrameList->size() > 1){
                            auto iterRight = std::prev(veloFrameList->end());
                            auto iterLeft = std::prev(std::prev(veloFrameList->end()));
                            iterRight->imuIntegrator.PreIntegration(iterLeft->timeStamp, iterLeft->bg, iterLeft->ba);
                        }

                        if (veloFrameList->size() == int(WINDOWSIZE / 1.5)) {
                                veloStartTime = veloFrameList->back().timeStamp;

                            }

                        if (!LidarIMUInited && veloFrameList->size() == WINDOWSIZE && veloFrameList->front().timeStamp >= veloStartTime){
                            std::cout<<"**************Start VELO IMU MAP Initialization!!!******************"<<std::endl;
                            if(TryMAPInitialization(veloFrameList))
                            {
                                LidarIMUInited = true;
                                veloPushCount = 0;
                                veloStartTime = 0;
                            }
                            std::cout<<"**************Finish VELO IMU MAP Initialization!!!******************"<<std::endl;
                        }

                    }
                }

                ROS_INFO_STREAM("Ending Velo cloud processing! ");
                ROS_INFO_STREAM("====================================================== \n\n");

            }


               // Processing the Hori cloud, init the IMU
            if(newHoriFullCloud)
            {
                lidarMode = 1;
                ROS_INFO_STREAM("[HoriPoseEstimation] New Hori Frame : " << hori_frame_cnt++ );
                if(Hori_IMU_Mode > 0 && _time_last_hori > 0){
                    // get IMU msg int the Specified time interval
                    vimuMsg.clear();
                    int countFail = 0;
                    // TODO check the timestamp here (Merged points!?)
                    while (!fetchImuMsgs(_time_last_hori, time_curr_hori, vimuMsg)) {
                        countFail++;
                        if (countFail > 4){
                            break;
                        }
                        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
                    }
                    if (countFail > 4){
                        ROS_WARN_STREAM("[Hori]: get vimuMsg, countFail "<< countFail);
                        _unionLidarMsgQueue.pop();
                        continue;
                    }
                }

                ROS_INFO_STREAM("[Hori]: get vimuMsg, size: " << vimuMsg.size() << " | vimuMsg.empty(): "  << vimuMsg.empty()
                                << ",  duration (s):" << time_curr_hori - _time_last_hori );
                // this lidar frame init
                Estimator::LidarFrame lidarFrame;
                lidarFrame.laserCloud = laserCloudFullHoriRes;
                lidarFrame.timeStamp = time_curr_hori;

                boost::shared_ptr<std::list<Estimator::LidarFrame>> hori_lidar_list;
                if(!vimuMsg.empty() && !first_hori_frame){
                    // If imu is not inited: init the position with IMU
                    if(!LidarIMUInited) {
                        ROS_INFO_STREAM("[Hori]: LidarIMUInited : " << LidarIMUInited);
                        // if get IMU msg successfully, use gyro integration to update delta_Rl
                        lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
                        lidarFrame.imuIntegrator.GyroIntegration(_time_last_hori);
                        delta_Rb = lidarFrame.imuIntegrator.GetDeltaQ().toRotationMatrix();
                        delta_Rl = exTlb.topLeftCorner(3, 3) * delta_Rb * exTlb.topLeftCorner(3, 3).transpose();

                        lidarFrame.P = transformAftMapped.topLeftCorner(3,3) * delta_tb
                                    + transformAftMapped.topRightCorner(3,1);
                        Eigen::Matrix3d m3d = transformAftMapped.topLeftCorner(3,3) * delta_Rb;
                        lidarFrame.Q = m3d;

                        lidarFrame.lidarType = 1;

                        hori_lidar_list.reset(new std::list<Estimator::LidarFrame>);
                        hori_lidar_list->push_back(lidarFrame);
                    }else{
                        // if get IMU msg successfully, use pre-integration to update delta lidar pose
                        lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
                        lidarFrame.imuIntegrator.PreIntegration(lidarFrameList->back().timeStamp, lidarFrameList->back().bg, lidarFrameList->back().ba);

                        const Eigen::Vector3d& Pwbpre = lidarFrameList->back().P;
                        const Eigen::Quaterniond& Qwbpre = lidarFrameList->back().Q;
                        const Eigen::Vector3d& Vwbpre = lidarFrameList->back().V;

                        const Eigen::Quaterniond& dQ =  lidarFrame.imuIntegrator.GetDeltaQ();
                        const Eigen::Vector3d& dP = lidarFrame.imuIntegrator.GetDeltaP();
                        const Eigen::Vector3d& dV = lidarFrame.imuIntegrator.GetDeltaV();
                        double dt = lidarFrame.imuIntegrator.GetDeltaTime();

                        lidarFrame.Q = Qwbpre * dQ;
                        lidarFrame.P = Pwbpre + Vwbpre*dt + 0.5*GravityVector*dt*dt + Qwbpre*(dP);
                        lidarFrame.V = Vwbpre + GravityVector*dt + Qwbpre*(dV);
                        lidarFrame.bg = lidarFrameList->back().bg;
                        lidarFrame.ba = lidarFrameList->back().ba;
                        lidarFrame.lidarType = 1;  // set the lidar type

                        Eigen::Quaterniond Qwlpre = Qwbpre * Eigen::Quaterniond(exRbl); // external Transform baselink to lidar
                        Eigen::Vector3d Pwlpre = Qwbpre * exPbl + Pwbpre;               // Lidar pose

                        Eigen::Quaterniond Qwl = lidarFrame.Q * Eigen::Quaterniond(exRbl);
                        Eigen::Vector3d Pwl = lidarFrame.Q * exPbl + lidarFrame.P;

                        delta_Rl = Qwlpre.conjugate() * Qwl;
                        delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
                        delta_Rb = dQ.toRotationMatrix();
                        delta_tb = dP;

                        lidarFrameList->push_back(lidarFrame);

                        if(lidarFrameList->size() > WINDOWSIZE)
                        // if(lidarFrameList->size() > 1)
                            lidarFrameList->pop_front();
                        hori_lidar_list = lidarFrameList;
                    }
                }else{
                    ROS_WARN("[Hori] Empty imu message");
                    first_hori_frame = false;
                    if(LidarIMUInited)
                        break;
                    else
                    {
                        // predict current lidar pose based on position
                        lidarFrame.P = transformAftMapped.topLeftCorner(3,3) * delta_tb
                                    + transformAftMapped.topRightCorner(3,1);
                        // ROS_INFO_STREAM("PoseEstimation::process " << "[Before Optim] lidarFrame.P : " <<  lidarFrame.P);

                        Eigen::Matrix3d m3d = transformAftMapped.topLeftCorner(3,3) * delta_Rb;
                        lidarFrame.Q = m3d;

                        hori_lidar_list.reset(new std::list<Estimator::LidarFrame>);
                        hori_lidar_list->push_back(lidarFrame);
                    }
                }

                if(lidarFrame.laserCloud->points.size() == 1){
                    ROS_WARN("Empty cloud from current lidar, continue !");
                    _unionLidarMsgQueue.pop();
                    continue;
                }


                bool detected_fast_rotation = false;
                bool huge_error_detected = false;
                if ( !vimuMsg.empty() && (  abs( vimuMsg.front()->angular_velocity.z) >  hori_rotate_th ||
                                            abs( vimuMsg.back()->angular_velocity.z) > hori_rotate_th)){
                    detected_fast_rotation = true;
                    ROS_WARN_STREAM("[HORI] Fast rotation detected !");
                    hori_lidar_list->clear();
                    if(veloFrameList->size() > 0){
                        hori_lidar_list->push_back(veloFrameList->front());
                        // hori_lidar_list->front().lidarType = 1;
                        _time_last_hori = veloFrameList->front().timeStamp;
                    }
                    else
                        _time_last_hori =  _time_last_velo;
                    newHoriFullCloud = false;

                    delta_Rl = velo_delta_Rl;
                    delta_tl = velo_delta_tl;

                    _unionLidarMsgQueue.pop();
                    continue;
                }

                // remove lidar distortion
                // RemoveLidarDistortion(laserCloudFullHoriRes, delta_Rl, delta_tl); //  方式不对。。。 todo: 统一的变换
                sensor_msgs::PointCloud2 laserCloudMsg;
                pcl::toROSMsg(*laserCloudFullHoriRes, laserCloudMsg);
                laserCloudMsg.header.frame_id = "/lio_world";
                laserCloudMsg.header.stamp = ros::Time().fromSec(time_curr_hori);
                pubHoriUndistortCloud.publish(laserCloudMsg);


                // TODO: failure detection, skip current info
                estimator->EstimateLidarPose(*hori_lidar_list, exTlb, GravityVector, lidarMode);

                if(estimator->failureDetected() && !veloFrameList->empty()){
                    ROS_WARN_STREAM("[HORI] Failure detected with current environment!");
                    hori_lidar_list->clear();
                    hori_lidar_list->push_back(veloFrameList->front());
                    // hori_lidar_list->front().lidarType = 1;
                    _time_last_hori = veloFrameList->front().timeStamp;
                    newHoriFullCloud = false;

                    delta_Rl = velo_delta_Rl;
                    delta_tl = velo_delta_tl;

                    _unionLidarMsgQueue.pop();
                    continue;
                }

                pcl::PointCloud<PointType>::Ptr laserCloudCornerMap(new pcl::PointCloud<PointType>());
                pcl::PointCloud<PointType>::Ptr laserCloudSurfMap(new pcl::PointCloud<PointType>());
                Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity(); // after mapped odometry

                // No failure detected, using the estimator pose
                transformTobeMapped.topLeftCorner(3,3) = hori_lidar_list->front().Q * exRbl;
                transformTobeMapped.topRightCorner(3,1) = hori_lidar_list->front().Q * exPbl + hori_lidar_list->front().P;

                // update delta transformation
                delta_Rb = transformAftMapped.topLeftCorner(3, 3).transpose() * hori_lidar_list->front().Q.toRotationMatrix();
                delta_tb = transformAftMapped.topLeftCorner(3, 3).transpose() * (hori_lidar_list->front().P - transformAftMapped.topRightCorner(3, 1));
                Eigen::Matrix3d Rwlpre = transformAftMapped.topLeftCorner(3, 3) * exRbl;
                Eigen::Vector3d Pwlpre = transformAftMapped.topLeftCorner(3, 3) * exPbl + transformAftMapped.topRightCorner(3, 1);
                delta_Rl = Rwlpre.transpose() * transformTobeMapped.topLeftCorner(3,3);
                delta_tl = Rwlpre.transpose() * (transformTobeMapped.topRightCorner(3,1) - Pwlpre);
                transformAftMapped.topLeftCorner(3,3) = hori_lidar_list->front().Q.toRotationMatrix();
                transformAftMapped.topRightCorner(3,1) = hori_lidar_list->front().P;

                // // publish odometry rostopic
                // pubOdometry(transformTobeMapped, hori_lidar_list->front().timeStamp);

                // publish lidar points
                int laserCloudFullResNum = hori_lidar_list->front().laserCloud->points.size();
                pcl::PointCloud<PointType>::Ptr laserCloudAfterEstimate(new pcl::PointCloud<PointType>());
                laserCloudAfterEstimate->reserve(laserCloudFullResNum);
                for (int i = 0; i < laserCloudFullResNum; i++) {
                    PointType temp_point;
                    pointAssociateToMap(&hori_lidar_list->front().laserCloud->points[i], &temp_point, transformTobeMapped);
                    laserCloudAfterEstimate->push_back(temp_point);
                }
                // Publish cloud of Horis
                // int cloudSensorType = hori_lidar_list->front().lidarType;
                // if(cloudSensorType < 3){
                // sensor_msgs::PointCloud2 laserCloudMsg;


                Eigen::Matrix3d  rot_mtx = veloTransformAftMapped.topLeftCorner(3,3);
                Eigen::Vector3d  curr_velo_trans = veloTransformAftMapped.topRightCorner(3,1);
                Eigen::Quaterniond curr_velo_quat(rot_mtx);
                // double curr_velo_yaw = curr_velo_quat.toRotationMatrix().eulerAngles(0, 1, 2 )[2];
                // double curr_hori_yaw = hori_lidar_list->front().Q.toRotationMatrix().eulerAngles(0, 1, 2 )[2];
                double angle_diff = hori_lidar_list->front().Q.angularDistance( curr_velo_quat);
                Eigen::Vector3d dis_vec = curr_velo_trans - transformAftMapped.topRightCorner(3,1);
                std::cout << curr_velo_trans << " VS " << transformAftMapped.topRightCorner(3,1) << std::endl;
                double dis =  dis_vec.x() * dis_vec.x() + dis_vec.y() * dis_vec.y();
                // double angle_diff = abs( curr_hori_yaw - curr_hori_yaw);
                if (angle_diff > 0.1 || dis > 0.1 ){
                    huge_error_detected = true;
                }
                if(!estimator->failureDetected() && !detected_fast_rotation && !huge_error_detected){
                    // sensor_msgs::PointCloud2 laserCloudMsg;
                    // pcl::toROSMsg(*laserCloudAfterEstimate, laserCloudMsg);
                    // laserCloudMsg.header.frame_id = "/lio_world";
                    // laserCloudMsg.header.stamp.fromSec(hori_lidar_list->front().timeStamp);
                    // pubFullLaserCloud.publish(laserCloudMsg);
                    // ROS_INFO_STREAM("[Hori]: Publish result point cloud \n");

                    // pcl::PointCloud<PointType> filteredFeaureCloud = * estimator->get_filtered_corner_map();

                    // pcl::PointCloud<pcl::PointXYZ>  points_cp;
                    // pcl::copyPointCloud(filteredFeaureCloud, points_cp);
                    // lineDetection(points_cp.makeShared());
                    // pcl::toROSMsg(filteredFeaureCloud, laserCloudMsg);
                    // laserCloudMsg.header.frame_id = "/lio_world";
                    // laserCloudMsg.header.stamp.fromSec(hori_lidar_list->front().timeStamp);
                    // pubHoriCornerMapFiltered.publish(laserCloudMsg);

                }else{
                    transformAftMapped = veloTransformAftMapped;
                    // hori_lidar_list->clear();
                    // hori_lidar_list->push_back(veloFrameList->front());
                    // _time_last_hori = veloFrameList->front().timeStamp;
                    // newHoriFullCloud = false;
                    sensor_msgs::PointCloud2 laserCloudMsg;
                    pcl::toROSMsg(*laserCloudAfterEstimate, laserCloudMsg);
                    laserCloudMsg.header.frame_id = "/lio_world";
                    laserCloudMsg.header.stamp.fromSec(hori_lidar_list->front().timeStamp);
                    pubFullLaserCloud.publish(laserCloudMsg);
                    ROS_INFO_STREAM("[Hori]: Publish result point cloud \n");

                    ROS_WARN_STREAM("[Hori] : Failure Detected ! Quit publishing points" <<
                                "->  estimator->failureDetected(): " << estimator->failureDetected() <<
                                " || -> detected_fast_rotation :" << detected_fast_rotation <<
                                " || Angle diff: " << angle_diff <<
                                " || dis diff: "    << dis <<
                                " || huge_error_detected" << huge_error_detected);
                    // continue;
                }


                surfCloudPtr = estimator->get_surf_map();
                pcl::toROSMsg(*surfCloudPtr, laserCloudMsg);
                laserCloudMsg.header.frame_id = "/lio_world";
                laserCloudMsg.header.stamp.fromSec(hori_lidar_list->front().timeStamp);
                pubHoriSurfMap.publish(laserCloudMsg);
                // if(!surfCloudPtr->empty())
                //     pcl::io::savePCDFileASCII ("/home/qing/hori_surf.pcd", *surfCloudPtr);
                ROS_INFO_STREAM("[HORI ]: Publish Surf Map\n\n");

                // if tightly coupled IMU message, start IMU initialization
                if(IMU_Mode > 1 && !LidarIMUInited){
                // if( !LidarIMUInited){
                    // update lidar frame pose
                    lidarFrame.P = transformTobeMapped.topRightCorner(3,1);
                    Eigen::Matrix3d m3d = transformTobeMapped.topLeftCorner(3,3);
                    lidarFrame.Q = m3d;

                    ROS_INFO_STREAM ("WINDOWSIZE :  " << WINDOWSIZE  << " | lidarFrameList->size(): " << lidarFrameList->size() );
                    // static int pushCount = 0;
                    if(pushCount == 0){
                        lidarFrameList->push_back(lidarFrame);
                        lidarFrameList->back().imuIntegrator.Reset();
                        if(lidarFrameList->size() > WINDOWSIZE)
                            lidarFrameList->pop_front();
                    }else{
                        lidarFrameList->back().laserCloud = lidarFrame.laserCloud;
                        lidarFrameList->back().imuIntegrator.PushIMUMsg(vimuMsg);
                        lidarFrameList->back().timeStamp = lidarFrame.timeStamp;
                        lidarFrameList->back().P = lidarFrame.P;
                        lidarFrameList->back().Q = lidarFrame.Q;
                    }
                    pushCount++;
                    ROS_INFO_STREAM ("LidarIMUInited  pushCount:  " << pushCount);
                    if (pushCount >= 3){
                        pushCount = 0;
                        if(lidarFrameList->size() > 1){
                            auto iterRight = std::prev(lidarFrameList->end());
                            auto iterLeft = std::prev(std::prev(lidarFrameList->end()));
                            iterRight->imuIntegrator.PreIntegration(iterLeft->timeStamp, iterLeft->bg, iterLeft->ba);
                        }

                        if (lidarFrameList->size() == int(WINDOWSIZE / 1.5)) {
                                startTime = lidarFrameList->back().timeStamp;

                            }

                        if (!LidarIMUInited && lidarFrameList->size() == WINDOWSIZE && lidarFrameList->front().timeStamp >= startTime){
                            std::cout<<"************** [Hori] Start IMU MAP Initialization!!!******************"<<std::endl;
                            if(TryMAPInitialization(lidarFrameList))
                            {
                                LidarIMUInited = true;
                                pushCount = 0;
                                startTime = 0;
                            }
                            std::cout<<"************** [Hori] Finish IMU MAP Initialization!!!******************"<<std::endl;
                        }

                    }
                }


                _time_last_hori = time_curr_hori;
                // _horiLidarMsgQueue.pop();
                newHoriFullCloud = false;

                ROS_INFO_STREAM("Ending Hori cloud processing! ");
                ROS_INFO_STREAM("====================================================== \n\n");
            } // END if new Hori cloud
            auto t1 = std::chrono::high_resolution_clock::now();
            double dt = 1.e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t1-tick).count();

            if(dt_last == 0){
                dt_last = dt;
                ROS_WARN_STREAM("Time cost : " << dt);
            }else{
                dt_last = (dt + dt_last) ;
                ROS_WARN_STREAM("Time cost : " << dt_last / processed_cnt);
            }
            _unionLidarMsgQueue.pop();
        }

        // _unionLidarMsgQueue
        if(0){
            time_curr_hori = _horiLidarMsgQueue.front()->header.stamp.toSec();

            while(_horiLidarMsgQueue.size() > 1 && _veloLidarMsgQueue.size() > 1
                    && abs( time_curr_hori - time_curr_velo) > 0.0001
                    && time_curr_hori < time_curr_velo ){
                _horiLidarMsgQueue.pop();
                ROS_WARN("POP Front");
            }

            std::unique_lock<std::mutex> lock_velo(_mutexVeloLidarQueue);
            if( _veloLidarMsgQueue.size() > 1 && !_imuMsgVector.empty()  ){
                std::cout << " " << std::endl;
                ROS_INFO_STREAM("##############################################################");
                ROS_INFO_STREAM("##############################################################");
                ROS_INFO_STREAM(" \n\n ====================== Velo Cloud processing ========================== ");
                double time_velo = _veloLidarMsgQueue.front()->header.stamp.toSec();

                if( !_imuMsgVector.empty() && time_velo < _imuMsgVector.back()->header.stamp.toSec()){
                    time_curr_velo = time_velo;
                    pcl::fromROSMsg(*_veloLidarMsgQueue.front(), *laserCloudFullVeloRes);

                    ROS_INFO_STREAM("[VeloProcessing] VELO Front timestamp : " << std::setprecision(15) << _veloLidarMsgQueue.front()->header.stamp.toSec());

                    _veloLidarMsgQueue.pop();
                    ROS_INFO_STREAM("[VeloProcessing] stamp | " << std::setprecision(15) << "last : " << _time_last_velo  << " -> current: " << time_curr_velo
                                                                << " | next: " << _veloLidarMsgQueue.front()->header.stamp.toSec());

                    newVeloFullCloud = true;
                    lidarMode = 2;
                }else{
                    newVeloFullCloud = false;
                }

                ROS_INFO_STREAM(" newVeloFullCloud -> " << newVeloFullCloud );
            }else{
                newVeloFullCloud = false;
            }
            lock_velo.unlock();

            std::unique_lock<std::mutex> lock_hori(_mutexHoriLidarQueue);
            if(_horiLidarMsgQueue.size() > 1 && abs( time_curr_hori - time_curr_velo) < 0.0001){ // Receive the first frame from velodyne
            // if(_horiLidarMsgQueue.size() > 1){ // if process in real time, no need time
                // get new lidar msg

                pcl::fromROSMsg(*_horiLidarMsgQueue.front(), *laserCloudFullHoriRes);
                ROS_INFO_STREAM("HORI timestamp : " << std::setprecision(15) << _horiLidarMsgQueue.front()->header.stamp.toSec());
                _horiLidarMsgQueue.pop();
                newHoriFullCloud = true;
                lidarMode = 1;

            }else
                newVeloFullCloud = false;
            lock_hori.unlock();
        }



      r.sleep();
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "union_PoseEstimation");
    // ros::NodeHandle nodeHandler("~");
    ros::NodeHandle nodeHandler ;

    if(!ros::param::get("~filter_parameter_corner",filter_parameter_corner )){
        filter_parameter_corner = 0.2;
    }
    if(!ros::param::get("~velo_rotate_threshold",velo_rotate_th )){
        velo_rotate_th = 1.5;
    }
    if(!ros::param::get("~hori_rotate_threshold",hori_rotate_th )){
        hori_rotate_th = 0.5;
    }
    if(!ros::param::get("~filter_parameter_surf",filter_parameter_surf )){
        filter_parameter_surf = 0.4;
    };
    if(!ros::param::get("~IMU_Mode",IMU_Mode)){
        IMU_Mode = 1;
    }
    if(!ros::param::get("~Hori_IMU_Mode",Hori_IMU_Mode)){
        Hori_IMU_Mode = 1;
    }
    if(!ros::param::get("~extrin_recali_times", extrin_recali_times)){
        extrin_recali_times = 30;
    }
    if(!ros::param::get("~Velo_Only_Mode", velo_only_mode)){
        velo_only_mode = false;
    }

    if(!ros::param::get("~Real_Time_Mode", real_time_mode)){
        real_time_mode = false;
    }



    if(!ros::param::get("~IMU_MAP_INIT",IMU_MAP_init)){};
    if(!IMU_MAP_init) LidarIMUInited = false;


    std::cout << filter_parameter_corner << " " << filter_parameter_surf << " "  << hori_rotate_th<< " "
            << velo_rotate_th << " " << IMU_Mode <<  " " << Hori_IMU_Mode << " " << IMU_MAP_init <<std::endl;
    // set extrinsic matrix between lidar & IMU
    std::vector<double> vecTlb;
    if(!ros::param::get("~Extrinsic_Tlb",vecTlb )){
        vecTlb =  {1.0, 0.0, 0.0, -0.05512,
                    0.0, 1.0, 0.0, -0.02226,
                    0.0, 0.0, 1.0,  0.0297,
                    0.0, 0.0, 0.0,  1.0};
        ROS_WARN_STREAM("Extrinsic_Tlb unavailable ! Using default param");
    }

    // filter_parameter_corner = 0.2;
    // filter_parameter_surf   = 0.4;

    Eigen::Matrix3d R;
    Eigen::Vector3d t;
        R << vecTlb[0], vecTlb[1], vecTlb[2],
            vecTlb[4], vecTlb[5], vecTlb[6],
            vecTlb[8], vecTlb[9], vecTlb[10];
        t << vecTlb[3], vecTlb[7], vecTlb[11];
    Eigen::Quaterniond qr(R);
    R = qr.normalized().toRotationMatrix();
    exTlb.topLeftCorner(3,3) = R;
    exTlb.topRightCorner(3,1) = t;
    exRlb = R;
    exRbl = R.transpose();
    exPlb = t;
    exPbl = -1.0 * exRbl * exPlb;


    // set extrinsic matrix between velodyne & horizon
    std::vector<double> vecTvh;
    //     ros::param::get("~Extrinsic_Tvh",vecTvh);
    //     tfVeloHori <<   vecTvh[0], vecTvh[1], vecTvh[2], vecTvh[3],
    //                     vecTvh[4], vecTvh[5], vecTvh[6], vecTvh[7],
    //                     vecTvh[8], vecTvh[9], vecTvh[10], vecTvh[11],
    //                     vecTvh[12], vecTvh[13], vecTvh[14], vecTvh[15];
    // ;
    // std::cout << "transformation_matrix Velo-> Hori: \n"<<tfVeloHori << std::endl;

    ros::Subscriber subHoriFullCloud = nodeHandler.subscribe<sensor_msgs::PointCloud2>("/hori_full_cloud", 1, horiFullCallBack);
    ros::Subscriber subVeloFullCloud = nodeHandler.subscribe<sensor_msgs::PointCloud2>("/velo_full_cloud", 1, veloFullCallBack);
    ros::Subscriber subUnionCloud = nodeHandler.subscribe<union_om::union_cloud>("/union_feature_cloud", 1, unionCloudHandler);

    ros::Subscriber sub_imu;

    if(IMU_Mode > 0)
        sub_imu = nodeHandler.subscribe("/livox/imu", 2000, imu_callback, ros::TransportHints().unreliable());
    if(IMU_Mode < 2)
        WINDOWSIZE = 3;
    else
        WINDOWSIZE = 3;

    pubFullLaserCloud   = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_full_cloud_mapped", 10);
    pubVeloFullLaserCloud   = nodeHandler.advertise<sensor_msgs::PointCloud2>("/velo_full_cloud_mapped", 10);
    pubVeloSurfMap          = nodeHandler.advertise<sensor_msgs::PointCloud2>("/velo_surf_map", 10);
    pubHoriSurfMap        = nodeHandler.advertise<sensor_msgs::PointCloud2>("/hori_surf_map", 10);
    pubMergedSurfMap       = nodeHandler.advertise<sensor_msgs::PointCloud2>("/merged_surf_map", 10);
    pubVeloCornerMap  = nodeHandler.advertise<sensor_msgs::PointCloud2>("/velo_corner_map", 10);
    pubHoriCornerMap  = nodeHandler.advertise<sensor_msgs::PointCloud2>("/hori_corner_map", 10);
    pubMergedCornerMap  = nodeHandler.advertise<sensor_msgs::PointCloud2>("/merged_corner_map", 10);
    pubHoriCornerMapFiltered = nodeHandler.advertise<sensor_msgs::PointCloud2>("/hori_corner_map_filtered", 10);

    pubVeloUndistortCloud  = nodeHandler.advertise<sensor_msgs::PointCloud2>("/velo_undistort_cloud_raw", 10);
    pubHoriUndistortCloud  = nodeHandler.advertise<sensor_msgs::PointCloud2>("/hori_undistort_cloud_raw", 10);

    pubSegmentedRGB     = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_segmented_cloud", 10);
    pubLaserOdometry    = nodeHandler.advertise<nav_msgs::Odometry> ("/livox_odometry_mapped", 5);
    pubLaserOdometryPath = nodeHandler.advertise<nav_msgs::Path> ("/livox_odometry_path_mapped", 5);

    tfBroadcaster = new tf::TransformBroadcaster();
    laserCloudFullRes.reset(new pcl::PointCloud<PointType>);
    laserCloudFullHoriRes.reset(new pcl::PointCloud<PointType>);
    laserCloudFullVeloRes.reset(new pcl::PointCloud<PointType>);
    surfCloudPtr.reset(new pcl::PointCloud<PointType>);
    cornerCloudPtr.reset(new pcl::PointCloud<PointType>);
    estimator = new Estimator(filter_parameter_corner, filter_parameter_surf);

    lidarFrameList.reset(new std::list<Estimator::LidarFrame>);
    veloFrameList.reset(new std::list<Estimator::LidarFrame>);

    std::thread thread_process{process};
    ros::spin();

    return 0;
}

