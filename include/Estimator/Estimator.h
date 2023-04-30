#ifndef LIO_LIVOX_ESTIMATOR_H
#define LIO_LIVOX_ESTIMATOR_H

#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <queue>
#include <iterator>
#include <future>
#include "MapManager/Map_Manager.h"
#include "utils/ceresfunc.h"
#include "IMUIntegrator/IMUIntegrator.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <chrono>

class Estimator{
	typedef pcl::PointXYZINormal PointType;
public:
	/** \brief slide window size */
	static const int SLIDEWINDOWSIZE = 5;

	/** \brief lidar frame struct */
	struct LidarFrame
	{
		pcl::PointCloud<PointType>::Ptr laserCloud;
		IMUIntegrator imuIntegrator;
		Eigen::Vector3d P; 	   // pose
		Eigen::Vector3d V;	   // speed
		Eigen::Quaterniond Q;  // orientation
		Eigen::Vector3d bg;	   //
		Eigen::Vector3d ba;
		double timeStamp;
		double timeOffset;
		int    lidarType;     // 1 Hori; 2 Velo

		LidarFrame(){
			P.setZero();
			V.setZero();
			Q.setIdentity();
			bg.setZero();
			ba.setZero();
			timeStamp = 0;
			timeOffset = 0;
			lidarType = 0;
		}
	};

	/** \brief point to line feature */
	struct FeatureLine
	{
		Eigen::Vector3d pointOri;
		Eigen::Vector3d lineP1;
		Eigen::Vector3d lineP2;
		double error;
		bool valid;
		FeatureLine(Eigen::Vector3d  po, Eigen::Vector3d  p1, Eigen::Vector3d  p2)
						:pointOri(std::move(po)), lineP1(std::move(p1)), lineP2(std::move(p2)){
			valid = false;
			error = 0;
		}
		double ComputeError(const Eigen::Matrix4d& pose){
			Eigen::Vector3d P_to_Map = pose.topLeftCorner(3,3) * pointOri + pose.topRightCorner(3,1);
			double l12 = std::sqrt((lineP1(0) - lineP2(0))*(lineP1(0) - lineP2(0)) + (lineP1(1) - lineP2(1))*
																						(lineP1(1) - lineP2(1)) + (lineP1(2) - lineP2(2))*(lineP1(2) - lineP2(2)));
			double a012 = std::sqrt(
							((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1)))
							* ((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1)))
							+ ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2)))
								* ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2)))
							+ ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2)))
								* ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))));
			error = a012 / l12;
		}
	};

	/** \brief point to plan feature */
	struct FeaturePlan{
		Eigen::Vector3d pointOri;
		double pa;
		double pb;
		double pc;
		double pd;
		double error;
		bool valid;
		FeaturePlan(const Eigen::Vector3d& po, const double& pa_, const double& pb_, const double& pc_, const double& pd_)
						:pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_){
			valid = false;
			error = 0;
		}
		double ComputeError(const Eigen::Matrix4d& pose){
			Eigen::Vector3d P_to_Map = pose.topLeftCorner(3,3) * pointOri + pose.topRightCorner(3,1);
			error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
		}
	};

	/** \brief point to plan feature */
	struct FeaturePlanVec{
		Eigen::Vector3d pointOri;
		Eigen::Vector3d pointProj;
		Eigen::Matrix3d sqrt_info;
		double error;
		bool valid;
		FeaturePlanVec(const Eigen::Vector3d& po, const Eigen::Vector3d& p_proj, Eigen::Matrix3d sqrt_info_)
						:pointOri(po), pointProj(p_proj), sqrt_info(sqrt_info_) {
			valid = false;
			error = 0;
		}
		double ComputeError(const Eigen::Matrix4d& pose){
			Eigen::Vector3d P_to_Map = pose.topLeftCorner(3,3) * pointOri + pose.topRightCorner(3,1);
			error = (P_to_Map - pointProj).norm();
		}
	};

	/** \brief non feature */
	struct FeatureNon{
		Eigen::Vector3d pointOri;
		double pa;
		double pb;
		double pc;
		double pd;
		double error;
		bool valid;
		FeatureNon(const Eigen::Vector3d& po, const double& pa_, const double& pb_, const double& pc_, const double& pd_)
						:pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_){
			valid = false;
			error = 0;
		}
		double ComputeError(const Eigen::Matrix4d& pose){
			Eigen::Vector3d P_to_Map = pose.topLeftCorner(3,3) * pointOri + pose.topRightCorner(3,1);
			error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
		}
	};

public:
	/** \brief constructor of Estimator
	*/
	Estimator(const float& filter_corner, const float& filter_surf);

	~Estimator();

		/** \brief Open a independent thread to increment MAP cloud
		*/
	[[noreturn]] void threadMapIncrement();

	/** \brief construct sharp feature Ceres Costfunctions
	* \param[in] edges: store costfunctions
	* \param[in] m4d: lidar pose, represented by matrix 4X4
	*/
	void processPointToLine(std::vector<ceres::CostFunction *>& edges,
							std::vector<FeatureLine>& vLineFeatures,
							const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
							const pcl::PointCloud<PointType>::Ptr& laserCloudCornerMap,
							const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
							const Eigen::Matrix4d& exTlb,
							const Eigen::Matrix4d& m4d);

	/** \brief construct Plan feature Ceres Costfunctions
	* \param[in] edges: store costfunctions
	* \param[in] m4d: lidar pose, represented by matrix 4X4
	*/
	void processPointToPlan(std::vector<ceres::CostFunction *>& edges,
							std::vector<FeaturePlan>& vPlanFeatures,
							const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
							const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
							const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
							const Eigen::Matrix4d& exTlb,
							const Eigen::Matrix4d& m4d);

	void processPointToPlanVec(std::vector<ceres::CostFunction *>& edges,
							   std::vector<FeaturePlanVec>& vPlanFeatures,
							   const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
							   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfMap,
							   const pcl::KdTreeFLANN<PointType>::Ptr& kdtree,
							   const Eigen::Matrix4d& exTlb,
							   const Eigen::Matrix4d& m4d,
							   bool& is_degenerate);

	void processNonFeatureICP(std::vector<ceres::CostFunction *>& edges,
							  std::vector<FeatureNon>& vNonFeatures,
							  const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
							  const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureLocal,
							  const pcl::KdTreeFLANN<PointType>::Ptr& kdtreeLocal,
							  const Eigen::Matrix4d& exTlb,
							  const Eigen::Matrix4d& m4d);

	/** \brief Transform Lidar Pose in slidewindow to double array
		* \param[in] lidarFrameList: Lidar Poses in slidewindow
		*/
	void vector2double(const std::list<LidarFrame>& lidarFrameList);

	/** \brief Transform double array to Lidar Pose in slidewindow
		* \param[in] lidarFrameList: Lidar Poses in slidewindow
		*/
	void double2vector(std::list<LidarFrame>& lidarFrameList);

	/** \brief estimate lidar pose by matching current lidar cloud with map cloud and tightly coupled IMU message
		* \param[in] lidarFrameList: multi-frames of lidar cloud and lidar pose
		* \param[in] exTlb: extrinsic matrix between lidar and IMU
		* \param[in] gravity: gravity vector
		*/
	void EstimateLidarPose(std::list<LidarFrame>& lidarFrameList,
						   const Eigen::Matrix4d& exTlb,
						   const Eigen::Vector3d& gravity,
						   int lidarMode);

	void Estimate(std::list<LidarFrame>& lidarFrameList,
				  const Eigen::Matrix4d& exTlb,
				  const Eigen::Vector3d& gravity,
				  bool& is_degenerate);

	pcl::PointCloud<PointType>::Ptr get_corner_map(){
		return map_manager->get_corner_map();
	}
	pcl::PointCloud<PointType>::Ptr get_surf_map(){
		return map_manager->get_surf_map();
	}
	pcl::PointCloud<PointType>::Ptr get_nonfeature_map(){
		return map_manager->get_nonfeature_map();
	}

	void MapIncrementLocal(const pcl::PointCloud<PointType>::Ptr& laserCloudCornerStack,
						   const pcl::PointCloud<PointType>::Ptr& laserCloudSurfStack,
						   const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureStack,
						   const Eigen::Matrix4d& transformTobeMapped
						    );


	void MapCornerFeatureFilter(pcl::PointCloud<PointType>::Ptr& map_corner_feature,
								pcl::PointCloud<PointType>::Ptr& map_corner_feature_filtered,
								pcl::PointCloud<PointType>::Ptr& curr_hori_corner,
								pcl::PointCloud<PointType>::Ptr& curr_hori_plane)
	{
		pcl::KdTreeFLANN<PointType>  hori_corner_kdtree;
		pcl::KdTreeFLANN<PointType>  hori_plane_kdtree;
		map_corner_feature_filtered.reset(new pcl::PointCloud<PointType>);

		hori_corner_kdtree.setInputCloud(map_corner_feature);
		hori_plane_kdtree.setInputCloud(curr_hori_plane);

		int points_num = map_corner_feature->points.size();
		PointType point_sel;
		std::vector<int>   cornerPointSearchInd;
    	std::vector<float> cornerPointSearchSqDis;
		std::vector<int>   planePointSearchInd;
    	std::vector<float> planePointSearchSqDis;

		for( int i = 0; i < points_num; i ++){
			point_sel.x = map_corner_feature->points[i].x;
			point_sel.y = map_corner_feature->points[i].y;
			point_sel.z = map_corner_feature->points[i].z;

			hori_corner_kdtree.nearestKSearch(point_sel, 1, cornerPointSearchInd, cornerPointSearchSqDis);
			hori_plane_kdtree.nearestKSearch(point_sel, 4, planePointSearchInd, planePointSearchSqDis);

			if(planePointSearchSqDis[4] < 0.1 && cornerPointSearchSqDis[0] < 0.1){
				map_corner_feature_filtered->push_back(map_corner_feature->points[i]);
			}
		}
	}


	double checkLocalizability( std::vector<Eigen::Vector3d> planeNormals);

	pcl::PointCloud<PointType>::Ptr get_filtered_corner_map(){
		return GlobalConerMapFiltered ;
	}

	bool   failureDetected();

private:
	/** \brief store map points */
	MAP_MANAGER* map_manager;

	double para_PR[SLIDEWINDOWSIZE][6];
	double para_VBias[SLIDEWINDOWSIZE][9];
	MarginalizationInfo *last_marginalization_info = nullptr;
	std::vector<double *> last_marginalization_parameter_blocks;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerLast;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfLast;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudNonFeatureLast;

	pcl::PointCloud<PointType>::Ptr laserCloudCornerFromLocal;  // corner feature points within local window (50 frames)
	pcl::PointCloud<PointType>::Ptr laserCloudSurfFromLocal;	// surf feature points within local window (50 frames)
	pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureFromLocal; // non feature points within local window (50 frames)
	pcl::PointCloud<PointType>::Ptr laserCloudCornerForMap;     // latest corner feature scans in sliding windows
	pcl::PointCloud<PointType>::Ptr laserCloudSurfForMap;		// latest surf   feature scans in sliding windows
	pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureForMap; // latest non-   feature scans in sliding windows
	Eigen::Matrix4d transformForMap;
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerStack; // vector of framelist of feature pointcloud
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfStack;   // vector of framelist of feature pointcloud
	std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudNonFeatureStack; // vector of framelist of feature pointcloud
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromLocal;         // kdtree of all feature points  within local window (50 frames)
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromLocal;           // kdtree of all feature points  within local window (50 frames)
	pcl::KdTreeFLANN<PointType>::Ptr kdtreeNonFeatureFromLocal;     // kdtree of all feature points  within local window (50 frames)

	pcl::VoxelGrid<PointType> downSizeFilterCorner;  	// global voxelgrid for corner
	pcl::VoxelGrid<PointType> downSizeFilterSurf; 		// global voxelgrid for surf
	pcl::VoxelGrid<PointType> downSizeFilterNonFeature; // global voxelgrid for NonFeature
	std::mutex mtx_Map;
	std::thread threadMap;

	pcl::KdTreeFLANN<PointType> CornerKdMap[10000];
	pcl::KdTreeFLANN<PointType> SurfKdMap[10000];
	pcl::KdTreeFLANN<PointType> NonFeatureKdMap[10000];

	pcl::PointCloud<PointType> GlobalSurfMap[10000];
	pcl::PointCloud<PointType> GlobalCornerMap[10000];
	pcl::PointCloud<PointType> GlobalNonFeatureMap[10000];

	pcl::PointCloud<PointType>::Ptr GlobalConerMapFiltered;

	int laserCenWidth_last = 10;
	int laserCenHeight_last = 5;
	int laserCenDepth_last = 10;

	static const int localMapWindowSize = 50;
	int localMapID = 0;
	pcl::PointCloud<PointType>::Ptr localCornerMap[localMapWindowSize];
	pcl::PointCloud<PointType>::Ptr localSurfMap[localMapWindowSize];
	pcl::PointCloud<PointType>::Ptr localNonFeatureMap[localMapWindowSize];

	int map_update_ID = 0;

	int map_skip_frame = 2; //every map_skip_frame frame update map
	double plan_weight_tan = 0.0;
	double thres_dist = 1.0;

	bool _fail_detected = false;
	Eigen::Vector3d last_velo_update_pose = {-1.0, -1.0, -1.0};
	Eigen::Vector3d last_hori_update_pose = {-1.0, -1.0, -1.0};
};

#endif //LIO_LIVOX_ESTIMATOR_H
