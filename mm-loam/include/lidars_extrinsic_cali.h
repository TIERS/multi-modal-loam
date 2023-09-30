 #include <time.h>
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h> 
 

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
 

#include <fstream>
#include <chrono>
#include <string>
#include <Eigen/Dense>
  
typedef  pcl::PointXYZ   PointT;
typedef  pcl::PointXYZI   PointType;

class LidarsCalibrater{
    private: 
        // ros::NodeHandle nh;  
        // ros::Subscriber sub_hori; 
        // ros::Publisher pub_hori;  
        // ros::Subscriber sub_velo; 
        // ros::Publisher pub_velo;  

        // Hori TF
        int hori_itegrate_frames = 5;
        pcl::PointCloud<PointT> hori_igcloud;
        Eigen::Matrix4f hori_tf_matrix; 
        Eigen::Matrix4f velo_hori_tf_matrix; 
        bool hori_tf_initd = false;  
  
        // Velo TF
        Eigen::Matrix4f velo_tf_matrix; 
        bool velo_tf_initd = false;
        bool velo_reveived= false;

        pcl::PointCloud<PointT> velo_cloud;


    public:
        LidarsCalibrater(){  
            // sub_hori = nh.subscribe<sensor_msgs::PointCloud2>("/livox_horizon", 1000, &LidarsCalibrater::hori_cloud_handler, this);
            // sub_velo = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1000, &LidarsCalibrater::velo_cloud_handler, this); 
  
            // pub_hori    = nh.advertise<sensor_msgs::PointCloud2>("/a_horizon", 1);
            // pub_velo    = nh.advertise<sensor_msgs::PointCloud2>("/a_velo", 1);    
        };
          
        ~LidarsCalibrater(){};

        // transform matrix to tf::Transform
        void matrix_to_transform(Eigen::Matrix4f & matrix, tf::Transform & trans){
            tf::Vector3 origin;
            origin.setValue(static_cast<double>(matrix(0,3)),static_cast<double>(matrix(1,3)),static_cast<double>(matrix(2,3)));

            tf::Matrix3x3 tf3d;
            tf3d.setValue(static_cast<double>(matrix(0,0)), static_cast<double>(matrix(0,1)), static_cast<double>(matrix(0,2)),
            static_cast<double>(matrix(1,0)), static_cast<double>(matrix(1,1)), static_cast<double>(matrix(1,2)),
            static_cast<double>(matrix(2,0)), static_cast<double>(matrix(2,1)), static_cast<double>(matrix(2,2)));

            tf::Quaternion tfqt;
            tf3d.getRotation(tfqt);

            trans.setOrigin(origin);
            trans.setRotation(tfqt);
        }

        // Remove far points to avoid downsampling error
        void removeFarPointCloud(const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out,  float thres) 
        {
            if (&cloud_in != &cloud_out) {
                cloud_out.header = cloud_in.header;
                cloud_out.points.resize(cloud_in.points.size());
            }

            size_t j = 0;

            for (size_t i = 0; i < cloud_in.points.size(); ++i) {
                if (cloud_in.points[i].x * cloud_in.points[i].x +
                        cloud_in.points[i].y * cloud_in.points[i].y +
                        cloud_in.points[i].z * cloud_in.points[i].z > thres * thres)
                    continue;
                cloud_out.points[j] = cloud_in.points[i];
                j++;
            }
            if (j != cloud_in.points.size()) {
                cloud_out.points.resize(j);
            }

            cloud_out.height = 1;
            cloud_out.width = static_cast<uint32_t>(j);
            cloud_out.is_dense = true;
        }
        

        Eigen::Matrix4f getVeloToHoriTFMatrix(){
             return velo_hori_tf_matrix;
         }

        Eigen::Matrix4f getHoriToVeloTFMatrix(){
             return hori_tf_matrix;
        }

        bool getIsTfAvaiable(){
            return hori_tf_initd;
        }

        // handle Horizon cloud msgs
        void hori_cloud_handler(const sensor_msgs::PointCloud2  pointCloudIn)
        {
            if(!velo_reveived) return;
            pcl::PointCloud<PointT>  full_cloud_in;
            pcl::fromROSMsg( pointCloudIn, full_cloud_in);
            pcl_conversions::toPCL(pointCloudIn.header, full_cloud_in.header); 

            if(hori_itegrate_frames > 0)
            {
                hori_igcloud += full_cloud_in; 
                hori_itegrate_frames--; 
                // ROS_INFO_STREAM("hori cloud integrating: " << hori_itegrate_frames);
                return;
            }else
            {
                if(!hori_tf_initd){
                    Eigen::AngleAxisf init_rot_x( 0.0 , Eigen::Vector3f::UnitX());
                    Eigen::AngleAxisf init_rot_y( 0.0 , Eigen::Vector3f::UnitY());
                    Eigen::AngleAxisf init_rot_z( 0.0 , Eigen::Vector3f::UnitZ());

                    Eigen::Translation3f init_trans(0.0,0.0,0.0);
                    Eigen::Matrix4f init_tf = (init_trans * init_rot_z * init_rot_y * init_rot_x).matrix();
 
                    // pcl::transformPointCloud (full_cloud , cloud_out, transformation_matrix);
                     ROS_INFO("\n\n\n  Calibrate Horizon ...");
                    calibrate_PCLICP(hori_igcloud.makeShared(), velo_cloud.makeShared(), hori_tf_matrix, true); 
                    // Eigen::Matrix3f rot_matrix = hori_tf_matrix.block(0,0,3,3);
                    // Eigen::Vector3f trans_vector = hori_tf_matrix.block(0,3,3,1);
                    
                    std::cout << "transformation_matrix Hori-> Velo: \n"<<hori_tf_matrix << std::endl;
                    Eigen::Matrix3f rot_matrix = hori_tf_matrix.block(0,0,3,3);
                    Eigen::Vector3f trans_vector = hori_tf_matrix.block(0,3,3,1);
                    velo_hori_tf_matrix.block(0,0,3,3) = rot_matrix.transpose();
                    velo_hori_tf_matrix.block(0,3,3,1) =  hori_tf_matrix.block(0,3,3,1) * -1;
                    velo_hori_tf_matrix.block(3,0,1,4) = hori_tf_matrix.block(3,0,1,4);
                    std::cout << "transformation_matrix Velo-> Hori: \n"<<velo_hori_tf_matrix << std::endl;

                    // std::cout << "hori -> base_link " << trans_vector.transpose()
                    //     << " " << rot_matrix.eulerAngles(2,1,0).transpose() << " /" << "hori_frame"
                    //     << " /" << "base_link" << " 10" << std::endl;

                    // publish result
                    pcl::PointCloud<PointT>  out_cloud;
                    out_cloud += hori_igcloud;
                    // pcl::transformPointCloud (hori_igcloud , out_cloud, hori_tf_matrix);

                    sensor_msgs::PointCloud2 hori_msg;
                    pcl::toROSMsg(out_cloud, hori_msg);
                    hori_msg.header.stamp = ros::Time::now();
                    hori_msg.header.frame_id = "base_link"; 
                    // pub_hori.publish(hori_msg); 

                    hori_tf_initd = true;
                }else
                {
                    // tf::Transform t_transform;
                    // matrix_to_transform(hori_tf_matrix,t_transform);
                    // tf_br.sendTransform(tf::StampedTransform(t_transform, ros::Time::now(), "hori_frame", "base_link"));
        
                    pcl::PointCloud<PointT>  out_cloud;
                    out_cloud += full_cloud_in;
                    // pcl::transformPointCloud (full_cloud_in , out_cloud, hori_tf_matrix);

                    sensor_msgs::PointCloud2 hori_msg;
                    pcl::toROSMsg(out_cloud, hori_msg);
                    // hori_msg.header.stamp = ros::Time::now();
                    double stamp_sec = pointCloudIn.header.stamp.toSec();
                    ros::Time stamp(stamp_sec);
                    hori_msg.header.stamp = stamp ;
                    hori_msg.header.frame_id = "base_link"; 
                    // pub_hori.publish(hori_msg); 
                }
            }  
        }

        // handle Velodyne cloud msgs
        void velo_cloud_handler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn)
        { 
            pcl::PointCloud<PointT>  full_cloud_in;
            pcl::fromROSMsg(*pointCloudIn, full_cloud_in);
            pcl_conversions::toPCL(pointCloudIn->header, full_cloud_in.header); 
              
            if(!hori_tf_initd){
                //  std::cout << "OS0 -> base_link " << trans_vector.transpose()
                //     << " " << rot_matrix.eulerAngles(2,1,0).transpose() << " /" << "os0_sensor"
                //     << " /" << "base_link" << " 10" << std::endl;
 
                Eigen::AngleAxisf init_rot_x( 0.0 , Eigen::Vector3f::UnitX());
                Eigen::AngleAxisf init_rot_y( 0.0 , Eigen::Vector3f::UnitY());
                Eigen::AngleAxisf init_rot_z( 0.0 , Eigen::Vector3f::UnitZ()); 
                Eigen::Translation3f init_trans(0.0,0.0,0.0);
                Eigen::Matrix4f init_tf = (init_trans * init_rot_z * init_rot_y * init_rot_x).matrix();

                Eigen::Matrix3f rot_matrix = init_tf.block(0,0,3,3);
                Eigen::Vector3f trans_vector = init_tf.block(0,3,3,1);

                pcl::PointCloud<PointT>  out_cloud;
                pcl::transformPointCloud (full_cloud_in , full_cloud_in, init_tf);
                
                velo_cloud.clear();
                velo_cloud += full_cloud_in;  
                velo_reveived = true;

                sensor_msgs::PointCloud2 velo_msg;
                pcl::toROSMsg(velo_cloud, velo_msg);
                // velo_msg.header.stamp = ros::Time::now();
                velo_msg.header.stamp = pointCloudIn->header.stamp;
                velo_msg.header.frame_id = "base_link"; 
                // pub_velo.publish(velo_msg); 
            }

            if(hori_tf_initd){
  
                pcl::PointCloud<PointT>  out_cloud;
                pcl::transformPointCloud (full_cloud_in , out_cloud, velo_hori_tf_matrix);

                sensor_msgs::PointCloud2 velo_msg;
                pcl::toROSMsg(out_cloud, velo_msg);
                // velo_msg.header.stamp = ros::Time::now();
                double stamp_sec = pointCloudIn->header.stamp.toSec() - 0.05;
                ros::Time stamp(stamp_sec); 
                velo_msg.header.stamp =  stamp;

                // velo_msg.header.stamp = pointCloudIn->header.stamp;
                velo_msg.header.frame_id = "base_link"; 
                // pub_velo.publish(velo_msg); 
                
            }  
        }

        // perform ICP matching
        void calibrate_PCLICP(pcl::PointCloud<PointT>::Ptr source_cloud,
        pcl::PointCloud<PointT>::Ptr target_cloud, Eigen::Matrix4f &tf_marix, bool save_pcd =true)
        {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            std::chrono::duration<double, std::ratio<1, 1000>> time_span =
            std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

            std::cout << "------------checking PCL GICP---------------- "<< std::endl;
            int pCount = source_cloud->size();

            pcl::PointCloud<PointT>::Ptr ds_cloud_in (new pcl::PointCloud<PointT> );
            pcl::PointCloud<PointT>::Ptr dstf_cloud_in (new pcl::PointCloud<PointT> );
            pcl::PointCloud<PointT>::Ptr ds_cloud_out (new pcl::PointCloud<PointT> ); 
            
            // // Remove the noise points
            pcl::PointCloud<PointT>::Ptr cloudin_filtered (new pcl::PointCloud<PointT>); 
            // pcl::StatisticalOutlierRemoval<PointT> sor;
            // sor.setInputCloud (source_cloud);
            // sor.setMeanK (50);
            // sor.setStddevMulThresh (1.0);
            // sor.filter (*cloudin_filtered); 
            // ROS_INFO_STREAM("CloudIn-> Removing noise clouds "<< source_cloud->size() << " "<<  cloudin_filtered->size() );
            removeFarPointCloud(*source_cloud, *cloudin_filtered, 50);
            ROS_INFO_STREAM("CloudIn-> Removing noise clouds "<< source_cloud->size() << " "<<  cloudin_filtered->size() );
 
            // Remove the noise points
            pcl::PointCloud<PointT>::Ptr cloudout_filtered (new pcl::PointCloud<PointT>);  
            // sor.setInputCloud (target_cloud);
            // sor.setMeanK (50);
            // sor.setStddevMulThresh (1.0);
            // sor.filter (*cloudout_filtered);
            // ROS_INFO_STREAM("CloudOut-> Removing noise clouds "<< target_cloud->size() << " "<<  cloudout_filtered->size() ); 
            removeFarPointCloud(*target_cloud, *cloudout_filtered, 50);
            ROS_INFO_STREAM("CloudOut-> Removing noise clouds "<< target_cloud->size() << " "<<  cloudout_filtered->size() ); 
        
            // Create the filtering object
            pcl::VoxelGrid< PointT> vox;
            vox.setInputCloud (cloudin_filtered);
            vox.setLeafSize (0.05f, 0.05f, 0.05f);
            vox.filter(*ds_cloud_in);
            // std::cout << "Source DS: " << source_cloud->size() << " ->  " << ds_cloud_in->size()<< std::endl;

            // Create the filtering object 
            vox.setInputCloud (cloudout_filtered);
            vox.setLeafSize (0.05f, 0.05f, 0.05f);
            vox.filter(*ds_cloud_out);
            // std::cout << "Target DS: " << target_cloud->size() << " -> " << ds_cloud_out->size()<< std::endl;
            
            // Init Rotation:
            // std::cout << "Rotating DS source  .... " << ds_cloud_in->size()<< std::endl; 
            // Eigen::Quaterniond rotate_tf =  Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())
            //                                 * Eigen::AngleAxisd(0,  Eigen::Vector3d::UnitY())
            //                                 * Eigen::AngleAxisd( M_PI/4, Eigen::Vector3d::UnitZ());
            // Eigen::Vector3d    trans_tf {0,0,0}; 
            // *dstf_cloud_in =  *dstf_cloud_in + *(transformCloud(*ds_cloud_in, rotate_tf, trans_tf));
            *dstf_cloud_in =  *dstf_cloud_in + *ds_cloud_in; 

            std::cout << "GICP start  .... " << ds_cloud_in->size() << " to "<< ds_cloud_out->size()<< std::endl;
            pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
            gicp.setTransformationEpsilon(0.001);
            gicp.setMaxCorrespondenceDistance(2);
            gicp.setMaximumIterations(500);
            gicp.setRANSACIterations(12);  
            gicp.setInputSource(dstf_cloud_in);
            gicp.setInputTarget(ds_cloud_out);

            pcl::PointCloud<PointT>::Ptr transformedP (new pcl::PointCloud<PointT>);

            t1 = std::chrono::steady_clock::now();
            gicp.align(*transformedP);
            t2 = std::chrono::steady_clock::now();
            time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
            // std::cout << "PCL gicp.align Time: " << time_span.count() << " ms."<< std::endl;
            std::cout << "has converged: " << gicp.hasConverged() << " score: " <<
                gicp.getFitnessScore() << std::endl; 

            auto transformation_matrix =  gicp.getFinalTransformation ();
            std::cout << "transformation_matrix:\n"<<transformation_matrix << std::endl;
            std::cout << std::endl;

            auto cloudSrc = dstf_cloud_in;
            auto cloudDst = ds_cloud_out;

            pcl::PointCloud<PointT> input_transformed;
            pcl::transformPointCloud (*cloudSrc, input_transformed, transformation_matrix);
        
            Eigen::Matrix3f rot_matrix = transformation_matrix.block<3,3>(0,0);
            Eigen::Vector3f euler = rot_matrix.eulerAngles(2, 1, 0); 
            // std::cout << "Rotation    : "<<  euler.x() << " " << euler.y() << " " << euler.z()<< std::endl;   
            // std::cout << "Rotation_matrix:\n"<<rot_matrix << std::endl; 

            tf_marix = transformation_matrix;

            if(save_pcd)
            {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudSrcRGB (new pcl::PointCloud<pcl::PointXYZRGB>(cloudSrc->size(),1));
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudDstRGB (new pcl::PointCloud<pcl::PointXYZRGB>(cloudDst->size(),1));
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudALL (new pcl::PointCloud<pcl::PointXYZRGB>(cloudSrc->size() + cloudDst->size(),1));
                
                // Fill in the CloudIn data
                for (int i = 0; i < cloudSrc->size(); i++)
                {
                    pcl::PointXYZRGB &pointin = (*cloudSrcRGB)[i];
                    pointin.x = (input_transformed)[i].x;
                    pointin.y = (input_transformed)[i].y;
                    pointin.z = (input_transformed)[i].z;
                    pointin.r = 255;
                    pointin.g = 0;
                    pointin.b = 0;
                    (*cloudALL)[i] = pointin;
                }
                for (int i = 0; i < cloudDst->size(); i++)
                {
                    pcl::PointXYZRGB &pointout = (*cloudDstRGB)[i];
                    pointout.x = (*cloudDst)[i].x;
                    pointout.y = (*cloudDst)[i].y;
                    pointout.z = (*cloudDst)[i].z;
                    pointout.r = 0;
                    pointout.g = 255;
                    pointout.b = 255;
                    (*cloudALL)[i+cloudSrc->size()] = pointout;
                } 
                // pcl::io::savePCDFile<pcl::PointXYZRGB> ("/home/qing/icp_ICP.pcd", *cloudALL);
            }
        } 
};

void matrixToTransform(Eigen::Matrix4f & matrix, tf::Transform & trans){
    tf::Vector3 origin;
    origin.setValue(static_cast<double>(matrix(0,3)),static_cast<double>(matrix(1,3)),static_cast<double>(matrix(2,3)));

    tf::Matrix3x3 tf3d;
    tf3d.setValue(static_cast<double>(matrix(0,0)), static_cast<double>(matrix(0,1)), static_cast<double>(matrix(0,2)),
    static_cast<double>(matrix(1,0)), static_cast<double>(matrix(1,1)), static_cast<double>(matrix(1,2)),
    static_cast<double>(matrix(2,0)), static_cast<double>(matrix(2,1)), static_cast<double>(matrix(2,2)));

    tf::Quaternion tfqt;
    tf3d.getRotation(tfqt);

    trans.setOrigin(origin);
    trans.setRotation(tfqt);
}

void removeFarPointCloud(const pcl::PointCloud<pcl::PointXYZI> &cloud_in, pcl::PointCloud<pcl::PointXYZI> &cloud_out, float thres) 
{
    if (&cloud_in != &cloud_out) {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i) {
        if (cloud_in.points[i].x * cloud_in.points[i].x +
                cloud_in.points[i].y * cloud_in.points[i].y +
                cloud_in.points[i].z * cloud_in.points[i].z > thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size()) {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

void removeNearPointCloud(const pcl::PointCloud<pcl::PointXYZINormal> &cloud_in, pcl::PointCloud<pcl::PointXYZINormal> &cloud_out, float thres) 
{
    cloud_out.clear();
    if (&cloud_in != &cloud_out) {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i) {
        if (cloud_in.points[i].x * cloud_in.points[i].x +
                cloud_in.points[i].y * cloud_in.points[i].y +
                cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size()) {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

void removeNearFarPoints(const pcl::PointCloud<pcl::PointXYZINormal> &cloud_in, pcl::PointCloud<pcl::PointXYZINormal> &cloud_out, float near_thres,float far_thres) 
{
    cloud_out.clear();
    if (&cloud_in != &cloud_out) {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i) {
        float dis = cloud_in.points[i].x * cloud_in.points[i].x +
                cloud_in.points[i].y * cloud_in.points[i].y +
                cloud_in.points[i].z * cloud_in.points[i].z;
        if (dis < near_thres * near_thres || dis > far_thres * far_thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size()) {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

void livoxToPCLCloud( livox_ros_driver::CustomMsg& livox_msg_in, pcl::PointCloud<pcl::PointXYZI>& out_cloud, int ratio = 1) 
{
    out_cloud.clear();
    for (unsigned int i = 0; i < livox_msg_in.point_num / ratio; ++i) {
        pcl::PointXYZI pt;
        pt.x = livox_msg_in.points[i].x;
        pt.y = livox_msg_in.points[i].y;
        pt.z = livox_msg_in.points[i].z; 
        pt.intensity =  livox_msg_in.points[i].reflectivity;
        // pt.intensity = livox_msg_in.timebase + livox_msg_in.points[i].offset_time;
        out_cloud.push_back(pt);
    }   
}

void calibratePCLICP(pcl::PointCloud<PointType>::Ptr source_cloud,
pcl::PointCloud<PointType>::Ptr target_cloud, Eigen::Matrix4f &tf_marix, bool save_pcd =true)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::ratio<1, 1000>> time_span =
    std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

    std::cout << "------------checking PCL GICP---------------- "<< std::endl;
    int pCount = source_cloud->size();

    pcl::PointCloud<PointType>::Ptr ds_cloud_in (new pcl::PointCloud<PointType> );
    pcl::PointCloud<PointType>::Ptr dstf_cloud_in (new pcl::PointCloud<PointType> );
    pcl::PointCloud<PointType>::Ptr ds_cloud_out (new pcl::PointCloud<PointType> ); 
    
    // // Remove the noise points
    pcl::PointCloud<PointType>::Ptr cloudin_filtered (new pcl::PointCloud<PointType>); 
    // pcl::StatisticalOutlierRemoval<PointType> sor;
    // sor.setInputCloud (source_cloud);
    // sor.setMeanK (50);
    // sor.setStddevMulThresh (1.0);
    // sor.filter (*cloudin_filtered); 
    // ROS_INFO_STREAM("CloudIn-> Removing noise clouds "<< source_cloud->size() << " "<<  cloudin_filtered->size() );
    removeFarPointCloud(*source_cloud, *cloudin_filtered, 50);
    ROS_INFO_STREAM("CloudIn-> Removing noise clouds "<< source_cloud->size() << " "<<  cloudin_filtered->size() );

    // Remove the noise points
    pcl::PointCloud<PointType>::Ptr cloudout_filtered (new pcl::PointCloud<PointType>);  
    // sor.setInputCloud (target_cloud);
    // sor.setMeanK (50);
    // sor.setStddevMulThresh (1.0);
    // sor.filter (*cloudout_filtered);
    // ROS_INFO_STREAM("CloudOut-> Removing noise clouds "<< target_cloud->size() << " "<<  cloudout_filtered->size() ); 
    removeFarPointCloud(*target_cloud, *cloudout_filtered, 50);
    ROS_INFO_STREAM("CloudOut-> Removing noise clouds "<< target_cloud->size() << " "<<  cloudout_filtered->size() ); 

    // Create the filtering object
    pcl::VoxelGrid< PointType> vox;
    vox.setInputCloud (cloudin_filtered);
    vox.setLeafSize (0.05f, 0.05f, 0.05f);
    vox.filter(*ds_cloud_in);
    // std::cout << "Source DS: " << source_cloud->size() << " ->  " << ds_cloud_in->size()<< std::endl;

    // Create the filtering object 
    vox.setInputCloud (cloudout_filtered);
    vox.setLeafSize (0.05f, 0.05f, 0.05f);
    vox.filter(*ds_cloud_out);
    // std::cout << "Target DS: " << target_cloud->size() << " -> " << ds_cloud_out->size()<< std::endl;
    
    // Init Rotation:
    // std::cout << "Rotating DS source  .... " << ds_cloud_in->size()<< std::endl; 
    // Eigen::Quaterniond rotate_tf =  Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())
    //                                 * Eigen::AngleAxisd(0,  Eigen::Vector3d::UnitY())
    //                                 * Eigen::AngleAxisd( M_PI/4, Eigen::Vector3d::UnitZ());
    // Eigen::Vector3d    trans_tf {0,0,0}; 
    // *dstf_cloud_in =  *dstf_cloud_in + *(transformCloud(*ds_cloud_in, rotate_tf, trans_tf));
    *dstf_cloud_in =  *dstf_cloud_in + *ds_cloud_in; 

    std::cout << "GICP start  .... " << ds_cloud_in->size() << " to "<< ds_cloud_out->size()<< std::endl;
    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> gicp;
    gicp.setTransformationEpsilon(0.001);
    gicp.setMaxCorrespondenceDistance(2);
    gicp.setMaximumIterations(500);
    gicp.setRANSACIterations(12);  
    gicp.setInputSource(dstf_cloud_in);
    gicp.setInputTarget(ds_cloud_out);

    pcl::PointCloud<PointType>::Ptr transformedP (new pcl::PointCloud<PointType>);

    t1 = std::chrono::steady_clock::now();
    gicp.align(*transformedP);
    t2 = std::chrono::steady_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
    // std::cout << "PCL gicp.align Time: " << time_span.count() << " ms."<< std::endl;
    std::cout << "====>>>>> has converged: " << gicp.hasConverged() << " ====>>>>> score: " <<
        gicp.getFitnessScore() << std::endl; 

    auto transformation_matrix =  gicp.getFinalTransformation ();
    std::cout << "transformation_matrix:\n"<<transformation_matrix << std::endl;
    std::cout << std::endl;

    auto cloudSrc = dstf_cloud_in;
    auto cloudDst = ds_cloud_out;

    pcl::PointCloud<PointType> input_transformed;
    pcl::transformPointCloud (*cloudSrc, input_transformed, transformation_matrix);

    Eigen::Matrix3f rot_matrix = transformation_matrix.block<3,3>(0,0);
    Eigen::Vector3f euler = rot_matrix.eulerAngles(2, 1, 0); 
    // std::cout << "Rotation    : "<<  euler.x() << " " << euler.y() << " " << euler.z()<< std::endl;   
    // std::cout << "Rotation_matrix:\n"<<rot_matrix << std::endl; 

    tf_marix = transformation_matrix;

    if(save_pcd)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudSrcRGB (new pcl::PointCloud<pcl::PointXYZRGB>(cloudSrc->size(),1));
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudDstRGB (new pcl::PointCloud<pcl::PointXYZRGB>(cloudDst->size(),1));
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudALL (new pcl::PointCloud<pcl::PointXYZRGB>(cloudSrc->size() + cloudDst->size(),1));
        
        // Fill in the CloudIn data
        for (int i = 0; i < cloudSrc->size(); i++)
        {
            pcl::PointXYZRGB &pointin = (*cloudSrcRGB)[i];
            pointin.x = (input_transformed)[i].x;
            pointin.y = (input_transformed)[i].y;
            pointin.z = (input_transformed)[i].z;
            pointin.r = 255;
            pointin.g = 0;
            pointin.b = 0;
            (*cloudALL)[i] = pointin;
        }
        for (int i = 0; i < cloudDst->size(); i++)
        {
            pcl::PointXYZRGB &pointout = (*cloudDstRGB)[i];
            pointout.x = (*cloudDst)[i].x;
            pointout.y = (*cloudDst)[i].y;
            pointout.z = (*cloudDst)[i].z;
            pointout.r = 0;
            pointout.g = 255;
            pointout.b = 255;
            (*cloudALL)[i+cloudSrc->size()] = pointout;
        } 
        // pcl::io::savePCDFile<pcl::PointXYZRGB> ("/home/qing/icp_ICP.pcd", *cloudALL);
    }
} 

