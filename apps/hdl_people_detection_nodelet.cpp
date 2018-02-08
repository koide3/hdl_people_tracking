#include <mutex>
#include <memory>
#include <iostream>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <hdl_people_tracking/ClusterArray.h>

#include <hdl_people_detection/people_detector.h>
#include <hdl_people_detection/background_subtractor.hpp>

namespace hdl_people_detection {

/**
 * @brief A nodelet to detect people using a 3D LIDAR
 */
class HdlPeopleDetectionNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlPeopleDetectionNodelet() {}
  virtual ~HdlPeopleDetectionNodelet() {}

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    // publishers
    backsub_points_pub = private_nh.advertise<sensor_msgs::PointCloud2>("backsub_points", 5);
    cluster_points_pub = private_nh.advertise<sensor_msgs::PointCloud2>("cluster_points", 5);
    human_points_pub = private_nh.advertise<sensor_msgs::PointCloud2>("human_points", 5);
    detection_markers_pub = private_nh.advertise<visualization_msgs::MarkerArray>("detection_markers", 5);

    backsub_voxel_points_pub = private_nh.advertise<sensor_msgs::PointCloud2>("backsub_voxel_points", 1, true);
    backsub_voxel_markers_pub = private_nh.advertise<visualization_msgs::Marker>("backsub_voxel_marker", 1, true);

    clusters_pub = private_nh.advertise<hdl_people_tracking::ClusterArray>("clusters", 10);

    // subscribers
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlPeopleDetectionNodelet::globalmap_callback, this);

    odom_sub.reset(new message_filters::Subscriber<nav_msgs::Odometry>(mt_nh, "/odom", 20));
    points_sub.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(mt_nh, "/velodyne_points", 20));
    sync.reset(new message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::PointCloud2>(*odom_sub, *points_sub, 20));
    sync->registerCallback(boost::bind(&HdlPeopleDetectionNodelet::callback, this, _1, _2));
  }

private:
  /**
   * @brief initialize_params
   */
  void initialize_params() {
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    NODELET_INFO("create people detector");
    detector.reset(new PeopleDetector(private_nh));
  }

  /**
   * @brief callback
   * @param odom_msg    sensor pose
   * @param points_msg  point cloud
   */
  void callback(const nav_msgs::OdometryConstPtr& odom_msg, const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    if(!globalmap) {
      NODELET_ERROR("globalmap has not been received!!");
      return;
    }

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    if(cloud->empty()) {
      NODELET_ERROR("cloud is empty!!");
      return;
    }

    // downsampling
    pcl::PointCloud<PointT>::Ptr downsampled(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*downsampled);
    downsampled->header = cloud->header;
    cloud = downsampled;

    // transform #cloud into the globalmap space
    const auto& position = odom_msg->pose.pose.position;
    const auto& orientation = odom_msg->pose.pose.orientation;
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 1>(0, 3) = Eigen::Vector3f(position.x, position.y, position.z);
    transform.block<3, 3>(0, 0) = Eigen::Quaternionf(orientation.w, orientation.x, orientation.y, orientation.z).toRotationMatrix();
    pcl::transformPointCloud(*cloud, *cloud, transform);
    cloud->header.frame_id = globalmap->header.frame_id;

    // background subtraction and people detection
    auto filtered = backsub->filter(cloud);
    auto clusters = detector->detect(filtered);

    publish_msgs(points_msg->header.stamp, filtered, clusters);
  }

  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    NODELET_INFO("background subtractor constructed");
    double backsub_resolution = private_nh.param<double>("backsub_resolution", 0.2);
    int backsub_occupancy_thresh = private_nh.param<int>("backsub_occupancy_thresh", 2);

    backsub.reset(new BackgroundSubtractor());
    backsub->setVoxelSize(backsub_resolution, backsub_resolution, backsub_resolution);
    backsub->setOccupancyThresh(backsub_occupancy_thresh);
    backsub->setBackgroundCloud(globalmap);

    backsub_voxel_markers_pub.publish(backsub->create_voxel_marker());
    backsub_voxel_points_pub.publish(backsub->voxels());
  }

private:
  /**
   * @brief publish messages
   * @param stamp
   * @param filtered
   * @param clusters
   */
  void publish_msgs(const ros::Time& stamp, const pcl::PointCloud<pcl::PointXYZI>::Ptr& filtered, const std::vector<Cluster::Ptr>& clusters) const {
    if(clusters_pub.getNumSubscribers()) {
      hdl_people_tracking::ClusterArrayPtr clusters_msg(new hdl_people_tracking::ClusterArray());
      clusters_msg->header.frame_id = globalmap->header.frame_id;
      clusters_msg->header.stamp = stamp;

      clusters_msg->clusters.resize(clusters.size());
      for(int i=0; i<clusters.size(); i++) {
        auto& cluster_msg = clusters_msg->clusters[i];
        cluster_msg.is_human = clusters[i]->is_human;
        cluster_msg.min_pt.x = clusters[i]->min_pt.x();
        cluster_msg.min_pt.y = clusters[i]->min_pt.y();
        cluster_msg.min_pt.z = clusters[i]->min_pt.z();

        cluster_msg.max_pt.x = clusters[i]->max_pt.x();
        cluster_msg.max_pt.y = clusters[i]->max_pt.y();
        cluster_msg.max_pt.z = clusters[i]->max_pt.z();

        cluster_msg.size.x = clusters[i]->size.x();
        cluster_msg.size.y = clusters[i]->size.y();
        cluster_msg.size.z = clusters[i]->size.z();

        cluster_msg.centroid.x = clusters[i]->centroid.x();
        cluster_msg.centroid.y = clusters[i]->centroid.y();
        cluster_msg.centroid.z = clusters[i]->centroid.z();
      }

      clusters_pub.publish(clusters_msg);
    }

    if(backsub_points_pub.getNumSubscribers()) {
      backsub_points_pub.publish(filtered);
    }

    if(cluster_points_pub.getNumSubscribers()) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr accum(new pcl::PointCloud<pcl::PointXYZI>());
      for(const auto& cluster : clusters) {
        std::copy(cluster->cloud->begin(), cluster->cloud->end(), std::back_inserter(accum->points));
      }
      accum->width = accum->size();
      accum->height = 1;
      accum->is_dense = false;

      accum->header.stamp = filtered->header.stamp;
      accum->header.frame_id = globalmap->header.frame_id;

      cluster_points_pub.publish(accum);
    }

    if(human_points_pub.getNumSubscribers()) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr accum(new pcl::PointCloud<pcl::PointXYZI>());
      for(const auto& cluster : clusters) {
        if(cluster->is_human){
          std::copy(cluster->cloud->begin(), cluster->cloud->end(), std::back_inserter(accum->points));
        }
      }
      accum->width = accum->size();
      accum->height = 1;
      accum->is_dense = false;

      accum->header.stamp = filtered->header.stamp;
      accum->header.frame_id = globalmap->header.frame_id;

      human_points_pub.publish(accum);
    }

    if(detection_markers_pub.getNumSubscribers()) {
      detection_markers_pub.publish(create_markers(stamp, clusters));
    }
  }

  visualization_msgs::MarkerArrayConstPtr create_markers(const ros::Time& stamp, const std::vector<Cluster::Ptr>& clusters) const {
    visualization_msgs::MarkerArrayPtr markers(new visualization_msgs::MarkerArray());
    markers->markers.reserve(clusters.size());

    for(int i=0; i<clusters.size(); i++) {
      if(!clusters[i]->is_human) {
        continue;
      }

      visualization_msgs::Marker cluster_marker;
      cluster_marker.header.stamp = stamp;
      cluster_marker.header.frame_id = globalmap->header.frame_id;
      cluster_marker.action = visualization_msgs::Marker::ADD;
      cluster_marker.lifetime = ros::Duration(0.5);
      cluster_marker.ns = (boost::format("cluster%d") % i).str();
      cluster_marker.type = visualization_msgs::Marker::CUBE;

      cluster_marker.pose.position.x = clusters[i]->centroid.x();
      cluster_marker.pose.position.y = clusters[i]->centroid.y();
      cluster_marker.pose.position.z = clusters[i]->centroid.z();
      cluster_marker.pose.orientation.w = 1.0;

      cluster_marker.color.r = 0.0;
      cluster_marker.color.g = 0.0;
      cluster_marker.color.b = 1.0;
      cluster_marker.color.a = 0.4;

      cluster_marker.scale.x = clusters[i]->size.x();
      cluster_marker.scale.y = clusters[i]->size.y();
      cluster_marker.scale.z = clusters[i]->size.z();

      markers->markers.push_back(cluster_marker);
    }

    return markers;
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  // subscribers
  std::unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> points_sub;
  std::unique_ptr<message_filters::TimeSynchronizer<nav_msgs::Odometry, sensor_msgs::PointCloud2>> sync;

  ros::Subscriber globalmap_sub;

  // publishers
  ros::Publisher backsub_points_pub;
  ros::Publisher backsub_voxel_points_pub;

  ros::Publisher cluster_points_pub;
  ros::Publisher human_points_pub;

  ros::Publisher detection_markers_pub;
  ros::Publisher backsub_voxel_markers_pub;

  ros::Publisher clusters_pub;

  // global map
  pcl::PointCloud<PointT>::Ptr globalmap;

  pcl::Filter<PointT>::Ptr downsample_filter;
  std::unique_ptr<BackgroundSubtractor> backsub;
  std::unique_ptr<PeopleDetector> detector;

};

}

PLUGINLIB_EXPORT_CLASS(hdl_people_detection::HdlPeopleDetectionNodelet, nodelet::Nodelet)
