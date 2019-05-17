#include <hdl_people_detection/people_detector.h>

#include <ros/ros.h>
#include <ros/package.h>
#include <hdl_people_detection/cluster_detector.hpp>
#include <hdl_people_detection/marcel_people_detector.hpp>

namespace hdl_people_detection {

PeopleDetector::PeopleDetector(ros::NodeHandle &nh) {
  min_pts = nh.param<int>("cluster_min_pts", 10);
  max_pts = nh.param<int>("cluster_max_pts", 8192);
  min_size.x() = nh.param<double>("cluster_min_size_x", 0.2);
  min_size.y() = nh.param<double>("cluster_min_size_y", 0.2);
  min_size.z() = nh.param<double>("cluster_min_size_z", 0.3);
  max_size.x() = nh.param<double>("cluster_max_size_x", 1.0);
  max_size.y() = nh.param<double>("cluster_max_size_y", 1.0);
  max_size.z() = nh.param<double>("cluster_max_size_z", 2.0);

  if(nh.param<bool>("enable_classification", true)) {
    std::string package_path = ros::package::getPath("hdl_people_tracking");
    classifier.reset(new KidonoHumanClassifier(package_path + "/data/boost_kidono.model", package_path + "/data/boost_kidono.scale"));
  }
}

PeopleDetector::~PeopleDetector() {

}

std::vector<Cluster::Ptr> PeopleDetector::detect(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud) const {
  MarcelPeopleDetector marcel(min_pts, max_pts, min_size, max_size);
  auto clusters = marcel.detect(cloud);

  for(auto& cluster : clusters) {
    //std::cout<<!classifier<<"\n";
    cluster->is_human = !classifier || classifier->predict(cluster->cloud);
  }

  return clusters;
}

}
