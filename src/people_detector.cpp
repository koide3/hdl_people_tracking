#include <hdl_people_detection/people_detector.h>

#include <ros/ros.h>
#include <ros/package.h>
#include <hdl_people_detection/cluster_detector.hpp>
#include <hdl_people_detection/marcel_people_detector.hpp>

namespace hdl_people_detection {

PeopleDetector::PeopleDetector() {
  std::string package_path = ros::package::getPath("hdl_people_tracking");
  classifier.reset(new KidonoHumanClassifier(package_path + "/data/boost_kidono.model", package_path + "/data/boost_kidono.scale"));
}

PeopleDetector::~PeopleDetector() {

}

std::vector<Cluster::Ptr> PeopleDetector::detect(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud) const {
  MarcelPeopleDetector marcel(10, 8192, Eigen::Array3f(0.2f, 0.2f, 0.3f), Eigen::Array3f(1.0f, 1.0f, 2.0f));
  auto clusters = marcel.detect(cloud);

  for(auto& cluster : clusters) {
    cluster->is_human = classifier->predict(cluster->cloud);
  }

  return clusters;
}

}
