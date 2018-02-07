#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace hdl_people_detection {

/**
 * @brief A class to hold cluster properties
 */
struct Cluster {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Cluster>;

  Cluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
    : is_human(false),
      cloud(cloud)
  {
    min_pt = max_pt = cloud->at(0).getArray3fMap();
    for(int i=1; i<cloud->size(); i++) {
      min_pt = cloud->at(i).getArray3fMap().min(min_pt);
      max_pt = cloud->at(i).getArray3fMap().max(max_pt);
    }

    size = max_pt - min_pt;
    centroid = (min_pt + max_pt) / 2.0f;
  }

  bool is_human;                                //
  Eigen::Array3f min_pt;                        //
  Eigen::Array3f max_pt;                        //
  Eigen::Array3f size;                          //
  Eigen::Array3f centroid;                      //
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;   //
};

}

#endif // CLUSTER_HPP
