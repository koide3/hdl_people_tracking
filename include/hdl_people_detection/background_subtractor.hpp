#ifndef BACKGROUND_SUBTRACTOR_HPP
#define BACKGROUND_SUBTRACTOR_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <visualization_msgs/Marker.h>

namespace hdl_people_detection {

/**
 * @brief A class to subtract a point cloud from another cloud
 *        It creates occupancy voxel map the cloud to subtract utilizing Eigen::SparseMatrix,
 *        and then remove each point of the clouds to be subtracted which is in an occupied voxel
 */
class BackgroundSubtractor {
public:
  using PointT = pcl::PointXYZI;

  BackgroundSubtractor() {}

  /**
   * @brief set voxel size
   * @param x
   * @param y
   * @param z
   */
  void setVoxelSize(float x, float y, float z) {
    occupancy_thresh = 1;
    voxel_size = Eigen::Vector3f(x, y, z);
    inverse_voxel_size = Eigen::Vector3f(1.0f / x, 1.0f / y, 1.0f / z);
  }

  /**
   * @brief set occupancy threshold
   * @param t  voxels containing larger number of points than this value, are considered occupied
   */
  void setOccupancyThresh(int t) {
    occupancy_thresh = t;
  }

  /**
   * @brief set background cloud
   * @param src_cloud
   */
  void setBackgroundCloud(pcl::PointCloud<PointT>::ConstPtr bg_cloud) {
    frame_id = bg_cloud->header.frame_id;

    min_pos = bg_cloud->front().getVector3fMap();
    max_pos = bg_cloud->front().getVector3fMap();
    for (auto pt = bg_cloud->begin(); pt != bg_cloud->end(); pt++){
      min_pos = min_pos.array().min(pt->getArray3fMap());
      max_pos = max_pos.array().max(pt->getArray3fMap());
    }

    Eigen::Array3f extents = (max_pos - min_pos).array().abs();
    grid_size = (extents * inverse_voxel_size.array()).unaryExpr([=](float v) { return ceil(v); }).cast<int>();
    occupancy_map.resize(grid_size[0] * grid_size[1] * grid_size[2]);

    addBackgroundCloud(bg_cloud);
  }

  /**
   * @brief add points to occupancy voxel map
   * @param bg_cloud
   */
  void addBackgroundCloud(pcl::PointCloud<pcl::PointXYZI>::ConstPtr bg_cloud) {
    for (auto pt = bg_cloud->begin(); pt != bg_cloud->end(); pt++) {
      int i = getMapIndex(getVoxelIndex(*pt));
      occupancy_map.coeffRef(i) ++;
    }
  }

  /**
   * @brief perform background subtraction
   * @param src_cloud
   * @return src_cloud - bg_cloud
   */
  pcl::PointCloud<pcl::PointXYZI>::Ptr filter(pcl::PointCloud<pcl::PointXYZI>::Ptr src_cloud) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr dst_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    dst_cloud->reserve(src_cloud->size() / 4);

    for (auto pt = src_cloud->begin(); pt != src_cloud->end(); pt++) {
      if (isOccupied(getVoxelIndex(*pt)) < occupancy_thresh){
        dst_cloud->push_back(*pt);
      }
    }

    dst_cloud->header = src_cloud->header;
    dst_cloud->width = dst_cloud->points.size();
    dst_cloud->height = 1;
    dst_cloud->is_dense = false;

    return dst_cloud;
  }

  /**
   * @brief centroids of occupied voxels
   * @return
   */
  pcl::PointCloud<pcl::PointXYZI>::Ptr voxels() const {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    cloud->reserve(occupancy_map.nonZeros() / 2);

    for (Eigen::SparseVector<int>::InnerIterator itr(occupancy_map); itr; ++itr) {
      if (itr.value() < occupancy_thresh) {
        continue;
      }

      int x = itr.index() % grid_size[0];
      int y = (itr.index() / grid_size[0]) % grid_size[1];
      int z = (itr.index() / (grid_size[0] * grid_size[1]));

      pcl::PointXYZI pt;
      pt.getArray3fMap() = Eigen::Array3f(x, y, z) * voxel_size + min_pos + voxel_size * 0.5f;
      pt.intensity = 0;

      cloud->push_back(pt);
    }

    cloud->header.frame_id = frame_id;
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = false;

    return cloud;
  }

  /**
   * @brief create visualization marker of voxels
   * @note  It seems rviz cannot deal with markers with very many primitives, so this method may not work well
   * @return
   */
  visualization_msgs::MarkerConstPtr create_voxel_marker() const {
    visualization_msgs::MarkerPtr marker(new visualization_msgs::Marker());
    marker->header.stamp = ros::Time(0);
    marker->header.frame_id = frame_id;
    marker->action = visualization_msgs::Marker::ADD;
    marker->lifetime = ros::Duration(0);
    marker->ns = "backsub_voxels";
    marker->type = visualization_msgs::Marker::CUBE_LIST;

    marker->pose.orientation.w = 1.0;

    marker->color.b = 1.0;
    marker->color.a = 0.5;

    marker->scale.x = voxel_size.x();
    marker->scale.y = voxel_size.y();
    marker->scale.z = voxel_size.z();

    marker->colors.reserve(8192);
    marker->points.reserve(8192);

    for (Eigen::SparseVector<int>::InnerIterator itr(occupancy_map); itr; ++itr) {
      if (itr.value() < occupancy_thresh) {
        continue;
      }

      int x = itr.index() % grid_size[0];
      int y = (itr.index() / grid_size[0]) % grid_size[1];
      int z = (itr.index() / (grid_size[0] * grid_size[1]));

      Eigen::Array3f pt = Eigen::Array3f(x, y, z) * voxel_size + min_pos + voxel_size * 0.5f;

      geometry_msgs::Point point;
      point.x = pt.x();
      point.y = pt.y();
      point.z = pt.z();
      marker->points.push_back(point);
    }

    return marker;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:

  template<typename PointT>
  Eigen::Vector3i getVoxelIndex(const PointT& pt) {
    return ((pt.getArray3fMap() - min_pos) * inverse_voxel_size).template cast<int>();
  }

  int getMapIndex(const Eigen::Vector3i& index) {
    return index[0] + index[1] * grid_size[0] + index[2] * grid_size[0] * grid_size[1];
  }

  int isOccupied(const Eigen::Vector3i& index) {
    if ((index.array() < 0).any() || (index.array() >= grid_size).any()){
      return false;
    }

    return occupancy_map.coeff(getMapIndex(index));
  }

  std::string frame_id;
  Eigen::Array3f min_pos;
  Eigen::Array3f max_pos;
  Eigen::Array3f voxel_size;
  Eigen::Array3f inverse_voxel_size;

  int occupancy_thresh;
  Eigen::Array3i grid_size;
  Eigen::SparseVector<int> occupancy_map;
};

}

#endif // BACKGROUND_SUBTRACTOR_HPP
