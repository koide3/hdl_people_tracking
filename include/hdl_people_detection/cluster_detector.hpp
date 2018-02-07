#ifndef CLUSTER_DETECTOR_HPP
#define CLUSTER_DETECTOR_HPP

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/impl/extract_clusters.hpp>

#include <hdl_people_detection/cluster.hpp>

namespace hdl_people_detection {

class ClusterDetector {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ClusterDetector(float min_dist, const Eigen::Array3f& min_size, const Eigen::Array3f& max_size, int min_n_pt = 10, int max_n_pt = 1024, float cluster_tolerance = 0.2f)
    : min_dist(min_dist),
    min_n_pt(min_n_pt),
    max_n_pt(max_n_pt),
    min_size(min_size),
    max_size(max_size),
    cluster_tolerance(cluster_tolerance)
  {
  }

  std::vector<Cluster::Ptr> detect(const pcl::PointCloud<pcl::PointXYZI>::Ptr& src_cloud) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr scaled(new pcl::PointCloud<pcl::PointXYZI>());
    scaled->resize(src_cloud->size());
    Eigen::Array3f scale(1.0f, 1.0f, 0.01f);
    for(int i=0; i<src_cloud->size(); i++) {
      scaled->at(i).getArray3fMap() = src_cloud->at(i).getArray3fMap() * scale;
    }
    scaled->width = scaled->size();
    scaled->height = 1;
    scaled->is_dense = false;

    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>());
    kdtree->setInputCloud(scaled);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::extractEuclideanClusters<pcl::PointXYZI>(*scaled, kdtree, cluster_tolerance, cluster_indices, min_n_pt, max_n_pt);

    std::vector<Cluster::Ptr> clusters;
    clusters.reserve(24);
    for (int i = 0; i < cluster_indices.size(); i++) {
      pcl::PointIndices::Ptr indices(new pcl::PointIndices(cluster_indices[i]));
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::ExtractIndices<pcl::PointXYZI> extract;
      extract.setInputCloud(src_cloud);
      extract.setIndices(indices);
      extract.filter(*cluster_cloud);

      Cluster::Ptr cluster(new Cluster(cluster_cloud));
      float dist = cluster->centroid.matrix().head<2>().norm();
      if ((cluster->size < min_size).any() || (cluster->size > max_size).any() || dist < min_dist) {
        continue;
      }

      clusters.push_back(cluster);
    }

    std::cout << "clusters: " << clusters.size() << std::endl;

    return clusters;
  }

private:
  float min_dist;
  int min_n_pt;
  int max_n_pt;
  float cluster_tolerance;
  Eigen::Array3f min_size;
  Eigen::Array3f max_size;
};

}

#endif // OBJECT_DETECTOR_HPP
