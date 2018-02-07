#ifndef MARCEL_PEOPLE_DETECTOR_HPP
#define MARCEL_PEOPLE_DETECTOR_HPP

#include <chrono>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

#include <kkl/alg/dp_means.hpp>
#include <hdl_people_detection/cluster.hpp>

namespace hdl_people_detection {

/**
 * @brief Marcel's cluster detector
 *        It first applies euclidean clustering, and then splits and re-merges the clusters
 *        to detect people who are very close to other objects
 * @see Marcel Haselich et al., "Confidence-based pedestrian tracking in unstructured environments using 3D laser distance measurements"
 * @see https://userpages.uni-koblenz.de/~agas/Documents/Haeselich2014CBP.pdf
 */
class MarcelPeopleDetector {
public:
  /**
   * @brief constructor
   * @param min_pt     minimum number of points in a cluster
   * @param max_pt     maximum number of points in a cluster
   * @param min_size   minimum size of a cluster
   * @param max_size   maximum size of a cluster
   */
  MarcelPeopleDetector(int min_pt, int max_pt, const Eigen::Array3f& min_size, const Eigen::Array3f& max_size)
    : min_pt(min_pt),
      max_pt(max_pt),
      min_size(min_size),
      max_size(max_size)
  {
  }

  /**
   * @brief detect clusters
   * @param cloud  background subtracted cloud
   * @return detected clusters
   */
  std::vector<Cluster::Ptr> detect(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) {
    const float lambda = 0.45;

    // project #cloud into XY-space
    pcl::PointCloud<pcl::PointXYZI>::Ptr scaled(new pcl::PointCloud<pcl::PointXYZI>());
    scaled->resize(cloud->size());
    Eigen::Array3f scale(1.0f, 1.0f, 0.01f);
    for(int i=0; i<cloud->size(); i++) {
      scaled->at(i).getArray3fMap() = cloud->at(i).getArray3fMap() * scale;
    }
    scaled->width = scaled->size();
    scaled->height = 1;
    scaled->is_dense = false;

    // euclidean clustering
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> object_clouds;
    auto candidate_clusters = clustering(scaled);

    // split and re-merge clusters to make each cluster contain only one person's points
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> detected;
    detected.reserve(candidate_clusters.size());
    for (int i = 0; i < candidate_clusters.size(); i++) {
      pcl::PointIndices::Ptr indices(new pcl::PointIndices(candidate_clusters[i]));
      pcl::ExtractIndices<pcl::PointXYZI> extract;
      extract.setInputCloud(cloud);
      extract.setIndices(indices);

      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      extract.filter(*cluster_cloud);

      if (cluster_cloud->empty()) {
        continue;
      }

      auto splitted = split(cluster_cloud, lambda);
      auto merged = merge(splitted);

      std::copy_if(merged.begin(), merged.end(), std::back_inserter(object_clouds), [&](const pcl::PointCloud<pcl::PointXYZI>::Ptr& c) { return isValid(c); });
    }

    std::vector<Cluster::Ptr> clusters(object_clouds.size());
    std::transform(object_clouds.begin(), object_clouds.end(), clusters.begin(), [=](const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) { return std::make_shared<Cluster>(cloud); });

    return clusters;
  }

private:
  bool isValid(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) const {
    std::pair<Eigen::Array3f, Eigen::Array3f> minmax(cloud->front().getArray3fMap(), cloud->front().getArray3fMap());
    for(int i=1; i<cloud->size(); i++) {
      minmax.first = minmax.first.min(cloud->at(i).getArray3fMap());
      minmax.second = minmax.second.max(cloud->at(i).getArray3fMap());
    }
    Eigen::Array3f size = minmax.second - minmax.first;
    size.x() = size.y() = size.head<2>().matrix().norm();

    return cloud->size() > min_pt && cloud->size() < max_pt && (size > min_size).all() && (size < max_size).all();
  }

  // apply euclidean clustering
  std::vector<pcl::PointIndices> clustering(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const {
    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>());
    kdtree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> extractor;
    extractor.setClusterTolerance(0.2);
    extractor.setMinClusterSize(16);
    extractor.setMaxClusterSize(8192 * 16);
    extractor.setInputCloud(cloud);
    extractor.setSearchMethod(kdtree);

    std::vector<pcl::PointIndices> cluster_indices;
    extractor.extract(cluster_indices);

    return cluster_indices;
  }

  // split clusters using DPmeans
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> split(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, float lambda) const {
    kkl::alg::DPmeans<float, 2> dp_means;

    std::vector<Eigen::Vector2f> points(cloud->size());
    std::transform(cloud->begin(), cloud->end(), points.begin(), [=](const pcl::PointXYZI& pt) { return pt.getVector3fMap().topLeftCorner(2, 1); });
    if (!dp_means.train(points, lambda, 0.01f, 128)) {
      // std::cerr << "warning : dp_means not converged!!" << std::endl;
    }

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> splitted(dp_means.centroids.size());
    for (int i = 0; i < splitted.size(); i++) {
      splitted[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
    }

    for (int i = 0; i < cloud->size(); i++) {
      pcl::PointXYZI pt = cloud->at(i);
      splitted[dp_means.labels[i]]->push_back(pt);
    }

    splitted.erase(std::remove_if(splitted.begin(), splitted.end(), [=](const pcl::PointCloud<pcl::PointXYZI>::Ptr& c) { return c->empty(); }), splitted.end());

    return splitted;
  }

  // re-merge clusters by gap detection
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> merge(std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clouds) const {
    std::vector<Eigen::Vector2f> means(clouds.size());
    std::transform(clouds.begin(), clouds.end(), means.begin(),
      [=](const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
        Eigen::Vector4f mean = Eigen::Vector4f::Zero();
        for(const auto& pt : cloud->points) {
          mean += pt.getVector4fMap();
        }
        return mean.head<2>();
      }
    );

    std::vector<int> merge_to(clouds.size(), -1);

    float dist_thresh = 0.8f;
    for (int i = 0; i < clouds.size(); i++) {
      for (int j = i + 1; j < clouds.size(); j++) {
        if (!clouds[i] || !clouds[j]) {
          continue;
        }

        if ((means[i] - means[j]).squaredNorm() > dist_thresh * dist_thresh) {
          continue;
        }

        if (!detectGap(means[i], means[j], clouds[i], clouds[j])) {
          if (merge_to[j] < 0 || (means[merge_to[j]] - means[j]).squaredNorm() > (means[i] - means[j]).squaredNorm()) {
            merge_to[j] = i;
          }
        }
      }
    }

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> merged(clouds.size());
    for (int i = 0; i < merge_to.size(); i++) {
      if (merge_to[i] < 0) {
        merged[i] = clouds[i];
        continue;
      }

      int to = merge_to[i];
      while (merge_to[to] != -1) {
        to = merge_to[to];
      }

      std::copy(clouds[i]->begin(), clouds[i]->end(), std::back_inserter(merged[to]->points));
      merged[to]->width = merged[to]->size();
      merged[to]->height = 1;
      merged[to]->is_dense = false;
    }

    merged.erase(std::remove_if(merged.begin(), merged.end(), [=](const pcl::PointCloud<pcl::PointXYZI>::Ptr& c) { return c == nullptr; }), merged.end());

    return merged;
  }

  // gap detection
  bool detectGap(const Eigen::Vector2f& mean1, const Eigen::Vector2f& mean2, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud1, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud2) const {
    Eigen::Vector2f major = (mean2 - mean1).normalized();

    std::vector<float> on_major(cloud1->size() + cloud2->size());
    std::transform(cloud1->begin(), cloud1->end(), on_major.begin(), [=](const pcl::PointXYZI& pt) { return major.dot(pt.getVector3fMap().topLeftCorner(2, 1)); });
    std::transform(cloud2->begin(), cloud2->end(), on_major.begin() + cloud1->size(), [=](const pcl::PointXYZI& pt) { return major.dot(pt.getVector3fMap().topLeftCorner(2, 1)); });

    float min_val = major.dot(mean1);
    float max_val = major.dot(mean2);
//		float min_val = *std::min_element(on_major.begin(), on_major.end());
//		float max_val = *std::max_element(on_major.begin(), on_major.end());

    float gap_thresh = 0.8f;
    int hist_size = 8;
    std::vector<int> hist(hist_size, 0);
    for (int i = 0; i < on_major.size(); i++) {
      int n = (on_major[i] - min_val) / (max_val - min_val) * hist_size;
      if (n >= 0 && n < hist.size()) {
        hist[n] ++;
      }
    }

    float average = static_cast<float>(std::accumulate(hist.begin(), hist.end(), 0)) / hist.size();

    return hist[3] + hist[4] < average * gap_thresh;
  }

private:
  int min_pt;
  int max_pt;
  Eigen::Array3f min_size;
  Eigen::Array3f max_size;
};

}

#endif
