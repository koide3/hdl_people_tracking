#ifndef KIDONO_FEATURE_EXTRACTOR_HPP
#define KIDONO_FEATURE_EXTRACTOR_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <boost/range/algorithm.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/impl/plane_clipper3D.hpp>

namespace hdl_people_detection {

/**
 * @brief A class to extract Kidono's features
 * @see Kiyosumi Kidono et al., "Pedestrian Recognition Using High-definition LIDAR"
 * @see http://www.aisl.cs.tut.ac.jp/~jun/pdffiles/kidono-iv2011.pdf
 */
class KidonoFeatureExtractor {
private:
  double distance_scale;
public:
  /**
   * @brief constructor
   * @param distance_scale
   */
  KidonoFeatureExtractor(double distance_scale = 1.0)
    : distance_scale(distance_scale)
  {}

  /**
   * @brief extract features
   * @param cloud  src cloud
   * @return extracted feature vector
   */
  std::vector<float> extract(pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud) const {
    std::vector<float> feature;		feature.reserve(256);
    feature.push_back(cloud->points.size());
    feature.push_back(minimumDistance(cloud) * distance_scale);

    Eigen::Vector3f mean;
    Eigen::Matrix3f covariance;

    boost::copy(covariance3d(cloud, mean, covariance), std::back_inserter(feature));

    auto centered_cloud = centeredCloud(cloud, mean);

    boost::copy(inertiaMoment3d(cloud), std::back_inserter(feature));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);

    Eigen::Vector3f e1 = solver.eigenvectors().col(2);
    Eigen::Vector3f e2 = solver.eigenvectors().col(1);
    Eigen::Vector3f e3 = solver.eigenvectors().col(0);

    if (e1.z() < 0.0f) {
      e1 = -e1;
    }

    boost::copy(covarianceIn3zones(centered_cloud, e1, e2, e3), std::back_inserter(feature));
    boost::copy(histogram2d(centered_cloud, e1, e2, 14, 7), std::back_inserter(feature));
    boost::copy(histogram2d(centered_cloud, e1, e3, 9, 5), std::back_inserter(feature));
    boost::copy(sliceFeature(centered_cloud, e1, e2, e3, 10), std::back_inserter(feature));
    boost::copy(intensityDistribution(centered_cloud, 25), std::back_inserter(feature));

    return feature;
  }

private:
  template<typename T>
  T square(T v) const { return v * v; }

  pcl::PointCloud<pcl::PointXYZI>::Ptr centeredCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, const Eigen::Vector3f& mean) const {
    pcl::PointCloud<pcl::PointXYZI>::Ptr centered_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    centered_cloud->resize(cloud->size());
    for (int i = 0; i < cloud->size(); i++) {
      centered_cloud->at(i).getVector3fMap() = cloud->at(i).getVector3fMap() - mean;
      centered_cloud->at(i).intensity = cloud->at(i).intensity;
    }
    centered_cloud->width = centered_cloud->size();
    centered_cloud->height = 1;
    centered_cloud->is_dense = false;

    return centered_cloud;
  }

  float minimumDistance(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const {
    float dist = cloud->front().getVector3fMap().squaredNorm();
    for (int i = 1; i < cloud->size(); i++) {
      dist = std::min(dist, cloud->at(i).getVector3fMap().squaredNorm());
    }
    return sqrtf(dist);
  }

  std::vector<float> covariance3d(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, Eigen::Vector3f& mean, Eigen::Matrix3f& covariance) const {
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    pcl::computeCovarianceMatrix(*cloud, centroid, covariance);

    mean = centroid.topLeftCorner(3, 1);

    std::vector<float> feature(6);
    feature[0] = covariance(0, 0);		feature[1] = covariance(0, 1);		feature[2] = covariance(0, 2);
    feature[3] = covariance(1, 1);		feature[4] = covariance(1, 2);
    feature[5] = covariance(2, 2);
    return feature;
  }

  std::vector<float> inertiaMoment3d(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& centered_cloud) const {
    std::vector<float> feature(6, 0.0f);

    for (int i = 0; i < centered_cloud->size(); i++) {
      const auto& pt = centered_cloud->at(i);
      feature[0] += square(pt.y) + square(pt.z);
      feature[1] += -pt.x * pt.y;
      feature[2] += -pt.x * pt.z;
      feature[3] += square(pt.x) + square(pt.z);
      feature[4] += -pt.y * pt.z;
      feature[5] += square(pt.x) + square(pt.y);
    }

    return feature;
  }

  std::vector<float> covariance2d(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, const Eigen::Vector3f& e1, const Eigen::Vector3f& e2) const {
    if (cloud->empty()){
      return std::vector<float>(3, 0.0f);
    }

    Eigen::MatrixXf ptdata(cloud->size(), 2);
    for (int i = 0; i < cloud->size(); i++) {
      ptdata(i, 0) = cloud->at(i).getVector3fMap().dot(e1);
      ptdata(i, 1) = cloud->at(i).getVector3fMap().dot(e2);
    }

    Eigen::Vector2f mean = ptdata.colwise().mean().transpose();
    Eigen::MatrixXf centered = ptdata.rowwise() - mean.transpose();
    Eigen::Matrix2f covariance = (centered.transpose() * centered) / cloud->size();

    std::vector<float> feature(3);
    feature[0] = covariance(0, 0);	feature[1] = covariance(0, 1);
    feature[2] = covariance(1, 1);

    return feature;
  }

  std::vector<float> covarianceIn3zones(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& centered_cloud, const Eigen::Vector3f& e1, const Eigen::Vector3f& e2, const Eigen::Vector3f& e3) const {
    pcl::PointCloud<pcl::PointXYZI>::Ptr upper_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr lower_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr right_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr left_cloud(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::PointIndices::Ptr indices(new pcl::PointIndices());

    pcl::PlaneClipper3D<pcl::PointXYZI> clipper(Eigen::Vector4f(e1.x(), e1.y(), e1.z(), 0.0f));
    clipper.clipPointCloud3D(*centered_cloud, indices->indices);

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(centered_cloud);
    extract.setIndices(indices);
    extract.filter(*upper_cloud);

    extract.setNegative(true);
    extract.filter(*lower_cloud);

    indices->indices.clear();
    clipper.setPlaneParameters(Eigen::Vector4f(e2.x(), e2.y(), e2.z(), 0.0f));
    clipper.clipPointCloud3D(*lower_cloud, indices->indices);

    extract.setInputCloud(lower_cloud);
    extract.setIndices(indices);
    extract.setNegative(false);
    extract.filter(*left_cloud);

    extract.setNegative(true);
    extract.filter(*right_cloud);

    std::vector<float> feature;		feature.reserve(9);
    boost::copy(covariance2d(upper_cloud, e1, e2), std::back_inserter(feature));
    boost::copy(covariance2d(left_cloud, e1, e2), std::back_inserter(feature));
    boost::copy(covariance2d(right_cloud, e1, e2), std::back_inserter(feature));

    return feature;
  }

  std::vector<float> histogram2d(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud, const Eigen::Vector3f& axis1, const Eigen::Vector3f& axis2, int bin1, int bin2) const {
    std::vector<Eigen::Vector2f> pts2d(cloud->size());
    for (int i = 0; i < cloud->size(); i++) {
      pts2d[i][0] = cloud->at(i).getVector3fMap().dot(axis1);
      pts2d[i][1] = cloud->at(i).getVector3fMap().dot(axis2);
    }

    Eigen::Array2f min_pt = pts2d.front();
    Eigen::Array2f max_pt = pts2d.front();
    for (int i = 1; i < pts2d.size(); i++) {
      min_pt = pts2d[i].array().min(min_pt);
      max_pt = pts2d[i].array().max(max_pt);
    }

    Eigen::Array2f size = max_pt - min_pt;
    Eigen::Array2f inv_size = Eigen::Array2f(bin1 - 0.1f, bin2 - 0.1f) / size;

    float weight = 1.0f / pts2d.size();
    std::vector<float> hist(bin1 * bin2, 0.0f);
    for (int i = 0; i < pts2d.size(); i++) {
      Eigen::Array2i index = ((pts2d[i].array() - min_pt) * inv_size).cast<int>();
      hist[index[0] + index[1] * bin1] += weight;
    }

    return hist;
  }

  std::vector<float> sliceFeature(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& centered_cloud, const Eigen::Vector3f& e1, const Eigen::Vector3f& e2, const Eigen::Vector3f& e3, int slice_n = 10) const {
    Eigen::Matrix3f cvt2eigenspace;
    cvt2eigenspace << e1.transpose(), e2.transpose(), e3.transpose();

    float e1_min = 9999.0f;
    float e1_max = -9999.0f;

    std::vector<Eigen::Vector3f> aligned_cloud(centered_cloud->size());
    for (int i = 0; i < centered_cloud->size(); i++) {
      aligned_cloud[i] = cvt2eigenspace * centered_cloud->at(i).getVector3fMap();
      e1_min = std::min(e1_min, aligned_cloud[i][0]);
      e1_max = std::max(e1_max, aligned_cloud[i][0]);
    }

    float height = e1_max - e1_min;
    float scale = (slice_n - 0.1f) / height;

    std::vector<Eigen::Array2f> min_pts(slice_n);
    std::vector<Eigen::Array2f> max_pts(slice_n);
    for (int i = 0; i < slice_n; i++) {
      min_pts[i] = Eigen::Array2f::Ones() * 9999.0f;
      max_pts[i] = Eigen::Array2f::Ones() * -9999.0f;
    }

    for (int i = 0; i < aligned_cloud.size(); i++) {
      int n = static_cast<int>((aligned_cloud[i][0] - e1_min) * scale);
      min_pts[n] = min_pts[n].min(aligned_cloud[i].bottomLeftCorner(2, 1).array());
      max_pts[n] = max_pts[n].max(aligned_cloud[i].bottomLeftCorner(2, 1).array());
    }

    std::vector<float> feature;
    feature.reserve(slice_n * 2);

    for (int i = 0; i < slice_n; i++){
      Eigen::Array2f w = max_pts[i] - min_pts[i];
      if (w[0] > 0.0f && w[1] > 0.0f) {
        feature.push_back(w[0]);
        feature.push_back(w[1]);
      }
      else {
        feature.push_back(0.0f);
        feature.push_back(0.0f);
      }
    }

    return feature;
  }

  std::vector<float> intensityDistribution(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& centered_cloud, int hist_n = 25) const {
    Eigen::ArrayXf intensity(centered_cloud->size());
    for (int i = 0; i < centered_cloud->size(); i++) {
      intensity[i] = centered_cloud->at(i).intensity;
    }

    float weight = 1.0f / intensity.size();
    std::vector<float> feature(hist_n + 2, 0.0f);

    float scale = (hist_n - 0.1f) / 255.0f;
    for (int i = 0; i < intensity.size(); i++) {
      int n = static_cast<int>(intensity[i] * scale);
      feature[n] += weight;
    }

    float mean = intensity.mean();
    float stddev = sqrt((intensity - mean).square().mean());

    feature[hist_n] = mean;
    feature[hist_n + 1] = stddev;

    return feature;
  }
};

}


#endif
