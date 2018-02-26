#ifndef KIDONO_HUMAN_CLASSIFIER_H
#define KIDONO_HUMAN_CLASSIFIER_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>
#include <kkl/ml/svm_scale.hpp>

namespace hdl_people_detection {

/**
 * @brief Kidono's features-based person classifier
 *        It classsifies clouds into human and other objects
 */
class KidonoHumanClassifier {
public:
  KidonoHumanClassifier(const std::string& modelfile, const std::string& scalefile);
  ~KidonoHumanClassifier();

  /**
   * @brief predict if a cloud is human
   * @param cloud  input cloud
   * @return if the cloud is human
   */
  bool predict(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const;

private:
#if CV_VERSION_EPOCH == 2
  using CvBoost = cv::Boost;
#else
  using CvBoost = cv::ml::Boost;
#endif

  cv::Ptr<CvBoost> boost;
  kkl::ml::SvmScale scale;
};

}

#endif // KIDONO_HUMAN_CLASSIFIER_H
