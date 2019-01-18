#include <hdl_people_detection/kidono_human_classifier.h>

#include <ros/ros.h>
#include <hdl_people_detection/kidono_feature_extractor.hpp>


namespace hdl_people_detection {

KidonoHumanClassifier::KidonoHumanClassifier(const std::string &modelfile, const std::string &scalefile)
  : scale(scalefile, 213)
{
#if CV_VERSION_EPOCH == 2
    boost = cv::Ptr<cv::Boost>(new cv::Boost());
    boost->load(modelfile.c_str());
    if(!boost->get_data()) {
      ROS_ERROR("failed to read boost model!!");
      ROS_ERROR("boost disabled");
      boost.release();
    }
#else
    boost = cv::ml::Boost::load<cv::ml::Boost>(modelfile);
    if(boost->empty()) {
      ROS_ERROR("failed to read boost model!!");
      ROS_ERROR("boost disabled");
      boost.release();
    }
#endif
}

KidonoHumanClassifier::~KidonoHumanClassifier() {}

bool KidonoHumanClassifier::predict(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud) const {
  if(boost.empty()) {
    return true;
  }

  KidonoFeatureExtractor extractor;
  auto feature = extractor.extract(cloud);
  auto scaled = scale.scaling(feature);

  return boost->predict(scaled) > 0.0f;
}

}
