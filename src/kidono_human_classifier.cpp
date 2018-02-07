#include <hdl_people_detection/kidono_human_classifier.h>

#include <hdl_people_detection/kidono_feature_extractor.hpp>


namespace hdl_people_detection {

KidonoHumanClassifier::KidonoHumanClassifier(const std::string &modelfile, const std::string &scalefile)
  : scale(scalefile, 213)
{
  boost.load(modelfile.c_str());
  if(!boost.get_data()) {
    std::cerr << "warning : failed to load boosting model!!" << std::endl;
    std::cerr << "        : boosting is disabled!!" << std::endl;
  }
}

KidonoHumanClassifier::~KidonoHumanClassifier() {}

bool KidonoHumanClassifier::predict(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud) const {
  if(!boost.get_data()) {
    return true;
  }

  KidonoFeatureExtractor extractor;
  auto feature = extractor.extract(cloud);
  auto scaled = scale.scaling(feature);

  return boost.predict(scaled) > 0.0f;
}

}
