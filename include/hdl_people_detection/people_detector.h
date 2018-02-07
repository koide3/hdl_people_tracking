#ifndef PEOPLE_DETECTOR_H
#define PEOPLE_DETECTOR_H

#include <memory>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <hdl_people_detection/cluster.hpp>
#include <hdl_people_detection/kidono_human_classifier.h>

namespace hdl_people_detection {

/**
 * @brief People detector based on Mercel's cluster detection and Kidono's person classifier
 * @see Marcel Haselich et al., "Confidence-based pedestrian tracking in unstructured environments using 3D laser distance measurements"
 * @see https://userpages.uni-koblenz.de/~agas/Documents/Haeselich2014CBP.pdf
 * @see Kiyosumi Kidono et al., "Pedestrian Recognition Using High-definition LIDAR"
 * @see http://www.aisl.cs.tut.ac.jp/~jun/pdffiles/kidono-iv2011.pdf
 */
class PeopleDetector {
public:
  PeopleDetector();
  ~PeopleDetector();

  std::vector<Cluster::Ptr> detect(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const;

private:
  std::unique_ptr<KidonoHumanClassifier> classifier;
};

}

#endif // PEOPLE_DETECTOR_H
