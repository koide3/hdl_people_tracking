#ifndef PEOPLE_TRACKER_HPP
#define PEOPLE_TRACKER_HPP

#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <boost/optional.hpp>
#include <opencv2/opencv.hpp>

#include <kkl/alg/data_association.hpp>
#include <kkl/alg/nearest_neighbor_association.hpp>

#include <hdl_people_tracking/Cluster.h>
#include <hdl_people_tracking/kalman_tracker.hpp>

namespace kkl {
  namespace alg {

/**
 * @brief definition of the distance between tracker and observation for data association
 */
template<>
boost::optional<double> distance(const std::shared_ptr<hdl_people_tracking::KalmanTracker>& tracker, const hdl_people_tracking::Cluster& observation) {
  Eigen::Vector3d pos(observation.centroid.x, observation.centroid.y, observation.centroid.z);
  double sq_mahalanobis = tracker->squaredMahalanobisDistance(pos);

  // gating
  if(sq_mahalanobis > pow(3.0, 2) || (tracker->position() - pos).norm() > 1.5) {
    return boost::none;
  }
  return -kkl::math::gaussianProbMul(tracker->position(), tracker->positionCov(), pos);
}
  }
}

namespace hdl_people_tracking {

/**
 * @brief People tracker
 */
class PeopleTracker {
public:
  PeopleTracker(ros::NodeHandle& private_nh) {
    id_gen = 0;
    human_radius = private_nh.param<double>("human_radius", 0.4);
    remove_trace_thresh = private_nh.param<double>("remove_trace_thresh", 1.0);

    data_association.reset(new kkl::alg::NearestNeighborAssociation<KalmanTracker::Ptr, Cluster>());
//    data_association.reset(new kkl::alg::GlobalNearestNeighborAssociation<KalmanTracker::Ptr, VisualDetection>());
  }

  /**
   * @brief predict people states
   * @param time  current time
   */
  void predict(const ros::Time& time) {
    for(auto& person : people) {
      person->predict(time);
    }
  }

  /**
   * @brief correct people states
   * @param time          current time
   * @param detections    detections
   */
  void correct(const ros::Time& time, const std::vector<Cluster>& detections) {
    // data association
    std::vector<bool> associated(detections.size(), false);
    auto associations = data_association->associate(people, detections);
    for(const auto& assoc : associations) {
      associated[assoc.observation] = true;
      const auto& observation = detections[assoc.observation].centroid;
      Eigen::Vector3d observation_pos(observation.x, observation.y, observation.z);
      people[assoc.tracker]->correct(time, observation_pos, detections[assoc.observation]);
    }

    // generate new tracks
    for(int i=0; i<detections.size(); i++) {
      if(!associated[i]) {
        // check if the detection is far from existing tracks
        const auto& observation = detections[i].centroid;
        Eigen::Vector3d observation_pos(observation.x, observation.y, observation.z);

        bool close_to_tracker = false;
        for(const auto& person : people) {

          if((person->position() - observation_pos).norm() < human_radius * 2.0) {
            close_to_tracker = true;
            break;
          }
        }

        if(close_to_tracker) {
          continue;
        }

        // generate a new track
        KalmanTracker::Ptr tracker(new KalmanTracker(id_gen++, time, observation_pos));
        people.push_back(tracker);
      }
    }

    // remove tracks with large covariance
    auto remove_loc = std::partition(people.begin(), people.end(), [&](const KalmanTracker::Ptr& tracker) {
      return tracker->positionCov().trace() < remove_trace_thresh;
    });
    removed_people.clear();
    std::copy(remove_loc, people.end(), std::back_inserter(removed_people));
    people.erase(remove_loc, people.end());
  }

public:
  long id_gen;                  // track ID which will be assigned to the next new track
  double human_radius;          // new tracks must be far from existing tracks than this value
  double remove_trace_thresh;   // tracks with larger covariance trace than this will be removed

  std::vector<KalmanTracker::Ptr> people;
  std::vector<KalmanTracker::Ptr> removed_people;
  std::unique_ptr<kkl::alg::DataAssociation<KalmanTracker::Ptr, Cluster>> data_association;
};

}

#endif // PEOPLE_TRACKER_HPP
