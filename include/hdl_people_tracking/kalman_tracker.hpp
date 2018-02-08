#ifndef KALMAN_TRACKER_HPP
#define KALMAN_TRACKER_HPP

#include <Eigen/Dense>
#include <boost/any.hpp>

#include <ros/ros.h>

#include <kkl/math/gaussian.hpp>
#include <kkl/alg/kalman_filter.hpp>


namespace hdl_people_tracking {

/**
 * @brief Kalman filter-based tracker with a constant velocity model
 */
class KalmanTracker {
  typedef kkl::alg::KalmanFilter<double, 6, 2, 3> KalmanFilter;
public:
  /**
   * @brief constructor
   * @param id            tracker ID
   * @param time          timestamp
   * @param init_pos      initial position
   * @param associated    associated detection
   */
  KalmanTracker(long id, const ros::Time& time, const Eigen::Vector3d& init_pos, boost::any associated = boost::any())
    : id_(id),
      correction_count(0),
      init_time(time),
      last_prediction_time(time),
      last_correction_time(time),
      last_associated(associated)
  {
    Eigen::Matrix<double, 6, 6> transition = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 6, 2> control = Eigen::Matrix<double, 6, 2>::Zero();
    Eigen::Matrix<double, 3, 6> measurement = Eigen::Matrix<double, 3, 6>::Zero();
    measurement.block<3, 3>(0, 0).setIdentity() * 0.2;

    Eigen::Matrix<double, 6, 6> process_noise = Eigen::Matrix<double, 6, 6>::Zero();
    process_noise.topLeftCorner(3, 3) = Eigen::Matrix3d::Identity() * 0.03;
    process_noise.bottomRightCorner(3, 3) = Eigen::Matrix3d::Identity() * 0.01;
    Eigen::Matrix3d measurement_noise = Eigen::Matrix3d::Identity() * 0.2;

    Eigen::Matrix<double, 6, 1> mean = Eigen::Matrix<double, 6, 1>::Zero();
    mean.head<3>() = init_pos;
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity() * 0.1;

    kalman_filter.reset(new KalmanFilter(transition, control, measurement, process_noise, measurement_noise, mean, cov));
  }
  ~KalmanTracker() {}

  using Ptr = std::shared_ptr<KalmanTracker>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
  /**
   * @brief predict the current state
   * @param time    current time
   */
  void predict(const ros::Time& time) {
    double difftime = (time - last_prediction_time).toSec();
    difftime = std::max(0.001, difftime);

    kalman_filter->transitionMatrix(0, 3) = difftime;
    kalman_filter->transitionMatrix(1, 4) = difftime;
    kalman_filter->transitionMatrix(2, 5) = difftime;

    kalman_filter->predict(Eigen::Matrix<double, 2, 1>::Zero());
    last_prediction_time = time;

    last_associated = boost::any();
  }

  /**
   * @brief correct the state with an observation
   * @param time    current time
   * @param pos     observed position
   * @param associated   associated detection
   */
  void correct(const ros::Time& time, const Eigen::Vector3d& pos, boost::any associated = boost::any()) {
    kalman_filter->correct(pos);

    correction_count++;
    last_correction_time = time;
    last_associated = associated;
  }

public:
  long id() const {
    return id_;
  }

  ros::Duration age(const ros::Time& time) const {
    return (time - init_time);
  }

  const ros::Time& lastCorrectionTime() const {
    return last_correction_time;
  }

  const boost::any& lastAssociated() const {
    return last_associated;
  }

  Eigen::Vector3d position() const {
    return kalman_filter->mean.head<3>();
  }

  Eigen::Vector3d velocity() const {
    return kalman_filter->mean.tail<3>();
  }

  Eigen::Matrix3d positionCov() const {
    return kalman_filter->cov.block<3, 3>(0, 0);
  }

  Eigen::Matrix3d velocityCov() const {
    return kalman_filter->cov.block<3, 3>(3, 3);
  }

  double squaredMahalanobisDistance(const Eigen::Vector3d& p) const {
    return kkl::math::squaredMahalanobisDistance<double, 3>(
          kalman_filter->mean.head<3>(),
          kalman_filter->cov.block<3, 3>(0, 0),
          p);
  }

  int correctionCount() const {
    return correction_count;
  }

private:
  long id_;

  int correction_count;
  ros::Time init_time;              // time when the tracker was initialized
  ros::Time last_prediction_time;   // tiem when prediction was performed
  ros::Time last_correction_time;   // time when correction was performed

  boost::any last_associated;       // associated detection data

  std::unique_ptr<KalmanFilter> kalman_filter;
};

}

#endif // KALMANTRACKER_HPP
