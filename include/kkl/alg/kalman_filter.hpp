/**
 * KalmanFilter.hpp
 * @author : koide
 * 13/08/04
 **/
#ifndef KKL_KALMAN_FILTER_HPP
#define KKL_KALMAN_FILTER_HPP

#include <Eigen/Dense>

namespace kkl{
  namespace alg{

/*****************************************************
 * Kalman filter
 * T : type of scalar
 * stateDim : dimension of the state
 * inputDim : dimension of the input
 * measurementDim : dimension of the observation
*****************************************************/
template<typename T, int stateDim, int inputDim, int measurementDim>
class KalmanFilter{
  const static int N = stateDim;
  const static int M = inputDim;
  const static int K = measurementDim;

  typedef Eigen::Matrix<T, N, 1> VectorN;
  typedef Eigen::Matrix<T, M, 1> VectorM;
  typedef Eigen::Matrix<T, K, 1> VectorK;
  typedef Eigen::Matrix<T, N, N> MatrixNN;
  typedef Eigen::Matrix<T, N, M> MatrixNM;
  typedef Eigen::Matrix<T, K, K> MatrixKK;
  typedef Eigen::Matrix<T, K, N> MatrixKN;
  typedef Eigen::Matrix<T, N, K> MatrixNK;
public:

  /*****************************************************************
   * constructor
   * transition		 : state transition matrix
   * control			 : input response matrix
   * measurement		 : observation matrix
   * processNoise		 : process noise covariance matrix
   * measurementNoise  : measurement noise covariance matrix
   * mean				 : initial mean
   * cov				 : initial covariance matrix
  *****************************************************************/
  KalmanFilter( const MatrixNN& transition, const MatrixNM& control, const MatrixKN& measurement, const MatrixNN& processNoise, const MatrixKK& measurementNoise, const VectorN& mean, const MatrixNN& cov )
    : mean( mean ),
    cov( cov ),
    transitionMatrix( transition ),
    controlMatrix( control ),
    measurementMatrix( measurement ),
    processNoiseCov( processNoise ),
    measurementNoiseCov( measurementNoise ){}

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /*****************************************************************
   * predict
   * control : input
  *****************************************************************/
  void predict( const VectorM& control ){
    const auto& A = transitionMatrix;
    const auto& B = controlMatrix;
    const auto& R = processNoiseCov;

    const auto& u = control;

    mean = A * mean + B * u;
    cov = A * cov * A.transpose() + R;
  }

  /*****************************************************************
   * correct
   * measurement : measurement vector
  *****************************************************************/
  void correct( const VectorK& measurement ){
    const auto& C = measurementMatrix;
    const auto& Q = measurementNoiseCov;

    kalmanGain = cov * C.transpose() * ( C * cov * C.transpose() + Q ).inverse();
    const auto& K = kalmanGain;

    mean = mean + K * ( measurement - C * mean );
    cov = ( MatrixNN::Identity() - K * C ) * cov;
  }

  /*			getter			*/
  const VectorN& getMean() const { return mean; }
  const MatrixNN& getCov() const { return cov; }

  const MatrixNN& getTransitionMatrix() const { return transitionMatrix; }
  const MatrixNM& getControlMatrix() const { return controlMatrix; }
  const MatrixKN& getMeasurementMatrix() const { return measurementMatrix; }

  const MatrixNN& getProcessNoiseCov() const { return processNoiseCov; }
  const MatrixKK& getMeasurementNoiseCov() const { return measurementNoiseCov; }

  const MatrixNK& getKalmanGain() const { return kalmanGain; }

  /*			setter			*/
  KalmanFilter& setMean( const VectorN& m ){ mean = m;			return *this; }
  KalmanFilter& setCov( const MatrixNN& s ){ cov = s;			return *this; }

  KalmanFilter& setTransitionMatrix( const MatrixNN& t ){ transitionMatrix = t;	return *this; }
  KalmanFilter& setControlMatrix( const MatrixNM& c ){ controlMatrix = c;			return *this; }
  KalmanFilter& setMeasurementMatrix( const MatrixKN& m ){ measurementMatrix = m;	return *this; }

  KalmanFilter& setProcessNoiseCov( const MatrixNN& p ){ processNoiseCov = p;			return *this; }
  KalmanFilter& setMeasurementNoiseCov( const MatrixKK& m ){ measurementNoiseCov = m;	return *this; }
public:
  VectorN mean;			// mean of the state
  MatrixNN cov;			// covariance of the state

  MatrixNN transitionMatrix;		//
  MatrixNM controlMatrix;			//
  MatrixKN measurementMatrix;		//
  MatrixNN processNoiseCov;		//
  MatrixKK measurementNoiseCov;	//

private:
  MatrixNK kalmanGain;
};
  }
}

#endif
