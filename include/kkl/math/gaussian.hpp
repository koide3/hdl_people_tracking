/**
* gaussian.hpp
* @author : koide
* 13/06/12
* 13/06/20 add GMM
* 13/08/03 rename
* 13/11/16 add univariate gauss distibution and univariate gaussian mixture
* 14/10/13 add KL divergence
* 14/10/16 add univariate L2 distance
* 14/11/11 IncrementalGaussianDistribution
* 15/02/13 make IncrementalGaussianDistribution faster using PartialPivLu
* 15/08/20 GaussianEstimater, IndependentGaussianEstimater
**/
#ifndef KKL_GAUSSIAN_HPP
#define KKL_GAUSSIAN_HPP

//#define _USE_BOOST_ERF

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#ifdef _USE_BOOST_ERF
#include <boost/math/special_functions/erf.hpp>
#endif

namespace kkl{
	namespace math{

    /**
     * @brief univariate gaussian PDF
     */
		template<typename T>
		T gaussianProbUni(T mean, T var, T x){
			const T dif = x - mean;
			return 1.0 / sqrt(2.0 * M_PI * var) * exp(-(dif * dif) / (2 * var));
		}

    /**
     * @brief multivariate gaussian PDF
     */
		template<typename T, int p>
		T gaussianProbMul(const Eigen::Matrix<T, p, 1>& mean, const Eigen::Matrix<T, p, p>& cov, const Eigen::Matrix<T, p, 1>& x) {
			const T sqrtDet = sqrt(cov.determinant());
			const Eigen::Matrix<T, p, 1> dif = x - mean;
			const T lhs = 1.0 / (pow(2.0 * M_PI, p / 2.0) * sqrtDet);
			const T rhs = exp(-0.5 * ((dif.transpose() * cov.inverse() * dif))(0, 0));
			return lhs * rhs;
		}

#ifdef _USE_BOOST_ERF
    /**
     * @brief univariate cumulative probability function
     */
		template<typename T>
		T gaussianCumulativeProbUni(T mean, T var, T x) {
			return 0.5 * (1 + boost::math::erf((x - mean) / sqrt(2 * var)));
		};
#endif

    /**
     * @brief compute an error ellipse
     */
    template<typename T, typename Matrix>
    Eigen::Matrix<T, 3, 1> errorEllipse(const Matrix& cov, double kai) {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 2, 2>> solver(cov);

      Eigen::Matrix<T, 3, 1> params;
      params[0] = std::sqrt(kai * kai * solver.eigenvalues()[1]);
      params[1] = std::sqrt(kai * kai * solver.eigenvalues()[0]);
      params[2] = std::atan2(solver.eigenvectors()(0, 2), solver.eigenvectors()(1, 2));

      return params;
    }

    /**
     * @brief squared mahalanobis distance
     */
		template<typename T, int p>
		T squaredMahalanobisDistance(const Eigen::Matrix<T, p, 1>& mean, const Eigen::Matrix<T, p, p>& cov, const Eigen::Matrix<T, p, 1>& x){
			const Eigen::Matrix<T, p, 1> diff = x - mean;
			double distance = diff.transpose() * cov.inverse() * diff;
			return distance;
		}

    /**
     * @brief univariate squared mahalanobis distance
     */
		template<typename T>
		T squaredMahalanobisDistanceUni(T mean, T var, T x) {
			T diff = mean - x;
			return diff * diff / var;
		}

  }
}

#endif
