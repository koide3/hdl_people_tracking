/**
 * dpmeans.hpp
 * @author koide
 * 16/06/07
 **/
#ifndef KKL_DPMEANS_HPP
#define KKL_DPMEANS_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace kkl {
  namespace alg {

/**
 * @brief DP-means
 * @ref http://people.eecs.berkeley.edu/~jordan/papers/kulis-jordan-icml12.pdf
 */
template<typename T, int dim>
class DPmeans {
  typedef Eigen::Matrix<T, dim, 1> VectorTd;

public:
  // constructor, destructor
  DPmeans() {}
  ~DPmeans() {}

  /**
   * @brief train
   * @param x           input data
   * @param lambda      cluster penalty prameter
   * @param criteria    convergence criteria
   * @param loop_limit  maximum number of iterations
   * @return
   */
  bool train(const std::vector<VectorTd>& x, T lambda, T criteria = 0.01, int loop_limit = 128) {
    labels.assign(x.size(), 0);
    centroids.clear();
    centroids.push_back(VectorTd::Zero());

    for (int i = 0; i < loop_limit; i++) {
      updateCentroids(x, criteria);
      if (updateLabels(x, lambda) ) {
        return true;
      }
    }

    return false;
  }

private:
  // update centroids acoording to labels
  bool updateCentroids(const std::vector<VectorTd>& x, T criteria) {
    int k = centroids.size();
    std::vector<int> accums(k, 0);
    std::vector<VectorTd> new_centroids(k);
    std::for_each(new_centroids.begin(), new_centroids.end(), [=](VectorTd& centroid) { centroid.setZero(); });

    // update centrods
    for (int i = 0; i < x.size(); i++) {
      accums[labels[i]]++;
      new_centroids[labels[i]] += x[i];
    }
    for (int i = 0; i < k; i++) {
      new_centroids[i] /= std::max( 1, accums[i] );
    }

    centroids.swap(new_centroids);

    // check if converged
    for (int i = 0; i < centroids.size(); i++) {
      if ((centroids[i] - new_centroids[i]).squaredNorm() > criteria * criteria) {
        return false;
      }
    }
    return true;
  }

  // update labels acoording to estimated centroids
  // if the distance between a point and the closest centroid is lager than lambda, the point is added as a new centroid
  bool updateLabels(const std::vector<VectorTd>& x, T lambda) {
    bool is_converged = true;
    bool k_incremented = false;
    // for each point
    for (int i = 0; i < x.size(); i++) {
      T min_d = (centroids[0] - x[i]).squaredNorm();
      int min_label = 0;

      // find the closest centroid
      for (int j = 1; j < centroids.size(); j++) {
        T d = (centroids[j] - x[i]).squaredNorm();
        if (d < min_d) {
          min_d = d;
          min_label = j;
        }
      }

      // check if the distance between the point and the closest centroid is larger than lambda
      if ( !k_incremented && min_d > lambda * lambda) {
        k_incremented = true;
        min_label = centroids.size();
        centroids.push_back(x[i]);

//				i = 0;	// re-calculate all labels since a new centroid is added
      }

      // if a label is changed, estimation is not converged
      if (labels[i] != min_label) {
        is_converged = false;
      }
      labels[i] = min_label;
    }

    return is_converged;
  }

public:
  std::vector<int> labels;			    // labels assigned to input data
  std::vector<VectorTd> centroids;	// estimated centroids
};

  }
}

#endif
