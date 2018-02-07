/**
 * GlobalNearestNeighborAssociation.hpp
 * @author koide
 * 15/05/29
**/
#ifndef KKL_GLOBAL_NEAREST_NEIGHBOR_ASSOCIATION_HPP
#define KKL_GLOBAL_NEAREST_NEIGHBOR_ASSOCIATION_HPP

#include <memory>
#include <boost/range/algorithm.hpp>

#include <kkl/alg/munkres.hpp>
#include <kkl/alg/data_association.hpp>

namespace kkl {
	namespace alg {

/******************************************
 * GlobalNearestNeighborAssociation
 * 
******************************************/
template<typename Tracker, typename Observation>
class GlobalNearestNeighborAssociation : public DataAssociation<Tracker, Observation> {
    typedef typename DataAssociation<Tracker, Observation>::Association Association;
public:
	// constructor, destructor
	GlobalNearestNeighborAssociation()
		: munkres( new Munkres<double>() ){}
	virtual ~GlobalNearestNeighborAssociation() {}

	// associate
    std::vector<Association> associate(const std::vector<Tracker>& trackers, const std::vector<Observation>& observations) override {
		const double HUGE_VALUE = 1000000.0;

		// create cost matrix between tracker and observation
		Eigen::MatrixXd cost_matrix;
		bool transposed = trackers.size() > observations.size();
		
		if (!transposed) {
			cost_matrix = Eigen::MatrixXd::Constant(trackers.size(), observations.size(), HUGE_VALUE + 1);
			for (int i = 0; i < trackers.size(); i++) {
				for (int j = 0; j < observations.size(); j++) {
					auto dist = distance<Tracker, Observation>(trackers[i], observations[j]);
					if (dist) {
						cost_matrix(i, j) = dist.get();
					}
				}
			}
		}
		else {
			cost_matrix = Eigen::MatrixXd::Constant(observations.size(), trackers.size(), HUGE_VALUE + 1);
			for (int i = 0; i < observations.size(); i++) {
				for (int j = 0; j < trackers.size(); j++) {
					auto dist = distance<Tracker, Observation>(trackers[j], observations[i]);
					if (dist) {
						cost_matrix(i, j) = dist.get();
					}
				}
			}
		}

		// solve combinatorial optimization using Munkres algorithm
		auto solution = munkres->solve(cost_matrix);

		std::vector<Association> associations;
		associations.reserve(trackers.size());

		for (int i = 0; i < solution.size(); i++) {
			int tracker;
			int observation;

			if (!transposed) {
				tracker = i;
				observation = solution[i];
			} else {
				tracker = solution[i];
				observation = i;
			}

			if (cost_matrix(i, solution[i]) < HUGE_VALUE) {
				associations.push_back(Association(tracker, observation, cost_matrix(i, solution[i])));
			}
		}

		return associations;
	}

private:
	std::unique_ptr<Munkres<double>> munkres;
};

	}
}

#endif
