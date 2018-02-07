/**
 * DataAssociation.hpp
 * @author koide
 * 15/05/29
 **/
#ifndef KKL_DATA_ASSOCIATION_HPP
#define KKL_DATA_ASSOCIATION_HPP

#include <vector>
#include <boost/optional.hpp>

namespace kkl {
	namespace alg {

/******************************************
 * distance
 * return : if tracker and observation is too far to associate return boost::none (gating), 
          : else return distance between tracker and observation
 * 
 * this function must be specialized for Tracker and Observation
******************************************/
template<typename Tracker, typename Observation>
boost::optional<double> distance(const Tracker& tracker, const Observation&  observation);

/******************************************
 * DataAssociation
 *
******************************************/
template<typename Tracker, typename Observation>
class DataAssociation {
public:
	/***************************
	 * Association
	 * 
  ***************************/
	struct Association {
	public:
		Association(int tracker, int observation, double distance)
			: tracker(tracker),
			observation(observation),
			distance( distance )
		{}

    bool operator< (const Association& rhs) const {
      return distance < rhs.distance;
    }

		int tracker;
		int observation;

		double distance;
	};

	// constructor, destructor
	DataAssociation() {}
	virtual ~DataAssociation() {}

	// associate
	virtual std::vector<Association> associate(const std::vector<Tracker>& trackers, const std::vector<Observation>& observations) = 0;
};

	}
}


#endif
