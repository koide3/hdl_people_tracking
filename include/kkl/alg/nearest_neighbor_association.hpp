/**
 * NearestNeighborAssociation.hpp
 * @author koide
 * 15/05/29
**/
#ifndef KKL_NEAREST_NEIGHBOR_ASSOCIATION_HPP
#define KKL_NEAREST_NEIGHBOR_ASSOCIATION_HPP

#include <algorithm>
#include <kkl/alg/data_association.hpp>

namespace kkl {
	namespace alg {

/******************************************
 * NearestNeighborAssociation
 *
******************************************/
template<typename Tracker, typename Observation>
class NearestNeighborAssociation : public DataAssociation<Tracker, Observation> {
  typedef typename DataAssociation<Tracker, Observation>::Association Association;
public:
	// constructor, destructor
	NearestNeighborAssociation() {}
	virtual ~NearestNeighborAssociation() {}

  // associate
  std::vector<Association> associate(const std::vector<Tracker>& trackers, const std::vector<Observation>& observations) override {
    if(trackers.empty() || observations.empty()) {
      return std::vector<Association>();
    }

    std::vector<Association> all_associations;
    all_associations.reserve(trackers.size() * 2);

    for(int i=0; i<trackers.size(); i++) {
      for(int j=0; j<observations.size(); j++) {
        auto dist = distance<Tracker, Observation>(trackers[i], observations[j]);
        if(dist) {
          all_associations.push_back(Association(i, j, dist.get()));
        }
      }
    }

    std::sort(all_associations.begin(), all_associations.end());

    std::vector<Association> associations;
    associations.reserve(trackers.size());
    while(!all_associations.empty()) {
      associations.push_back(all_associations.front());
      auto remove_loc = std::remove_if(all_associations.begin(), all_associations.end(), [&](const Association& assoc) {
        return assoc.tracker == associations.back().tracker || assoc.observation == associations.back().observation;
      });
      all_associations.erase(remove_loc, all_associations.end());
    }

    return associations;
  }
};

	}
}

#endif
