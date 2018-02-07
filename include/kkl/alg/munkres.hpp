/**
 * Munkres.hpp
 * @author : koide
 * 14/11/03
 **/
#ifndef KKL_MUNKRES_HPP
#define KKL_MUNKRES_HPP

#include <cfloat>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

namespace kkl {
	namespace alg {

/*********************************************
 * ゼロ判定用テンプレート関数
 * いらないかも
*********************************************/
template<typename T>
bool isZero(T value) {
	return value == 0;
}
template<>
bool isZero(float value) {
	return abs(value) <= FLT_EPSILON;
}
template<>
bool isZero(double value) {
	return abs(value) <= DBL_EPSILON;
}

/************************************************
 * Munkres
 * 組み合わせ最適化問題を解く
 * http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
 * 
 * Munkres<int> munkres
 * auto ans = munkres.solve( cost )
************************************************/
template<typename T>
class Munkres {
public:
	/*********************************************************
	 * solve
	 * cost : 組み合わせ問題のコスト行列 (cost.rows <= cost.cols)
	 * ret : 組み合わせ最適解
	********************************************************/
	Eigen::VectorXi solve(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& cost_) {
		assert( cost_.rows() <= cost_.cols() );
		cost = cost_;
		K = std::min(cost.rows(), cost.cols());

		starred = Eigen::MatrixXi::Zero(cost.rows(), cost.cols());
		primed = Eigen::MatrixXi::Zero(cost.rows(), cost.cols());
		covered_rows = Eigen::VectorXi::Zero(cost.rows());
		covered_cols = Eigen::VectorXi::Zero(cost.cols());

		// step 1
		cost.colwise() -= cost.rowwise().minCoeff();

		// step 2
		for (int i = 0; i < cost.rows(); i++) {
			for (int j = 0; j < cost.cols(); j++) {
				if (isZero(cost(i, j)) && starred.row(i).count() == 0 && starred.col(j).count() == 0){
					starred(i, j) = 1;
				}
			}
		}

		// step3 ~ step 6
		step3();

		// done, find the starred zeros
		Eigen::VectorXi ret(K);
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < cost.cols(); j++) {
				if (starred(i, j)) {
					ret(i) = j;
					break;
				}
			}
		}

		return ret;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
	/*****************************
	 * step3
	*****************************/
	void step3() {
		for (int i = 0; i < cost.cols(); i++) {
			if (starred.col(i).count() != 0) {
				covered_cols[i] = 1;
			}
		}
		// if K columns are covered, go to done
		if (covered_cols.count() == K) {
			return;
		}
		// otherwise, go to step 4
		return step4();
	}

	/*****************************
	 * step4
	*****************************/
	void step4() {
		for (int i = 0; i < cost.rows(); i++) {
			for (int j = 0; j < cost.cols(); j++){
				// find a noncovered zero and prime it
				if (!covered_rows[i] && !covered_cols[j] && isZero(cost(i, j))){
					primed(i, j) = 1;

					// if there is no starred zero in the row, go to step 5
					if (starred.row(i).count() == 0) {
						return step5(i, j);
					}
					// otherwise, cover this row and uncover the column containing the starred zero
					covered_rows[i] = 1;
					for (int k = 0; k < cost.cols(); k++) {
						if (starred(i, k)) {
							covered_cols[k] = 0;
						}
					}
				}
			}
		}

		T smallest = -1;
		// continue in this manner until there are no uncovered zeros left
		// save the smallest uncovered value and go to step 6
		for (int i = 0; i < cost.rows(); i++){
			for (int j = 0; j < cost.cols(); j++) {
				if (!covered_rows[i] && !covered_cols[j] && isZero(cost(i, j))){
					return step4();
				}

				if (!covered_rows[i] && !covered_cols[j] &&
					(smallest > cost(i, j) || smallest < 0.0)) {
					smallest = cost(i, j);
				}
			}
		}

		return step6(smallest);
	}

	/*****************************
	 * step5
	*****************************/
	void step5(int z0_row, int z0_col) {
		std::vector<std::pair<int, int>> series(1);
		series[0] = std::make_pair(z0_row, z0_col);

		bool done = false;
		while (!done) {
			// z1 : the starred zero in the column of z0
			bool added = false;
			const auto& z0 = series.back();
			for (int i = 0; i < cost.rows(); i++) {
				if (starred(i, z0.second)) {
					added = true;
					series.push_back(std::make_pair(i, z0.second));
					break;
				}
			}
			if (!added) {
				break;
			}

			const auto& z1 = series.back();
			// z2 : the primed zero in the rows of z1
			for (int i = 0; i < cost.cols(); i++) {
				if (primed(z1.first, i)){
					series.push_back(std::make_pair(z1.first, i));
					break;
				}
			}

			done = true;
			const auto& z2 = series.back();
			for (int i = 0; i < cost.rows(); i++){
				if (starred(i, z2.second)) {
					done = false;
				}
			}
		}

		for (int i = 0; i < series.size(); i++) {
			if (i % 2 == 0) {
				starred(series[i].first, series[i].second) = 1;
			}
			else {
				starred(series[i].first, series[i].second) = 0;
			}
		}

		primed.setZero();
		covered_rows.setZero();
		covered_cols.setZero();

		return step3();
	}

	/*****************************
	 * step6
	*****************************/
	void step6(T smallest) {
		for (int i = 0; i < cost.rows(); i++) {
			if (covered_rows[i]) {
				cost.row(i).array() += smallest;
			}
		}
		for (int i = 0; i < cost.cols(); i++) {
			if (!covered_cols[i]) {
				cost.col(i).array() -= smallest;
			}
		}

		return step4();
	}

	/*****************************
	 * showState
	 * show all matrices
	*****************************/
	void showState() const {
		std::cout << "--- cost ---" << std::endl << cost << std::endl;
		std::cout << "--- starred ---" << std::endl << starred << std::endl;
		std::cout << "--- primed ---" << std::endl << primed << std::endl;

		Eigen::MatrixXi covered(covered_rows.size(), covered_cols.size());
		for (int i = 0; i < covered.rows(); i++) {
			for (int j = 0; j < covered.cols(); j++) {
				covered(i, j) = covered_rows[i] + covered_cols[j];
			}
		}
		std::cout << "--- covered ---" << std::endl << covered << std::endl;
	}

private:
	int K;
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> cost;

	Eigen::MatrixXi starred;
	Eigen::MatrixXi primed;

	Eigen::VectorXi covered_rows;
	Eigen::VectorXi covered_cols;
};

	}
}

#endif