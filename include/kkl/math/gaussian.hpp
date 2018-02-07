/**
* gaussian.hpp
* @author : koide
* 13/06/12
* 13/06/20 GMM追加
* 13/08/03 ファイル名変更
* 13/11/16 1変量ガウス分布，1変量混合ガウス分布追加
* 14/10/13 KL情報量追加
* 14/10/16 1変量L2距離追加
* 14/11/11 IncrementalGaussianDistribution を追加
* 15/02/13 IncrementalGaussianDistributionの計算をPartialPivLuを使って高速化
* 15/08/20 GaussianEstimater，IndependentGaussianEstimater追加
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

		/**************************************************************
		* 1変量ガウス確率
		**************************************************************/
		template<typename T>
		T gaussianProbUni(T mean, T var, T x){
			const T dif = x - mean;
			return 1.0 / sqrt(2.0 * M_PI * var) * exp(-(dif * dif) / (2 * var));
		}

		/**************************************************************
		* 多変量ガウス確率
		**************************************************************/
		template<typename T, int p>
		T gaussianProbMul(const Eigen::Matrix<T, p, 1>& mean, const Eigen::Matrix<T, p, p>& cov, const Eigen::Matrix<T, p, 1>& x) {
			const T sqrtDet = sqrt(cov.determinant());
			const Eigen::Matrix<T, p, 1> dif = x - mean;
			const T lhs = 1.0 / (pow(2.0 * M_PI, p / 2.0) * sqrtDet);
			const T rhs = exp(-0.5 * ((dif.transpose() * cov.inverse() * dif))(0, 0));
			return lhs * rhs;
		}

#ifdef _USE_BOOST_ERF
		/**************************************************************
		* 正規分布の累積分布関数
		**************************************************************/
		template<typename T>
		T gaussianCumulativeProbUni(T mean, T var, T x) {
			return 0.5 * (1 + boost::math::erf((x - mean) / sqrt(2 * var)));
		};
#endif

    template<typename T, typename Matrix>
    Eigen::Matrix<T, 3, 1> errorEllipse(const Matrix& cov, double kai) {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 2, 2>> solver(cov);

      Eigen::Matrix<T, 3, 1> params;
      params[0] = std::sqrt(kai * kai * solver.eigenvalues()[1]);
      params[1] = std::sqrt(kai * kai * solver.eigenvalues()[0]);
      params[2] = std::atan2(solver.eigenvectors()(0, 2), solver.eigenvectors()(1, 2));

      return params;
    }

		/**************************************************************
		* マハラノビス距離
		**************************************************************/
		template<typename T, int p>
		T squaredMahalanobisDistance(const Eigen::Matrix<T, p, 1>& mean, const Eigen::Matrix<T, p, p>& cov, const Eigen::Matrix<T, p, 1>& x){
			const Eigen::Matrix<T, p, 1> diff = x - mean;
			double distance = diff.transpose() * cov.inverse() * diff;
			return distance;
		}

		/**************************************************************
		* 1変量マハラノビス距離
		**************************************************************/
		template<typename T>
		T squaredMahalanobisDistanceUni(T mean, T var, T x) {
			T diff = mean - x;
			return diff * diff / var;
		}

		/**************************************************************
		* ガウシアン当てはめ
		**************************************************************/
		template<typename T, int p>
		std::pair<Eigen::Matrix<T, p, 1>, Eigen::Matrix<T, p, p>> fitGaussian(const std::vector<Eigen::Matrix<T, p, 1>>& data_) {
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data(data_.size(), p);
			for (int i = 0; i < data_.size(); i++) {
				data.row(i) = data_[i];
			}

			Eigen::Matrix<T, p, 1> mean = data.colwise().mean();
			for (int i = 0; i < data_.size(); i++) {
				data.row(i) -= mean;
			}
			Eigen::Matrix<T, p, p> cov = (data.transpose() * data) / data_.size();

			return std::make_pair(mean, cov);
		}

		/**************************************************************
		* ガウシアン当てはめ，iterator版
		**************************************************************/
		template<typename T, int p, typename Iterator>
		std::pair<Eigen::Matrix<T, p, 1>, Eigen::Matrix<T, p, p>> fitGaussian(Iterator first, Iterator last) {
			int size = last - first;
			Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data(size, p);

			for (int i = 0; i < size; i++) {
				data.row(i) = *(first + i);
			}

			Eigen::Matrix<T, p, 1> mean = data.colwise().mean();
			for (int i = 0; i < size; i++) {
				data.row(i) -= mean;
			}
			Eigen::Matrix<T, p, p> cov = (data.transpose() * data) / size;

			return std::make_pair(mean, cov);
		}

		/**************************************************************
		* 一変量正規分布間のKL情報量
		**************************************************************/
		template<typename T>
		T klDivergenceUni(T mean_p, T var_p, T mean_q, T var_q) {
			return log(sqrt(var_q / var_p)) + (var_p + (mean_p - mean_q) * (mean_p - mean_q)) / (2 * var_q) - 0.5;
		}

		/**************************************************************
		* 多変量正規分布間のKL情報量
		**************************************************************/
		template<typename T, int p>
		T klDivergenceMul(const Eigen::Matrix<T, p, 1>& mean_p, const Eigen::Matrix<T, p, p>& var_p, const Eigen::Matrix<T, p, 1>& mean_q, const Eigen::Matrix<T, p, p>& var_q) {
			auto mean_diff = (mean_p - mean_q).eval();
			return 0.5 * (log(var_q.determinant() / var_p.determinant())
				+ (var_q.inverse() * var_p).trace()
				+ mean_diff.transpose() * var_q.inverse() * mean_diff
				- p);
		}

		/**************************************************************
		* 一変量正規分布間のL2距離
		**************************************************************/
		template<typename T>
		T l2DistanceUni(T mean_p, T var_p, T mean_q, T var_q) {
			double mean = (var_q * mean_p + var_p * mean_q) / (var_p + var_q);
			double var = (var_p * var_q) / (var_p + var_q);
			double A = mean * mean - (var_q * mean_p * mean_p + var_p * mean_q * mean_q) / (var_p + var_q);

			return 1 / (2 * sqrt(M_PI * var_p))
				+ 1 / (2 * sqrt(M_PI * var_q))
				- sqrt(2 * M_PI * var) / (M_PI * sqrt(var_p * var_q)) * exp(A / (2 * var));
		}


		/**************************************************************
		* 1変量ガウス分布
		* T : スカラ型
		**************************************************************/
		template<typename T>
		class GaussianDistributionUni{
		public:
			/******************************************************************
			* コンストラクタ
			* mean : 平均
			* sigmaSq : 分散
			******************************************************************/
			GaussianDistributionUni(T mean, T sigmaSq)
				: mean(mean),
				sigmaSq(sigmaSq),
				invSqrt2PiSigma(1.0 / sqrt(2.0 * M_PI * sigmaSq)),
				inv2SigmaSq(1.0 / (2.0 * sigmaSq))
			{}

			/******************************************************************
			* 確率を出す
			* x : 変数
			******************************************************************/
			T prob(T x) const {
				const T diff = x - mean;
				return invSqrt2PiSigma * exp(-(diff * diff) * inv2SigmaSq);
			}

			/******************************************************************
			* 最大確率(平均点の確率)を出す
			******************************************************************/
			T maxProb() const {
				return prob(mean);
			}

			/******************************************************************
			* () オペレータのオーバーライド
			* 確率を返す
			******************************************************************/
			T operator() (T x) const {
				return prob(x);
			}

			const T mean;
			const T sigmaSq;
		private:
			const T invSqrt2PiSigma;
			const T inv2SigmaSq;
		};
		/**************************************************************
		* 多変量ガウス分布
		* T : スカラ型
		* p : 次元数
		**************************************************************/
		template<typename T, int p>
		class GaussianDistribution{
		public:
			typedef Eigen::Matrix<T, p, 1> VectorTp;	// p次元ベクトル
			typedef Eigen::Matrix<T, p, p> MatrixTp;	// pxp行列

			/******************************************************************
			* コンストラクタ
			* mean : 平均
			* cov : 分散共分散
			******************************************************************/
			GaussianDistribution(const VectorTp& mean, const MatrixTp& cov)
				: mean_(mean),
				cov_(cov),
				det(cov.determinant()),
				lhm(1.0 / (pow(2.0 * M_PI, p / 2.0) * sqrt(det))),
				invCov(cov.inverse()),
				valid(abs(det) > 0.0001)
			{
			}

			/******************************************************************
			* マハラノビス距離
			* x : 変数
			******************************************************************/
			T mahalanobisDistance(const VectorTp& x) const {
				const auto diff = (x - mean_).eval();
				return sqrt((diff.transpose() * invCov * diff)(0, 0));
			}

			/******************************************************************
			* 確率を求める
			* x : 変数
			******************************************************************/
			T prob(const VectorTp& x) const {
				const auto diff = (x - mean_).eval();
				return lhm * exp(-0.5 * (diff.transpose() * invCov * diff)(0, 0));
			}

			/******************************************************************
			* 最大確率(平均点の確率)を求める
			******************************************************************/
			T maxProb() const {
				return prob(mean_);
			}

			/******************************************************************
			* = オペレータのオーバーライド
			******************************************************************/
			T operator() (const VectorTp& x) const {
				return prob(x);
			}

			/******************************************************************
			* アクセサ
			******************************************************************/
			const VectorTp& mean() const { return mean_; }
			const MatrixTp& cov() const { return cov_; }


			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		private:
			VectorTp mean_;	// 平均ベクトル
			MatrixTp cov_;	// 分散共分散
			T det;			// cov.determinant()
			T lhm;			// 1.0 / ( 2pi^(p/2) |cov|^0.5 )
			MatrixTp invCov;	// cov.inverse()
		public:
			bool valid;
		};

		/*******************************************************
		* IncrementalGaussianDistribution
		* インクリメンタルに更新できるガウス分布
		*******************************************************/
		template<typename T>
		class IncrementalGaussianDistribution {
		public:
			typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorTx;
			typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixTx;

			/*******************************************************
			* constructor
			*******************************************************/
			IncrementalGaussianDistribution(int dim)
				: dim(dim),
				accum_w(0),
				mean_(dim),
				cov_(dim, dim),
				accum_mean_(dim),
				accum_cov_(dim, dim),
				updated(true),
				det(0),
				lhm(0),
				inv_cov(dim, dim)
			{
				mean_.setZero();
				cov_.setZero();
				accum_mean_.setZero();
				accum_cov_.setZero();
			}

			/*******************************************************
			* constructor
			*******************************************************/
			IncrementalGaussianDistribution(double accum_w, const VectorTx& mean_, const MatrixTx& cov_)
				: dim(dim),
				accum_w(accum_w),
				mean_(mean_),
				cov_(cov_),
				accum_mean_(accum_w * mean_),
				accum_cov_(accum_w * cov_),
				updated(true),
				det(0),
				lhm(0),
				inv_cov(cov_.rows(), cov_.cols())
			{}

			/*******************************************************
			* reset
			* 平均とかリセット
			*******************************************************/
			void reset() {
				accum_w = 0;
				mean_.setZero();
				cov_.setZero();
				accum_mean_.setZero();
				accum_cov_.setZero();
				updated = true;
			}

			/*******************************************************
			* add
			* データを分布に追加
			* w : 重み
			* x : データ
			*******************************************************/
			template<typename VectorT>
			void add(double w, const VectorT& x) {
				auto diff = (x - mean_).eval();

				accum_mean_ += w * x;
				accum_cov_ += w * diff * diff.transpose();
				accum_w += w;

				mean_ = accum_mean_ / accum_w;

				updated = true;
			}

			/*******************************************************
			* mean
			*******************************************************/
			const VectorTx& mean() const {
				return mean_;
			}

			/*******************************************************
			* cov
			* 遅延評価なので，constでない
			*******************************************************/
			const MatrixTx& cov() {
				update();
				return cov_;
			}

			/*******************************************************
			* cumulativeWeight
			*******************************************************/
			double cumulativeWeight() const {
				return accum_w;
			}

			/*******************************************************
			* prob
			* 確率密度
			* 遅延評価なので，constでない
			*******************************************************/
			template<typename VectorT>
			double prob(const VectorT& x) {
				update();
				const auto diff = (x - mean_).eval();
				return dim <= 4 ?
					lhm * exp(-0.5 * diff.dot(inv_cov * diff)) :
					lhm * exp(-0.5 * diff.dot(pplu.solve(diff)));
			}

			/*******************************************************
			* operator()
			* 確率密度
			*******************************************************/
			template<typename VectorT>
			double operator() (const VectorT& x) {
				return prob(x);
			}

		private:
			T fast_determinant(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m) {
				switch (m.rows()){
				case 2:
					return Eigen::internal::determinant_impl<Eigen::Matrix<T, 2, 2>>::run(m);
				case 3:
					return Eigen::internal::determinant_impl<Eigen::Matrix<T, 3, 3>>::run(m);
				case 4:
					return Eigen::internal::determinant_impl<Eigen::Matrix<T, 4, 4>>::run(m);
				default:
					return m.determinant();
				}
			}

			void fast_inverse(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& dst) {
				switch (m.rows()){
				case 2:
					dst = Eigen::internal::inverse_impl<Eigen::Matrix<T, 2, 2>>(m);
					break;
				case 3:
					dst = Eigen::internal::inverse_impl<Eigen::Matrix<T, 3, 3>>(m);
					break;
				case 4:
					dst = Eigen::internal::inverse_impl<Eigen::Matrix<T, 4, 4>>(m);
					break;
				default:
					dst = m.inverse();
					break;
				}
			}


			/*******************************************************
			* update
			* 確率密度計算用の変数を評価する
			*******************************************************/
			void update() {
				if (!updated) {
					return;
				}
				updated = false;

				cov_ = accum_cov_ / accum_w;
				// 4次元以下であれば公式を用いた行列式，逆行列計算
				if (dim <= 4) {
					det = fast_determinant(cov_);
					fast_inverse(cov_, inv_cov);
				}
				// 4次元より大きければpartial piv lu decompositionを使用
				else {
					pplu.compute(cov_);
					det = pplu.determinant();
				}

				lhm = 1.0 / (pow(2.0 * M_PI, mean_.size() / 2.0) * sqrt(det));
			}

			const int dim;
			double accum_w;		// 累積重み
			VectorTx mean_;		// 平均
			MatrixTx cov_;		// 分散
			VectorTx accum_mean_;
			MatrixTx accum_cov_;

			bool updated;
			double det;			// det( cov )
			double lhm;			// 1.0 / ( 2pi^(p/2) |cov|^0.5 )
			MatrixTx inv_cov;	// cov.inverse()
			Eigen::PartialPivLU<MatrixTx> pplu;
		};

    template<typename T>
    class IncrementalGaussianDistributionUni {
    public:
      IncrementalGaussianDistributionUni()
        : accum_w(0.0),
          accum_wx(0.0),
          accum_wx_sq(0.0),
          mean_(0.0),
          var_(0.0)
      {}

      void add(T w, T x) {
        accum_w += w;
        accum_wx += w * x;
        accum_wx_sq += w * (x*x);

        mean_ = accum_wx / accum_w;
        var_ = (accum_w * mean_ * mean_ - 2 * mean_ * accum_wx + accum_wx_sq) / accum_w;
      }

      double cumulativeWeight() const {
        return accum_w;
      }

      double mean() const {
        return mean_;
      }

      double var() const {
        return var_;
      }

      double prob(T x) const {
        return gaussianProbUni(mean_, var_, x);
      }

      double operator() (T x) const {
        return prob(x);
      }

    private:

    private:
      double accum_w;
      double accum_wx;
      double accum_wx_sq;

      double mean_;
      double var_;
    };

		/**************************************************************
		* 1変量GMM
		* T : スカラ型
		**************************************************************/
		template<typename T>
		class GaussianMixtureModelUni{
		public:
			typedef GaussianDistributionUni<T> Gaussian;

			/**************************************************************
			* コンストラクタ
			* pi : 各ガウシアンの混合比
			* mu : 各ガウシアンの平均
			* sigmaSq : 各ガウシアンの分散
			**************************************************************/
			GaussianMixtureModelUni(const std::vector<T>& pi, const std::vector<T>& mu, const std::vector<T>& sigmaSq)
				: pi(pi)
			{
				gaussians.reserve(mu.size());
				for (size_t i = 0; i<mu.size(); i++){
					gaussians.push_back(Gaussian(mu[i], sigmaSq[i]));
				}
			}

			/**************************************************************
			* 確率
			*
			**************************************************************/
			T prob(T x) const{
				T accum = 0;
				for (size_t i = 0; i<pi.size(); i++){
					accum += pi[i] * gaussians[i](x);
				}
				return accum;
			}

			/**************************************************************
			* 最大確率の和
			* 数学的な意味はあまりない，描画時の正規化に使う
			**************************************************************/
			T maxProb() const{
				T maxprob = 0;
				for (size_t i = 0; i<pi.size(); i++){
					maxprob = std::max(maxprob, gaussians[i].maxProb());
				}
				return maxprob;
			}

			/**************************************************************
			* ()演算子，確率を返す
			**************************************************************/
			T operator() (T x) const {
				return prob(x);
			}
		private:
			std::vector<T> pi;
			std::vector<Gaussian> gaussians;
		};

		/**************************************************************
		* GMM
		* T : スカラ型， p : 次元数
		**************************************************************/
		template<typename T, int p>
		class GaussianMixtureModel{
		public:
			typedef Eigen::Matrix<T, p, 1> VectorTp;	// p次元ベクトル
			typedef Eigen::Matrix<T, p, p> MatrixTp;	// pxp行列
			typedef GaussianDistribution<T, p> Gaussian;

			/**************************************************************
			* コンストラクタ
			**************************************************************/
			GaussianMixtureModel(const std::vector<T>& weights, const std::vector<VectorTp>& mean, const std::vector<MatrixTp>& cov)
				: weights(weights){
				gaussians.reserve(mean.size());
				for (size_t i = 0; i<mean.size(); i++){
					gaussians.push_back(Gaussian(mean[i], cov[i]));
				}
			}

			/**************************************************************
			* 確率密度
			**************************************************************/
			T prob(const VectorTp& x) const{
				T accum = 0;
				for (size_t i = 0; i<weights.size(); i++){
					accum += weights[i] * gaussians[i](x);
				}
				return accum;
			}

			/**************************************************************
			* 最大確率の和
			**************************************************************/
			T totalMaxProb() const{
				T accum = 0;
				for (size_t i = 0; i<weights.size(); i++){
					accum += gaussians[i].maxProb();
				}
				return accum;
			}

			/**************************************************************
			* 演算子のオーバーライド
			**************************************************************/
			T operator() (const VectorTp& x) const {
				return prob(x);
			}

			const std::vector<T>& getWeights() const { return weights; }
			const std::vector< Gaussian, Eigen::aligned_allocator< Gaussian > >& getGaussians() const { return gaussians; }

		private:
			std::vector< T > weights;				// 正規分布重み
			std::vector< Gaussian, Eigen::aligned_allocator< Gaussian > > gaussians;	// 正規分布
		};

		/***************************************
		* GaussianEstimater
		* カルマンフィルタを使った一次元正規分布推定器
		***************************************/
		class GaussianEstimater {
		public:
			/***********************************
			* GaussianEstimater
			* init_mean : 平均初期値
			* init_var : 分散初期値
			* init_P : 初期値の分散
			***********************************/
      GaussianEstimater(double init_mean = 0.0, double init_var = 1.0, double init_P = 1000.0)
				: P(init_P), mean(init_mean), var(init_var)
			{}

			/***********************************
			* update
			* w : 重み
			* f : 入力
			***********************************/
			void update(double w, double f) {
				const double R = 0.01;
        double K = std::min(1.0 - 1e-6, w * P / (P + R));
				mean = K * f + (1 - K) * mean;
        var = K * (f - mean) * (f - mean) + (1 - K) * var;
				P = (1 - K) * P;
			}

			/***********************************
			* prob
			* f : 入力
			* return : 推定された正規分布からfが生起する確率
			***********************************/
			double prob(double f) const {
				return 1.0 / sqrt(2 * M_PI * var) * exp(-(f - mean) * (f - mean) / (2 * var));
			}

			/***********************************
			* operator()
			* f : 入力
			* return : 推定された正規分布からfが生起する確率
			***********************************/
			double operator()(double f) const {
				return prob(f);
			}

    public:
			double P;
			double mean;
			double var;
		};

		/*******************************************************
		* IndependentGaussianEstimater
		* 各次元が独立している仮定でのガウシアン推定
		* "On-line Boosting and Vision"からの実装
		*******************************************************/
		template<typename T>
		class IndependentGaussianEstimater {
			typedef Eigen::Array<T, Eigen::Dynamic, 1> ArrayT;
		public:
			IndependentGaussianEstimater(int dim, double process_noise = 0.01)
				: R(process_noise)
			{
        P = 1000.0;
        mean = ArrayT::Zero(dim);
				var = ArrayT::Constant(dim, 1.0);
			}

			IndependentGaussianEstimater(const ArrayT& init_mean, const ArrayT& init_var, double init_P = 1000.0, double process_noise = 0.01)
                : mean(init_mean),
                var(init_var),
				P(init_P),
				R(process_noise)
			{}

      void add(T weight, const ArrayT& data) {
        double K = std::min(1.0, P / (P + R) * weight);
				mean = K * data + (1 - K) * mean;
				var = K * (data - mean).square() + (1 - K) * var;
				P = (1 - K) * P;
			}

			const ArrayT& getMean() const { return mean; }
			const ArrayT& getVar() const { return var; }

      double squaredMahalanobisDistance(const ArrayT& f) const {
        return ((f - mean).pow(2) / var).sum();
      }

			double prob(const ArrayT& f) const {
				double lhs = sqrt(2.0 * M_PI);
        return (1.0 / (lhs * var.sqrt()) * (-(f - mean).square() / (2 * var)).exp()).prod();

			}

			double operator() (const ArrayT& f) const {
				return prob(f);
			}

			EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
			double P;
			const double R;
			ArrayT mean;
			ArrayT var;
		};

	}
}

#endif
