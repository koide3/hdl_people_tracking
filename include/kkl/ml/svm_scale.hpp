#ifndef KKL_SVM_SCALE_HPP
#define KKL_SVM_SCALE_HPP

#include <array>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

namespace kkl {
  namespace ml {

struct SvmScale{
public:
  using vec2d = std::array<double, 2>;

  vec2d range;
  std::vector< vec2d > minmax;

  std::vector< vec2d > offsetScale;	//	( scaled = os[0] + value * os[1] )

  // constructor, destructor
  SvmScale(const std::string& scalefile, int featureCount = 5) {
    std::ifstream ifs(scalefile.c_str());
    if (ifs.fail()){
      std::cerr << "error : cannot open scaling file!!" << std::endl;
      std::cerr << "error : " << scalefile << std::endl;
      return;
    }

    std::string token;
    ifs >> token;
    if (token != "x"){
      std::cerr << "error : file format error!!" << std::endl;
      std::cerr << "error : " << scalefile << " is unsupported" << std::endl;
      return;
    }

    minmax.reserve(featureCount);
    offsetScale.reserve(featureCount);
    ifs >> range[0] >> range[1];

    vec2d lu, os;
    double index;
    while (!ifs.eof()){
      ifs >> index >> lu[0] >> lu[1];
      if (ifs.eof()){
        break;
      }
      minmax.push_back(lu);

      os[1] = (range[0] - range[1]) / (lu[0] - lu[1]);
      os[0] = range[0] - os[1] * lu[0];
      offsetScale.push_back(os);
    }
  }
  ~SvmScale(){}

  cv::Mat scaling(const cv::Mat& feature) const {
    cv::Mat scaled(feature.size(), 1, CV_32FC1);
    for (size_t i = 0; i<feature.rows; i++) {
      scaled.at<float>(i) = offsetScale[i][0] + feature.at<float>(i) * offsetScale[i][1];
    }

    return scaled;
  }

  cv::Mat scaling(const std::vector<float>& feature) const {
    cv::Mat scaled(feature.size(), 1, CV_32FC1);
    for (size_t i = 0; i<feature.size(); i++) {
      scaled.at<float>(i) = offsetScale[i][0] + feature[i] * offsetScale[i][1];
    }

    return scaled;
  }
};

  }
}

#endif
