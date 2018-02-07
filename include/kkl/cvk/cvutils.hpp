#ifndef CVUTILS_HPP
#define CVUTILS_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/circular_buffer.hpp>

namespace cvk {

inline boost::circular_buffer<cv::Scalar> create_color_palette(int n, double scale = 255.0) {
  std::vector<cv::Vec3b> palette;
  palette.reserve(n);

  for(int i=0; i<n; i++) {
    palette.push_back(cv::Vec3b((180.0 / (n+1)) * i, 220, 220));
  }
  cv::cvtColor(palette, palette, CV_HSV2BGR);

  boost::circular_buffer<cv::Scalar> hsv(n);
  double s = scale / 255.0;
  for(const auto& col: palette) {
    hsv.push_back(cv::Scalar(col[0] * s, col[1] * s, col[2] * s));
  }
  return hsv;
}

inline cv::Rect clip_roi(const cv::Rect& rect, const cv::Size& size) {
  int top = std::max(0, rect.y);
  int left = std::max(0, rect.x);
  int bottom = std::min(size.height, rect.y + rect.height);
  int right = std::min(size.width, rect.x + rect.width);
  return cv::Rect(left, top, right - left, bottom - top);
}

inline cv::Rect enlarge_rect(const cv::Rect& rect, double scale) {
  double dsize = (scale - 1.0) / 2.0;
  return cv::Rect(rect.x - rect.width * dsize, rect.y - rect.height * dsize, rect.width * scale, rect.height * scale);
}

inline cv::Rect shift_rect(const cv::Rect& rect, const cv::Point& pt) {
  return cv::Rect(rect.tl() + pt, rect.size());
}

}

#endif // CVUTILS_HPP
