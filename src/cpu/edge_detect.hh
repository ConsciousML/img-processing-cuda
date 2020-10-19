#include <iostream>
#include <valarray>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


std::valarray<double> conv_mask(cv::Mat image, int x, int y, int conv_size);
cv::Mat convolution(cv::Mat image, int conv_size);
cv::Mat conv_with_mask(cv::Mat image, int conv_size);
