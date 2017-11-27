#include <iostream>
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

cv::Mat knn(cv::Mat image, int conv_size, double h_param);
cv::Mat knn_grey(cv::Mat image, int conv_size, double h_param);
