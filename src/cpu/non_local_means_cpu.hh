#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

cv::Mat non_local_means_cpu(cv::Mat image, int conv_size, int block_radius, double weight_decay);
