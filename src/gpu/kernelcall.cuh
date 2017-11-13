#include "kernel.cuh"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>

Rgb *img_to_device(cv::Mat img);
void device_to_img(Rgb *device_img, cv::Mat& img);
Rgb *empty_img_device(cv::Mat img);
void kernel_non_local_means_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size, float weight_decay);
void kernel_shared_conv_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size);
void kernel_conv_host(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size);
void kernel_pixelize_host(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size);
void kernel_shared_conv_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size);
