#include "kernel.cuh"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>


Rgb *img_to_device(cv::Mat img);
Rgb *empty_img_device(cv::Mat img);
void device_to_img(Rgb *device_img, cv::Mat& img);

double *img_to_device_grey(cv::Mat img);
double *empty_img_device_grey(cv::Mat img);
void device_to_img_grey(Rgb *device_img, cv::Mat& img);

void kernel_edge_detect(Rgb* device_img, double* img, int width, int height, int conv_size, double otsu_threshold);
void kernel_shared_knn_host(Rgb* device_img, Rgb* img, int width, int height, int r, double h_param);
void kernel_knn_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size, double h_param);
void kernel_nlm_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size, int block_size, double h_param);
void kernel_shared_conv_host(Rgb* device_img, Rgb* img, int width, int height, int strel_size);
void kernel_conv_host(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size);
void kernel_pixelize_host(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size);
void kernel_shared_conv_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size);
void kernel_shared_knn_host(Rgb* device_img, Rgb* img, int width, int height, int conv_size, double h_param);
