#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


struct Rgb {
    __host__ __device__ Rgb() {}
    __host__ __device__ Rgb(double x, double y, double z) : r(x), g(y), b(z) {}
    __host__ __device__ Rgb(cv::Vec3b v) : r(v[0]), g(v[1]), b(v[2]) {}
    __host__ __device__ void div(int x)
    {
        r /= x;
        g /= x;
        b /= x;
    }
    double r;
    double g;
    double b;
};
__global__ void hysterysis(Rgb *device_img, int* changed, int width, int height, double t);
__global__ void non_max_suppr(Rgb *device_img, double* img, int width, int height, double otsu_threshold);
__global__ void sobel_conv(Rgb *device_img, double* img, int width, int height, int conv_size);
__global__ void shared_knn(Rgb* device_img, Rgb* img, int width, int height, int conv_size, double h_param);
__global__ void nlm(Rgb* device_img, Rgb* img, int width, int height, int conv_size, int block_radius, double h_param);
__global__ void knn(Rgb* device_img, Rgb* img, int width, int height, int conv_size, double h_param);
__global__ void kernel_shared_conv(Rgb* device_img, Rgb* img, int width, int height, int strel_size);
__global__ void kernel_conv(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size);
__global__ void kernel_pixelize(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size);



