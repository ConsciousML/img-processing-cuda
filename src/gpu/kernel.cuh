#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

struct Rgb {
    __host__ __device__ Rgb() {}
    __host__ __device__ Rgb(float x, float y, float z) : r(x), g(y), b(z) {}
    __host__ __device__ Rgb(cv::Vec3b v) : r(v[0]), g(v[1]), b(v[2]) {}
    __host__ __device__ void div(int x)
    {
        r /= x;
        g /= x;
        b /= x;
    }
    float r;
    float g;
    float b;
};

__global__ void non_local_means_gpu(Rgb* device_img, Rgb* img, int conv_size, float weight_decay);
__global__ void kernel_shared_conv(Rgb* device_img, Rgb* img, int width, int height);
__global__ void kernel_conv(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size);
__global__ void kernel_pixelize(Rgb* device_img, Rgb* img, int rows, int cols, int conv_size);
Rgb *img_to_device(cv::Mat img);
void device_to_img(Rgb *device_img, cv::Mat& img);
Rgb *empty_img_device(cv::Mat img);



