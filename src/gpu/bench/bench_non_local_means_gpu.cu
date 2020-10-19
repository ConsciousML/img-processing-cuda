#include <iostream>
#include <string>
#include <valarray>

#include "timer.hh"
#include "../kernelcall.cuh"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;


int main()
{
    std::ofstream  myfile;
    myfile.open ("edge_gpu.txt",  std::ofstream::out | std::ofstream::app);
    Mat image;
    Mat res;
    std::string path_image("../../../../pictures/temple01.jpg");
    cv::Mat gray;
    image = imread(path_image, CV_LOAD_IMAGE_UNCHANGED);
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    if (!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return 1;
    }
    int width = image.rows;
    int height = image.cols;

    Rgb* device_dst;
    double* device_img;
    Rgb* out;

    device_dst = empty_img_device(gray);
    device_img = img_to_device_grey(gray);

    double param_decay = 150.0;
    float milliseconds = 0;
    cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
                kernel_edge_detect(device_dst, device_img, width, height, 1, 91);
                cudaEventRecord(stop, 0);

                cudaThreadSynchronize();
                cudaEventElapsedTime(&milliseconds, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                myfile << milliseconds / 1000 << std::endl;
    myfile.close();
}
