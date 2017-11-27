#include "../kernelcall.cuh"
#include <iostream>
#include "timer.hh"
#include <string>
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;
int main()
{
    std::ofstream  myfile;
    myfile.open ("nlm_gpu.txt",  std::ofstream::out | std::ofstream::app);
    Mat image;
    Mat res;
    image;
    std::string path_image("../../../../pictures/lenna.jpg");
    image = imread(path_image, CV_LOAD_IMAGE_UNCHANGED);
    if (!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return 1;
    }
    int width = image.rows;
    int height = image.cols;

    Rgb* device_dst;
    Rgb* device_img;
    Rgb* out;

    device_dst = empty_img_device(image);
    device_img = img_to_device(image);
    out = (Rgb*)malloc(width * height * sizeof (Rgb));

    double param_decay = 150.0;
    float milliseconds = 0;
    cudaEvent_t start, stop;
    for (size_t i = 2; i <= 5; i = i + 1)
    {
        for (size_t j = 2; j <= 6 ; j = j + 1)
        {
            {
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, 0);
		kernel_nlm_host(device_dst, device_img, width, height, i, j, param_decay);
                cudaEventRecord(stop, 0);

                cudaThreadSynchronize();
                cudaEventElapsedTime(&milliseconds, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                myfile << milliseconds / 1000 << std::endl;
            }
        }

    }
    myfile.close();

    myfile.open ("nlm_gpu2.txt",  std::ofstream::out | std::ofstream::app);
    std::string path_image2("../../../../pictures/Lenna514.png");
    Mat image2;
    image2 = imread(path_image2, CV_LOAD_IMAGE_UNCHANGED);

    if (!image2.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return 1;
    }
    for (size_t i = 2; i <= 5; i = i + 1)
    {
        for (size_t j = 2; j <= 6 ; j = j + 1)
        {
            {
                cudaEventRecord(start, 0);
                kernel_nlm_host(device_dst, device_img, width, height, i, j, param_decay);
                cudaEventRecord(stop, 0);

                cudaThreadSynchronize();
                cudaEventElapsedTime(&milliseconds, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                myfile << milliseconds / 1000 << std::endl;
            }
        }
    }
    myfile.close();
}
