#include <stdio.h>
#include <iostream>
#include <valarray>
#include <assert.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

struct Rgb {
    Rgb(float x, float y, float z) : r(x), g(y), b(z) {}
    Rgb(cv::Vec3b v) : r(v[0]), g(v[1]), b(v[2]) {}
    float r;
    float g;
    float b;
};


__global__ void foo(int *a, int N) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
    a[i]=i;
}

int test_kernel()
{
  int N=4097;
  int threads=128;
  int blocks=(N+threads-1)/threads;
  int *a;

  cudaMallocManaged(&a,N * sizeof (int));
  foo<<<blocks,threads>>>(a, N);
  cudaDeviceSynchronize();

  for (int i=0;i<10;i++)
    printf("%d\n",a[i]);

  return 0;
}

Rgb *img_to_device(cv::Mat img)
{
    Rgb *device_img;
    int width = img.rows;
    int height = img.cols;
    cudaMallocManaged(&device_img, width * height * sizeof (Rgb));

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            device_img[j + i * width] = Rgb(img.at<cv::Vec3b>(i, j));

    return device_img;
}

cv::Mat conv(cv::Mat& host_img)
{
    //img_proc
    //cudaDeviceSynchronize();
    Rgb *device_img = img_to_device(host_img);
    cudaFree(device_img);
    return host_img;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("usage: main <Image_Path> <Conv_size>");
        return 1;
    }
    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if (!image.data)
    {
        printf("Could not open or find the image");
        return 1;
    }
    image = conv(image);

    cv::namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    cv::imshow("Display Window", image);
    cv::waitKey(0);
    return 0;
}
