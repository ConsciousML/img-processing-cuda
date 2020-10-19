#include <stdio.h>
#include <iostream>
#include <valarray>
#include <assert.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "kernelcall.cuh"

#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define STREL_SIZE 5
#define R (STREL_SIZE / 2)
#define BLOCK_W (TILE_WIDTH + (2 * R))
#define BLOCK_H (TILE_HEIGHT + (2 * R))


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "usage: main <Image_Path> <Func_name>" << std::endl;
        return 1;
    }
    std::string func_name = argv[2];
    if (func_name == "knn" && argc < 5)
    {
        std::cout << "usage: main <Image_Path> knn <Conv_size> <Weight_Decay_Param>" << std::endl;
        return 1;
    }
    if (func_name == "nlm" && argc < 6)
    {
        std::cout << "usage: main <Image_Path> knn <Conv_size> <Block_radius> <Weight_Decay_Param>" << std::endl;
        return 1;
    }
    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if (!image.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return 1;
    }

    int width = image.rows;
    int height = image.cols;

    Rgb* device_dst;
    Rgb* device_img;
    Rgb* out;
    Rgb* device_dst_grey;
    double* device_img_grey;
    Rgb* out_grey;

    cv::Mat grey_img;
    device_dst = empty_img_device(image);
    device_img = img_to_device(image);
    out = (Rgb*)malloc(width * height * sizeof (Rgb));

    if (func_name == "pixelize")
        kernel_pixelize_host(device_dst, device_img, width, height, std::stoi(argv[3]));
    else if (func_name == "conv")
        kernel_conv_host(device_dst, device_img, width, height, std::stoi(argv[3]));
    else if (func_name == "shared_conv")
        kernel_shared_conv_host(device_dst, device_img, width, height, std::stoi(argv[3]));
    else if (func_name == "knn")
        kernel_knn_host(device_dst, device_img, width, height, std::stoi(argv[3]), std::stod(argv[4]));
    else if (func_name == "shared_knn")
        kernel_shared_knn_host(device_dst, device_img, width, height, std::stoi(argv[3]), std::stod(argv[4]));
    else if (func_name == "nlm")
        kernel_nlm_host(device_dst, device_img, width, height, std::stoi(argv[3]), std::stoi(argv[4]), std::stod(argv[5]));
    else if (func_name == "edge_detect")
    {
        cv::cvtColor(image, grey_img, cv::COLOR_BGR2GRAY);
        device_dst_grey = empty_img_device(grey_img);
        device_img_grey = img_to_device_grey(grey_img);
        out_grey = (Rgb*)malloc(width * height * sizeof (Rgb));
        cv::Mat dst;
        double otsu_threshold = cv::threshold(grey_img, dst, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        kernel_edge_detect(device_dst_grey, device_img_grey, width, height, 1, otsu_threshold);
    }
    else
    {
        std::cout << "error: function name '" << func_name << "' is not known." << std::endl;
        cudaFree(device_dst);
        cudaFree(device_img);
        free(out);
        return 1;
    }

    cudaDeviceSynchronize();
    cv::namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    if (func_name != "edge_detect")
    {
        cudaMemcpy(out, device_dst, height * width * sizeof (Rgb), cudaMemcpyDeviceToHost);

        device_to_img(out, image);

        cudaFree(device_dst);
        cudaFree(device_img);
        free(out);
        cv::imshow("Display Window", image);
    }
    else
    {
        cudaMemcpy(out_grey, device_dst_grey, height * width * sizeof (Rgb), cudaMemcpyDeviceToHost);

        device_to_img_grey(out_grey, grey_img);

        cudaFree(device_dst_grey);
        cudaFree(device_img_grey);
        free(out_grey);
        cv::imshow("Display Window", grey_img);
    }
    cv::waitKey(0);
    grey_img.release();
    image.release();
    return 0;
}
