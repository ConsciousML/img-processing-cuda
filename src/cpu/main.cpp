#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "non_local_means_cpu.hh"
#include "utils.hh"
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("usage: main <Image_Path> <Conv_size>\n");
        return 1;
    }
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if (!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return 1;
    }

    auto res = non_local_means_cpu(image, stoi(argv[2]));

    namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    imshow("Display Window", res);
    waitKey(0);
    return 0;
}
