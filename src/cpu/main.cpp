#include <iostream>
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "non_local_means_cpu.hh"
#include "knn.hh"
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        printf("usage: main <Image_Path> <Func_name> <Conv_size> <Weight_Decay_Param>\n");
        return 1;
    }
    string func_name = argv[2];
    if (func_name == "nlm" and argc < 6)
    {
        printf("usage: main <Image_Path> nlm <Conv_size> <Block_radius> <Weight_Decay_Param>\n");
        return 1;

    }
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);
    if (!image.data)
    {
        cout << "Could not open or find the image" << std::endl;
        return 1;
    }

    Mat res;
    if (func_name == "nlm")
    {
        cout << stoi(argv[4]);
        res = non_local_means_cpu(image, stoi(argv[3]), stoi(argv[4]), stod(argv[5]));
    }
    else if (func_name == "knn")
        res = knn(image, stoi(argv[3]), stof(argv[4]));
    namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
    imshow("Display Window", res);
    waitKey(0);
    imwrite("output.jpg", res);
    return 0;
}
