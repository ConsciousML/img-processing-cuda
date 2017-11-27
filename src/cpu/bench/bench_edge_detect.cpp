#include <iostream>
#include "timer.hh"
#include <string>  
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "edge_detect.hh"
#include "edge_detect.hh"
using namespace std;
using namespace cv;
int main()
{
  double t = 1;
  std::ofstream  myfile;
  myfile.open ("edge.txt",  std::ofstream::out | std::ofstream::app);
  Mat image;
  Mat res;
  image;
  std::string path_image("../../../pictures/lenna.jpg");
  image = imread(path_image, 0);
  double param_decay = 150.0;
  {
    t = 1;
    scoped_timer timer(t, myfile);
    res = conv_with_mask(image, 1);
  }
  std::string path_image2("../../../pictures/Lenna514.png");
  Mat image2;
  image2 = imread(path_image2, 0);
  if (!image2.data)
  {
    cout << "Could not open or find the image" << std::endl;
    return 1;
  }
  {
    t = 1;
    scoped_timer timer(t, myfile);
    res = conv_with_mask(image2, 1);
  }
  std::string path_image3("../../../pictures/my_face.jpg");
  Mat image3;
  image3 = imread(path_image3, 0);
  if (!image3.data)
  {
    cout << "Could not open or find the image" << std::endl;
    return 1;
  }
  {
    t = 1;
    scoped_timer timer(t, myfile);
    res = conv_with_mask(image3, 1);
  }
  std::string path_image4("../../../pictures/temple01.jpg");
  Mat image4;
  image4 = imread(path_image4, 0);
  if (!image4.data)
  {
    cout << "Could not open or find the image" << std::endl;
    return 1;
  }
  {
    t = 1;
    scoped_timer timer(t, myfile);
    res = conv_with_mask(image4, 1);
  }
  myfile.close();
}
