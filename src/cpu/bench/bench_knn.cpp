#include <iostream>
#include "timer.hh"
#include <string>  
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "edge_detect.hh"
#include "knn.hh"
using namespace std;
using namespace cv;
int main()
{
  double t = 1;
  std::ofstream  myfile;
  myfile.open ("knn.txt",  std::ofstream::out | std::ofstream::app);
  Mat image;
  Mat res;
  image;
  std::string path_image("../../../pictures/lenna.jpg");
  image = imread(path_image, CV_LOAD_IMAGE_UNCHANGED);
  double param_decay = 150.0;
  for(size_t j = 2; j <= 24; j = j + 2)
  {
    {
      t = 1;
      scoped_timer timer(t, myfile);
      res = knn(image, j, param_decay);
    }
  }
  std::string path_image2("../../../pictures/Lenna514.png");
  Mat image2;
  image2 = imread(path_image2, CV_LOAD_IMAGE_UNCHANGED);
  if (!image2.data)
  {
    cout << "Could not open or find the image" << std::endl;
    return 1;
  }
  for(size_t j = 2; j <= 24; j = j + 2)
  {
    {
      t = 1;
      scoped_timer timer(t, myfile);
      res = knn(image2, j, param_decay);
    }
  }
  myfile.close();
}
