#include <iostream>
#include "timer.hh"
#include <string>  
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "edge_detect.hh"
#include "edge.hh"
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
  for(size_t j = 2; j <= 8; j = j + 1)
  {
    {
      t = 1;
      scoped_timer timer(t, myfile);
      res = knn_grey(image, j, 150.0);
      res = conv_with_mask(res, 1);
    }
  }
  std::string path_image2("../../../pictures/Lenna514.jpg");
  Mat image2;
  image2 = imread(path_image2, 0);
  if (!image2.data)
  {
    cout << "Could not open or find the image" << std::endl;
    return 1;
  }
  for(size_t j = 2; j <= 8; j = j + 1)
  {
    {
      t = 1;
      scoped_timer timer(t, myfile);
      res = knn_grey(image, j, 150.0);
      res = conv_with_mask(res, 1);
    }
  }
  myfile.close();
}
