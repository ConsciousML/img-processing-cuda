#include "non_local_means_cpu.hh"
#include <iostream>
#include "timer.hh"
#include <string>  
#include <valarray>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "edge_detect.hh"
using namespace std;
using namespace cv;
int main()
{
  double t = 1;
  std::ofstream  myfile;
  
  myfile.open ("nlm.txt",  std::ofstream::out | std::ofstream::app);
  Mat image;
  Mat res;
  image;
  std::string path_image("../../../pictures/lenna.jpg");
  image = imread(path_image, CV_LOAD_IMAGE_UNCHANGED);
  if (!image.data)
  {
    cout << "Could not open or find the image" << std::endl;
    return 1;
  }
  //  size_t conv_size = 2;
  // size_t radius = 2;
  double param_decay = 150.0;
  for(size_t i = 2; i <= 5; i = i + 1)
  {
    for(size_t j = 2; j <= 6 ; j = j + 1)
    {
      {
        t = 1;
        scoped_timer timer(t, myfile);
        res = non_local_means_cpu(image, i, j, param_decay);
      }
    }

  }
  myfile.close();

  myfile.open ("nlm2.txt",  std::ofstream::out | std::ofstream::app);
  std::string path_image2("../../../pictures/Lenna514.png");
  Mat image2;
  image2 = imread(path_image2, CV_LOAD_IMAGE_UNCHANGED);
  
  if (!image2.data)
  {
    cout << "Could not open or find the image" << std::endl;
    return 1;
  }
  for(size_t i = 2; i <= 5; i = i + 1)
  {
    for(size_t j = 2; j <= 6 ; j = j + 1)
    {
      {
        t = 1;
        scoped_timer timer(t, myfile);
        res = non_local_means_cpu(image2, i, j, param_decay);
      }
    }
  }
  myfile.close();
}
