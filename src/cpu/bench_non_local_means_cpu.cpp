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
  std::string path_image("./main ../../../pictures/lenna.jpg");
//  size_t conv_size = 2;
 // size_t radius = 2;
  double param_decay = 150.0;
  for(size_t i = 2; i < 32; i = i * 2)
  {
    for(size_t j = 2; j < 32; j = j * 2)
    {
      {
        scoped_timer timer(t, myfile);
        res = non_local_means_cpu(image, i, j, param_decay);
      }
    }

  }
