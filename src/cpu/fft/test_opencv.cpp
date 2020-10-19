#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <valarray>
#include <complex>
//#include <utils.hh>
using namespace std;
using namespace cv;
std::complex<double> subDFT(Mat& I, int m, int n);
Mat DFT(Mat& I);
std::complex<double> subDFT(Mat& I, int m, int n)
{
  int nRows = I.rows;
  int nCols = I.cols;
  std::complex<double> i(0,1);
  double pi = std::acos(-1);
  Mat I2(nRows,nCols ,CV_32FC1, 1);
  std::complex<double> sum(0,0);
  for (int r = 0; r < nRows; ++r)
  {
    for (int c = 0; c < nCols; ++c)
    {
      auto pix = I.at<uchar>(r,c);
      sum +=  double(pix) * std::exp(-2.0 *  i * pi * double((((r * m) /nRows) + ((c * n) /nCols ))));
    }
  }
  return sum;
}
Mat DFT(Mat& I)
{
  int nRows = I.rows;
  int nCols = I.cols;
  const int channels = I.channels();
  std::complex<double> B[nRows][nCols];
  for (int r = 0; r < nRows; ++r)
  {
    for (int c = 0; c < nCols; ++c)
    {
      B[r][c] = subDFT(I, r, c);
    }
  }
  return cv::Mat(nRows, nCols, CV_64FC2, &B);
}
int main(int argc, char** argv)
{
  if (argc != 2)
  {
    printf("usage: main <Image_Path> \n");
    return 1;
  }
  Mat image;
  image = imread(argv[1],  CV_LOAD_IMAGE_GRAYSCALE);
  if (!image.data)
  {
    cout << "Could not open or find the image" << std::endl;
    return 1;
  }

  auto res = DFT(image);
  namedWindow("Display Window", CV_WINDOW_AUTOSIZE);
  imshow("Display Window", res);
  waitKey(0);
  return 0;
}
