#include "utils.hh"
#include <iostream>
#include <complex>


Mat& subDFT(Mat& I)
{
  CV_Assert(I.depth() == CV_8U);
  const int channels = I.channels();
  int nRows = I.rows;
  int nCols = I.cols;
  const std::complex<double> i(0, 1);
  const double pi = std::acos(-1);
  switch(channels)
  {
    case 1:
      {
        Mat I2(nrows, 3,CV_32FC3, 1);
        for (int r = 0; i < nRows; ++r)
        {
          for (int c = 0; c < nCols; ++c)
          {
           auto pix = I.at<uchar>(i,j);
           I2.at<uchar>(i,j) = std::exp(-2 * i * pi )
          }
        }
        break;
      }
    case 3:
      {
        Mat I2(nrows, 3,CV_32FC3, Scalar(1,1,1));
        for (int i = 0; i < nRows; ++i)
        {
          cv::Vec3b* pixel = I.ptr<Vec3b>(i); // point to first pixel in row
          cv::Vec3f* pixeld = I.ptr<Vec3f>(i); // point to first pixel in row
          for (int j = 0; j < nCols; ++j)
          {
          }
        }
      }
  }
  return I;
}