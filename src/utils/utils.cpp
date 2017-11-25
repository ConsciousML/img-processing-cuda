#include "utils.hh"
#include <iostream>
#include <complex>
Mat& subDFT(Mat& I)
{
  // accept only char type matrices
  //Mat I;
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
          //std::cout <<"hhhAAAA" << channels;
          for (int j = 0; j < nCols; ++j)
          {
            //f(pixel[j], i, j);
          }
        }
      }
  }
  return I;
}
/*
   Functor::Functor(Mat &img)
   {
   img_ = img;
   }
   uchar & Functor::operator()(uchar & elt, int i, int j)
   {
   i = i + j;
   return elt;
   }
   Vec3b & Functor::operator()(Vec3b & elt, int i, int j)
   {
   i = i + j;
   return elt;
   }
   DFT::DFT(Mat &img):Functor::Functor(img)
   {}
   uchar & DFT::operator()(uchar & elt, int i, int j)
   {
   double a = 
   }
 */
/*Vec3b & DFT::operator()(Vec3b & elt, int i, int j)
  {
  int r, g, b;
  r = elt[2];
  g = elt[1];
  b = elt[0];

  }*/
/*
   GrayScale::GrayScale(Mat img)
   :Functor::Functor(img)
   {
   }*/
/*
   uchar & DFT:operator()(uchar & elt)
   {
   }*/
/*
   Vec3b & DFT::operator()(Vec3b & elt) 
   {
   }
   uchar & GrayScale::operator()(uchar & elt)
   {
   return elt;
   }
   Vec3b & GrayScale::operator()(Vec3b & elt)
   {
//std::cout <<"hello\n";
//Vec3b elt2(elt);
int r, g, b;
r = elt[2];
g = elt[1];
b = elt[0];
int avg =     (r + g + b) / 3;
//std::cout << avg << std::endl;
//std::cout << r << " " << g << " " << b << std::endl;
elt[2] = avg ;
elt[1] = avg ;
elt[0] = avg ;
return elt;
}*/
