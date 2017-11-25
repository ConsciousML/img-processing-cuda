#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

Mat& DFT(Mat& I2);

/*class Functor {
  public:
    Functor(Mat &img);
    virtual uchar & operator()(uchar & elt, int i, int j) ; 
    virtual Vec3b & operator()(Vec3b & elt, int i, int j) ; 
  private:
    Mat  img_;
};*/
/*
class GrayScale: public Functor{
    Vec3b & operator()(Vec3b & elt, int i, int j) ; 
    //  private:
    // Mat imgp_; 
};
*/
/*class DFT: public Functor{
  public:
    DFT(Mat &img):Functor(img)
    {
    }
    uchar & operator()(uchar & elt, int i, int j) ; 
    Vec3b & operator()(Vec3b & elt, int i, int j) ; 
};
*/
