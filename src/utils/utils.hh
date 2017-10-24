#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

class Functor {
  public:
  Functor(Mat &img);
  virtual uchar operator()(uchar elt) ; 
  virtual Vec3b operator()(Vec3b elt) ; 
  private:
  Mat  img_;
};

class GrayScale: public Functor{
  public:
  GrayScale(Mat img);
  virtual uchar operator()(uchar elt) override  ; 
  virtual Vec3b operator()(Vec3b elt) override  ; 
  private:
  Mat imgp_; 
};

Mat& Iterator(Mat& I, Functor f);



