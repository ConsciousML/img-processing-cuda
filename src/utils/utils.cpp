#include "utils.hh"
Mat& Iterator(Mat& I, Functor f)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);
    const int channels = I.channels();
    //int nRows = I.rows;
    //int nCols = I.cols;
    switch(channels)
    {
    case 1:
        {
            MatIterator_<uchar> it, end;
            for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
                f(*it);
            break;
        }
    case 3:
        {
            MatIterator_<Vec3b> it, end;
            for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
            {
		f(*it);
            }
        }
    }

    return I;
}
Functor::Functor(Mat &img)
{
 img_ = img;
}
uchar Functor::operator()(uchar elt)
{
  return elt;
}
Vec3b Functor::operator()(Vec3b elt)
{
  return elt;
}
GrayScale::GrayScale(Mat img)
:Functor::Functor(img)
{
  imgp_ = img.clone();
}
uchar GrayScale::operator()(uchar elt)
{
  return elt;
}
Vec3b GrayScale::operator()(Vec3b elt)
{
  return elt;
}
