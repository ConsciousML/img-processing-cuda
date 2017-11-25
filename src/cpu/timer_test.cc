#include "timer.hh"
#include <iostream>
int main()
{
  double t = 1;
   double &b = t;
   {
    scoped_timer timer(t);
    //do your stuff there
    long a = 0;
    for(int i = 0; i < 1000; i++)
    {
      for(int j = 0; j < 1000; j++)
      {
        a = a + 1;
      }
    }
  }
}
