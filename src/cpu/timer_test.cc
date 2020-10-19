#include "timer.hh"
#include <iostream>
#include <fstream>


int main()
{
  double t = 1;
  std::ofstream  myfile;
  myfile.open ("time_rec.txt",  std::ofstream::out | std::ofstream::app);

  {
    scoped_timer timer(t, myfile);
    long a = 0;
    for(int i = 0; i < 1000; i++)
    {
      for(int j = 0; j < 1000; j++)
      {
        a = a + 1;
      }
    }
  }
  myfile.close();

}
