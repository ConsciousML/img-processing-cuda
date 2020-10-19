#ifndef TIMER_HH_
#define TIMER_HH_

#include <chrono>
#include <iostream>
#include <fstream>

using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;


struct scoped_timer {
  scoped_timer(double & s, std::ofstream & f) : seconds(s), t0(steady_clock::now()),myfile(f) {
  }

  ~scoped_timer()
  {
    seconds = std::chrono::duration_cast<std::chrono::milliseconds>
      (steady_clock::now() - t0).count();
    myfile  << seconds << " ms" << std::endl;
  }
  double  & seconds;
  steady_clock::time_point t0;
  std::ofstream  & myfile;

};

#endif
