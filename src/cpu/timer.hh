#ifndef TIMER_HH_
#define TIMER_HH_

#include <chrono>
#include <iostream>
#include <fstream>

using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;
struct scoped_timer {
  /* Init the timer: get the current time */
  scoped_timer(double & s, std::ofstream & f) : seconds(s), t0(steady_clock::now()),myfile(f) {
  }

  /* set time diff between now and start in the double ref (seconds) */
  /* use maximal precision (nanoseconds) for counts                  */
  /* hint: seconds = double(nanoseconds count) / 1e9 */
  ~scoped_timer()
  {
    seconds = std::chrono::duration_cast<std::chrono::microseconds>
      (steady_clock::now() - t0).count();
   // myfile.open ("time_rec.txt",  std::ofstream::out | std::ofstream::app);
   myfile  << seconds << " ms" << std::endl;
  //std::cout <<"open time_rec.txt to see time exec record"<<std::endl;
    //myfile.close();
  }
  double  & seconds;
  steady_clock::time_point      t0;
  std::ofstream  & myfile;

};

#endif
