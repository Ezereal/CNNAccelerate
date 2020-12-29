#ifndef __TIMER_H_
#define __TIMER_H_
#include <sys/time.h>
#include <string>
#include <iostream>
#include <chrono>

class Timer {
public:
    Timer(){}

    void start_timer()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    void end_timer()
    {
        end = std::chrono::high_resolution_clock::now();
    }

    void compute_timer()
    {
        auto duration = std::chrono::duration<double>(end - start).count();
        std::cout << "consuming time(ms): " << duration * 1000 << std::endl;
    }

    ~Timer(){}
    
private:
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
};

#endif
