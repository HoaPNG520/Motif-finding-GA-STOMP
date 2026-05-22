#pragma once
// =============================================================================
//  timer.cuh -- Lightweight timing instrumentation
//
//  CudaTimer  : wraps cudaEvent_t for accurate GPU kernel timing (ms)
//  WallTimer  : wraps clock() for host-side elapsed time (ms)
//
//  Usage:
//    CudaTimer t;
//    t.start();
//    my_kernel<<<grid,block>>>(...);
//    float ms = t.stop();   // blocks until kernel finishes
//
//  TimingReport : accumulated timing breakdown printed at the end of a GA run
// =============================================================================

#include "common.cuh"
#include <time.h>

// -----------------------------------------------------------------------------
//  GPU event timer
// -----------------------------------------------------------------------------
struct CudaTimer {
    cudaEvent_t _start, _stop;

    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&_start));
        CUDA_CHECK(cudaEventCreate(&_stop));
    }
    ~CudaTimer() {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

    void start() { CUDA_CHECK(cudaEventRecord(_start)); }

    // Returns elapsed milliseconds; blocks until the stop event is reached.
    float stop() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(_stop));
        CUDA_CHECK(cudaEventSynchronize(_stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, _start, _stop));
        return ms;
    }
};

// -----------------------------------------------------------------------------
//  Host wall-clock timer (milliseconds)
// -----------------------------------------------------------------------------
struct WallTimer {
    clock_t _start;
    void  start() { _start = clock(); }
    float stop()  {
        return 1000.0f * (float)(clock() - _start) / (float)CLOCKS_PER_SEC;
    }
};

// -----------------------------------------------------------------------------
//  Accumulated timing breakdown for one GA run
// -----------------------------------------------------------------------------
struct TimingReport {
    float total_wall_ms;        // wall time for the entire GA run
    float stomp_total_ms;       // sum of all STOMP GPU kernel times
    float precompute_total_ms;  // sum of all stats+QT-init kernel times
    float fitness_total_ms;     // sum of CPU fitness evaluation times
    float ga_overhead_ms;       // selection + crossover + mutation time
    int   n_stomp_calls;        // total STOMP evaluations

    void print() const;
    void save_csv(const char* path) const;
};
