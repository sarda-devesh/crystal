// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <string>
#include <cuda.h>
#include <cstdlib>
#include <cub/util_allocator.cuh>
#include "cub/test/test_util.h"

#include "crystal/crystal.cuh"

#include "gpu_utils.h"
#include "tpch_utils.h"

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define NUM_BLOCKS 640
#define NUM_TRIALS 10

/**
 * Globals, constants and typedefs
 */
bool  g_verbose = true;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

__global__ void QueryKernel(int* d_l_shipdate, float* d_l_discount, float* d_l_quantity, 
float* d_l_extendedprice, float* total, int lo_num_entries) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while(idx < lo_num_entries) {
      // Check the conditions for the query
      int curr_ship_date = d_l_shipdate[idx]; 
      if(curr_ship_date >= 8766 && curr_ship_date < 9131) {
        float curr_discount = d_l_discount[idx];
        if(curr_discount >= 0.05 && curr_discount <= 0.070001) {
          if(d_l_quantity[idx] < 24.0) {
            atomicAdd(total, d_l_extendedprice[idx] * curr_discount);
          }
        }
      }
      
      idx += gridDim.x * blockDim.x;
    }
}

float runQuery(int* d_l_shipdate, float* d_l_discount, float* d_l_quantity, float* d_l_extendedprice, 
int lo_num_entries, cub::CachingDeviceAllocator&  g_allocator, float* results, float* timings, int trial_num) {
    // Setup all the timing stuff
    cudaEvent_t gpu_start, gpu_stop; 
    cudaEventCreate(&gpu_start); 
    cudaEventCreate(&gpu_stop);
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_ms;
    start = high_resolution_clock::now();
    cudaEventRecord(gpu_start, 0);

    // Allocate space for the result on the GPU
    float* total = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**) &total, sizeof(float)));
    cudaMemset(total, 0, sizeof(float));

    // Run the kernel
    QueryKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_l_shipdate, d_l_discount, d_l_quantity, 
    d_l_extendedprice, total, LO_LEN);

    // Record that the kernel has finished
    float time_query;
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&time_query, gpu_start, gpu_stop);

    // Get the total amount of reveue
    float revenue;
    CubDebugExit(cudaMemcpy(&revenue, total, sizeof(float), cudaMemcpyDeviceToHost));

    // Record that we finished the query
    end = high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Log the results
    results[trial_num] = revenue;
    timings[2 * trial_num] = time_query;
    timings[2 * trial_num + 1] = (float) duration_ms.count();

    CubDebugExit(g_allocator.DeviceFree(total));
    return time_query;
}

#define MAX_ALLOWED_DIFF 500
void verifySameResult(float* results, int num_trials) {
  float expected = results[0];
  for(int i = 1; i < num_trials; i++) {
    float diff = std::abs(expected - results[i]);
    if(diff > MAX_ALLOWED_DIFF) {
      std::cerr << "Idx " << i << " has value " << results[i] << " but idx 0 has value " << expected 
      << ", which have a diff of " << diff << std::endl;
      exit(1);
    }
  }
}

void printTimings(float* timings, int num_trials) {
  float gpuTotalTime = 0.0;
  float overallTotalTime = 0.0;
  for(int i = 0; i < num_trials; i += 2) {
    gpuTotalTime += timings[i];
    overallTotalTime += timings[i + 1];
  }
  std::cout << "Average time taken GPU: " << gpuTotalTime/num_trials << std::endl;
  std::cout << "Average time taken overall: " << overallTotalTime/num_trials << std::endl;
} 

/**
 * Main
 */
int main(int argc, char** argv)
{
  // Get the data directory
  CommandLineArgs args(argc, argv);
  string data_dir = std::string(BASE_PATH) + std::string(DATA_DIR);
  
  // Read the columns
  int *h_l_shipdate = loadColumn<int>(data_dir, "lineitem", "l_shipdate", LO_LEN);
  float *h_l_discount = loadColumn<float>(data_dir, "lineitem", "l_discount", LO_LEN);
  float *h_l_quantity = loadColumn<float>(data_dir, "lineitem", "l_quantity", LO_LEN);
  float *h_l_extendedprice = loadColumn<float>(data_dir, "lineitem", "l_extendedprice", LO_LEN);

  // Begin the debugger
  CubDebugExit(args.DeviceInit());

  // Copy the data to the GPU
  int *d_l_shipdate = loadToGPU<int>(h_l_shipdate, LO_LEN, g_allocator);
  float *d_l_discount = loadToGPU<float>(h_l_discount, LO_LEN, g_allocator);
  float *d_l_quantity = loadToGPU<float>(h_l_quantity, LO_LEN, g_allocator);
  float *d_l_extendedprice = loadToGPU<float>(h_l_extendedprice, LO_LEN, g_allocator);

  // Perform multiple trials
  float* results = new float[NUM_TRIALS];
  float* timings = new float[2 * NUM_TRIALS];
  for(int trial_num = 0; trial_num < NUM_TRIALS; trial_num++) {
    runQuery(d_l_shipdate, d_l_discount, d_l_quantity, d_l_extendedprice, 
    LO_LEN, g_allocator, results, timings, trial_num);
  }

  // Output the timings
  verifySameResult(results, NUM_TRIALS);
  printTimings(timings, NUM_TRIALS);

  // Free the memory
  CubDebugExit(cudaFree(d_l_shipdate));
  CubDebugExit(cudaFree(d_l_discount)); 
  CubDebugExit(cudaFree(d_l_quantity));
  CubDebugExit(cudaFree(d_l_extendedprice));  
  return 0;
}

