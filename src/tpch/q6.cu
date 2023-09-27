// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <string>
#include <cuda.h>
#include <cub/util_allocator.cuh>
#include "cub/test/test_util.h"

#include "crystal/crystal.cuh"

#include "gpu_utils.h"
#include "tpch_utils.h"

using std::cout;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

/**
 * Globals, constants and typedefs
 */
bool  g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

__global__ void BasicQuery(int* d_l_shipdate, float* d_l_discount, float* d_l_quantity, float* d_l_extendedprice, 
int lo_num_entries, float* revenue) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < lo_num_entries) {
      int ship_date = d_l_shipdate[idx];
      float discount = d_l_discount[idx];
      float quantity = d_l_quantity[idx];
      if(ship_date >= 8766 && ship_date < 9131 && discount >= 0.05 && discount <= 0.07 && quantity < 24) {
        atomicAdd(revenue, discount * d_l_extendedprice[idx]);
      }
    }
}

float runQuery(int* d_l_shipdate, float* d_l_discount, float* d_l_quantity, float* d_l_extendedprice, 
int lo_num_entries, cub::CachingDeviceAllocator&  g_allocator) {
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
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**) &d_sum, sizeof(float)));
    cudaMemset(d_sum, 0, sizeof(float));

    // Call the kernel
    int blocks_per_thread = (lo_num_entries + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    BasicQuery<<<blocks_per_thread,THREADS_PER_BLOCK>>>(d_l_shipdate, d_l_discount, d_l_quantity, 
      d_l_extendedprice, lo_num_entries, d_sum);

    // Record that the kernel has finished
    float time_query;
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&time_query, gpu_start, gpu_stop);

    // Get the total amount of reveue
    float revenue;
    CubDebugExit(cudaMemcpy(&revenue, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // Record that we finished the query
    end = high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

    // Log the results
    std::cout << "Revenue: " << revenue << std::endl;
    std::cout << "Time taken GPU: " << time_query << " ms" << std::endl;
    std::cout << "Time taken total: " << duration_ms.count() << " ms" << std::endl;

    CLEANUP(d_sum);
    return time_query;
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
  std::cout << "** Loaded all the data **" << std::endl;

  // Begin the debugger
  CubDebugExit(args.DeviceInit());

  // Move the data to the GPU
  int *d_l_shipdate = loadToGPU<int>(h_l_shipdate, LO_LEN, g_allocator);
  float *d_l_discount = loadToGPU<float>(h_l_discount, LO_LEN, g_allocator);
  float *d_l_quantity = loadToGPU<float>(h_l_quantity, LO_LEN, g_allocator);
  float *d_l_extendedprice = loadToGPU<float>(h_l_extendedprice, LO_LEN, g_allocator);
  std::cout << "** Loaded all the data to the GPU **" << std::endl;

  // Run the query
  runQuery(d_l_shipdate, d_l_discount, d_l_quantity, d_l_extendedprice, LO_LEN, g_allocator);

  return 0;
}

