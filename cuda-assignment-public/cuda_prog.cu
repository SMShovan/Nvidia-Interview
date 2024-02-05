#include <stdio.h>
#include <limits.h>

bool checkResults(float *gold, float *d_data, int dimx, int dimy, float rel_tol) {
  for (int iy = 0; iy < dimy; ++iy) {
    for (int ix = 0; ix < dimx; ++ix) {
      int idx = iy * dimx + ix;

      float gdata = gold[idx];
      float ddata = d_data[idx];

      if (isnan(gdata) || isnan(ddata)) {
        printf("Nan detected: gold %f, device %f\n", gdata, ddata);
        return false;
      }

      float rdiff;
      if (fabs(gdata) == 0.f)
        rdiff = fabs(ddata);
      else
        rdiff = fabs(gdata - ddata) / fabs(gdata);

      if (rdiff > rel_tol) {
        printf("Error solutions don't match at iy=%d, ix=%d.\n", iy, ix);
        printf("gold: %f, device: %f\n", gdata, ddata);
        printf("rdiff: %f\n", rdiff);
        return false;
      }
    }
  }
  return true;
}

void computeCpuResults(float *g_data, int dimx, int dimy, int niterations,
                       int nreps) {
  for (int r = 0; r < nreps; r++) {
    printf("Rep: %d\n", r);
#pragma omp parallel for
    for (int iy = 0; iy < dimy; ++iy) {
      for (int ix = 0; ix < dimx; ++ix) {
        int idx = iy * dimx + ix;

        float value = g_data[idx];

        for (int i = 0; i < niterations; i++) {
          if (ix % 4 == 0) {
            value += sqrtf(logf(value) + 1.f);
          } else if (ix % 4 == 1) {
            value += sqrtf(cosf(value) + 1.f);
          } else if (ix % 4 == 2) {
            value += sqrtf(sinf(value) + 1.f);
          } else if (ix % 4 == 3) {
            value += sqrtf(tanf(value) + 1.f);
          }
        }
        g_data[idx] = value;
      }
    }
  }
}



__global__ void kernel_A(float *g_data, int dimx, int dimy, int niterations) {
  //Removal of outer two loops and replaced with respective code to maintain accuracy
  
  // Using shared memory to improve memory access patterns

  /*
  Shared memory speeds up the access of data. 
  As the block is (32 * 32), the size of shared memory 
  is (32 * (32 +1)).  Here additional element in each 
  row helps to resolve the Bank conflict when we were r
  eading from VRAM to shared memory (cache) and writing 
  from Cache to VRAM in a transposed way. 
  */
  const int paddedWidth = 33;
  extern __shared__ float shared_data[32 * 33];

  // Calculate global thread coordinates
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = iy * dimx + ix;

  // Load data into shared memory for better coalescence
  /*
  ### 2.5 Memory Coalesce and Fixing Warp Divergence

  We know when threads in the warp do different things, 
  it hinders the performance. We cannot avoid the conditional 
  statement altogether. However, we can make sure the threads 
  in a warp do the same instruction execution. However, when memory
   is read from VRAM to the Cache, in a cycle a set of values is 
   read (locality). If threads in warp work on the data in the Cache 
   (shared memory), we can avoid multiple cycles to fetch the data into the Cache. 

  #### Reason for Warp Divergence

  Initially, the conditional statements were given in such a way that 
  for a single row, the operations on the data for *nterations* are 
  varied for the consecutive column. For example, if the column (x-axis) 
  is divisible by 4, then log function, otherwise if the reminder is 
  one, cos function, and so on. However, as the GPUs are row-major. 
  That means threads in a warp will follow row-major order, each thread 
  working on consecutive data will process different functions which 
  cause warp divergence, which is a hindrance to performance. 

  #### Solution

  One idea is to transpose while reading into Cache. Transpose 
  reading and writing can cause Bank conflict which has been handled 
  by padding 1 additional memory space for each row in shared memory 
  allocation. In subsection 2.4, we have shown the transpose reading. 

  #### Discussion

  Due to memory coalesce, less number of clock cycle is needed to 
  get the data from VRAM to Cache.  The threads in Warp will perform 
  the same operations in row-major order for transposing the 32* 32 matrix in Cache.  
  */
  if (ix < dimx && iy < dimy) {
    shared_data[threadIdx.x * paddedWidth + threadIdx.y] = g_data[idx];
  }

  // Synchronize threads to ensure all data is loaded into shared memory
  __syncthreads();

  // Perform calculations on data in shared memory
  if (ix < dimx && iy < dimy) {
    float value = shared_data[threadIdx.y * paddedWidth + threadIdx.x];
    // Mod operator replacement with bitwise AND
    int iy_mod_4 = iy & 3;
    /*
    ### 2.6 Loop Unrolling
    As the *niterations* are small, we can use loop unrolling to utilize the vectorized operation. 
    We cannot parallelize this portion due to having dependency of ith iteration on (i-1)th iteration. 
    */
    #pragma unroll
    for (int i = 0; i < niterations; i++) {
      float temp;
      switch (iy_mod_4) {
        // Usage of intrinsic function
        case 0: temp = sqrtf(__logf(value) + 1.f); break;
        case 1: temp = sqrtf(__cosf(value) + 1.f); break;
        case 2: temp = sqrtf(__sinf(value) + 1.f); break;
        case 3: temp = sqrtf(__tanf(value) + 1.f); break;
      }
      value += temp;
    }
    shared_data[threadIdx.y * paddedWidth + threadIdx.x] = value;
  }

  // Synchronize threads again before writing back to global memory
  __syncthreads();

  // Write back the results from shared memory to global memory
  if (ix < dimx && iy < dimy) {
    g_data[idx] = shared_data[threadIdx.x * paddedWidth + threadIdx.y];
  }
}

/*
Necessary GPU Device Specification

Name: Tesla V100-SXM2-32GB

GPU count: 1

* Maximum number of threads per block: 1024
* Maximum dimension size of thread block: (1024, 1024, 64)
* Total amount of shared memory per block: 49152 Bytes
* Warp size: 32
* Global Memory: 32 GB
* Maximum dimension of grid size: (2147483647, 65535, 65535)

*/

void launchKernel(float * d_data, int dimx, int dimy, int niterations) {
  // Only change the contents of this function and the kernel(s). You may
  // change the kernel's function signature as you see fit. 

  //query number of SMs
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  //int num_sms = prop.multiProcessorCount;
  // int num_th = (int)ceil(sqrtf(prop.maxThreadsPerBlock));
  int num_th = 32;
  int num_bkx = (int)ceil(dimx/num_th);
  int num_bky = (int)ceil(dimy/num_th);
  
  dim3 block(num_th, num_th);
  dim3 grid(num_bkx , num_bky);
  kernel_A<<<grid, block>>>(d_data, dimx, dimy, niterations);

  /*
  Given the input matrix dimension as follows, 
  $$
  (x, y) = (8 \times 1024, 8 \times 1024) = (8192, 8192)
  $$
  As the block can have 1024 threads, we can take the shape of the block as 
  $$
  (32 \times 32) = 1024
  $$
  32 is exactly divisible by 8192. So, the grid size will be 
  $$
  ((8192/32),(8192/32)) = (256, 256)
  $$

  ### Finding: For both of the cases we got integers. So, 
  due to the grid and block for the input matrix, 
  there will not be any *warp divergence* problem. 
  There can be other satisfiable shapes possible for 
  both grid and block dimensions, however, the performance
  is better for these dimensions upon trial and error 
  basis due to high *occupancy*. 
  */
}

float timing_experiment(float *d_data,
                        int dimx, int dimy, int niterations, int nreps) {
  float elapsed_time_ms = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  for (int i = 0; i < nreps; i++) {
    launchKernel(d_data, dimx, dimy, niterations);
  }
  cudaEventRecord(stop, 0);
  cudaDeviceSynchronize();
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  elapsed_time_ms /= nreps;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return elapsed_time_ms;
}

int main() {
  int dimx = 8 * 1024;
  int dimy = 8 * 1024;

  int nreps = 10;
  int niterations = 5;

  int nbytes = dimx * dimy * sizeof(float);

  float *d_data = 0, *h_data = 0, *h_gold = 0;
  cudaMalloc((void **)&d_data, nbytes);
  if (0 == d_data) {
    printf("couldn't allocate GPU memory\n");
    return -1;
  }
  printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));
  h_data = (float *)malloc(nbytes);
  h_gold = (float *)malloc(nbytes);
  if (0 == h_data || 0 == h_gold) {
    printf("couldn't allocate CPU memory\n");
    return -2;
  }
  printf("allocated %.2f MB on CPU\n", 2.0f * nbytes / (1024.f * 1024.f));
  for (int i = 0; i < dimx * dimy; i++) h_gold[i] = 1.0f + 0.01*(float)rand()/(float)RAND_MAX;
  cudaMemcpy(d_data, h_gold, nbytes, cudaMemcpyHostToDevice);

  timing_experiment(d_data, dimx, dimy, niterations, 1);
  printf("Verifying solution\n");

  cudaMemcpy(h_data, d_data, nbytes, cudaMemcpyDeviceToHost);

  float rel_tol = .001;
  computeCpuResults(h_gold, dimx, dimy, niterations, 1);
  bool pass = checkResults(h_gold, h_data, dimx, dimy, rel_tol);

  if (pass) {
    printf("Results are correct\n");
  } else {
    printf("FAIL:  results are incorrect\n");
  }  

  float elapsed_time_ms = 0.0f;
 
  elapsed_time_ms = timing_experiment(d_data, dimx, dimy, niterations,
                                      nreps);
  printf("A:  %8.2f ms\n", elapsed_time_ms);

  printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

  if (d_data) cudaFree(d_data);
  if (h_data) free(h_data);

  cudaDeviceReset();

  return 0;
}

