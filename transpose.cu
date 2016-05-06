#include <stdio.h>
#include <assert.h>

//const int TILE_DIM = 32;
//const int BLOCK_ROWS = 32;
const int thread = 32;
const int NUM_REPS = 100;
const int nx = 1000;
const int ny = 1000;


//Uni10 Transpose
__global__ void transposeUni10(const double *A, size_t M, size_t N, double *AT)
{
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if(y < M && x < N)
    AT[x * M + y] = A[y * N + x];
}

//Yours Transpose
__global__ void transpose10(const double *A, size_t M, size_t N, double *AT)
{
  //
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// Check errors and print GB/s
void postprocess(const double *ref, const double *res, int n, double ms)
{
  bool passed = true;
  for (int i = 0; i < n; i++)
    if (res[i] != ref[i]) {
      printf("%d %f %f\n", i, res[i], ref[i]);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  if (passed)
    printf("%20.2f\n", 2 * n * sizeof(double) * 1e-6 * NUM_REPS / ms );
}

int main(int argc, char **argv)
{
  size_t M = nx;
  size_t N = ny;
  const int mem_size = nx*ny*sizeof(double);

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("\nDevice : %s\n", prop.name);
  //printf("Matrix size: %d %d, Block size: %d %d, Tile size: %d %d\n", 
  //       nx, ny, TILE_DIM, BLOCK_ROWS, TILE_DIM, TILE_DIM);
  
  checkCuda( cudaSetDevice(devId) );

  double *h_idata = (double*)malloc(mem_size);
  double *h_cdata = (double*)malloc(mem_size);
  double *h_tdata = (double*)malloc(mem_size);
  double *gold    = (double*)malloc(mem_size);
  
  double *d_idata, *d_cdata, *d_tdata;
  checkCuda( cudaMalloc(&d_idata, mem_size) );
  checkCuda( cudaMalloc(&d_cdata, mem_size) );
  checkCuda( cudaMalloc(&d_tdata, mem_size) );

  // host
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      h_idata[j*nx + i] = j*nx + i;

  // correct result for error checking
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++)
      gold[j*nx + i] = h_idata[i*nx + j];
  
  // device
  checkCuda( cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );
  
  // events for timing
  cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms;

  // ------------
  // time kernels
  // ------------
  printf("%25s%25s\n", "Routine", "Bandwidth (GB/s)");
  
  // --------------
  // transposeUni10 (Naive)
  // --------------

  dim3 dimGrid( (N + thread - 1) / thread, (M + thread - 1) / thread, 1);
  dim3 dimBlock(thread, thread, 1);
  printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

  printf("%25s", "uni10 transpose");
  checkCuda( cudaMemset(d_tdata, 0, mem_size) );
  // warmup
  transposeUni10<<<dimGrid, dimBlock>>>(d_idata, M, N, d_tdata);
  checkCuda( cudaEventRecord(startEvent, 0) );
  for (int i = 0; i < NUM_REPS; i++)
     transposeUni10<<<dimGrid, dimBlock>>>(d_idata, M, N, d_tdata);
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  checkCuda( cudaMemcpy(h_tdata, d_tdata, mem_size, cudaMemcpyDeviceToHost) );
  postprocess(gold, h_tdata, nx * ny, ms);

  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaFree(d_tdata) );
  checkCuda( cudaFree(d_cdata) );
  checkCuda( cudaFree(d_idata) );
  free(h_idata);
  free(h_tdata);
  free(h_cdata);
  free(gold);
}
