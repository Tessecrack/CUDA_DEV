#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

__global__ void rgb_to_grayscale(unsigned char* image, int full_size_image)
{
  int i = 3 * (threadIdx.x + blockIdx.x*blockDim.x);
  if (i > full_size_image) return;
  int sum = (image[i] + image[i + 1] + image[i + 2]) / 3;
  image[i] = sum;
  image[i + 1] = sum;
  image[i + 2] = sum;
}

void ProcessImageCPU();
void ProcessImageGPU();

int main( int argc, char** argv )
{
    ProcessImageCPU();
    ProcessImageGPU();
    return 0;
}

void ProcessImageGPU()
{
  Mat image;
  image = imread("pic.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
  if(! image.data )                              // Check for invalid input
  {
    cout <<  "Could not open or find the image" << std::endl ;
  }
  unsigned char* imageGray;
  int full_size_image = image.rows * image.cols * 3;

  cudaEvent_t startCUDA, stopCUDA;
  float elapsedTimeCUDA;
  cudaEventCreate(&startCUDA);
  cudaEventCreate(&stopCUDA);

  CHECK(cudaMalloc(&imageGray, full_size_image));

  CHECK(cudaMemcpy(imageGray, image.data, full_size_image, cudaMemcpyHostToDevice));

  cudaEventRecord(startCUDA,0);

  rgb_to_grayscale<<<(full_size_image/3 + 255)/256, 256>>>(imageGray, full_size_image);

  cudaEventRecord(stopCUDA,0);
  cudaEventSynchronize(stopCUDA);
  CHECK(cudaGetLastError());

  cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

  cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
  cout << "CUDA memory throughput = " << 3*full_size_image*sizeof(float)/elapsedTimeCUDA/1024/1024/1.024 << " Gb/s\n";

  CHECK(cudaMemcpy(image.data, imageGray, full_size_image, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(imageGray));
  imwrite("pic2GPU.jpg", image);
  namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Display window", image );                   // Show our image inside it.
  waitKey(0);
}

void ProcessImageCPU()
{
  Mat image;

  image = imread("pic.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
  if(! image.data )                              // Check for invalid input
  {
    cout <<  "Could not open or find the image" << std::endl ;
  }
  clock_t start = clock();
  for(int i = 0; i < image.rows; i++)
  {
      //pointer to 1st pixel in row
      Vec3b* p = image.ptr<Vec3b>(i);
      for (int j = 0; j < image.cols; j++)
      {
        int sum = 0;
          for (int ch = 0; ch < 3; ch++)
          {
              sum += p[j][ch];  // – channel ch
          }
          for (int ch = 0; ch < 3; ch++)
          {
            p[j][ch] = sum / 3;
          }
      }
  }
  clock_t end = clock();
  double seconds = (double)(end - start) / CLOCKS_PER_SEC;
  imwrite("pic2CPU.jpg",image);
  cout <<"CPU time: " << seconds *1000 << "ms" << endl;
  //show image
  namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Display window", image );                   // Show our image inside it.
  waitKey(0);
}
