#include <cuda_runtime.h>
#include <iostream>
#include <ctime>
#include "device_launch_parameters.h"
#include <limits.h>

#define PRINT_MATRIX true
#define CHECK(value) {\
    cudaError_t _m_cudaStat = value;\
    if (_m_cudaStat != cudaSuccess) {\
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);\
    } }

#define MAX_MEMORY_VECTOR 104857600 //100 Mb

#define COUNT_OF_ELEMENTS_IN_SYSTEM 1024 //Count of elements in system of vectors
#define COUNT_OF_VECTORS_IN_SYSTEM 12 //Count of vectors in system
#define COUNT_OF_ELEMENTS_IN_VECTOR (COUNT_OF_ELEMENTS_IN_SYSTEM / COUNT_OF_VECTORS_IN_SYSTEM) //Count of elements in one vector
#define SIZE_GRAM_MATRIX  (COUNT_OF_VECTORS_IN_SYSTEM * COUNT_OF_VECTORS_IN_SYSTEM)       

using namespace std;

inline void Info()
{
	cout << "Size of system: " << COUNT_OF_ELEMENTS_IN_SYSTEM 
		<< "\nCount of vectors: " << COUNT_OF_VECTORS_IN_SYSTEM
		<< "\nCount of elements in one vector: " << COUNT_OF_ELEMENTS_IN_VECTOR << endl;
}
void InfoResult(unsigned char*, unsigned char*);

void PrintSystemOfVectors(unsigned char*);

void PrintVector(unsigned char*, size_t);

unsigned char* GetRandomSystemOfVectors();

unsigned char* GetGramMatrixCPU(unsigned char* systemOfVectors, float& time);

unsigned char* GetGramMatrixGPU(unsigned char* systemOfVectors, float& time);

bool IsEqual(unsigned char* firstVector, unsigned char* secondVector, size_t size);

void Check(unsigned char* matrix_Host, unsigned char* matrix_Device);

__global__ void calculate_GramMatrix_GPU(unsigned char* systemOfVectors, unsigned char* gramMatrix)
{
	__shared__ unsigned char cache[1024];
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= SIZE_GRAM_MATRIX) return;
	for (int j = 0; j < COUNT_OF_ELEMENTS_IN_VECTOR; j++)
	{
		cache[threadIdx.x] +=
			systemOfVectors[(index / COUNT_OF_VECTORS_IN_SYSTEM) * COUNT_OF_ELEMENTS_IN_VECTOR + j] *
			systemOfVectors[(index % COUNT_OF_VECTORS_IN_SYSTEM) * COUNT_OF_ELEMENTS_IN_VECTOR + j];
	}
	__syncthreads();
	gramMatrix[index] = cache[threadIdx.x];
}
int main()
{
	Info();
	float timeCPU = 0.0f, timeGPU = 0.0f;
	unsigned char* systemOfVectors = GetRandomSystemOfVectors();
	bool isForPrint = COUNT_OF_ELEMENTS_IN_SYSTEM <= 2048;
	if (isForPrint) PrintSystemOfVectors(systemOfVectors);

	cout << "\nSize Gram matrix: " << SIZE_GRAM_MATRIX << "\n\n";

	unsigned char* matrixGramCPU = GetGramMatrixCPU(systemOfVectors, timeCPU);

	unsigned char* matrixGramGPU = GetGramMatrixGPU(systemOfVectors, timeGPU);

	Check(matrixGramCPU, matrixGramGPU);

	cout << "\n--------\n";
	cout << "Time CPU: " << timeCPU << endl;
	cout << "Time GPU: " << timeGPU << endl;
	cout << "\n--------\n";
	InfoResult(matrixGramCPU, matrixGramGPU);
	cin.get();
	return 0;
}
unsigned char* GetGramMatrixGPU(unsigned char* systemOfVectors, float& time_d)
{
	cout << "\n---------\n";
	unsigned char* matrixGram = new unsigned char[SIZE_GRAM_MATRIX];

	int memoryForGramMatrix = sizeof(unsigned char) * SIZE_GRAM_MATRIX;
	int memoryForBigVector = sizeof(unsigned char) * COUNT_OF_ELEMENTS_IN_SYSTEM;

	for (int i = 0; i < SIZE_GRAM_MATRIX; i++)
		matrixGram[i] = 0;

	unsigned char* systemOfVectors_GPU; 
	unsigned char* matrixGram_GPU;

	cudaEvent_t startCUDA, stopCUDA;
	CHECK(cudaEventCreate(&startCUDA));
	CHECK(cudaEventCreate(&stopCUDA));

	CHECK(cudaMalloc(&systemOfVectors_GPU, memoryForBigVector));
	CHECK(cudaMalloc(&matrixGram_GPU, memoryForGramMatrix));

	CHECK(cudaMemcpy(systemOfVectors_GPU, systemOfVectors, memoryForBigVector, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(matrixGram_GPU, matrixGram, memoryForGramMatrix, cudaMemcpyHostToDevice));

	CHECK(cudaEventRecord(startCUDA, 0));

	cout << "Calculate on DEVICE...\n";
	int countOfBlocks = (SIZE_GRAM_MATRIX + 1023) / 1024;
	calculate_GramMatrix_GPU<<<countOfBlocks, 1024>>>(systemOfVectors_GPU, matrixGram_GPU);
	cout << "Count of blocks: " << countOfBlocks << endl;
	

	cudaEventRecord(stopCUDA, 0);
	cudaEventSynchronize(stopCUDA);

	cudaEventElapsedTime(&time_d, startCUDA, stopCUDA);
	time_d /= 1000;
	CHECK(cudaMemcpy(matrixGram, matrixGram_GPU, memoryForGramMatrix, cudaMemcpyDeviceToHost));
	cout << "Done\n";
	
	cudaFree(systemOfVectors_GPU);
	cudaFree(matrixGram_GPU);
	return matrixGram;
}

unsigned char* GetGramMatrixCPU(unsigned char* systemOfVectors, float& time_h)
{
	unsigned char* matrixGram = new unsigned char[SIZE_GRAM_MATRIX];
	for (int i = 0; i < SIZE_GRAM_MATRIX; i++) matrixGram[i] = 0;
	cout << "Calculate on HOST...\n";
	time_h = clock();
	for (int i = 0; i < SIZE_GRAM_MATRIX; i++)
	{
		int currentRow = (i / COUNT_OF_VECTORS_IN_SYSTEM) * COUNT_OF_VECTORS_IN_SYSTEM;
		int shiftCol  = (i / COUNT_OF_VECTORS_IN_SYSTEM);
		int currentIndexMainDiag = currentRow + shiftCol;
		if (i < currentIndexMainDiag) continue;
		unsigned char temp = 0;
		for (int j = 0; j < COUNT_OF_ELEMENTS_IN_VECTOR; j++)
			temp +=
			systemOfVectors[(i / COUNT_OF_VECTORS_IN_SYSTEM) * COUNT_OF_ELEMENTS_IN_VECTOR + j] *
			systemOfVectors[(i % COUNT_OF_VECTORS_IN_SYSTEM) * COUNT_OF_ELEMENTS_IN_VECTOR + j];
		matrixGram[currentIndexMainDiag + (i - currentIndexMainDiag) * COUNT_OF_VECTORS_IN_SYSTEM] = matrixGram[i] = temp;
	}
	cout << "Done\n";
	time_h /= CLOCKS_PER_SEC;
	return matrixGram;
}
void Check(unsigned char* matrix_Host, unsigned char* matrix_Device)
{
	cout << "\nCheck...\n";
	if (IsEqual(matrix_Host, matrix_Device, SIZE_GRAM_MATRIX))
		cout << "That's right! :)\n";
	else cout << "Wrong! :(\n";
}
bool IsEqual(unsigned char* firstVector, unsigned char* secondVector, size_t size)
{	
	for (int i = 0; i < size; i++)
		if (firstVector[i] != secondVector[i])
			return false;
	return true;
}
unsigned char* GetRandomSystemOfVectors()
{
	unsigned char* vector = new unsigned char[COUNT_OF_ELEMENTS_IN_SYSTEM];
	for (int i = 0; i < COUNT_OF_ELEMENTS_IN_SYSTEM; i++)
		vector[i] = rand() % 9 + 1;
	return vector;
}
void InfoResult(unsigned char* matrix_Host, unsigned char* matrix_Device)
{
	cout << "\nGram matrix CPU: " << endl;
	PrintVector(matrix_Host, SIZE_GRAM_MATRIX);

	cout << "\nGram matrix GPU: " << endl;
	PrintVector(matrix_Device, SIZE_GRAM_MATRIX);
}

void PrintSystemOfVectors(unsigned char* systemOfVectors)
{
	bool step = COUNT_OF_ELEMENTS_IN_SYSTEM < 10;
	cout << "\nBig vector:\n\n";
	for (int i = 0, j = 0; i < COUNT_OF_ELEMENTS_IN_SYSTEM; i++, j++)
	{
		if (j == COUNT_OF_ELEMENTS_IN_VECTOR && step)
		{
			cout << endl;
			j = 0;
		}
		cout << (int)systemOfVectors[i] << " ";
	}
	cout << endl;
}
void PrintVector(unsigned char* vector, size_t size)
{
	if (PRINT_MATRIX)
	{
		for (int i = 0; i < COUNT_OF_VECTORS_IN_SYSTEM; i++)
		{
			for (int j = 0; j < COUNT_OF_VECTORS_IN_SYSTEM; j++)
			{
				cout << (int)vector[i * COUNT_OF_VECTORS_IN_SYSTEM + j] << "\t";
			}
			cout << endl;
		}
	}
	else
	{
		for (int i = 0; i < size; i++)
			cout << (int)vector[i] << " ";
		cout << endl;
	}
}


