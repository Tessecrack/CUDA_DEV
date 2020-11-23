#include <cuda_runtime.h>
#include <iostream>
#include <ctime>
#include "device_launch_parameters.h"
#include <limits.h>

#define CHECK(value) {\
    cudaError_t _m_cudaStat = value;\
    if (_m_cudaStat != cudaSuccess) {\
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);\
    } }

#define MEMORY_VECTOR 104857600

/*
	104857600
	4294967296
*/
/*
done  (1 block, sizeGramMatrix = 1024, countOfVectors = 32)
crush (1 block, sizeGramMatrix = 1089, countOfVectors = 33)
*/

#define SIZE_BIG_VECTOR 104857600
#define COUNT_OF_VECTORS 64

using namespace std;

static const size_t COUNT_OF_ELEMENTS = (int)SIZE_BIG_VECTOR / (int)COUNT_OF_VECTORS; // in one vector

inline void Info()
{
	cout << "Size big vector: " << SIZE_BIG_VECTOR 
		<< "\nCount of vectors: " << COUNT_OF_VECTORS
		<< "\nCount of elements in one vector: " << COUNT_OF_ELEMENTS << endl;
}

void PrintBigVector(unsigned char* bigVector);
void PrintVector(unsigned char* vector, size_t size);
unsigned char* GetRandomBigVector();
unsigned char* GetGramMatrixCPU(unsigned char* bigVector, float& time);
unsigned char* GetGramMatrixGPU(unsigned char* bigVector, float& time);
bool IsEqual(unsigned char* firstVector, unsigned char* secondVector, size_t size);


__global__
void calculate_GramMatrix_GPU(unsigned char* bigVector, unsigned char* gramMatrix, size_t sizeGramMatrix,
	size_t countOfElementsInVector, size_t countOfVectors)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= sizeGramMatrix) return;
	for (int j = 0; j < countOfElementsInVector; j++)
		gramMatrix[index] +=
			bigVector[(index / countOfVectors) * countOfElementsInVector + j] *
			bigVector[(index % countOfVectors) * countOfElementsInVector + j];
}

int main()
{
	Info();
	float timeCPU = 0.0f, timeGPU = 0.0f;

	unsigned char* bigVector = GetRandomBigVector();

	bool isForPrint = SIZE_BIG_VECTOR <= 2048;

	if (isForPrint) PrintBigVector(bigVector);

	size_t sizeGramMatrix = COUNT_OF_VECTORS * COUNT_OF_VECTORS;
	cout << "\nSize Gram matrix: " << sizeGramMatrix << endl;

	unsigned char* matrixGramCPU = GetGramMatrixCPU(bigVector, timeCPU);
	cout << "\nGram matrix CPU: " << endl;
	if (isForPrint) PrintVector(matrixGramCPU, sizeGramMatrix);

	unsigned char* matrixGramGPU = GetGramMatrixGPU(bigVector, timeGPU);
	cout << "\nGram matrix GPU: " << endl;
	if (isForPrint) PrintVector(matrixGramGPU, sizeGramMatrix);

	cout << "\nCheck...\n";
	if (IsEqual(matrixGramCPU, matrixGramGPU, sizeGramMatrix))
		cout << "That's right! :)\n";
	else cout << "Wrong! :(\n";

	cout << "\n--------\n";
	cout << "Time CPU: " << timeCPU << endl;
	cout << "Time GPU: " << timeGPU << endl;

	cin.get();
	return 0;
}
unsigned char* GetGramMatrixGPU(unsigned char* bigVector, float& time)
{
	int sizeGramMatrix = COUNT_OF_VECTORS * COUNT_OF_VECTORS;

	unsigned char* matrixGram = new unsigned char[sizeGramMatrix];

	int memoryForGramMatrix = sizeof(unsigned char) * sizeGramMatrix;
	int memoryForBigVector = sizeof(unsigned char) * SIZE_BIG_VECTOR;

	for (int i = 0; i < sizeGramMatrix; i++)
		matrixGram[i] = 0;

	unsigned char* bigVector_GPU; 
	unsigned char* matrixGram_GPU;

	cudaEvent_t startCUDA, stopCUDA;
	CHECK(cudaEventCreate(&startCUDA));
	CHECK(cudaEventCreate(&stopCUDA));

	CHECK(cudaMalloc(&bigVector_GPU, memoryForBigVector));
	CHECK(cudaMalloc(&matrixGram_GPU, memoryForGramMatrix));

	CHECK(cudaMemcpy(bigVector_GPU, bigVector, memoryForBigVector, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(matrixGram_GPU, matrixGram, memoryForGramMatrix, cudaMemcpyHostToDevice));

	CHECK(cudaEventRecord(startCUDA, 0));

	calculate_GramMatrix_GPU<<<sizeGramMatrix, 1024>>>(bigVector_GPU, matrixGram_GPU, sizeGramMatrix,
		COUNT_OF_ELEMENTS, COUNT_OF_VECTORS);

	CHECK(cudaEventRecord(stopCUDA, 0));
	CHECK(cudaEventSynchronize(stopCUDA));
	CHECK(cudaEventElapsedTime(&time, startCUDA, stopCUDA));

	CHECK(cudaMemcpy(matrixGram, matrixGram_GPU, memoryForGramMatrix, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(bigVector_GPU));
	CHECK(cudaFree(matrixGram_GPU));
	return matrixGram;
}

unsigned char* GetGramMatrixCPU(unsigned char* bigVector, float& time)
{
	int sizeGramMatrix = COUNT_OF_VECTORS * COUNT_OF_VECTORS;
	unsigned char* matrixGram = new unsigned char[sizeGramMatrix];

	time = clock();
	for (int i = 0; i < sizeGramMatrix; i++)
	{
		matrixGram[i] = 0;
		for (int j = 0; j < COUNT_OF_ELEMENTS; j++)
			matrixGram[i] +=
				bigVector[(i / COUNT_OF_VECTORS) * COUNT_OF_ELEMENTS + j] *
				bigVector[(i % COUNT_OF_VECTORS) * COUNT_OF_ELEMENTS + j];
	}
	time /= CLOCKS_PER_SEC;
	return matrixGram;

}

bool IsEqual(unsigned char* firstVector, unsigned char* secondVector, size_t size)
{	
	for (int i = 0; i < size; i++)
	{
		if (firstVector[i] != secondVector[i])
			return false;
	}
	return true;
}
unsigned char* GetRandomBigVector()
{
	unsigned char* vector = new unsigned char[SIZE_BIG_VECTOR];
	for (int i = 0; i < SIZE_BIG_VECTOR; i++)
		vector[i] = rand() % 9 + 1;
	return vector;
}


void PrintBigVector(unsigned char* bigVector)
{
	bool step = SIZE_BIG_VECTOR < 10;
	cout << "\nBig vector:\n\n";
	for (int i = 0, j = 0; i < SIZE_BIG_VECTOR; i++, j++)
	{
		if (j == COUNT_OF_ELEMENTS && step)
		{
			cout << endl;
			j = 0;
		}
		cout << (int)bigVector[i] << " ";
	}
	cout << endl;
}
void PrintVector(unsigned char* vector, size_t size)
{
	for (int i = 0; i < size; i++)
		cout << (int)vector[i] << " ";
	cout << endl;
}


