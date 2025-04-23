#ifdef __linux__
#include <openacc.h>
#endif
#include <opencl.hpp>
#include <omp.h>
#include <iostream>
#include <chrono>
using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

struct msClock
{
	typedef std::chrono::high_resolution_clock clock;
	std::chrono::time_point<clock> t1, t2;
	void Start() { t1 = high_resolution_clock::now(); }
	void Stop() { t2 = high_resolution_clock::now(); }
	double ElapsedTime()
	{
		duration<double, std::milli> ms_doubleC = t2-t1;
		return ms_doubleC.count();
	}
}
Clock;

void NullMat(TYPE*, int, int);
TYPE diffMat(TYPE*, TYPE*, int , int);
void initialiseMat(TYPE*, int, int);
void displayMat(TYPE*, int, int);
void matMult(TYPE*, TYPE*, TYPE*, int, int, int, int);
void matMultOMP(TYPE*, TYPE*, TYPE*, int, int, int, int);
#ifdef __linux__
void matMultOACC(TYPE*, TYPE*, TYPE*, int, int, int, int);
#endif

int main()
{
	vector<Device_Info> test;
	// compile OpenCL C code for the device with the given id
	Device device(select_device_with_id(0), "MatMult_Kernels.cl");

	// size of vectors
	int rA = 1024;
	int cA = 1024;
	int rB = cA;
	int cB = 1024;

	// allocate memory on both host and device
	Memory<TYPE> A(device, rA*cA, 1);
	Memory<TYPE> B(device, rB*cB, 1);
	Memory<TYPE> C(device, rA*cB, 1);
	Memory<TYPE> gpuC(device, rA*cB, 1);

	// initialise memory
	initialiseMat(A.data(), rA, cA);
	initialiseMat(B.data(), rB, cB);

	displayMat(A.data(),rA,cA);
	displayMat(B.data(),rB,cB);

	Clock.Start();
	matMult(C.data(),A.data(),B.data(),rA,cA,rB,cB);
	Clock.Stop();
	//cout << "Time taken (single): " << Clock.ElapsedTime() << " ms." << endl;
	print_info("Time taken (single): "+to_string(Clock.ElapsedTime(), 3)+" ms.");

	displayMat(C.data(),rA,cB);

	Clock.Start();
	matMultOMP(C.data(),A.data(),B.data(),rA,cA,rB,cB);
	Clock.Stop();
	print_info("Time taken (OMP): "+to_string(Clock.ElapsedTime(), 3)+" ms.");

	#ifdef __linux__
	Clock.Start();
	matMultOACC(C.data(),A.data(),B.data(),rA,cA,rB,cB);
	Clock.Stop();
	print_info("Time taken (OACC): "+to_string(Clock.ElapsedTime(), 3)+" ms.");
	#endif

	// ================================= OpenCL Mat Mult =========================================
	// kernel that runs on the device
	Kernel MatMultKernel(device, rA*cB, "MatMultKernel", gpuC, A, B, rA, cA, rB, cB);	// 1 D Kernel
	Kernel MatMultKernel2D(device, rA, cB, "MatMultKernel2D", gpuC, A, B, rA, cA, rB, cB);	// 2 D Kernel
	//Kernel MatMultKernel2D(device, rA, cB, true, 8, "MatMultKernel2D", gpuC, A, B, rA, cA, rB, cB);	// 2 D Kernel explicit workgroup size

	print_info("Value before kernel execution: C[0] = "+to_string(gpuC[0]));

	Clock.Start();
	// copy data from host memory to device memory
	A.write_to_device();
	B.write_to_device();
	// run add_kernel on the device
	//MatMultKernel.run();
	MatMultKernel2D.run();
	// copy data from device memory to host memory
	gpuC.read_from_device();
	Clock.Stop();
	print_info("Time taken (OpenCL): "+to_string(Clock.ElapsedTime(), 3)+" ms.");
	// =========================================================================================

	displayMat(gpuC.data(),rA,cB);

	print_info("CPU and GPU diff: "+to_string(diffMat(C.data(),gpuC.data(),rA,cB)));
	print_info("Value after kernel execution: C[0] = "+to_string(gpuC[0]));
	
	//wait();
	return 0;
}

TYPE diffMat(TYPE* M1, TYPE* M2, int rM, int cM)
{
	TYPE diff = 0;
	for (int i=0; i<rM; i++)
		for (int j=0; j<cM; j++)
			diff = diff + abs(M2[j+i*cM]-M1[j+i*cM]);
	return diff;
}

void NullMat(TYPE* M, int rM, int cM)
{
	for (int i=0; i<rM; i++)
		for (int j=0; j<cM; j++)
			M[j+i*cM] = 0;
}

void initialiseMat(TYPE* M, int rM, int cM)
{
	for (int i=0; i<rM; i++)
		for (int j=0; j<cM; j++)
			M[j+i*cM] = (TYPE)i+(TYPE)j*((j%3)-1);
}

void displayMat(TYPE* M, int rM, int cM)
{
	// Don't display large matrices
	if (rM > 5 || cM > 5)
		return;
	for (int i=0; i<rM; i++)
	{
		for (int j=0; j<cM; j++)
			print_info(to_string(M[j+i*cM])+" ");
		print_info("\n");
	}
}

void matMult(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
	for (int i=0; i<rA; i++)
	{
		for (int j=0; j<cB; j++)
		{
			TYPE Sum = 0;
			for (int k=0; k<cA; k++)
				Sum = Sum+A[k+i*cA]*B[j+k*cB];
			C[j+i*cB] = Sum;
		}
	}
}

void matMultOMP(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
#pragma omp parallel //num_threads(16)
	{
#pragma omp for
		for (int i=0; i<rA; i++)
		{
			for (int j=0; j<cB; j++)
			{
				TYPE Sum = 0;
				for (int k=0; k<cA; k++)
					Sum = Sum+A[k+i*cA]*B[j+k*cB];
				C[j+i*cB] = Sum;
			}
		}
	}
}

#ifdef __linux__
void matMultOACC(TYPE* C, TYPE* A, TYPE* B, int rA, int cA, int rB, int cB)
{
#pragma acc data copy(A,B,C)
	{
#pragma acc kernels
		for (int i=0; i<rA; i++)
		{
			for (int j=0; j<cB; j++)
			{
				TYPE Sum = 0;
				for (int k=0; k<cA; k++)
					Sum = Sum+A[k+i*cA]*B[j+k*cB];
				C[j+i*cB] = Sum;
			}
		}
	}
}
#endif
