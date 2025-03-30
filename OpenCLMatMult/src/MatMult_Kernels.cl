#ifdef _WIN32
#include <kernel.hpp>	// only for syntax highlighting
#endif
#define TYPE float
#define A(i,j) A[j+i*cB]
#define B(i,j) B[j+i*cB]
#define C(i,j) C[j+i*cB]

kernel void MatMultKernel(global TYPE* C, global TYPE* A, global TYPE* B, int rA, int cA, int rB, int cB)
{
	const uint n = get_global_id(0);
	const uint i = n/cB;
	const uint j = n%cB;

	TYPE Sum = 0;
	for (int k=0; k<cA; k++)
		Sum = Sum + A(i,k)* B(k,j);
	C(i,j) = Sum;
}

kernel void MatMultKernel2D(global TYPE* C, global TYPE* A, global TYPE* B, int rA, int cA, int rB, int cB)
{
	const uint i = get_global_id(0);
	const uint j = get_global_id(1);

	TYPE Sum = 0;
	for (int k=0; k<cA; k++)
		Sum = Sum + A(i,k)* B(k,j);
	C(i,j) = Sum;
}
