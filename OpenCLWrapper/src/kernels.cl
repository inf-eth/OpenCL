#ifdef _WIN32
#include <kernel.hpp>	// only for syntax highlighting
#endif
#define TYPE float
#define A(i,j) A[j+i*Cols]
#define B(i,j) B[j+i*Cols]
#define C(i,j) C[j+i*Cols]

kernel void add_kernel(global TYPE* A, global TYPE* B, global TYPE* C)
{
	// equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	const uint n = get_global_id(0);
	C[n] = A[n]+B[n];
}

kernel void add_kernelsudo2D(global TYPE* A, global TYPE* B, global TYPE* C, uint Rows, uint Cols)
{
	// equivalent to
	// for(uint i=0u; i<Rows; i++)
	// 	   for (uint j=0u; j<Cols; j++)
	
	const uint n = get_global_id(0);
	const uint i = n/Cols;
	const uint j = n%Cols;

	C(i,j) = A(i,j)+B(i,j);
}

kernel void add_kernel2D(global TYPE* A, global TYPE* B, global TYPE* C, uint Row, uint Cols)
{
	// equivalent to
	// for(uint i=0u; i<Rows; i++)
	// 	   for (uint j=0u; j<Cols; j++)

	const uint i = get_global_id(0);
	const uint j = get_global_id(1);
	C(i,j) = A(i,j)+B(i,j);
}
