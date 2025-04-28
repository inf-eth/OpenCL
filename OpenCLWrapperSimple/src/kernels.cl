#ifdef _WIN32
#include <kernel.hpp>	// only for syntax highlighting
#endif

kernel void add_kernel(global const float* A, global const float* B, global float* C, const float x)
{
	// equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	const uint n = get_global_id(0);
	//C[n] = A[n]+B[n]+x;
	C[0] = C[0]+1;
}
