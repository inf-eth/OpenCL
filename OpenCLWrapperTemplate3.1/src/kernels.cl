#ifdef _WIN32
#include <kernel.hpp>	// only for syntax highlighting
#endif

kernel void OpenCLWrapperKernel(global double* input, global double* output, const double Multiplier)
{
	// equivalent to "for(int i=0; i<N; i++) {", but executed in parallel
	const int i = get_global_id(0);
	output[i] = Multiplier * input[i];
}
