#include <opencl.hpp>

int main()
{
	// compile OpenCL C code for the fastest available device
	//Device device(select_device_with_most_flops(), "kernels.cl");
	Device device(select_device_with_id(0), "kernels.cl");	// This is useful if you want a particular device

	// size of vectors
	const uint N = 1024u;
	// allocate memory on both host and device
	Memory<float> A(device, N);
	Memory<float> B(device, N);
	Memory<float> C(device, N);
	float x = 5;

	// kernel that runs on the device
	Kernel add_kernel(device, N, "add_kernel", A, B, C, x);
	// default workgroup size can be changed by passing a dummy bool after size(s) followed by workgroup size.
	// For example, to change 1D workgroup size (from default of 64) to 16 you can write:
	// Kernel add_kernel(device, N, true, 16, "add_kernel", A, B, C);

	// initialize memory
	for(uint n=0u; n<N; n++)
	{
		A[n] = 3.0f;
		B[n] = 2.0f;
		C[n] = 1.0f;
	}

	print_info("Value before kernel execution: C[0] = "+to_string(C[0]));

	// copy data from host memory to device memory
	A.write_to_device();
	B.write_to_device();

	// run add_kernel on the device
	add_kernel.run();

	// copy data from device memory to host memory
	C.read_from_device();

	print_info("Value after kernel execution: C[0] = "+to_string(C[0]));

	//wait();	// Might need this if executing directly from exe
	return 0;
}
