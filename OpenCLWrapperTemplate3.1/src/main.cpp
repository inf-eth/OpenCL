#include <opencl.hpp>

int main()
{
	// compile OpenCL C code for the fastest available device
	//Device device(select_device_with_most_flops(), "kernels.cl");
	Device device(select_device_with_id(0), "kernels.cl");	// This is useful if you want a particular device

	// size of vectors
	const uint N = 256u;
	// allocate memory on both host and device
	Memory<double> input(device, N);
	Memory<double> output(device, N);
	const double Multiplier = 2.0;

	// kernel that runs on the device
	Kernel OpenCLWrapperKernel(device, N, "OpenCLWrapperKernel", input, output, Multiplier);
	// default workgroup size can be changed by passing a dummy bool after size(s) followed by workgroup size.
	// For example, to change 1D workgroup size (from default of 64) to 16 you can write:
	// Kernel add_kernel(device, N, true, 16, "add_kernel", A, B, C);

	// initialize memory
	for(uint i=0u; i<N; i++)
		input[i] = (double)(rand()%100);

	print_info("Value before kernel execution: input[0] = "+to_string(input[0],2));

	// copy data from host memory to device memory
	input.write_to_device();

	// run add_kernel on the device
	OpenCLWrapperKernel.run();

	// copy data from device memory to host memory
	output.read_from_device();

	print_info("Value after kernel execution: output[0] = "+to_string(output[0],2));

	//wait();	// Might need this if executing directly from exe
	return 0;
}
