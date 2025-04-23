#include <opencl.hpp>
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// Timing
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

int main()
{
	// compile OpenCL C code for the fastest available device
	Device device(select_device_with_most_flops(), "kernels.cl");
	//Device device(select_device_with_id(0), "kernels.cl");	// This is useful if you want a particular device
	
	// Get the flops of a particular device.
	// vector<Device_Info> test;
	// test = get_devices();
	// print_info("device 1: " + to_string(test[0].tflops) + ", device 2: " + to_string(test[1].tflops));

	// ####################################### 1D Addition ##########################################
	print_info("Doing 1D addition.");

	// size of vectors
	const uint N = 1024u;
	// allocate memory on both host and device
	Memory<float> A(device, N);
	Memory<float> B(device, N);
	Memory<float> C(device, N);

	// kernel that runs on the device
	Kernel add_kernel(device, N, "add_kernel", A, B, C);
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
	Clock.Start();
	add_kernel.run();
	Clock.Stop();
	print_info("add_kernel time: "+to_string(Clock.ElapsedTime(),3)+" ms.");

	// copy data from device memory to host memory
	C.read_from_device();

	print_info("Value after kernel execution: C[0] = "+to_string(C[0]));
	// #####################################################################################################
	// ######################################## 2D Addition ################################################
	print_info("Doing 2D additions.");

	// size of matrices
	const uint Rows = 1024u;
	const uint Cols = 1024u;
	Memory<float> A2D(device, Rows*Cols);
	Memory<float> B2D(device, Rows*Cols);
	Memory<float> C2D(device, Rows*Cols);

	// kernels that runs on the device
	// Notice we pass the product of Rows*Cols to sudo kernel and pass Rows and Cols as separate args to 2D kernel
	// For sudo kernel, there is a single global thread id in kernel function ranging from 0 to Rows*Cols
	// For 2D kernel there will be two global threads ids in kernel function over 0 to Rows and 0 to Cols
	Kernel add_kernelsudo2D(device, Rows*Cols, "add_kernelsudo2D", A2D, B2D, C2D, Rows, Cols);
	Kernel add_kernel2D(device, Rows, Cols, "add_kernel2D", A2D, B2D, C2D, Rows, Cols);

	// initialize memory
	for(uint i=0u; i<Rows; i++)
	{
		for (uint j=0u; j<Cols; j++)
		{
			A2D[j+i*Cols] = 3.0f;
			B2D[j+i*Cols] = 2.0f;
			C2D[j+i*Cols] = 1.0f;
		}
	}

	print_info("Value before sudo 2D kernel execution: C2D[0] = "+to_string(C2D[0]));

	// copy data from host memory to device memory
	A2D.write_to_device();
	B2D.write_to_device();

	// run add_kernel on the device
	Clock.Start();
	add_kernelsudo2D.run();
	Clock.Stop();
	print_info("add_kernelsudo2D time: "+to_string(Clock.ElapsedTime(),3)+" ms.");

	// copy data from device memory to host memory
	C2D.read_from_device();

	print_info("Value after sudo 2D kernel execution: C2D[0] = "+to_string(C2D[0]));

	// initialize memory
	for(uint i=0u; i<Rows; i++)
	{
		for (uint j=0u; j<Cols; j++)
		{
			A2D[j+i*Cols] = 3.0f;
			B2D[j+i*Cols] = 2.0f;
			C2D[j+i*Cols] = 1.0f;
		}
	}

	print_info("Value before 2D kernel execution: C2D[0] = "+to_string(C2D[0]));

	// copy data from host memory to device memory
	A2D.write_to_device();
	B2D.write_to_device();

	// run add_kernel on the device
	Clock.Start();
	add_kernel2D.run();
	Clock.Stop();
	print_info("add_kernel2D time: "+to_string(Clock.ElapsedTime(),3)+" ms.");

	// copy data from device memory to host memory
	C2D.read_from_device();

	print_info("Value after 2D kernel execution: C2D[0] = "+to_string(C2D[0]));

	//wait();	// Might need this if executing directly from exe
	return 0;
}
