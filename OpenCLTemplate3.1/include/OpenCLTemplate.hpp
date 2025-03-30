#ifndef OpenCLTemplate_H_
#define OpenCLTemplate_H_

#define PRECISION double

#include <string>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl.h>
#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

class COpenCLTemplate
{
private:
	// Size of problem domain
	const unsigned int Width;

	// Host data arrays.
	PRECISION* input;
	PRECISION* output;

	// Scalar.
	const PRECISION Multiplier;

	// Device data arrays.
	cl_mem d_input;
	cl_mem d_output;

	// ============ OPENCL related parameters ===========
	// OPENCL context/device/program
	cl_context context;
	cl_device_id *devices;
	cl_command_queue commandQueue;
	cl_program program;
	cl_kernel kernel;
	// ==================================================

	// Timer variables.
	typedef std::chrono::high_resolution_clock clock;
	std::chrono::time_point<clock> tStart;
	std::chrono::time_point<clock> tEnd;
	PRECISION tDelta;
	bool tPaused;

public:
	COpenCLTemplate(unsigned int=256U, const PRECISION=2.0);	// Default width of problem is 256 and multiplier is 2.

	// Memory allocation and initialisation.
	int AllocateMemoryCPU();
	int InitialiseCPU();
	int InitialiseCL();			// Search and allocate a device.
	int AllocateMemoryGPU();
	int InitialiseCLKernelsGPU(); // Build/attach kernels to respective kernel functions and set arguments.
	int RunCLKernels();

	// Complete run encapsulating all the sub-functions.
	int CompleteRun();

	// Timing.
	void StartTimer();
	void StopTimer();
	void ResetTimer();
	PRECISION GetElapsedTime();

	std::string convertToString(const char * filename);
	int DisplayArray(const unsigned int, PRECISION*);

	int SafeCall(cl_int, const char []=NULL);

	int CleanupCPU();
	int CleanupCL();
	int CleanupGPU();
	~COpenCLTemplate ();
};
#endif // #ifndef OpenCLTemplate_H_
