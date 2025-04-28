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

kernel void MatMultKernelMulti2D(global TYPE* C, global TYPE* A, global TYPE* B, int rA, int cA, int rB, int cB, const int OffsetI)
{
	const uint i = get_global_id(0)+OffsetI;
	const uint j = get_global_id(1);

	if (i<rA && j<cB)
	{
		TYPE Sum = 0;
		for (int k=0; k<cA; k++)
			Sum = Sum + A(i,k)* B(k,j);
		C(i,j) = Sum;
	}
}

kernel void MatMultKernel2D(global TYPE* C, global TYPE* A, global TYPE* B, int rA, int cA, int rB, int cB)
{
	const uint i = get_global_id(0);
	const uint j = get_global_id(1);

	if (i<rA && j<cB)
	{
		TYPE Sum = 0;
		for (int k=0; k<cA; k++)
			Sum = Sum + A(i,k)* B(k,j);
		C(i,j) = Sum;
	}
}

#define BLKI 16
#define BLKJ 16
kernel void MatMultKernel2DLocal(global TYPE* C, global TYPE* A, global TYPE* B, int rA, int cA, int rB, int cB)//, const int BLKI, const int BLKJ, local TYPE* lA, local TYPE* lB)
{
	const uint i = get_global_id(0);
	const uint j = get_global_id(1);
	const uint li = get_local_id(0);
	const uint lj = get_local_id(1);

	__private TYPE SubSum = 0;
	__local TYPE lA[BLKI][BLKJ];
	__local TYPE lB[BLKI][BLKJ];

	int gridrows = rB/BLKI;
	//int gridcols = cA/BLKJ;

	for (int blockINo=0, blockJNo=0; blockINo<gridrows; blockINo++, blockJNo++)
	{
		int iOffset = get_group_id(0) * BLKI;
		int jOffset = get_group_id(1) * BLKJ;

		int ii = iOffset+li;
		int jj = blockJNo*BLKJ+lj;
		//lA[lj+li*BLKJ] = A[jj+ii*cA];
		lA[li][lj] = A[jj+ii*cA];

		ii = blockINo*BLKI+li;
		jj = jOffset+lj;
		//lB[lj+li*BLKJ] = B[jj+ii*cB];
		lB[li][lj] = B[jj+ii*cB];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k=0; k<BLKJ; k++)
			SubSum = SubSum + lA[li][k] * lB[k][lj];
			//SubSum = SubSum + lA[k+li*BLKJ] * lB[lj+k*BLKJ];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[j+i*cB] = SubSum;
}

kernel void MatMultKernel2DLocalPass(global TYPE* C, global TYPE* A, global TYPE* B, int rA, int cA, int rB, int cB, const int BLKII, const int BLKJJ, local TYPE* lA, local TYPE* lB)
{
	const uint i = get_global_id(0);
	const uint j = get_global_id(1);
	const uint li = get_local_id(0);
	const uint lj = get_local_id(1);

	__private TYPE SubSum = 0;

	int gridrows = rB/BLKII;
	//int gridcols = cA/BLKJ;

	for (int blockINo=0, blockJNo=0; blockINo<gridrows; blockINo++, blockJNo++)
	{
		int iOffset = get_group_id(0) * BLKII;
		int jOffset = get_group_id(1) * BLKJJ;

		int ii = iOffset+li;
		int jj = blockJNo*BLKJJ+lj;
		lA[lj+li*BLKJJ] = A[jj+ii*cA];

		ii = blockINo*BLKII+li;
		jj = jOffset+lj;
		lB[lj+li*BLKJJ] = B[jj+ii*cB];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int k=0; k<BLKJJ; k++)
			SubSum = SubSum + lA[k+li*BLKJJ] * lB[lj+k*BLKJJ];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[j+i*cB] = SubSum;
}
