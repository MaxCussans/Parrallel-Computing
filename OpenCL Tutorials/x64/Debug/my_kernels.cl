
__kernel void reduce_add(__global float* A) 
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = 1; i < N; i*=2) 
	{ 
		if (id % (i*2) == 0)
		{ 	
			A[id] += A[id+i];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void maximum(__global int* A, __global int* B)
{
	int id = get_global_id(0);
	int N = get_local_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE); //ensure all values are copied before continuing

	if(id < N)
	{
		if(A[id] < B[0])
		{
			B[0] = A[id];
		}
	}
}


