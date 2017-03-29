

__kernel void reduce_add1(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE); //ensure all values are copied to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}

}

__kernel void reduce_add2(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	int result = 0;
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//ensure all values are copied to local memory

	for(int i =1; i < N; i++)
	{ 		
			result += scratch[i];
	
			B[0] += result;
			barrier(CLK_LOCAL_MEM_FENCE);	
	}
	

}

__kernel void maximum(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//ensure all values are copied to local memory

	for(int i =1; i < N; i++)
	{ 
	//compare all values with current max
		if(scratch[lid] > B[0])
		{
			B[0] = scratch[lid];
			barrier(CLK_LOCAL_MEM_FENCE);
		}

	}

}

__kernel void minimum(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//ensure all values are copied to local memory

	for(int i =1; i < N; i++)
	{ 
	//compare all values with current min
		if(scratch[lid] < B[0])
		{
			B[0] = scratch[lid];
			barrier(CLK_LOCAL_MEM_FENCE);
		}

	}

}


//value - mean then square each value and divide by the sample THEN sqrt ~5.98

__kernel void variance(__global const int* A, __global int* B, int mean, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	
	barrier(CLK_LOCAL_MEM_FENCE);//ensure all values are copied to local memory

	while(id < N)
	{
		B[id] = scratch[lid] - mean;
	}
	

}


