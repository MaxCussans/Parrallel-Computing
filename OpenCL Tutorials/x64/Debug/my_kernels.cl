
//__kernel void reduce_add(__global float* A) 
//{
//	int id = get_global_id(0);
//	int N = get_global_size(0);
//
//	for (int i = 1; i < N; i*=2) 
//	{ 
//		if (id % (i*2) == 0)
//		{ 	
//			A[id] += A[id+i];
//		}
//		barrier(CLK_GLOBAL_MEM_FENCE);
//	}
//}

__kernel void reduce_add(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}

}




