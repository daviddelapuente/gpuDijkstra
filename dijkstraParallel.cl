__kernel void dijkstraParallel(
__global const int* G,
__global int* U,
__global int* F,
__global int* deltas,
__global int * d)
{
    // get index of the work item
	int index = get_global_id(0);
	int dlt=INT_MAX;

    // add the vector elements
    //z[index] = x[index] + y[index];
}