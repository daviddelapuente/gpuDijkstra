#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "CL/opencl.h"
#include <vector>
#include <fstream>
#include <ctime>
#include <chrono>
#include "AOCLUtils/aocl_utils.h"



using namespace aocl_utils;
using namespace std::chrono;

std::ofstream myfile;
// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
cl_device_id device; // num_devices elements
cl_context context = NULL;
cl_command_queue queue; // num_devices elements
cl_program program = NULL;
cl_kernel initKernel;
cl_kernel relaxKernel;
cl_kernel updateKernel;
#if USE_SVM_API == 0
cl_mem input_G_buf;
cl_mem dijkstraInfo_buf;
#endif /* USE_SVM_API == 0 */

// Problem data.
std::unique_ptr<int[]> G;
std::unique_ptr<int[]> d;
std::unique_ptr<int[]> U;
std::unique_ptr<int[]> F;
std::unique_ptr<int[]> delta;
std::unique_ptr<int[]> dijkstraInfo;

std::unique_ptr<int[]> initConectGraph(int n,int maxEdgeValue);
std::unique_ptr<int[]> initGraphZero(int n);
std::unique_ptr<int[]> dijkstraCPU(int* G,int source,int n);
std::unique_ptr<int[]> dijkstraCPU2(int* G, int source, int n);
std::unique_ptr<int[]> dijkstraGPU(int source, int n);
std::unique_ptr<int[]> fastInit(int n);

int minDistance(int* dist, bool* sptSet, int n);
std::unique_ptr<int[]> initVec(int n,int x);

// Function prototypes
float rand_float2();
bool init_opencl(int n);

void cleanup();


//run experiments

void runProblemInGpu();
void runProblemInCpu();

void runProblemInCpu2();
void initDijkstra();
void relaxDijkstra(int n);
int computeTmin(int n);
void updateDijkstra(int n);
void openFile(std::string filename);
void closeFile();
void write(std::string text);
void writeTime(int i, int n, duration<double, std::milli> time_span);
int mode = 2;
// Entry point.
int main(int argc, char** argv) {
    if (mode == 0) {
        //run the problem in the CPU
        runProblemInCpu();
    }
    else if (mode == 1) {
        //run the problem in the CPU version 2
        runProblemInCpu2();
    }
    else if (mode == 2) {
        //run the problem in the GPU with CPU version 2 algorithm
        runProblemInGpu();
    }
    
    return 0;
}


//this create a graph with n nodes and 0 edges
std::unique_ptr<int[]> initGraphZero(int n) {
    std::unique_ptr<int[]> Z(new int[n * n]);
    for (int i = 0; i < n * n; i++) {
        Z[i] = 0;
    }
    return Z;
}



//return the adjacent matrix for a random Graph.
std::unique_ptr<int[]> initConectGraph(int n,int maxEdgeValue) {
    //first we init the graph G with no edges
    std::unique_ptr<int[]> G = initGraphZero(n);

    //first we create a conected graph with n-1 edges
    //we asume that first node is in the graph
    for (int i = 1; i < n; i++) {
        //we conect the i node with some edge j in the conected graph (this is rand()%i)
        int j = rand() % i;
        //set the value of the edge to something grater than 0;
        G[j + i * n] = rand() % maxEdgeValue + 1;
    }

    //now we will add edges randomly between nodes
    //if G is conected, add a random edge will keep G conected
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //we filter G[i][i] cuz we dont want auto edges
            if (i != j) {
                //with a 50% of probability we add the edge to the graph
                if ((rand() % 100) > 50) {
                    G[j + i * n] = rand() % maxEdgeValue + 1;
                }
            }
        }
    }

    return G;
}


// code adapted from https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/ 
int minDistance(int* dist, bool* sptSet, int n){
    // Initialize min value 
    int min = INT_MAX, min_index;
    for (int v = 0; v < n; v++) {
        if (sptSet[v] == false && dist[v] <= min) {
            min = dist[v], min_index = v;
        }
    }
    return min_index;
}

std::unique_ptr<int[]> dijkstraCPU(int* G,int src ,int n) {
    //distances[i] hold the shortest distance between src and i
    std::unique_ptr<int[]> distances(new int[n]);
    
    //sptSet[i] will be true if vertex i is included in shortest path tree or shortest distance from src to i is finalized 
    std::unique_ptr<bool[]> sptSet(new bool[n]);
    
    // Initialize all distances as INFINITE and stpSet[] as false 
    for (int i = 0; i < n; i++) {
        distances[i] = INT_MAX;
        sptSet[i] = false;
    }

    distances[0] = 0;

    // Find shortest path for all vertices 
    for (int count = 0; count < n - 1; count++){

        // Pick the minimum distance vertex from the set of vertices not 
        // yet processed. u is always equal to src in the first iteration. 
        int u = minDistance(distances.get(), sptSet.get(),n);

        // Mark the picked vertex as processed 
        sptSet[u] = true;

        // Update dist value of the adjacent vertices of the picked vertex. 
        for (int v = 0; v < n; v++) {
            // Update dist[v] only if is not in sptSet, there is an edge from 
            // u to v, and total weight of path from src to  v through u is 
            // smaller than current value of dist[v] 
            if (!sptSet[v] && G[v+n*u] && distances[u] != INT_MAX && distances[u] + G[v+n*u] < distances[v]) {
                distances[v] = distances[u] + G[v+n*u];
            }
        }
    }
    
    return distances;
}

std::unique_ptr<int[]> dijkstraCPU2(int* G, int src, int n) {

    std::unique_ptr<int[]> U = initVec(n, 1);
    U[0] = 0;

    std::unique_ptr<int[]> F = initVec(n, 0);
    F[0] = 1;

    std::unique_ptr<int[]> delta = initVec(n, INT_MAX);
    delta[0] = 0;

    int T = INT_MAX;

    int ucount = 1;


    std::unique_ptr<int[]> g = initVec(n, INT_MAX);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (G[j + i * n] > 0) {
                g[i] = std::min(g[i], G[j + i*n]);
            }
        }

    }

    //parallel dijstra
    while (ucount < n) {


        //relaxation

        for (int i = 0; i < n; i++) {
            if (F[i] == 1) {
                for (int j = 0; j < n; j++) {
                    //son vecinos
                    if (G[j + i * n] > 0) {
                        if (U[j] == 1) {
                            delta[j] = std::min(delta[j], delta[i] + G[j + i * n]);
                        }
                    }
                }
            }
        }

        //T min

        T = INT_MAX;
        for (int i = 0; i < n; i++) {
            if (U[i] == 1) {
                //calculamos DELTAu=minimo de las aristas de u
                if (delta[i] != INT_MAX) {
                    T = std::min(T, g[i] + delta[i]);

                }
            }
        }

        
        //update

        for (int i = 0; i < n; i++) {
            F[i] = 0;
            if (U[i] == 1 && delta[i] <= T) {
                U[i] = 0;
                F[i] = 1;
            }
        }

        ucount = 0;
        for (int i = 0; i < n; i++) {
            if (U[i] == 0) {
                ucount++;
            }
        }

    }

    return delta;
}

void openFile(std::string filename) {
    myfile.open(filename);
}

void closeFile() {
    myfile.close();
}

void write(std::string text) {
    myfile << text;
    myfile << std::endl;
}

void writeTime(int i, int n, duration<double, std::milli> time_span) {
    //myfile << i << ") " << "numero de elementos=" << n << "  tiempo=" << time_span.count();
    //myfile << std::endl;
    myfile << time_span.count() << "," << std::endl;
}

void runProblemInCpu() {
    std::cout << "runing problem in CPU" << std::endl;
    std::string file = "cpu1";
    openFile(file);
    write("inicia analisis en CPU alg1:");

    int k = 40;
    for (int i = 0; i < k; i++) {
        high_resolution_clock::time_point t1;
        high_resolution_clock::time_point t2;


        int n = 100+100*i;
        int maxEdgeValue = 100;
        G = initConectGraph(n, maxEdgeValue);

        t1 = high_resolution_clock::now();
        int source = 0;
        d = dijkstraCPU(G.get(), source, n);

        t2 = high_resolution_clock::now();
        duration<double, std::milli> time_span = t2 - t1;
        writeTime(i+1, n, time_span);
    }
    



    closeFile();
}


void runProblemInCpu2() {
    std::cout << "runing problem in CPU version 2" << std::endl;
    std::string file = "cpu2";
    openFile(file);
    write("inicia analisis en CPU alg2:");
    
    int k = 40;
    for (int i = 0; i < k; i++) {
        high_resolution_clock::time_point t1;
        high_resolution_clock::time_point t2;

        int n = 100 + 100 * i;
        int maxEdgeValue = 100;
        G = initConectGraph(n, maxEdgeValue);

        t1 = high_resolution_clock::now();
        int source = 0;
        d = dijkstraCPU2(G.get(), source, n);

        t2 = high_resolution_clock::now();
        duration<double, std::milli> time_span = t2 - t1;
        writeTime(i + 1, n, time_span);
    }

    closeFile();
}


void runProblemInGpu() {
    std::cout << "runing problem in GPU" << std::endl;

    std::string file = "gp1";
    openFile(file);
    write("inicia analisis en gpu:");
    int k = 40;
    for (int i = 0; i < k; i++) {
        high_resolution_clock::time_point t1;
        high_resolution_clock::time_point t2;

        int n = 100 + 100 * i;
        int maxEdgeValue = 100;
        G = initConectGraph(n, maxEdgeValue);
        dijkstraInfo = fastInit(n);

        t1 = high_resolution_clock::now();
        int source = 0;
        init_opencl(n);
        dijkstraGPU(source, n);
        t2 = high_resolution_clock::now();
        duration<double, std::milli> time_span = t2 - t1;
        writeTime(i + 1, n, time_span);
        cleanup();
    }

   
    closeFile();
}


/////// HELPER FUNCTIONS ///////
cl_platform_id findPlatform2() {
    cl_int status;
    // Get number of platforms.
    cl_uint num_platforms;
    status = clGetPlatformIDs(0, NULL, &num_platforms);
    // Get a list of all platform ids.
    scoped_array<cl_platform_id> pids(num_platforms);
    status = clGetPlatformIDs(num_platforms, pids, NULL);
    return pids[0];
}


cl_device_id* getDevices2(cl_platform_id pid, cl_device_type dev_type, cl_uint* num_devices) {
    cl_int status;
    status = clGetDeviceIDs(pid, dev_type, 0, NULL, num_devices);
    cl_device_id* dids = new cl_device_id[*num_devices];
    status = clGetDeviceIDs(pid, dev_type, *num_devices, dids, NULL);
    return dids;
}


std::string getDeviceName2(cl_device_id did) {
  cl_int status;
  size_t sz;
  status = clGetDeviceInfo(did, CL_DEVICE_NAME, 0, NULL, &sz);
  scoped_array<char> name(sz);
  status = clGetDeviceInfo(did, CL_DEVICE_NAME, sz, name, NULL);
  return name.get();
}


std::string getBoardBinaryFile2(const char* prefix, cl_device_id device) {
    // First check if <prefix>.aocx exists. Use it if it does.
    std::string file_name = std::string(prefix) + ".cl";
    // Now get the name of the board. For Intel(R) FPGA SDK for OpenCL(TM) boards,
    // the name of the device is presented as:
    //  <board name> : ...
    std::string device_name = getDeviceName2(device);
    // Now search for the " :" in the device name.
    size_t end = device_name.find(" :");
    if (end != std::string::npos) {
        std::string board_name(device_name, 0, end);

        // Look for a AOCX with the name <prefix>_<board_name>_<version>.aocx.
        file_name = std::string(prefix) + "_" + board_name + "_" + "161" + ".aocx";
    }
    // At this point just use <prefix>.aocx. This file doesn't exist
    // and this should trigger an error later.
    return std::string(prefix) + ".aocx";
}

void oclContextCallback2(const char* errinfo, const void*, size_t, void*) {
    printf("Context callback: %s\n", errinfo);
}


unsigned char* loadBinaryFile2(const char* file_name, size_t* size) {
    // Open the File
    FILE* fp;
    if (fopen_s(&fp, file_name, "rb") != 0) {
        return NULL;
    }
    // Get the size of the file
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    // Allocate space for the binary
    unsigned char* binary = new unsigned char[*size];
    // Go back to the file start
    rewind(fp);
    // Read the file into the binary
    if (fread((void*)binary, *size, 1, fp) == 0) {
        delete[] binary;
        fclose(fp);
        return NULL;
    }
    return binary;
}

cl_program createProgramFromBinary2(cl_context context, const char* binary_file_name, const cl_device_id* devices, unsigned num_devices) {
    // Early exit for potentially the most common way to fail: AOCX does not exist.
    // Load the binary.
    size_t binary_size;
    scoped_array<unsigned char> binary(loadBinaryFile2(binary_file_name, &binary_size));
    scoped_array<size_t> binary_lengths(num_devices);
    scoped_array<unsigned char*> binaries(num_devices);
    for (unsigned i = 0; i < num_devices; ++i) {
        binary_lengths[i] = binary_size;
        binaries[i] = binary;
    }
    cl_int status;
    scoped_array<cl_int> binary_status(num_devices);
    cl_program program = clCreateProgramWithBinary(context, num_devices, devices, binary_lengths,
        (const unsigned char**)binaries.get(), binary_status, &status);
    return program;
}

int ReadSourceFromFile2(const char* fileName, char** source, size_t* sourceSize){
    int errorCode = CL_SUCCESS;
    FILE* fp = NULL;
    fopen_s(&fp, fileName, "rb");
    if (fp == NULL){
        errorCode = CL_INVALID_VALUE;
    }
    else {
        fseek(fp, 0, SEEK_END);
        *sourceSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        *source = new char[*sourceSize];
        if (*source == NULL){
            errorCode = CL_OUT_OF_HOST_MEMORY;
        }
        else {
            fread(*source, 1, *sourceSize, fp);
        }
    }
    return errorCode;
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float2() {
    return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}















// Initializes the OpenCL objects.
bool init_opencl(int n) {
    cl_int status;

    printf("Initializing OpenCL\n");

    // Get the OpenCL platform.
    platform = findPlatform2();
    if (platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    }

    // Query the available OpenCL device.
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;

    devices.reset(getDevices2(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    device = devices[0];


    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback2, NULL, &status);

    // Create the program for all device. Use the first device as the
    // representative device (assuming all device are of the same type).
    
    //init program
    char* source[3];


    size_t src_size[3];


    ReadSourceFromFile2("initDijkstra.cl", &source[0], &src_size[0]);
    ReadSourceFromFile2("relaxDijkstra.cl",&source[1],&src_size[1]);
    ReadSourceFromFile2("updateDijkstra.cl", &source[2], &src_size[2]);

    program = clCreateProgramWithSource(context, 3, (const char**)source, src_size, &status);
    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);



    // Command queue.
    queue = clCreateCommandQueueWithProperties(context, device,NULL, &status);

    // Kernels.

    //init Kernel
    initKernel = clCreateKernel(program, "initDijkstra", &status);
    printf("Using kernel: %s\n", "initDijkstra");
        
    //relax kernel
    relaxKernel = clCreateKernel(program, "relaxDijkstra", &status);
    printf("Using kernel: %s\n", "relaxDijkstra");

    //update kernel
    updateKernel = clCreateKernel(program, "updateDijkstra", &status);
    printf("Using kernel: %s\n", "updateDijkstra");


    // Input buffers.

    input_G_buf= clCreateBuffer(context, CL_MEM_READ_ONLY,
        n * n * sizeof(int), NULL, &status);

    dijkstraInfo_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
        (4 * n + 2) * sizeof(int), NULL, &status);

    return true;
}

std::unique_ptr<int[]> initVec(int n,int x) {
    std::unique_ptr<int[]> distances(new int[n]);
    for (int i = 0; i < n; i++) {
        distances[i] = x;
    }
    return distances;
}

std::unique_ptr<int[]> fastInit(int n) {
    std::unique_ptr<int[]> distances(new int[4*n+2]);
    distances[0] = n;
    return distances;
}

void initDijkstra(int source, int n) {
    cl_int status;
    cl_event kernel_event;
    cl_event finish_event;
    cl_event write_event[2];

    status = clEnqueueWriteBuffer(queue, input_G_buf, CL_FALSE,
        0, n * n * sizeof(int), G.get(), 0, NULL, &write_event[0]);

    status = clEnqueueWriteBuffer(queue, dijkstraInfo_buf, CL_FALSE,
        0, (4 * n + 2 )* sizeof(int), dijkstraInfo.get(), 0, NULL, &write_event[1]);



    unsigned argi = 0;
    status = clSetKernelArg(initKernel, argi++, sizeof(cl_mem), &input_G_buf);
    status = clSetKernelArg(initKernel, argi++, sizeof(cl_mem), &dijkstraInfo_buf);

    const size_t global_work_size = n;
    printf("Launching for device %d (%zd elements)\n", 0, global_work_size);

    status = clEnqueueNDRangeKernel(queue, initKernel, 1, NULL,
        &global_work_size, NULL, 2, write_event, &kernel_event);

    // Read the result.
    status = clEnqueueReadBuffer(queue, dijkstraInfo_buf, CL_TRUE,
        0, (4 * n + 2) * sizeof(int), dijkstraInfo.get(), 1, &kernel_event, &finish_event);


    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseEvent(kernel_event);
    //clReleaseEvent(finish_event);
}

void relaxDijkstra(int n) {
    cl_int status;
    cl_event kernel_event;
    cl_event finish_event;
    cl_event write_event[2];

    status = clEnqueueWriteBuffer(queue, input_G_buf, CL_FALSE,
        0, n * n * sizeof(int), G.get(), 0, NULL, &write_event[0]);


    status = clEnqueueWriteBuffer(queue, dijkstraInfo_buf, CL_FALSE,
        0, (4 * n + 2) * sizeof(int), dijkstraInfo.get(), 0, NULL, &write_event[1]);

    unsigned argi = 0;
    status = clSetKernelArg(relaxKernel, argi++, sizeof(cl_mem), &input_G_buf);
    status = clSetKernelArg(relaxKernel, argi++, sizeof(cl_mem), &dijkstraInfo_buf);
    

    const size_t global_work_size = n;
    

    status = clEnqueueNDRangeKernel(queue, relaxKernel, 1, NULL,
        &global_work_size, NULL, 2, write_event, &kernel_event);

    // Read the result.
    status = clEnqueueReadBuffer(queue, dijkstraInfo_buf, CL_TRUE,
        0, (4 * n + 2) * sizeof(int), dijkstraInfo.get(), 1, &kernel_event, &finish_event);


    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
    clReleaseEvent(kernel_event);
    clReleaseEvent(finish_event);
}

int computeTmin(int n) {

    int T = INT_MAX;

    for (int i = 0; i < n; i++) {
        //U[i]==1
        if (dijkstraInfo[i*4+2] == 1) {
            //delta[i]!=int_max
            if (dijkstraInfo[i*4+2+2] != INT_MAX) {
                T = std::min(T, dijkstraInfo[i * 4 + 3 + 2] + dijkstraInfo[i * 4 + 2 + 2]);
            }
        }
    }

    return T;
}

void updateDijkstra(int n) {
    cl_int status;
    cl_event kernel_event;
    cl_event finish_event;
    cl_event write_event[1];

    status = clEnqueueWriteBuffer(queue, dijkstraInfo_buf, CL_FALSE,
        0, (4 * n + 2) * sizeof(int), dijkstraInfo.get(), 0, NULL, &write_event[0]);

    unsigned argi = 0;
    status = clSetKernelArg(updateKernel, argi++, sizeof(cl_mem), &dijkstraInfo_buf);

    const size_t global_work_size = n;


    status = clEnqueueNDRangeKernel(queue, updateKernel, 1, NULL,
        &global_work_size, NULL, 1, write_event, &kernel_event);

    // Read the result.
    status = clEnqueueReadBuffer(queue, dijkstraInfo_buf, CL_TRUE,
        0, (4 * n + 2) * sizeof(int), dijkstraInfo.get(), 1, &kernel_event, &finish_event);


    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(kernel_event);
    clReleaseEvent(finish_event);
}



std::unique_ptr<int[]> dijkstraGPU(int source, int n) {


    //initialize dijkstra

    initDijkstra(source, n);

    //compute for every node

    int T = INT_MAX;
    int ucount = 0;

    while (ucount < n) {

        relaxDijkstra(n);
        T=computeTmin(n);
        dijkstraInfo[1] = T;
        updateDijkstra(n);
     
        ucount = 0;
        for (int i = 0; i < n; i++) {
            if (dijkstraInfo[i * 4 + 2] == 0) {
                ucount++;
            }
        }


    }



    return NULL;
}







// Free the resources allocated during initialization
void cleanup() {
    
    if (initKernel) {
        clReleaseKernel(initKernel);
    }

    if (relaxKernel) {
        clReleaseKernel(relaxKernel);
    }

    if (updateKernel) {
        clReleaseKernel(updateKernel);
    }


    if (queue) {
        clReleaseCommandQueue(queue);
    }

    if (dijkstraInfo_buf) {
       clReleaseMemObject(dijkstraInfo_buf);
    }
    
    if (input_G_buf) {
        clReleaseMemObject(input_G_buf);
    }

    if (program) {
        clReleaseProgram(program);
    }

    if (context) {
        clReleaseContext(context);
    }
}
