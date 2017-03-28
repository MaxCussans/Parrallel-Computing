#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>

#include <CL/cl.hpp>
#include "Utils.h"

using namespace std;

void print_help() { 
	cerr << "Application usage:" << endl;

	cerr << "  -p : select platform " << endl;
	cerr << "  -d : select device" << endl;
	cerr << "  -l : list all platforms and devices" << endl;
	cerr << "  -h : print this message" << endl;
}


float* read(const char* filepath, int length)
{
	float* temp = new float[length];

	FILE* data = fopen(filepath, "r");

	for(int i = 0; i < length; i++)
	{
		fscanf(data, "%*s %*lf %*lf %*lf %*lf %f", &temp[i]);
	}
	fclose(data);

	return temp;
}





int main(int argc, char **argv) 
{
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	
	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	
	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		try {
			program.build();
		}
		//display kernel building errors
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - memory allocation
		//host - input


		const int datasize = 1873106;
		char* filepath = "temp_lincolnshire.txt";
		float* temperatures;
		temperatures = read(filepath, datasize);
		
		size_t vector_elements = datasize;//number of elements
		size_t vector_size = datasize*sizeof(int);//size in bytes

		//host - output

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);


		//Part 5 - device operations

		//5.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &temperatures[0]);

		//5.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "global_max");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_A);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);


		
			cout << temperatures[0] << endl;
		
	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	system("PAUSE");
	return 0;
}

