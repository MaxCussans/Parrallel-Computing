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


int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}


	


	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		//Part 4 - memory allocation
		//host - input
		const int datasize = 1873106;
		char* filepath = "temp_lincolnshire.txt";
		float* temperatures;
		temperatures = read(filepath, datasize);
		std::vector<mytype> temps(datasize, 0);

		for (int i = 0; i < datasize; i++)
		{
			temps[i] = temperatures[i];
		}



		size_t local_size = 128;
		cout << "Local Size = " << local_size << endl;
		
		size_t padding_size = datasize % local_size;
		
		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) 
		{
			//create an extra vector with neutral values
			std::vector<int> temperatures_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			temps.insert(temps.end(), temperatures_ext.begin(), temperatures_ext.end());
		}
		
		size_t input_elements = datasize;//number of input elements
		size_t input_size = datasize * sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;
		
		//host - output
		std::vector<mytype> B(input_elements);
		size_t output_size = B.size() * sizeof(mytype);//size in bytes
		
													   //device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		
		//Part 5 - device operations
		
		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &temps[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		
															 //5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_max = cl::Kernel(program, "maximum");
		kernel_max.setArg(0, buffer_A);
		kernel_max.setArg(1, buffer_B);
		
		
		queue.enqueueNDRangeKernel(kernel_max, cl::NDRange(input_elements), cl::NDRange(local_size));
		
		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

			cout << B[0] << endl;
		
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	system("PAUSE");
	return 0;
}


