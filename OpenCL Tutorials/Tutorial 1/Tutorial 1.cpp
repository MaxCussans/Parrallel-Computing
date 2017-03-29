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
		
		size_t input_elements = temps.size();//number of input elements
		size_t input_size = datasize * sizeof(mytype);//size in bytes
		size_t nr_groups = input_elements / local_size;
		
		//host - output
		std::vector<mytype> B(1);
		std::vector<mytype> C(datasize);
		std::vector<mytype> D(datasize);
		std::vector<mytype> E(datasize);
		std::vector<mytype> F(datasize);
		std::vector<mytype> G(datasize);
		std::vector<mytype> H(1);
		std::vector<mytype> I(datasize);
		std::vector<mytype> J(1);
		int mean;

		size_t output_size = B.size() * sizeof(mytype);//size in bytes
		size_t output_sizeD = D.size() * sizeof(mytype);//size in bytes
		size_t output_sizeF = F.size() * sizeof(mytype);//size in bytes
		size_t output_sizeH = H.size() * sizeof(mytype);//size in bytes
		size_t output_sizeJ = J.size() * sizeof(mytype);//size in bytes
													   //device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		//Part 5 - device operations
		
		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &temps[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
		
															 //5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "reduce_add1");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, cl::Local(local_size * sizeof(mytype)));
		
		
		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		
		
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		mean = B[0] / datasize;



		cl::Buffer buffer_C(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_sizeD);

	
		queue.enqueueWriteBuffer(buffer_C, CL_TRUE, 0, input_size, &temps[0]);
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_sizeD);

		//variance
		cl::Kernel kernel_var = cl::Kernel(program, "squaredDifference");
		kernel_var.setArg(0, buffer_C);
		kernel_var.setArg(1, buffer_D);
		kernel_var.setArg(2, mean);
		kernel_var.setArg(3, datasize);


		queue.enqueueNDRangeKernel(kernel_var, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//COPY RESULTS
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_sizeD, &D[0]);
		
		

		cl::Buffer buffer_E(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_F(context, CL_MEM_READ_WRITE, output_sizeF);


		queue.enqueueWriteBuffer(buffer_E, CL_TRUE, 0, input_size, &D[0]);
		queue.enqueueFillBuffer(buffer_F, 0, 0, output_sizeF);

		
		kernel_add.setArg(0, buffer_E);
		kernel_add.setArg(1, buffer_F);
		kernel_add.setArg(2, cl::Local(local_size * sizeof(mytype)));


		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

		//COPY RESULTS
		queue.enqueueReadBuffer(buffer_F, CL_TRUE, 0, output_sizeD, &F[0]);

		cl::Buffer buffer_G(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_H(context, CL_MEM_READ_WRITE, output_sizeH);
		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_G, CL_TRUE, 0, input_size, &temps[0]);
		queue.enqueueFillBuffer(buffer_H, 0, 0, output_sizeH);

		cl::Kernel kernel_min = cl::Kernel(program, "minimum");
		kernel_min.setArg(0, buffer_G);
		kernel_min.setArg(1, buffer_H);
		kernel_min.setArg(2, cl::Local(local_size * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));


		queue.enqueueReadBuffer(buffer_H, CL_TRUE, 0, output_sizeH, &H[0]);

		cl::Buffer buffer_I(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_J(context, CL_MEM_READ_WRITE, output_sizeJ);
		//Part 5 - device operations

		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_I, CL_TRUE, 0, input_size, &temps[0]);
		queue.enqueueFillBuffer(buffer_J, 0, 0, output_sizeJ);

		cl::Kernel kernel_max = cl::Kernel(program, "maximum");
		kernel_max.setArg(0, buffer_I);
		kernel_max.setArg(1, buffer_J);
		kernel_max.setArg(2, cl::Local(local_size * sizeof(mytype)));

		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));


		queue.enqueueReadBuffer(buffer_J, CL_TRUE, 0, output_sizeJ, &J[0]);

			cout << "Minimum:" << H[0] << endl;
			
			cout << "Maximum:" << J[0] << endl;

			cout << "S.D:" << sqrt(F[0] / datasize) << endl;
			
			cout << "Mean:" << mean << endl;

			
		
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	system("PAUSE");
	return 0;
}


