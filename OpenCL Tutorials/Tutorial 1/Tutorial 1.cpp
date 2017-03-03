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

void readFile(char* filepath, vector<string> place, vector<int> year, vector<int> month, vector<int> day, vector<string> time, vector <double> temperature)
{
	ifstream file;
	string buffer;
	string txt;

	if (filepath != nullptr)
	{
		file.open(filepath);
	}
	if (file.is_open())
	{
		while (getline(file, buffer))
		{
			//counter for columns in txt file
			short columnCount = 0;
			string tempPlace;
			int tempYear;
			int tempMonth;
			int tempDay;
			string tempTime;
			double tempTemperature;

			//Start at the beginning of each line
			for (int i = 0; i < buffer.size(); i++)
			{
				//ignore space characters
				if (buffer[i] != ' ')
				{
					//parse
					txt += buffer[i];
				}
				//if at place column
				else if (buffer[i] == ' ' && columnCount == 0)
				{
					//parse place
					tempPlace = txt;
					columnCount++;
					txt = "";
				}
				//if at year column
				else if (buffer[i] == ' ' && columnCount == 1)
				{
					//parse year
					tempYear = stoi(txt);
					columnCount++;
					txt = "";
				}
				//if at month column
				else if (buffer[i] == ' ' && columnCount == 2)
				{
					//parse month
					tempMonth = stoi(txt);
					columnCount++;
					txt = "";
				}
				//if at day column
				else if (buffer[i] == ' ' && columnCount == 3)
				{
					//parse day
					tempDay = stoi(txt);
					columnCount++;
					txt = "";
				}
				//if at time column
				else if (buffer[i] == ' ' && columnCount == 4)
				{
					//parse time
					tempTime = txt;
					columnCount++;
					txt = "";
				}
				//if at end of line
				else if (buffer[i] == ' ' && columnCount == 5)
				{
					//parse temperature
					tempTemperature = stod(txt);
					txt = "";
				}
			}
			//fill each vector with the corresponding values
			place.push_back(tempPlace);
			year.push_back(tempYear);
			month.push_back(tempMonth);
			day.push_back(tempDay);
			time.push_back(tempTime);
			temperature.push_back(tempTemperature);
		}
		//finished parsing
		file.close();
	}

}


int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
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
		vector<int> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 }; //C++11 allows this type of initialisation
		vector<int> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };

		vector<string> place;
		vector<int> year;
		vector<int> month;
		vector<int> day; 
		vector<string> time; 
		vector <double> temperature;
		char* filepath = "temp_lincolnshire.txt";
		readFile(filepath, place, year, month, day, time, temperature);
		
		size_t vector_elements = A.size();//number of elements
		size_t vector_size = A.size()*sizeof(int);//size in bytes

		//host - output
		vector<int> C(vector_elements);

		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//Part 5 - device operations

		//5.1 Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &A[0]);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0]);

		//5.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel_add = cl::Kernel(program, "add");
		kernel_add.setArg(0, buffer_A);
		kernel_add.setArg(1, buffer_B);
		kernel_add.setArg(2, buffer_C);

		queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);

		//5.3 Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);

		cout << "A = " << A << endl;
		cout << "B = " << B << endl;
		cout << "C = " << C << endl;
	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	return 0;
}

