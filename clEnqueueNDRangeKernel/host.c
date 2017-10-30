#pragma warning (disable : 4996)

// Import libraries
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <sys\timeb.h>
#include <conio.h>

// Defining variables
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
#define MATH_PI 3.14159265358979323846

// Device and platform info variables
int i, j;
char* value;
size_t valueSize;
cl_uint platformCount;
cl_platform_id* platforms;
cl_uint deviceCount;
cl_device_id* devices;
cl_uint maxComputeUnits;

// Debugging strings
cl_int ret;
char string[MEM_SIZE];

// Source file variables
FILE *fp;
char fileName[] = "./propagate.cl";
char *source_str;
size_t source_size;

// Simulation parameters
float rho = 2000;               // particle density
float D = 0.001;     					  // particle diameter
float mu = 0.0000193; 					// fluid viscosity
float g = -0.001;   					  // gravitational acceleration
float flowMag = 0.14;   				// maximum velocity of the flow field
float vortexFreq = 1; 					// scalar to define the vortex density
float tauMultiplier = 1; 				// scalar to change the tau value
float timeDurationFactor = 99;  // defines length of simulation

int NUMPART = 1000000;          // number of particles
int LOCAL_SIZE = 1;             // local work size
int log_step = 50;              // log files for intermediate particle position and velocities written every 2*log_step

// GPU and host buffers for particle position and velocity
float *buffer;
float *hpos;
float *hvel;
cl_mem gposold;
cl_mem gvelold;
cl_mem gposnew;
cl_mem gvelnew;

// GPU and host buffers for simulation variables
float *h_tau;
float *h_tstep;
cl_mem g_tau;
cl_mem g_tstep;


// PROGRAM BEGINS HERE
int main() {

	// load source code containing kernel
	fopen_s(&fp, fileName, "r");
	if (!fp) {
			fprintf(stderr, "Failed to load kernel.\n");
			exit(1);
		}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* THIS PART PRINTS ALL THE AVAILABLE PLATFORMS AND COMPUTE DEVICES OF A SYSTEM */
	clGetPlatformIDs(0, NULL, &platformCount);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);

	if (platformCount == 0) {
		printf("No OpenCL platforms available!\n");
	}

	for (i = 0; i < platformCount; i++) {
		printf("--------------\n");
		printf(" PLATFORM %d \n", i);
		printf("--------------\n");
		// get all devices
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
		// for each device print critical attributes
		for (j = 0; j < deviceCount; j++) {
			// print device name
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
			printf("%d. Device: %s\n", j + 1, value);
			free(value);

			// print hardware device version
			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
			printf("   %d.%d Hardware version: %s\n", j + 1, 1, value);
			free(value);

			// print software driver version
			clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
			printf("   %d.%d Software version: %s\n", j + 1, 2, value);
			free(value);

			// print c version supported by compiler for device
			clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
			printf("   %d.%d OpenCL C version: %s\n", j + 1, 3, value);
			free(value);

			// device type
			cl_device_type devtype;
			clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(devtype), &devtype, NULL);
			if (devtype == CL_DEVICE_TYPE_CPU)		printf("   %d.%d Device type: CPU\n", j + 1, 4);
			else if (devtype == CL_DEVICE_TYPE_GPU)		printf("   %d.%d Device type: GPU\n", j + 1, 4);
			else if (devtype == CL_DEVICE_TYPE_ACCELERATOR)	printf("   %d.%d Device type: ACCELERATOR\n", j + 1, 4);
			else						printf("   %d.%d Device type: UNKNOWN\n", j + 1, 4);

			// print parallel compute units
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
			printf("   %d.%d Parallel compute units: %d\n", j + 1, 5, maxComputeUnits);

			// max work group size
			size_t devcores2;
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(devcores2), &devcores2, NULL);
			printf("   %d.%d Max work group size: %u\n", j + 1, 6, (cl_uint)devcores2);

			// max work item size
			size_t work_item_size[3] = { 0, 0, 0 };
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_item_size), &work_item_size, NULL);
			printf("   %d.%d Max work items: (%d, %d, %d)\n", j + 1, 7, work_item_size[0], work_item_size[1], work_item_size[2]);

			// max clock frequency
			cl_uint devfreq;
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(devfreq), &devfreq, NULL);
			printf("   %d.%d Max clock frequency: %u\n", j + 1, 8, devfreq);

			// memory in MB
			cl_ulong devmem;
			clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(devmem), &devmem, NULL);
			cl_uint devmemMB = (cl_uint)(devmem / 1000000);
			printf("   %d.%d Device global memory (MB): %u\n", j + 1, 9, devmemMB);

			// double precision support?
			cl_int supported;
			clGetDeviceInfo(devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(supported), &supported, NULL);
			printf("   %d.%d Double precision supported: %s\n\n", j + 1, 10, supported ? "YES" : "NO");

		}
		free(devices);
	}
	free(platforms);
	printf("Press any key to continue.\n")
	getch();

	// select OpenCL platform and device
	clGetPlatformIDs(0, NULL, &platformCount);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);

	// Change the first argument of clGetDeviceIDs to the desired platform from initial system diagnosis, default is set to platforms[0]
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

	// Change the first argument of clGetDeviceInfo to the desired device from initial system diagnosis, default is set to devices[0]
	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(devices[0], CL_DEVICE_NAME, valueSize, value, NULL);
	printf("Default computing device selected: %s\n\n", value);
	free(value);

	// Create OpenCL context
	cl_context context = NULL;
	context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, &ret);
	printf("[INIT] Create OpenCL context: ");
	if ((int)ret == 0) {
			printf("SUCCESS\n");
		}
	else {
		printf("FAILED\n");
		_getch();
		return 1;
	}

	// Create command queue
	cl_command_queue queue = NULL;
	queue = clCreateCommandQueue(context, devices[0], 0, &ret);
	printf("[INIT] Create command queue: ");
	if ((int)ret == 0) {
			printf("SUCCESS\n");
		}
	else {
		printf("FAILED\n");
		_getch();
		return 1;
	}

	// Create memory buffers
	gposold = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMPART * sizeof(cl_float4), NULL, &ret);
	gvelold = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMPART * sizeof(cl_float4), NULL, &ret);
	gposnew = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMPART * sizeof(cl_float4), NULL, &ret);
	gvelnew = clCreateBuffer(context, CL_MEM_READ_WRITE, NUMPART * sizeof(cl_float4), NULL, &ret);
	printf("[INIT] Create memory buffers: ");
	if ((int)ret == 0) {
		printf("SUCCESS\n");
	}
	else {
		printf("FAILED (%d)\n", ret);
		_getch();
		return 1;
	}

	// Create kernel program from source
	cl_program program = NULL;
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	printf("[INIT] Create kernel program: ");
	if ((int)ret == 0) {
		printf("SUCCESS\n");
	}
	else {
		printf("FAILED (%d)\n", ret);
		_getch();
		return 1;
	}

	// Build kernel program
	ret = clBuildProgram(program, 1, &devices[0], NULL, NULL, NULL);
	printf("[INIT] Build kernel program: ");
	if ((int)ret == 0) {
		printf("SUCCESS\n");
	}
	else {
		printf("FAILED (%d)\n", ret);
		// Determine the size of the log
		size_t log_size;
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		// Allocate memory for the log
		char *log = (char *)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);
		_getch();
		return 1;
	}

	// Create OpenCL kernel
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "propagate", &ret);
	printf("[INIT] Create OpenCL kernel: ");
	if ((int)ret == 0) {
		printf("SUCCESS\n");
	}
	else {
		printf("FAILED (%d)\n", ret);
		_getch();
		return 1;
	}

	// Calculate the TGV constants
	float A = flowMag;
	float a = vortexFreq;
	float B = A;
	float b = a;
	float C = -2 * A;
	float c = a;
	printf("[ SIM] Defining TGV values: SUCCESS\n");

	// Define characteristic values of simulation
	float tau = tauMultiplier * (rho * D * D) / (18 * mu);
	float tstep = 0.1 * tau;
	float vT = -g * tau;
	float finalT = timeDurationFactor * tau;
	printf("[ SIM] Defining characteristic values of simulation: SUCCESS\n\n"
		"             Tau value: %6f\n"
		"             Time step: %6f\n"
		"             Terminal velocity: %6f\n"
		"             Final time: %6f\n\n", tau, tstep, vT, finalT);

	// Generate random initial values for position
	srand(0);
	hpos = malloc(sizeof(float) * 4 * NUMPART);
	for (int ip = 0; ip < NUMPART; ip++) {
		hpos[ip * 4 + 0] = 3.1 * (float)((float)rand() / (float)RAND_MAX) - 1.6;
		hpos[ip * 4 + 1] = 3.1 * (float)((float)rand() / (float)RAND_MAX) - 1.6;
		hpos[ip * 4 + 2] = 3.1 * (float)((float)rand() / (float)RAND_MAX) - 1.6;
		hpos[ip * 4 + 3] = 0.0f;
	}
	printf("[ SIM] Create %d initial position values: SUCCESS\n", NUMPART);

	// Define initial particle velocities based on local TGV velocities
	hvel = malloc(sizeof(float) * 4 * NUMPART);
	for (int ip = 0; ip < NUMPART; ip++) {
		hvel[ip * 4 + 0] = A * cos(a * ((float)hpos[ip * 4 + 0] + ((float)MATH_PI / (a * 2)))) * sin(b * ((float)hpos[ip * 4 + 1] + ((float)MATH_PI / (2 * a)))) * sin(c * ((float)hpos[ip * 4 + 2] + ((float)MATH_PI / (2 * a))));
		hvel[ip * 4 + 1] = B * sin(a * ((float)hpos[ip * 4 + 0] + ((float)MATH_PI / (a * 2)))) * cos(b * ((float)hpos[ip * 4 + 1] + ((float)MATH_PI / (2 * a)))) * sin(c * ((float)hpos[ip * 4 + 2] + ((float)MATH_PI / (2 * a))));
		hvel[ip * 4 + 2] = C * sin(a * ((float)hpos[ip * 4 + 0] + ((float)MATH_PI / (a * 2)))) * sin(b * ((float)hpos[ip * 4 + 1] + ((float)MATH_PI / (2 * a)))) * cos(c * ((float)hpos[ip * 4 + 2] + ((float)MATH_PI / (2 * a))));
		hvel[ip * 4 + 3] = 0.0f;
	}
	printf("[ SIM] Create %d initial velocity values: SUCCESS\n", NUMPART);
	printf("  POSITION: %6f %6f %6f %6f\n", hpos[0], hpos[1], hpos[2], hpos[3]);
	printf("  VELOCITY: %6f %6f %6f %6f\n", hvel[0], hvel[1], hvel[2], hvel[3]);

	// Write initial values to file
	FILE *fd = fopen("INITIAL VALUES.txt", "w");
	for (int i = 0; i<NUMPART; i++)
	{
		fprintf(fd, "%e %e %e %e", hpos[i * 4 + 0], hpos[i * 4 + 1], hpos[i * 4 + 2], hpos[i * 4 + 3]);
		fprintf(fd, " %e %e %e %e\n", hvel[i * 4 + 0], hvel[i * 4 + 1], hvel[i * 4 + 2], hvel[i * 4 + 3]);
	}
	fclose(fd);

	// pass relevant values to memory buffer on the GPU
	ret = clEnqueueWriteBuffer(queue, gposold, CL_TRUE, 0, NUMPART * sizeof(cl_float4), hpos, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(queue, gvelold, CL_TRUE, 0, NUMPART * sizeof(cl_float4), hvel, 0, NULL, NULL);
	printf("[ SIM] Copying values to GPU memory buffers: ");
	if ((int)ret == 0) {
		printf("SUCCESS\n");
	}
	else {
		printf("FAILED (%d)\n", ret);
		_getch();
		return 1;
	}

	// set OpenCL kernel parameters
	ret = clSetKernelArg(kernel, 4, sizeof(float), &tau);
	ret = clSetKernelArg(kernel, 5, sizeof(float), &A);
	ret = clSetKernelArg(kernel, 6, sizeof(float), &a);
	ret = clSetKernelArg(kernel, 7, sizeof(float), &B);
	ret = clSetKernelArg(kernel, 8, sizeof(float), &b);
	ret = clSetKernelArg(kernel, 9, sizeof(float), &C);
	ret = clSetKernelArg(kernel, 10, sizeof(float), &c);
	ret = clSetKernelArg(kernel, 11, sizeof(float), &tstep);
	printf("[ SIM] Set OpenCL kernel parameters: ");
	if ((int)ret == 0) {
		printf("SUCCESS\n");
	}
	else {
		printf("FAILED (%d)\n", ret);
		_getch();
		return 1;
	}\

	// SIMULATION BEGINS HERE
	int count = 0;
	int *particle;

	printf("\nPress any key to start simulation.\n\n");
	getch();

	// start logging simulation time
	struct timeb start, end;
	int diff;
	int i = 0;
	ftime(&start);

	for (float time = 0; time < finalT; time += 2 * tstep) {
		// If an intermediate log file is required
		if (count % log_step == 0) {
			// copy results from memory buffer
			ret = clEnqueueReadBuffer(queue, gposold, CL_TRUE, 0, NUMPART * sizeof(cl_float4), hpos, 0, NULL, NULL);
			ret = clEnqueueReadBuffer(queue, gvelold, CL_TRUE, 0, NUMPART * sizeof(cl_float4), hvel, 0, NULL, NULL);
			printf("[ SIM] Finished round of particles for timestep: %3f / %3f\n, writing intermediate results for this step to file.", time, finalT);
			printf("  POSITION: %6f %6f %6f %6f\n", hpos[0], hpos[1], hpos[2], hpos[3]);
			printf("  VELOCITY: %6f %6f %6f %6f\n", hvel[0], hvel[1], hvel[2], hvel[3]);
			// write intermediate results to file
			char filename[500];
			sprintf(filename, "SIM_RUN_%d.txt", count * 2);
			fd = fopen(filename, "w");
			for (int i = 0; i<NUMPART; i++) {
				fprintf(fd, "%e %e %e %e ", hpos[i * 4 + 0], hpos[i * 4 + 1], hpos[i * 4 + 2], hpos[i * 4 + 3]);
				fprintf(fd, "%e %e %e %e\n", hvel[i * 4 + 0], hvel[i * 4 + 1], hvel[i * 4 + 2], hvel[i * 4 + 3]);
			}
			fclose(fd);
		}
		// Reverse the kernel arguments
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &gposold);
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &gvelold);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &gposnew);
		ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &gvelnew);
		// Execute kernel over NDRange
		ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, NUMPART, 0, NULL, NULL);
		// Wait for this step to finish
		clFinish(queue);

		// Reverse the kernel arguments (new > old; old > new)
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &gposnew);
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &gvelnew);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &gposold);
		ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &gvelold);
		// Execute kernel over NDRange
		ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, NUMPART, 0, NULL, NULL);
		// Wait for this step to finish
		clFinish(queue);
	}

	// Log simulation time
	ftime(&end);
	diff = (int)(1000.0 * (end.time - start.time)
		+ (end.millitm - start.millitm));

	printf("\nOperation took %f seconds\n", diff / 1000.);

	getch();
}
