#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef INTEROP_VIZ

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#include <OpenGL/OpenGL.h>
#include "OpenCL/cl.h"
#include "OpenCL/cl_gl.h"
#include "OpenCL/cl_gl_ext.h"
#include "OpenCL/cl_ext.h"
#include "OpenGL/CGLDevice.h"
#define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
#include <GL/gl.h>
#include <GL/glx.h>
#include "CL/cl_gl.h"
#endif

#endif

#include "cl_utils.h"
#include "yamlOutput.h"

#define GPU 1

#if (RUNTIME_API)

/*
   typedef struct float4
   {
   float x,y,z,w;
   } float4;
   typedef struct float3
   {
   float x,y,z,w;
   } float3;
   */

cl_uint          num_platforms;
cl_platform_id*  platformId; 
cl_device_id     deviceId;
cl_context       context; 
cl_command_queue commandq;
cl_program       program;
cl_uint          num_devices;
cl_uint          num_cpus;
cl_uint          num_gpus;

#endif

static char * loadProgramSourceFromFile(const char *filename)
{
   struct stat statbuf;
   FILE        *fh;
   char        *source;

   fh = fopen(filename, "r");
   if (fh == 0)
      return 0;

   stat(filename, &statbuf);
   source = (char *) malloc(statbuf.st_size + 1);
   fread(source, statbuf.st_size, 1, fh);
   source[statbuf.st_size] = '\0';

   return source;
}

// given a file path, loads a program file and builds the program
int buildProgramFromFile(cl_program* program, 
      char* filename, 
      cl_context context, 
      cl_device_id deviceId)
{
   int err;

   // Load programs from separate kernels file

   printf("Loading program '%s'...\n", filename);

   char *source = loadProgramSourceFromFile(filename);
   if(!source)
   {
      printf("Failed to load compute program from file!\n");
      return EXIT_FAILURE;    
   }

   // define the right type for the code
   const char *srcPrec  = "CL_REAL_T";
   const char *srcPrec4 = "CL_REAL4_T";
#ifdef SINGLE
   const char myPrec[128]  =  "float    "; /* note padded with zeros to length of srcPrec */
   const char myPrec4[128] =  "float4    "; /* note padded with zeros to length of srcPrec4 */
#else
   const char myPrec[128]  =  "double   ";  /* note padded with zeros to length of srcPrec */
   const char myPrec4[128] =  "double4   ";  /* note padded with zeros to length of srcPrec4 */
#endif
   char *mydef = source;
   int rep_count = 0;
   while((mydef = strstr(mydef,srcPrec4)) != NULL) 
   {
      strncpy(mydef,myPrec4,strlen(srcPrec4));
      rep_count +=1;
   }
   mydef = source;
   printf("Replaced %d instances of %s with %s in %s\n", rep_count, srcPrec4, myPrec4, filename);
   while((mydef = strstr(mydef,srcPrec)) != NULL) 
   {
      memcpy(mydef,myPrec,strlen(srcPrec));
      rep_count +=1;
   }
   printf("Replaced %d instances of %s with %s in %s\n", rep_count, srcPrec, myPrec, filename);
   rep_count = 0;

   // Create the compute program from the source buffer
   //

   *program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &err);
   if (!program || err != CL_SUCCESS)
   {
      printf("%s\n", source);
      printf("Failed to create compute program!\n");
      return EXIT_FAILURE;
   }

   // Build the program executable
   //
   char* OPT_STRING;
#if (USE_CHEBY)
   OPT_STRING = "-DUSE_CHEBY=1";
#else
   OPT_STRING = "-DUSE_CHEBY=0";
#endif
   err = clBuildProgram(*program, 0, NULL, OPT_STRING, NULL, NULL);
   if (err != CL_SUCCESS)
   {
      size_t len;
      char buffer[2048];

      printf("Failed to build program executable!\n");
      clGetProgramBuildInfo(*program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      exit(1);
   }
   else
   {
      printf("Module %s built\n", filename);
   }
   return 0;
}

#if (RUNTIME_API)
int oclInit(int gpu_request)
{
   int err = 0;
   char device_name[80];

   printf("************************************************************************\n");
   printf("Initializing OpenCL...\n");

   // Query the number of platforms
   err = clGetPlatformIDs(0, NULL, &num_platforms);
   if (err != CL_SUCCESS)
   {
      printf("Failed to get number of platforms!\n");
      return EXIT_FAILURE;
   } 
   else 
   {
      printf("Found %d OpenCL platforms\n", num_platforms);

      platformId = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
      for (int i = 0; i < num_platforms; i++) 
      {
	 err = clGetPlatformIDs(num_platforms, &platformId[i], NULL);
	 if (err != CL_SUCCESS)
	 {
	    printf("Failed to get platform ID!\n");
	    return EXIT_FAILURE;
	 } 
      }
   }

   // Check for OpenCL capable CPUs
   num_cpus = 0;
   err = clGetDeviceIDs(platformId[0], CL_DEVICE_TYPE_CPU, 0, NULL, &num_cpus);
   if (err != CL_SUCCESS)
   {
      printf("No OpenCL CPU devices available \n");
   }
   else
   {
      printf("Found %d OpenCL CPUs on this platform\n", num_cpus);
   }

   // Check for OpenCL capable GPUs
   num_gpus = 0;
   err = clGetDeviceIDs(platformId[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_gpus);
   if (err != CL_SUCCESS)
   {
      printf("No OpenCL GPU devices available \n");
   }
   else
   {

      printf("Found %d OpenCL GPUs on this platform\n", num_gpus);
   }

   // Set total number of OpenCL devices
   num_devices = num_cpus + num_gpus;

   if (num_devices == 0) 
   {
      printf("No OpenCL-capable devices are available\n");
      return 0;
   }

   int usingGPU = 0;
   if (gpu_request) 
   { // branch for request for GPU
      if (num_gpus > 0) 
      {

	 printf("Connecting to GPU as compute device\n");
	 usingGPU = 1;
	 err = clGetDeviceIDs(platformId[0], CL_DEVICE_TYPE_GPU, 1, &deviceId, NULL);
	 if (err != CL_SUCCESS)
	 {
	    printf("Failed to create a device group!\n");
	    return EXIT_FAILURE;
	 }
      }
      else
      {
	 // If we asked for a GPU but none is available, let the user know
	 if (num_cpus > 0 ) 
	 {
	    printf("****************************************\n");
	    printf("*No GPUs available, falling back to CPU*\n");
	    printf("****************************************\n");

	    err = clGetDeviceIDs(platformId[0], CL_DEVICE_TYPE_CPU, 1, &deviceId, NULL);
	    if (err != CL_SUCCESS)
	    {
	       printf("Failed to create a device group!\n");
	       return EXIT_FAILURE;
	    }
	 }
      }
   }
   else
   {// branch for no GPU request
      if (num_cpus > 0 ) 
      {
	 printf("Connecting to CPU as compute device\n");
	 err = clGetDeviceIDs(platformId[0], CL_DEVICE_TYPE_CPU, 1, &deviceId, NULL);
	 if (err != CL_SUCCESS)
	 {
	    printf("Failed to create a device group!\n");
	    return EXIT_FAILURE;
	 }
      }
      else
      {
	 if (num_gpus > 0) 
	 { 
	    // If we asked for a CPU but none is available, let the user know
	    printf("****************************************\n");
	    printf("*No CPUs available, falling back to GPU*\n");
	    printf("****************************************\n");

	    usingGPU = 1;
	    err = clGetDeviceIDs(platformId[0], CL_DEVICE_TYPE_GPU, 1, &deviceId, NULL);
	    if (err != CL_SUCCESS)
	    {
	       printf("Failed to create a device group!\n");
	       return EXIT_FAILURE;
	    }
	 }
      }
   }

   err = clGetDeviceInfo(deviceId , CL_DEVICE_NAME, 80*sizeof(char), &device_name, NULL);
   printf("Device type is %s\n", device_name);
   fprintf(yamlFile, "Device type is %s\n", device_name);

   cl_ulong global_mem_size;
   err = clGetDeviceInfo(deviceId , CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
   printf("Global memory size is %.2f GB\n", (float)global_mem_size/1024.0/1024.0/1024.0);
   fprintf(yamlFile, "  Global memory size is %.2f GB\n", (float)global_mem_size/1024.0/1024.0/1024.0);

   size_t max_work_group_size;
   err = clGetDeviceInfo(deviceId , CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
   printf("Maximum work group size is %lu \n", (max_work_group_size));
   fprintf(yamlFile, "  Maximum work group size is %lu \n", (max_work_group_size));

   cl_uint max_compute_units;
   err = clGetDeviceInfo(deviceId , CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &max_compute_units, NULL);
   printf("Maximum compute units is %u \n", (max_compute_units));
   fprintf(yamlFile, "  Maximum compute units is %u \n", (max_compute_units));

   cl_uint max_clock_freq;
   err = clGetDeviceInfo(deviceId , CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &max_clock_freq, NULL);
   printf("Maximum clock freq is %u \n", (max_clock_freq));
   fprintf(yamlFile, "  Maximum clock freq is %u \n", (max_clock_freq));

   // Create a compute context 
   printf("Creating a compute context\n");

#ifdef INTEROP_VIZ

#if defined (__APPLE__) || defined(MACOSX)
   CGLContextObj kCGLContext = CGLGetCurrentContext();
   CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
   cl_context_properties cpsGL[] = { CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup, 0 };

   if (usingGPU == 1) context = clCreateContext(cpsGL, 0, 0, NULL, NULL, &err);
   else context = clCreateContext(cpsGL, 1, &deviceId, NULL, NULL, &err);
#else
   GLXContext gGlCtx = glXGetCurrentContext();
   cl_context_properties cpsGL[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformId[0],
      CL_GLX_DISPLAY_KHR, (intptr_t) glXGetCurrentDisplay(),
      CL_GL_CONTEXT_KHR, (intptr_t) gGlCtx, 0};
   context = clCreateContext(cpsGL, 1, &deviceId, NULL, NULL, &err);
#endif

#else
   context = clCreateContext(0, 1, &deviceId, NULL, NULL, &err);
#endif

   if (!context)
   {
      printf("Failed to create a compute context! %d\n", err);
      return EXIT_FAILURE;
   }

   // Create a command queue
   printf("Creating a command queue\n");
   commandq = clCreateCommandQueue(context, deviceId, CL_QUEUE_PROFILING_ENABLE, &err);
   if (!commandq)
   {
      printf("Failed to create a command queue!\n");
      return EXIT_FAILURE;
   }
   printf("Initialization done\n");
   return 0;
}

// release program, commnand queue and then the context
int oclCleanup()
{
   printf("Cleaning up OpenCL...");
   clReleaseProgram(program);
   clReleaseCommandQueue(commandq);
   clReleaseContext(context);
   printf("done\n");
   return 0;
}

// given a file path, loads a program file and builds the program
int oclBuildProgramFromFile(char* filename)
{
   int err;

   // Load programs from separate kernels file

   printf("Loading program '%s'...\n", filename);

   char *source = loadProgramSourceFromFile(filename);
   if(!source)
   {
      printf("Failed to load compute program from file!\n");
      return EXIT_FAILURE;    
   }

   // Create the compute program from the source buffer
   //
   program = clCreateProgramWithSource(context, 1, (const char **) & source, NULL, &err);
   if (!program || err != CL_SUCCESS)
   {
      printf("%s\n", source);
      printf("Failed to create compute program!\n");
      return EXIT_FAILURE;
   }

   // Build the program executable
   //
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err != CL_SUCCESS)
   {
      size_t len;
      char buffer[2048];

      printf("Failed to build program executable!\n");
      clGetProgramBuildInfo(program, deviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      exit(1);
   }
   return 0;
}

void oclCreateReadBuffer(cl_mem *device_buffer, int size)
{
   *device_buffer = clCreateBuffer(context,  CL_MEM_READ_ONLY,  size, NULL, NULL);
   if (!device_buffer)
   {
      printf("Failed to allocate device memory!\n");
      exit(1);
   }
}

void oclCreateWriteBuffer(cl_mem *device_buffer, int size)
{
   *device_buffer = clCreateBuffer(context,  CL_MEM_WRITE_ONLY,  size, NULL, NULL);
   if (!device_buffer)
   {
      printf("Failed to allocate device memory!\n");
      exit(1);
   }
}

void oclCreateReadWriteBuffer(cl_mem *device_buffer, int size)
{
   *device_buffer = clCreateBuffer(context,  CL_MEM_READ_WRITE,  size, NULL, NULL);
   if (!device_buffer)
   {
      printf("Failed to allocate device memory!\n");
      exit(1);
   }
}

void oclCopyToDevice(void* host_buffer, cl_mem device_buffer, int size, int offset)
{
   int err = clEnqueueWriteBuffer(commandq, device_buffer, CL_TRUE, offset, size, host_buffer, 0, NULL, NULL);
   if (err != CL_SUCCESS)
   {
      printf("Failed to write to device array!\n");
      printf("Error: %s\n", print_cl_errstring(err));
      exit(1);
   }
}

void oclCopyToHost(cl_mem device_buffer, void* host_buffer, int size, int offset)
{
   int err = clEnqueueReadBuffer(commandq, device_buffer, CL_TRUE, offset, size, host_buffer, 0, NULL, NULL);
   if (err != CL_SUCCESS)
   {
      printf("Failed to write to host array!\n");
      exit(1);
   }
}
#endif

int ParseGPURequest(int argc, char* argv[])
{
   // check to see if the command line asks for GPU, otherwise default to CPU
   int gpu_request = 0;
   if (argc > 1) 
   {
      printf("%s\n", argv[1]);
      gpu_request = atoi(argv[1]);
   }
   return gpu_request;
}

char *print_cl_errstring(cl_int err) 
{
   switch (err) 
   {
      case CL_SUCCESS:                          return strdup("Success!");
      case CL_DEVICE_NOT_FOUND:                 return strdup("Device not found.");
      case CL_DEVICE_NOT_AVAILABLE:             return strdup("Device not available");
      case CL_COMPILER_NOT_AVAILABLE:           return strdup("Compiler not available");
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return strdup("Memory object allocation failure");
      case CL_OUT_OF_RESOURCES:                 return strdup("Out of resources");
      case CL_OUT_OF_HOST_MEMORY:               return strdup("Out of host memory");
      case CL_PROFILING_INFO_NOT_AVAILABLE:     return strdup("Profiling information not available");
      case CL_MEM_COPY_OVERLAP:                 return strdup("Memory copy overlap");
      case CL_IMAGE_FORMAT_MISMATCH:            return strdup("Image format mismatch");
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return strdup("Image format not supported");
      case CL_BUILD_PROGRAM_FAILURE:            return strdup("Program build failure");
      case CL_MAP_FAILURE:                      return strdup("Map failure");
      case CL_INVALID_VALUE:                    return strdup("Invalid value");
      case CL_INVALID_DEVICE_TYPE:              return strdup("Invalid device type");
      case CL_INVALID_PLATFORM:                 return strdup("Invalid platform");
      case CL_INVALID_DEVICE:                   return strdup("Invalid device");
      case CL_INVALID_CONTEXT:                  return strdup("Invalid context");
      case CL_INVALID_QUEUE_PROPERTIES:         return strdup("Invalid queue properties");
      case CL_INVALID_COMMAND_QUEUE:            return strdup("Invalid command queue");
      case CL_INVALID_HOST_PTR:                 return strdup("Invalid host pointer");
      case CL_INVALID_MEM_OBJECT:               return strdup("Invalid memory object");
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return strdup("Invalid image format descriptor");
      case CL_INVALID_IMAGE_SIZE:               return strdup("Invalid image size");
      case CL_INVALID_SAMPLER:                  return strdup("Invalid sampler");
      case CL_INVALID_BINARY:                   return strdup("Invalid binary");
      case CL_INVALID_BUILD_OPTIONS:            return strdup("Invalid build options");
      case CL_INVALID_PROGRAM:                  return strdup("Invalid program");
      case CL_INVALID_PROGRAM_EXECUTABLE:       return strdup("Invalid program executable");
      case CL_INVALID_KERNEL_NAME:              return strdup("Invalid kernel name");
      case CL_INVALID_KERNEL_DEFINITION:        return strdup("Invalid kernel definition");
      case CL_INVALID_KERNEL:                   return strdup("Invalid kernel");
      case CL_INVALID_ARG_INDEX:                return strdup("Invalid argument index");
      case CL_INVALID_ARG_VALUE:                return strdup("Invalid argument value");
      case CL_INVALID_ARG_SIZE:                 return strdup("Invalid argument size");
      case CL_INVALID_KERNEL_ARGS:              return strdup("Invalid kernel arguments");
      case CL_INVALID_WORK_DIMENSION:           return strdup("Invalid work dimension");
      case CL_INVALID_WORK_GROUP_SIZE:          return strdup("Invalid work group size");
      case CL_INVALID_WORK_ITEM_SIZE:           return strdup("Invalid work item size");
      case CL_INVALID_GLOBAL_OFFSET:            return strdup("Invalid global offset");
      case CL_INVALID_EVENT_WAIT_LIST:          return strdup("Invalid event wait list");
      case CL_INVALID_EVENT:                    return strdup("Invalid event");
      case CL_INVALID_OPERATION:                return strdup("Invalid operation");
      case CL_INVALID_GL_OBJECT:                return strdup("Invalid OpenGL object");
      case CL_INVALID_BUFFER_SIZE:              return strdup("Invalid buffer size");
      case CL_INVALID_MIP_LEVEL:                return strdup("Invalid mip-map level");
      default:                                  return strdup("Unknown");
   }
}

// Queries the total number of OpenCL platforms available.
// Then creates a list of platforms, and for each platform, prints the name.
// Creates a context and command queue for the chosen device.
int initCL(cl_uint* num_platforms, 
      cl_platform_id* platformId, 
      cl_device_id* deviceId, 
      cl_context* context, 
      cl_command_queue* commandq)
{
   int err = 0;
   char device_name[80];

   printf("Connecting to compute devices\n");

   // Connect to a compute device
   //
   err = clGetPlatformIDs(0, NULL, num_platforms);
   if (err != CL_SUCCESS)
   {
      printf("Failed to get number of platforms!\n");
      return EXIT_FAILURE;
   } 
   else 
   {
      printf("Found %d OpenCL platforms\n", *num_platforms);

      platformId = (cl_platform_id*)malloc(*num_platforms * sizeof(cl_platform_id));
      for (int i = 0; i < *num_platforms; i++) 
      {
	 err = clGetPlatformIDs(*num_platforms, &platformId[i], NULL);
	 if (err != CL_SUCCESS)
	 {
	    printf("Failed to get platform ID!\n");
	    return EXIT_FAILURE;
	 } 
      }
   }

   err = clGetDeviceIDs(platformId[0], GPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 0, NULL, &num_devices);
   if (err != CL_SUCCESS)
   {
      printf("No CPU devices available \n");
   }
   printf("Found %d devices on this platform\n", num_devices);


   err = clGetDeviceIDs(platformId[0], GPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, deviceId, NULL);
   if (err != CL_SUCCESS)
   {
      printf("Failed to create a device group!\n");
      return EXIT_FAILURE;
   }

   err = clGetDeviceInfo(*deviceId , CL_DEVICE_NAME, 80*sizeof(char), &device_name, NULL);

   printf("Device type is %s\n", device_name);

   // Create a compute context 
   //
   *context = clCreateContext(0, 1, deviceId, NULL, NULL, &err);
   if (!context)
   {
      printf("Failed to create a compute context!\n");
      return EXIT_FAILURE;
   }

   // Create a command queue
   //
   *commandq = clCreateCommandQueue(*context, *deviceId, CL_QUEUE_PROFILING_ENABLE, &err);
   if (!commandq)
   {
      printf("Failed to create a command queue!\n");
      return EXIT_FAILURE;
   }
   if (err != CL_SUCCESS)
   {
      printf("Failed to create a command queue!\n");
      return EXIT_FAILURE;
   }
   return 0;
}


