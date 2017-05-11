# OpenCL Particle-Fluid Simulation
This project is a particle-fluid simulation, programmed in C an OpenCL for the author's FEEG3003 Individual Project module.

   ![Intro](http://chow.ch/i/7uLBF.png)

## Requirements
The development enviroment(s) and OpenCL libraries are required to compile and run the source code:
  - IDE: [Visual Studio]
  - For NVidia GPUs: [CUDA Toolkit]
  - For AMD GPUs:  [AMD APP SDK]

## Getting started
Below are the steps to setup OpenCL in Visual Studio. Once this has been done, any of the source files can be modified, compiled and run within the development environment.

### 1. Creating an OpenCL project 
  - In Visual Studio, create a new project by navigating under the ‘File’ menu, selecting ‘New’ > Project’ > ‘Visual C++’ > ‘Win32 Console Application’. 
  - Give the project a name, and in the application creation wizard select ‘Next’
  - Under ‘Additional options’ tick the ‘Empty project’ checkbox and select ‘Finish’

### 2. Including OpenCL libraries
  -  In the solution explorer, right click on the project name and select ‘Properties’
  - From the ‘Configuration’ drop down menu, select ‘All Configurations’
  - On the left menu, navigate to ‘Configuration Properties’ > ‘C/C++’ > ‘General’
  - The ‘Additional Include Directories’ field has to be filled  as per the OpenCL SDK downloaded in step 1:
  
    CUDA Toolkit: `$(CUDA_INC_PATH)`
  
    AMD APP SDK: `$(AMDAPPSDKROOT)\include`

### 3. Adding OpenCL linkers
  - Without closing the option menu in Step 2, navigate to ‘Linker’ > ‘General’
  - In the ‘Additional Dependencies’ field, fill the following (without quotations) based on the GPU manufacturer:
  
    Nvidia: `$(CUDA_LIB_PATH)`
    
    AMD: `$(AMDAPPSDKROOT)\lib\x86`   (‘x86 replaced with x86_64 if using a 64-bit system)
    
  - Under ‘Linker’ > ‘Input’, select ‘Edit’ from the dropdown in ‘Additional Dependencies’ field, a new menu dialog should appear. Enter “`OpenCL.lib`” (without quotations) in this dialog and select OK. 

### 4. Import source files
  - Import the `host.c` and `propagate.cl` source files from either the clEnqueueTask or clEnqueueNDRange folders to the Source Files in Solution Explorer pane in Visual Studio.
  - Press the run button. If configured correctly, a console window should pop-up printing information on the compatible OpenCL platforms and devices on a system. 
  
    ![Log](http://chow.ch/i/PwFL5.png)
  - Further documentation on each of the simulation parameters and how to configure simulation variables are given in the respective folders.

   [Visual Studio]: <https://www.visualstudio.com/downloads/>
   [CUDA Toolkit]: <https://developer.nvidia.com/cuda-downloads>
   [AMD APP SDK]: <http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/>
