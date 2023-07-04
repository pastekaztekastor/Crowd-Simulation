/*******************************************************************************
* File Name: kernel.cuh
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: 
*******************************************************************************/

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include "simulation.hpp"
#include <cuda_runtime.h>

/*
  _____       _ _   
 |_   _|     (_) |  
   | |  _ __  _| |_ 
   | | | '_ \| | __|
  _| |_| | | | | |_ 
 |_____|_| |_|_|\__|
                    
*/
void            initKernelParam   (kernelParam * _kernelParam, simParam _simParam, settings _settings);

/*
  _  __                    _     
 | |/ /                   | |    
 | ' / ___ _ __ _ __   ___| |___ 
 |  < / _ \ '__| '_ \ / _ \ / __|
 | . \  __/ |  | | | |  __/ \__ \
 |_|\_\___|_|  |_| |_|\___|_|___/

*/
__global__ void kernel_model1_GPU (kernelParam _kernelParam, simParam _simParam, settings _settings);
__global__ void kernel_costMap_GPU(kernelParam _kernelParam, simParam _simParam, settings _settings);

/*
  ______                       _    
 |  ____|                     | |   
 | |__  __  ___ __   ___  _ __| |_  
 |  __| \ \/ / '_ \ / _ \| '__| __| 
 | |____ >  <| |_) | (_) | |  | |_  
 |______/_/\_\ .__/ \___/|_|   \__| 
             | |                    
             |_|                    
*/
void            mapKernelToSim    (kernelParam _kernelParam, simParam * _simParam, settings _settings);
void            popKernelToSim    (kernelParam _kernelParam, simParam * _simParam, settings _settings);
void            pInKernelToSim    (kernelParam _kernelParam, simParam * _simParam, settings _settings);

/*
  ______             
 |  ____|            
 | |__ _ __ ___  ___ 
 |  __| '__/ _ \/ _ \
 | |  | | |  __/  __/
 |_|  |_|  \___|\___|
                     
*/
void            destroyKernel     (kernelParam * _kernelParam, settings _settings);

#endif