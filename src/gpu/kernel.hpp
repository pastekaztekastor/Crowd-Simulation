/*******************************************************************************
* File Name: kernel.cuh
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: 
*******************************************************************************/

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include "main.hpp"
#include <cuda_runtime.h>

void model1_GPU(
    unsigned int ***    populationPosition,         // (*) Because change
    unsigned int ***    map,                        // (*) Because change
    unsigned int *      simPIn,                     // (*) Because change
    unsigned int **     cost,                       // useless
    unsigned int *      simExit, 
    unsigned int        simDimX, 
    unsigned int        simDimY, 
    unsigned int        simDimP, 
    unsigned int        settings_print );

__global__ void kernel_model1_GPU(
    unsigned int ***    dev_populationPosition,     // (*) Because change
    unsigned int ***    dev_map,                    // (*) Because change
    unsigned int *      dev_simPIn,                 // (*) Because change
    unsigned int **         cost,                   // useless
    unsigned int *          simExit, 
    unsigned int            simDimX, 
    unsigned int            simDimY,
    unsigned int            simDimP
);

#endif