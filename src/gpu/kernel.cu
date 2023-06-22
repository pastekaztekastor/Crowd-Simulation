/*******************************************************************************
* File Name: kernel.cu
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: 
*******************************************************************************/

#include "kernel.hpp"

void initKernel( 
    unsigned int **     populationPosition, // (*) // change
    unsigned int **     map,                // (*) // change
    unsigned int *      simPIn,             // (*) // change
    unsigned int *      cost, 
    unsigned int *      simExit,  
    unsigned int        simDimX,  
    unsigned int        simDimY,  
    unsigned int        simDimP, 
    unsigned int        settings_print 
){
    
}

void destroyKernel(){
    
}
__global__ void kernel_model1_GPU(
    unsigned int **     dev_populationPosition,     // (*) Because change
    unsigned int **     dev_map,                    // (*) Because change
    unsigned int *      dev_simPIn,                 // (*) Because change
    unsigned int *          cost,                   // useless
    unsigned int *          simExit, 
    unsigned int            simDimX, 
    unsigned int            simDimY,
    unsigned int            simDimP
){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < simDimP){
        // position de l'individue tid
        unsigned int x = (*dev_populationPosition)[2*tid + 0];
        unsigned int y = (*dev_populationPosition)[2*tid + 1];
        // Delta à ajouté à la position pour avoir la position next step
        int deltaX =  simExit[0]-x;
        int deltaY =  simExit[1]-y;
        int moovX = deltaX / max(abs(deltaX), abs(deltaY));
        int moovY = deltaY / max(abs(deltaX), abs(deltaY));
        // on regarde si la case est disponible 
        if((*dev_map[ simDimY * (y+moovY) + (x + moovX)]) == 0){ // if is EMPTY
            // conditionné par le atomique 
            (*dev_populationPosition)[2*tid + 0] = x+moovX;
            (*dev_populationPosition)[2*tid + 1] = y+moovY;
    
            // Temporaire
            (*dev_map)[simDimY * (y+moovY) + (x + moovX)]    = 1;
            (*dev_map)[simDimY * y + x]                = 0;
            //atomicExch( &(*dev_map)[y+deltaY][x+deltaX]  , 1); // HUMAN
            //atomicExch( &(*dev_map)[y][x]                , 0); // EMPTY
        }
    }
}

// -1 vide
// index : humain
// .. 