/*******************************************************************************
* File Name: kernel.cu
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: 
*******************************************************************************/

#include "kernel.hpp"

void model1_GPU( 
    unsigned int ***    populationPosition, 
    unsigned int ***    map, 
    unsigned int *      simPIn, 
    unsigned int **     cost, 
    unsigned int *      simExit,  
    unsigned int        simDimX,  
    unsigned int        simDimY,  
    unsigned int        simDimP, 
    unsigned int        settings_print 
){
    unsigned int ***    dev_populationPosition  = nullptr;
    unsigned int ***    dev_map                 = nullptr;
    unsigned int *      dev_simPIn              = nullptr;
    //unsigned int *      dev_outMove             = nullptr;

    //unsigned int        outMove                 = 0;

    cudaMalloc((void**) &dev_populationPosition , 2 * sizeof(unsigned int) * simDimP); 
    cudaMalloc((void**) &dev_map                , simDimX * simDimY * sizeof(unsigned int ));
    cudaMalloc((void**) &dev_simPIn             , sizeof(unsigned int));
    //cudaMalloc((void**) &dev_outMove            , sizeof(unsigned int));

    cudaMemcpy(dev_populationPosition, populationPosition, (2 * sizeof(unsigned int) * simDimP)         , cudaMemcpyHostToDevice);
    cudaMemcpy(dev_map               , map               , (simDimX * simDimY * sizeof(unsigned int ))  , cudaMemcpyHostToDevice);
    cudaMemcpy(dev_simPIn            , simPIn            , sizeof(unsigned int)                         , cudaMemcpyHostToDevice);

    unsigned int nb_threads = 16;
    dim3 blocks((simDimP + (nb_threads-1))/nb_threads);
    dim3 threads(nb_threads);

    kernel_model1_GPU<<<blocks,threads>>>(dev_populationPosition, dev_map, dev_simPIn, cost, simExit, simDimX, simDimY, simDimP);

    //cudaMemcpy(outMove           , dev_outMove           , sizeof(unsigned int)                      , cudaMemcpyDeviceToHost);
    cudaMemcpy(populationPosition, dev_populationPosition, 2 * sizeof(unsigned int) * simDimP        , cudaMemcpyDeviceToHost);
    cudaMemcpy(map               , dev_map               , simDimX * simDimY * sizeof(unsigned int ) , cudaMemcpyDeviceToHost);
    
    cudaFree(dev_populationPosition);
    cudaFree(dev_map);
    cudaFree(dev_simPIn);
    //cudaFree(dev_outMove);
}
__global__ void kernel_model1_GPU(
    unsigned int ***    dev_populationPosition,     // (*) Because change
    unsigned int ***    dev_map,                    // (*) Because change
    unsigned int *      dev_simPIn,                 // (*) Because change
    unsigned int **         cost,                   // useless
    unsigned int *          simExit, 
    unsigned int            simDimX, 
    unsigned int            simDimY,
    unsigned int            simDimP
){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < simDimP){
        // position de l'individue tid
        unsigned int x = (*dev_populationPosition)[tid][0];
        unsigned int y = (*dev_populationPosition)[tid][1];
        // Delta à ajouté à la position pour avoir la position next step
        int deltaX =  simExit[0]-x;
        int deltaY =  simExit[1]-y;
        int moovX = deltaX / max(abs(deltaX), abs(deltaY));
        int moovY = deltaY / max(abs(deltaX), abs(deltaY));
        // on regarde si la case est disponible 
        if((*dev_map[y+moovY][x+moovX]) == 0){ // if is EMPTY
            (*dev_populationPosition)[tid][0] = x+moovX;
            (*dev_populationPosition)[tid][1] = y+moovY;
    
            // Temporaire
            (*dev_map)[y+moovY][x+moovX]    = 1;
            (*dev_map)[y][x]                = 0;
            //atomicExch( &(*dev_map)[y+deltaY][x+deltaX]  , 1); // HUMAN
            //atomicExch( &(*dev_map)[y][x]                , 0); // EMPTY
        }
    }
}