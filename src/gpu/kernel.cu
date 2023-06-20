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
    enum _Element ***   map, 
    unsigned int *      simPIn, 
    unsigned int **     cost, 
    unsigned int *      simExit,  
    unsigned int        simDimX,  
    unsigned int        simDimY,  
    unsigned int        simDimP, 
    unsigned int        settings_print 
){

    unsigned int ***    dev_populationPosition;
    enum _Element ***   dev_map;
    unsigned int *      dev_simPIn;
    unsigned int **     dev_cost;
    unsigned int *      dev_simDimP, 
    unsigned int *      dev_simExit;
    unsigned int *      dev_outMove;
    //unsigned int        outMove = 0;

    cudaMalloc((void**) &dev_populationPosition , sizeof(unsigned int *** ));
    cudaMalloc((void**) &dev_map                , sizeof(enum _Element ***));
    cudaMalloc((void**) &dev_simPIn             , sizeof(unsigned int *   ));
    cudaMalloc((void**) &dev_cost               , sizeof(unsigned int **  ));
    cudaMalloc((void**) &dev_simExit            , sizeof(unsigned int *   ));
    cudaMalloc((void**) &dev_outMove            , sizeof(unsigned int *   ));

    cudaMemcpy(dev_populationPosition, populationPosition, sizeof( unsigned int ***  ) , cudaMemcpyHostToDevice);
    cudaMemcpy(dev_map               , map               , sizeof( enum _Element *** ) , cudaMemcpyHostToDevice);
    cudaMemcpy(dev_simPIn            , simPIn            , sizeof( unsigned int *    ) , cudaMemcpyHostToDevice);
    cudaMemcpy(dev_cost              , cost              , sizeof( unsigned int **   ) , cudaMemcpyHostToDevice);
    cudaMemcpy(dev_simExit           , simExit           , sizeof( unsigned int *    ) , cudaMemcpyHostToDevice);

    unsigned int nb_threads = 16;
    dim3 blocks((simDimX + (nb_threads-1))/nb_threads, (simDimY + (nb_threads-1))/nb_threads);
    dim3 threads(nb_threads,nb_threads);

    kernel_model1_GPU<<<blocks,threads>>>(dev_populationPosition, dev_map, dev_map, dev_simPIn, dev_cost, dev_simExit, simDimX, simDimY, simDimP);

    cudaMemcpy(outMove           , dev_outMove           , sizeof( unsigned int      ) , cudaMemcpyDeviceToHost);
    
    cudaFree(dev_populationPosition);
    cudaFree(dev_map);
    cudaFree(dev_simPIn);
    cudaFree(dev_cost);
    cudaFree(dev_simExit);
    cudaFree(dev_outMove);
}

__global__ void kernel_model1_GPU(
    unsigned int ***    dev_populationPosition,     // (*) Because change
    enum _Element ***   dev_map,                    // (*) Because change
    unsigned int *      dev_simPIn,                 // (*) Because change
    unsigned int **     dev_cost,                   // useless
    unsigned int *      dev_simExit, 
    unsigned int            simDimX, 
    unsigned int            simDimY,
    unsigned int            simDimP
){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < simDimP){
        // position de l'individue tid
        unsigned int x = dev_populationPosition[tid][0];
        unsigned int y = dev_populationPosition[tid][1];
        // Delta à ajouté à la position pour avoir la position next step
        int deltaX =  dev_simExit[0]-x / max(dev_simExit[0]-x, dev_simExit[1]-y);
        int deltaY =  dev_simExit[1]-y / max(dev_simExit[0]-x, dev_simExit[1]-y);
        
    }
}

int cudaTest(unsigned int max) {
    int size = max;
    int a[size], b[size], c[size]; // Tableaux sur le CPU
    int *dev_a, *dev_b, *dev_c; // Tableaux sur le GPU

    // Allocation de mémoire sur le GPU
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    // Initialisation des tableaux sur le CPU
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Copie des tableaux depuis le CPU vers le GPU
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Configuration du nombre de blocs et de threads par bloc
    int nb_threads = 16;
    int numBlocks = ((size + (nb_threads-1))/nb_threads, (size + (nb_threads-1))/nb_threads);
    int threadsPerBlock = nb_threads;

    // Appel du kernel CUDA
    addArrays<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, size);

    // Copie du résultat depuis le GPU vers le CPU
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Affichage du résultat
    for (int i = 0; i < size; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Libération de la mémoire du GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}

__global__ void addArrays(int* a, int* b, int* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}