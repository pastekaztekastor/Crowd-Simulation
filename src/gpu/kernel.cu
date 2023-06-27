/*******************************************************************************
* File Name: kernel.cu
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: 
*******************************************************************************/

#include "kernel.hpp"

/*
  _____       _ _   
 |_   _|     (_) |  
   | |  _ __  _| |_ 
   | | | '_ \| | __|
  _| |_| | | | | |_ 
 |_____|_| |_|_|\__|
                    
*/
void initKernelParam(kernelParam * _kernelParam, simParam _simParam, settings _settings){
    if( _settings.print > 2 )cout << endl << " ### Init kernel params ###" << endl;

    if( _settings.print > 2 )cout << " \t> Malloc";
    cudaMalloc( &_kernelParam->populationPosition , 2 * sizeof(uint) * _simParam.nbIndividual); 
    cudaMalloc( &_kernelParam->map                , _simParam.dimension.x * _simParam.dimension.y * sizeof(uint ));
    cudaMalloc( &_kernelParam->pInSim             , sizeof(uint));
    if( _settings.print > 2 )cout  << " OK " << endl;
    if( _settings.print > 2 )cout << " \t> Copy";
    cudaMemcpy( (void**) _kernelParam->populationPosition, _simParam.populationPosition, (2 * sizeof(uint) * _simParam.nbIndividual)                     , cudaMemcpyHostToDevice);
    cudaMemcpy( (void**) _kernelParam->map               , _simParam.map               , (_simParam.dimension.x * _simParam.dimension.y * sizeof(uint))  , cudaMemcpyHostToDevice);
    cudaMemcpy( (void**) _kernelParam->pInSim            , &_simParam.pInSim           , sizeof(uint)                                                    , cudaMemcpyHostToDevice);
    if( _settings.print > 2 )cout  << " OK " << endl;
    if( _settings.print > 2 )cout << " \t> Threads & blocks" ;
    _kernelParam->nb_threads = 32;
    _kernelParam->blocks = ((_simParam.nbIndividual + (_kernelParam->nb_threads-1))/_kernelParam->nb_threads);
    _kernelParam->threads = (_kernelParam->nb_threads);
    if( _settings.print > 2 )cout  << " OK " << endl;
}

/*
  _  __                    _     
 | |/ /                   | |    
 | ' / ___ _ __ _ __   ___| |___ 
 |  < / _ \ '__| '_ \ / _ \ / __|
 | . \  __/ |  | | | |  __/ \__ \
 |_|\_\___|_|  |_| |_|\___|_|___/

*/
__global__ void kernel_model1_GPU(kernelParam _kernelParam, simParam _simParam, settings _settings){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if(tid < _simParam.nbIndividual)
    {
        if ( _kernelParam.populationPosition[tid].x > -1 && _kernelParam.populationPosition[tid].y > -1){
            // printf(" frame %d id %d\n", _simParam.nbFrame, tid);
            // position de l'individue tid
            uint2 pos    = make_uint2(_kernelParam.populationPosition[tid].x, _kernelParam.populationPosition[tid].y);
            int2  delta  = make_int2(_simParam.exit.x - pos.x, _simParam.exit.y - pos.y);
            uint  maxDim = max(abs(delta.x), abs(delta.y));
            int2  move   = make_int2(delta.x / (int) maxDim, delta.y / (int) maxDim);
            // on regarde si la case est disponible 
            //printf("atomicExch ")
            int oldValue = atomicExch(&_kernelParam.map[_simParam.dimension.x * (pos.y + move.y) + (pos.x + move.x)], tid);
            switch (oldValue)
            {
            case __MAP_EMPTY__:
                _kernelParam.populationPosition[tid] = make_int2(pos.x + move.x, pos.y + move.y);           // Position dans populationPosition
                atomicExch(&_kernelParam.map[_simParam.dimension.x * (pos.y) + (pos.x)], __MAP_EMPTY__); 
                break;
            case __MAP_EXIT__:
                _kernelParam.populationPosition[tid] =  __MAP_HUMAN_QUITE__;                                    // Position dans populationPosition
                atomicExch(&_kernelParam.map[_simParam.dimension.x * (pos.y) + (pos.x)], __MAP_EMPTY__); 
                _kernelParam.map[_simParam.dimension.x * (pos.y + move.y) + (pos.x + move.x)] =  __MAP_EXIT__;  // Valeur de la nouvelle position map qui doit rester la sortie
                (*_kernelParam.pInSim) --;
                break;
            default:
                break;
            }
        }
    }
}
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
void mapKernelToSim(kernelParam _kernelParam, simParam * _simParam, settings _settings){
    if( _settings.print > 2 )cout <<endl<< " \t> mapKernelToSim " << endl;
    cudaMemcpy(_simParam->map, _kernelParam.map, _simParam->dimension.x * _simParam->dimension.y * sizeof(uint), cudaMemcpyDeviceToHost);
    if( _settings.print > 2 )cout  << " OK " << endl;
}
void popKernelToSim(kernelParam _kernelParam, simParam * _simParam, settings _settings){
    if( _settings.print > 2 )cout <<endl<< " \t> popKernelToSim " << endl;
    cudaMemcpy(_simParam->populationPosition, _kernelParam.populationPosition, 2 * sizeof(uint) * _simParam->nbIndividual, cudaMemcpyDeviceToHost);
    if( _settings.print > 2 )cout  << " OK " << endl;
}
void pInKernelToSim(kernelParam _kernelParam, simParam * _simParam, settings _settings){
    if( _settings.print > 2 )cout <<endl<< " \t> mapKernelToSim " << endl;
    cudaMemcpy(&_simParam->pInSim, _kernelParam.pInSim, sizeof(uint), cudaMemcpyDeviceToHost);
    if( _settings.print > 2 )cout  << " OK " << endl;
} 

/*
  ______             
 |  ____|            
 | |__ _ __ ___  ___ 
 |  __| '__/ _ \/ _ \
 | |  | | |  __/  __/
 |_|  |_|  \___|\___|
                     
*/
void destroyKernel(kernelParam * _kernelParam, settings _settings){}