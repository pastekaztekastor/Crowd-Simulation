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
    cudaMalloc( &_kernelParam->populationPosition , sizeof(float3) * _simParam.nbIndividual); 
    cudaMalloc( &_kernelParam->cost               , _simParam.dimension.x * _simParam.dimension.y * sizeof(uint)); 
    cudaMalloc( &_kernelParam->map                , _simParam.dimension.x * _simParam.dimension.y * sizeof(int));
    cudaMalloc( &_kernelParam->pInSim             , sizeof(uint));
    if( _settings.print > 2 )cout  << " OK " << endl;
    if( _settings.print > 2 )cout << " \t> Copy";
    cudaMemcpy( (void**) _kernelParam->populationPosition, _simParam.populationPosition, (sizeof(float3) * _simParam.nbIndividual)                             , cudaMemcpyHostToDevice);
    cudaMemcpy( (void**) _kernelParam->cost              , _simParam.cost              , (_simParam.dimension.x * _simParam.dimension.y * sizeof(uint))  , cudaMemcpyHostToDevice);
    cudaMemcpy( (void**) _kernelParam->map               , _simParam.map               , (_simParam.dimension.x * _simParam.dimension.y * sizeof(int))   , cudaMemcpyHostToDevice);
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
                _kernelParam.populationPosition[tid] = make_float3(pos.x + move.x, pos.y + move.y, 0.f);           // Position dans populationPosition
                atomicExch(&_kernelParam.map[_simParam.dimension.x * (pos.y) + (pos.x)], __MAP_EMPTY__); 
                break;
            case __MAP_EXIT__:
                _kernelParam.populationPosition[tid] =  __MAP_HUMAN_QUITE__;                                                    // Position dans populationPosition
                atomicExch(&_kernelParam.map[_simParam.dimension.x * (pos.y) + (pos.x)], __MAP_EMPTY__); 
                _kernelParam.map[_simParam.dimension.x * (pos.y + move.y) + (pos.x + move.x)] =  __MAP_EXIT__;                  // Valeur de la nouvelle position map qui doit rester la sortie
                (*_kernelParam.pInSim) --;
                break;
            default:
                _kernelParam.populationPosition[tid] = make_float3(pos.x, pos.y,_kernelParam.populationPosition[tid].z + 1.f);  // Position dans populationPosition
                break;
            }
        }
    }
}


__global__ void kernel_costMap_GPU(kernelParam _kernelParam, simParam _simParam, settings _settings){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < _simParam.nbIndividual){
        if ( _kernelParam.populationPosition[tid].x > -1 && _kernelParam.populationPosition[tid].y > -1){
            if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL TID = %d \n", tid);
            // position de l'individue tid
            uint2 pos     = make_uint2(_kernelParam.populationPosition[tid].x, _kernelParam.populationPosition[tid].y);
            uint  cost    = _kernelParam.cost[pos.y * _simParam.dimension.x + pos.x];
            uint2 nextPos = pos;
            if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL position : [%d,%d] avec un cout de %d\n", pos.x,pos.y,cost);
            
            // Définir les déplacements possibles (haut)
            int2 newPos  = make_int2((int)pos.x + 0, (int)pos.y + -1);
            if (newPos.x >=0 && newPos.x < _simParam.dimension.x && newPos.y >=0 && newPos.y < _simParam.dimension.y){
                uint  newCost = _kernelParam.cost[newPos.y * _simParam.dimension.x + newPos.x];
                if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL position : [%d,%d] avec un cout de %d\n", newPos.x,newPos.y,newCost);
                // Vérifier si les nouvelles sont plus intéréssante.
                if (newCost <= cost) {
                    if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf("     Meilleur déplacement en haut \n");
                    nextPos = make_uint2(newPos.x,newPos.y);
                }
            }// Définir les déplacements possibles (bas)
            newPos  = make_int2((int)pos.x + 0, (int)pos.y + 1);
            if (newPos.x >=0 && newPos.x < _simParam.dimension.x && newPos.y >=0 && newPos.y < _simParam.dimension.y){
                uint  newCost = _kernelParam.cost[newPos.y * _simParam.dimension.x + newPos.x];
                if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL position : [%d,%d] avec un cout de %d\n", newPos.x,newPos.y,newCost);
                // Vérifier si les nouvelles sont plus intéréssante.
                if (newCost <= cost) {
                    if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf("     Meilleur déplacement en bas \n");
                    nextPos = make_uint2(newPos.x,newPos.y);
                }
            }// Définir les déplacements possibles (gauche)
            newPos  = make_int2((int)pos.x + -1, (int)pos.y + 0);
            if (newPos.x >=0 && newPos.x < _simParam.dimension.x && newPos.y >=0 && newPos.y < _simParam.dimension.y){
                uint  newCost = _kernelParam.cost[newPos.y * _simParam.dimension.x + newPos.x];
                if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL position : [%d,%d] avec un cout de %d\n", newPos.x,newPos.y,newCost);
                // Vérifier si les nouvelles sont plus intéréssante.
                if (newCost <= cost) {
                    if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf("     Meilleur déplacement à gauche \n");
                    nextPos = make_uint2(newPos.x ,newPos.y);
                }
            }// Définir les déplacements possibles (droite)
            newPos  = make_int2((int)pos.x + 1, (int)pos.y + 0);
            if (newPos.x >=0 && newPos.x < _simParam.dimension.x && newPos.y >=0 && newPos.y < _simParam.dimension.y){
                uint  newCost = _kernelParam.cost[newPos.y * _simParam.dimension.x + newPos.x];
                if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL position : [%d,%d] avec un cout de %d\n", newPos.x,newPos.y,newCost);
                // Vérifier si les nouvelles sont plus intéréssante.
                if (newCost <= cost) {
                    if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf("     Meilleur déplacement à droite \n");
                    nextPos = make_uint2(newPos.x,newPos.y);
                }
            }
            if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL : nest pos = [%d,%d]\n", nextPos.x, nextPos.y);
            // on regarde si la case est disponible 
            int oldValue = atomicExch(&_kernelParam.map[_simParam.dimension.x * (nextPos.y) + (nextPos.x)], tid);
            if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL : on regarde si la case est disponible : %d\n", oldValue);
            switch (oldValue)
            {
            case __MAP_EMPTY__:
                _kernelParam.populationPosition[tid] = make_float3(nextPos.x, nextPos.y, 0.f);           // Position dans populationPosition
                if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL : __MAP_EMPTY__ -> \n");
                atomicExch(&_kernelParam.map[_simParam.dimension.x * (pos.y) + (pos.x)], __MAP_EMPTY__); 
                break;
            case __MAP_EXIT__:
                _kernelParam.populationPosition[tid] =  __MAP_HUMAN_QUITE__;                                    // Position dans populationPosition
                if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL : __MAP_EXIT__ -> \n");
                atomicExch(&_kernelParam.map[_simParam.dimension.x * (pos.y) + (pos.x)], __MAP_EMPTY__); 
                _kernelParam.map[_simParam.dimension.x * (nextPos.y) + (nextPos.x)] =  __MAP_EXIT__;  // Valeur de la nouvelle position map qui doit rester la sortie
                (*_kernelParam.pInSim) --;
                break;
            default:
                if (_settings.print >= __DEBUG_PRINT_DEBUG__) printf(" @@@ KERNEL : default -> \n");
                _kernelParam.populationPosition[tid] = make_float3(pos.x, pos.y,_kernelParam.populationPosition[tid].z + 1.f); // Position dans populationPosition
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
    cudaMemcpy(_simParam->populationPosition, _kernelParam.populationPosition, sizeof(float3) * _simParam->nbIndividual, cudaMemcpyDeviceToHost);
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