/*******************************************************************************
* File Name: main.cpp
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: main file of the crowd simulation with parallelization on GPU. Contains only the main program.
*******************************************************************************/

// Include necessary libraries here
#include "kernel.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    simParam _simParam;
    settings _settings;
    kernelParam _kernelParam;

    if( _settings.print > 2 )cout  << " ### Init simulation ###" << endl;
    srand(time(NULL));
    initSimSettings(argc, argv, &_simParam, &_settings);
    initPopulationPositionMap(&_simParam, _settings);
    //initKernelParam(&_kernelParam, _simParam, _settings);
    //printMap(_simParam, _settings);

    while (_simParam.isFinish == 0){
        if (_simParam.pInSim == 0) _simParam.isFinish = 1; 

        _simParam.nbFrame ++;
        progressBar(_simParam.nbIndividual - _simParam.pInSim, _simParam.nbIndividual, 100, _simParam.nbFrame);
        shuffleIndex(&_simParam, _settings);
        
        // MODEL
        switch (_settings.model){
            case 0: // MODEL : sage ignorant
                // TO DO
                // kernel_model1_GPU<<<blocks,threads>>>(dev_populationPosition, dev_map, dev_simPIn, cost, simExit, simDim, simDimP);
                
                for (size_t tid = 0; tid < _simParam.nbIndividual; tid++)
                {
                    if (_simParam.populationPosition[tid].x > -1 && _simParam.populationPosition[tid].y > -1){
                        // position de l'individue tid
                        uint2 pos    = make_uint2(_simParam.populationPosition[tid].x, _simParam.populationPosition[tid].y);
                        int2  delta  = make_int2(_simParam.exit.x - pos.x, _simParam.exit.y - pos.y);
                        uint  maxDim = max(abs(delta.x), abs(delta.y));
                        int2  move   = make_int2(delta.x / (int) maxDim, delta.y / (int) maxDim);
                        // on regarde si la case est disponible 
                        if(_simParam.map[ _simParam.dimension.x * (pos.y+move.y) + (pos.x + move.x)] == -1){ // if is EMPTY
                            _simParam.populationPosition[tid] = make_int2(pos.x + move.x, pos.y + move.y);     // Position dans populationPosition
                            _simParam.map[_simParam.dimension.x * pos.y + pos.x]                        = -1;  // Valeur de l'ancien position map
                            _simParam.map[_simParam.dimension.x * (pos.y+move.y) + (pos.x + move.x)]    = tid; // Valeur de la nouvelle position map
                        }
                        else if(_simParam.map[ _simParam.dimension.x * (pos.y+move.y) + (pos.x + move.x)] == -2){ // Sorti
                            _simParam.populationPosition[tid] = make_int2(-1,-1);                              // Position dans populationPosition
                            _simParam.map[_simParam.dimension.x * pos.y + pos.x]                        = -1;  // Valeur de l'ancien position map
                            _simParam.pInSim --;
                        }
                    }
                }
                break;

            case 1: // MDOEL : Impatient ignorant
            case 2: // MDOEL : Forcée
            case 3: // MDOEL : Conne de vision
            case 4: // MDOEL : Meilleur coût
            case 5: // MDOEL : Meilleur déplacement 
            default:
                _simParam.pInSim--;
                break;
        }
        // EXPORT 
        switch (_settings.exportType){
            case 1:
                // TO DO  
                break;
            
            default:
                break;
        }
    }
    
    if( _settings.print > 2 )cout <<endl<< " \t> Cuda Copy ";
    //cudaMemcpy(outMove           , dev_outMove           , sizeof(uint)                      , cudaMemcpyDeviceToHost);
    //cudaMemcpy(populationPosition, dev_populationPosition, 2 * sizeof(uint) * simDimP        , cudaMemcpyDeviceToHost);
    //cudaMemcpy(map               , dev_map               , simDim.x * simDim.y * sizeof(uint ) , cudaMemcpyDeviceToHost);
    if( _settings.print > 2 )cout  << " OK " << endl;

    cout << endl;
    cout << "solved on " << _simParam.nbFrame << " frames";
    //printMap(_simParam, _settings);

    return 0;
}
