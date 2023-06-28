/*******************************************************************************
* File Name: main.cpp
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: main file of the crowd simulation with parallelization on GPU. Contains only the main program.
*******************************************************************************/

// Include necessary libraries here
#include "kernel.hpp"


int main(int argc, char const *argv[])
{
    simParam _simParam;
    settings _settings;
    kernelParam _kernelParam;

    if( _settings.print > 2 )cout  << " ### Init simulation ###" << endl;
    srand(time(NULL));
    initSimSettings(argc, argv, &_simParam, &_settings);
    initPopulationPositionMap(&_simParam, _settings);
    
    printMap(_simParam, _settings);
    printPopulationPosition(_simParam, _settings);
   
    initKernelParam(&_kernelParam, _simParam, _settings);
    

    while (_simParam.isFinish == 0 && _simParam.nbFrame < pow(_simParam.nbIndividual,2)){
        _simParam.nbFrame ++;
        
        //cout << "------------ FRAME " << _simParam.nbFrame << " ------------" << endl;
        
        if (_simParam.pInSim == 0) _simParam.isFinish = 1; 

        //progressBar(_simParam.nbIndividual - _simParam.pInSim, _simParam.nbIndividual, 100, _simParam.nbFrame);
        //shuffleIndex(&_simParam, _settings);
        
        // MODEL
        switch (_settings.model){
            case 0: // MODEL : sage ignorant
                kernel_model1_GPU<<<_kernelParam.blocks,_kernelParam.threads>>>(_kernelParam, _simParam, _settings);
                break;
            case 1: // MDOEL : Impatient ignorant
            case 2: // MDOEL : Forcée
            case 3: // MDOEL : Conne de vision
            case 4: // MDOEL : Meilleur coût
            case 5: // MDOEL : Meilleur déplacement 
            default:
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
<<<<<<< HEAD
        // mapKernelToSim(_kernelParam, &_simParam, _settings);
        popKernelToSim(_kernelParam, &_simParam, _settings);
=======
        mapKernelToSim(_kernelParam, &_simParam, _settings);
        // popKernelToSim(_kernelParam, &_simParam, _settings);
>>>>>>> parent of 7988a53... Ajout de la lib HDF5 + parsseur IMAGE
        pInKernelToSim(_kernelParam, &_simParam, _settings);
        cout << _simParam.pInSim << endl;
        // printPopulationPosition(_simParam, _settings);
<<<<<<< HEAD
        // printMap(_simParam, _settings);
        exportPopulationPosition2HDF5(_simParam, _settings);
=======
        printMap(_simParam, _settings);
>>>>>>> parent of 7988a53... Ajout de la lib HDF5 + parsseur IMAGE
    }
    
    cout << endl << "solved on " << _simParam.nbFrame << " frames" << endl << endl;
    //printMap(_simParam, _settings);

    

    return 0;
}
