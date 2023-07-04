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
    simParam        _simParam;
    settings        _settings;
    kernelParam     _kernelParam;
    exportData      _exportData;

    if( _settings.print >= __DEBUG_PRINT_STEP__ )cout  << " ### Init simulation ###" << endl;
    srand(time(NULL));
    initSimSettings(argc, argv, &_simParam, &_settings);
    initPopulationPositionMap(&_simParam, _settings);
    if(_settings.print >= __DEBUG_PRINT_DEBUG__ ) printMap(_simParam, _settings);
    if(_settings.print >= __DEBUG_PRINT_DEBUG__ ) printPopulationPosition(_simParam, _settings);
    initExportData(_simParam, &_exportData, _settings);
    initCostMap(&_simParam, _settings);
    if(_settings.print >= __DEBUG_PRINT_DEBUG__ ) printCostMap(_simParam, _settings);
    
    // printMap(_simParam, _settings);
    // printPopulationPosition(_simParam, _settings);
   
    initKernelParam(&_kernelParam, _simParam, _settings);
    
    if( _settings.print >= __DEBUG_PRINT_STEP__ )cout  << " ### Launch simulation ###" << endl;
    //while (_simParam.isFinish == 0 ){
    while (_simParam.nbFrame <= _simParam.nbIndividual*3 ){
        _simParam.nbFrame ++;
        
        if( _settings.print >= __DEBUG_PRINT_DEBUG__ )cout << "------------ FRAME " << _simParam.nbFrame << " ------------" << endl;
        
        if (_simParam.pInSim <= 0) _simParam.isFinish = 1; 

        if(_settings.print <= __DEBUG_PRINT_ALL__) progressBar(_simParam.nbIndividual - _simParam.pInSim, _simParam.nbIndividual, 100, _simParam.nbFrame);
        //shuffleIndex(&_simParam, _settings);
        
        // MODEL
        switch (_settings.model){
            case 0: // MODEL : sage ignorant
                kernel_model1_GPU<<<_kernelParam.blocks,_kernelParam.threads>>>(_kernelParam, _simParam, _settings);
                break;
            case 1: // MDOEL : Impatient ignorant
                kernel_costMap_GPU<<<_kernelParam.blocks,_kernelParam.threads>>>(_kernelParam, _simParam, _settings);
                break;
            case 2: // MDOEL : Forcée
            case 3: // MDOEL : Conne de vision
            case 4: // MDOEL : Meilleur coût
            case 5: // MDOEL : Meilleur déplacement 
            default:
                break;
        }
        // exportData 
        switch (_settings.exportDataType){
            case 1:
                // TO DO  
                break;
            
            default:
                break;
        }

        popKernelToSim(_kernelParam, &_simParam, _settings);
        pInKernelToSim(_kernelParam, &_simParam, _settings);
        

        if(_settings.print >= __DEBUG_PRINT_DEBUG__ ) cout << "  -  P IN : " << _simParam.pInSim << endl;
        if(_settings.print >= __DEBUG_PRINT_DEBUG__ ) printMap(_simParam, _settings);
        if(_settings.print >= __DEBUG_PRINT_DEBUG__ ) printPopulationPosition(_simParam, _settings);

        switch (_settings.exportDataType)
        {
        case  __EXPORT_TYPE_ALL__:
                exportDataFrameVideo(_simParam, &_exportData, _settings);
                exportDataFrameValue(_simParam, &_exportData, _settings);
            break;
        case  __EXPORT_TYPE_VIDEO__:
                exportDataFrameVideo(_simParam, &_exportData, _settings);
            break;
        case  __EXPORT_TYPE_VALUE__:
                exportDataFrameValue(_simParam, &_exportData, _settings);
            break;
        
        default:
            break;
        }
    }
    
        
    if( _settings.print >= __DEBUG_PRINT_ALL__ )cout << endl<< "solved on " << _simParam.nbFrame << " Frames" << endl << endl;
    
    if( _settings.print >= __DEBUG_PRINT_STEP__ )cout  << " ### Export simulation ###" << endl;
    switch (_settings.exportDataType)
    {
    case  __EXPORT_TYPE_ALL__:
            saveExportDataVideo(_simParam, _exportData, _settings);
            saveExportDataValue(_simParam, _exportData, _settings);
        break;
    case  __EXPORT_TYPE_VIDEO__:
            saveExportDataVideo(_simParam, _exportData, _settings);
        break;
    case  __EXPORT_TYPE_VALUE__:
            saveExportDataValue(_simParam, _exportData, _settings);
        break;
    
    default:
        break;
    }
    cout << endl;
    if( _settings.print >= __DEBUG_PRINT_STEP__ )cout  << " ### Free memory ###" << endl << std::numeric_limits<uint>::max() << " " << UINT_MAX;
    // TO DO 
    cout << endl;
    return 0;
}
