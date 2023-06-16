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
    unsigned int **     populationPosition       = nullptr; // Position table of all the individuals in the simulation [[x,y],[x,y],...]
    unsigned int **     cost                     = nullptr; //
    enum _Element **    map                      = nullptr; // 2D map composed of the enum _Element
    unsigned int *      simExit                  = nullptr; // [x,y] coordinate of simulation output
    unsigned int *      populationIndex          = nullptr; // List of individual indexes so as not to disturb the order of the individuals
    unsigned int        simDimX                  = 10;      // Simulation x-dimension
    unsigned int        simDimY                  = 10;      // Simulation y-dimension
    unsigned int        simDimG                  = 10;      // Number of generation that the program will do before stopping the simulation
    unsigned int        simDimP                  = 10;      // Number of individuals who must evolve during the simulation
    unsigned int        simPIn                   = simDimP; //
    unsigned int        simIsFinish              = 0;       //
    unsigned int        settings_print           = 2;       // For display the debug [lvl to print]
    unsigned int        settings_debugMap        = 0;       // For display map
    unsigned int        settings_model           = 0;       //
    unsigned int        settings_exportType      = 0;       //
    unsigned int        settings_exportFormat    = 0;       //
    unsigned int        settings_finishCondition = 0;       //
    std::string         settings_dir             = "bin/";  // For chose the directory to exporte bin files
    std::string         settings_dirName         = "";      //
    std::string         settings_fileName        = "";      //

    // 
    //unsigned int (*pModel) (int *** ,enum _Element *** ,int * ,int ** ,int * ,int ,int ,int);
    // 
    //unsigned int (*pExport) ();

    initSimParam(
        argc,
        argv,
        &simDimX,
        &simDimY,
        &simDimP,
        &simPIn,
        &simDimG,
        &settings_print,
        &settings_debugMap,
        &settings_model,
        &settings_exportType,
        &settings_exportFormat,
        &settings_finishCondition,
        &settings_dir,
        &settings_dirName,
        &settings_fileName
    );

    if( settings_print > 2 )std::cout  << " ### Init simulation ###" << std::endl;
    srand(time(NULL));
    initSimExit( &simExit, simDimX, simDimY, settings_print );
    initPopulationPositionMap( &populationPosition, &map, simExit, simDimX, simDimY, simDimP, settings_print );
    initPopulationIndex( &populationIndex, simDimP, settings_print );
    initCost( &cost, map, simExit, simDimX, simDimY, settings_print );
    
    if( settings_print > 2 )std::cout << std::endl << " ### Start simulation ###" << std::endl;
    while (simIsFinish == 0){
        if (simPIn == 0) simIsFinish = 1; 
        
        progressBar(simDimP - simPIn, simDimP, 100, 0);
        shuffleIndex(&populationIndex, simDimP, 0);
        
        // MODEL
        switch (settings_model){
            case 0: // MODEL : sage ignorant
                // TO DO
                simPIn--;
                break;

            case 1: // MDOEL : Impatient ignorant
            case 2: // MDOEL : Forcée
            case 3: // MDOEL : Conne de vision
            case 4: // MDOEL : Meilleur coût
            case 5: // MDOEL : Meilleur déplacement 
            default:
                simPIn--;
                break;
        } 
        // EXPORT 
        switch (settings_model){
            case 1:
                // TO DO  
                break;
            
            default:
                break;
        } 
    }
    
    std::cout << std::endl << "CUDA TEST ADDITION" << std::endl;

    cudaTest();

    if( settings_print > 2 )std::cout  << std::endl  << std::endl << "### Memory free ###" << std::endl;
    freePopulationPosition (&populationPosition, simDimP, settings_print);
    freePopulationIndex (&populationIndex, settings_print);
    // freeCost (&cost, simDimY, settings_print);
    freeMap (&map, simDimY, settings_print);

    return 0;
}
