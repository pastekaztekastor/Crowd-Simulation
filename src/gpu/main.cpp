/*******************************************************************************
* File Name: main.cpp
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: main file of the crowd simulation with parallelization on GPU. Contains only the main program.
*******************************************************************************/

// Include necessary libraries here
#include "main.hpp"

int main(int argc, char const *argv[])
{
    int **              populationPosition       = nullptr; // Position table of all the individuals in the simulation [[x,y],[x,y],...]
    int **              cost                     = nullptr; //
    enum _Element **    map                      = nullptr; // 2D map composed of the enum _Element
    int *               simExit                  = nullptr; // [x,y] coordinate of simulation output
    int *               populationIndex          = nullptr; // List of individual indexes so as not to disturb the order of the individuals
    int                 simDimX                  = 10;      // Simulation x-dimension
    int                 simDimY                  = 10;      // Simulation y-dimension
    int                 simDimG                  = 10;      // Number of generation that the program will do before stopping the simulation
    int                 simDimP                  = 10;      // Number of individuals who must evolve during the simulation
    int                 simPIn                   = simDimP; //
    int                 settings_print           = 0;       // For display the debug [lvl to print]
    int                 settings_debugMap        = 0;       // For display map
    int                 settings_model           = 0;       //
    int                 settings_exportType      = 0;       //
    int                 settings_exportFormat    = 0;       //
    int                 settings_finishCondition = 0;       //
    string              settings_dir             = "bin/";  // For chose the directory to exporte bin files
    string              settings_dirName         = "";      //
    string              settings_fileName        = "";      //

    simParamInit(
        argc,
        argv,
        &simDimX,
        &simDimY,
        &simDimP,
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

    return 0;
}
