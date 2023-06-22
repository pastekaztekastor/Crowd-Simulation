/*******************************************************************************
* File Name: main.hpp
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: main header
*******************************************************************************/

// Include necessary libraries here
#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>        // chdir
#include <sys/stat.h>      // mkdir
#include <cuda_runtime.h>
/*
  _____       _ _ 
 |_   _|     (_) |  
   | |  _ __  _| |_ 
   | | | '_ \| | __|
  _| |_| | | | | |_ 
 |_____|_| |_|_|\__|

*/

// Declare functions and classes here
// Launch simulation
void    initSimParam(
        int argc,
        char const *argv[],
        uint2 * simDim,                     // (*) Because change
        uint * simDimP,                     // (*) Because change
        uint * simPIn,                      // (*) Because change
        uint * simDimG,                     // (*) Because change
        uint * settings_print,              // (*) Because change
        uint * settings_debugMap,           // (*) Because change
        uint * settings_model,              // (*) Because change
        uint * settings_exportType,         // (*) Because change
        uint * settings_exportFormat,       // (*) Because change
        uint * settings_finishCondition,    // (*) Because change
        std::string * settings_dir,                 // (*) Because change
        std::string * settings_dirName,             // (*) Because change
        std::string * settings_fileName             // (*) Because change
);
    /**
     * @brief 
     * 
     * @param argc                                  ()
     * @param argv                                  ()
     * @param simDim     ()
     * @param simDimP                               ()
     * @param simPIn                                ()
     * @param simDimG                               ()
     * @param settings_print                        ()
     * @param settings_debugMap                     ()
     * @param settings_model                        ()
     * @param settings_exportType                   ()
     * @param settings_exportFormat                 ()
     * @param settings_finishCondition              ()
     * @param settings_dir                          ()
     * @param settings_dirName                      ()
     * @param settings_fileName                     ()
    */

void initPopulationPositionMap(
    uint2        ** populationPosition,            // (*) Because change
    int          ** map,                           // (*) Because change
    uint2        simExit,
    uint2 simDim,
    uint simDimP,
    uint settings_print
);
    /**
     * @brief 
     * 
     * @param populationPosition                    ()
     * @param map                                   ()
     * @param simExit                               ()
     * @param simDim                               ()
     * @param simDimP                               ()
     * @param settings_print                        ()
    */

void initCost(
    uint ** cost,                           // (*) Because change
    int          * map,
    uint2        simExit,   
    uint2 simDim,
    uint settings_print
);
    /**
     * @brief 
     * 
     * @param cost                                  ()
     * @param map                                   ()
     * @param simExit                               ()
     * @param simDim,                               ()
     * @param settings_print                        ()
    */

void initSimExit(
    uint2        * simExit,                         // (*) Because change
    uint2 simDim,
    uint settings_print
);
    /**
     * @brief 
     * 
     * @param simExit                               ()
     * @param simDim()
     * @param settings_print                        ()
    */

void initPopulationIndex(
    uint ** populationIndex,                // (*) Because change
    uint simDimP,
    uint settings_print
);
    /**
     * @brief 
     * 
     * @param populationIndex                       ()
     * @param simDimP                               ()
    */

/*
   _____      _   _            
  / ____|    | | | |           
 | (___   ___| |_| |_ ___ _ __ 
  \___ \ / _ \ __| __/ _ \ '__|
  ____) |  __/ |_| ||  __/ |   
 |_____/ \___|\__|\__\___|_|   
                               
*/

void setSimExit(
    uint2        * simExit,                         // (*) Because change
    uint posX,
    uint posY,
    uint settings_print
);
    /**
     * @brief 
     * 
     * @param simExit                               ()
     * @param posX                                  ()
     * @param posY                                  ()
     * @param settings_print                        ()
    */

void setPopulationPositionMap(
    uint2        ** populationPosition,              // (*) Because change
    uint ** map,                            // (*) Because change
    uint2        simExit,
    uint2 simDim,
    uint settings_print
);
    /**
     * @brief 
     * 
     * @param populationPosition                    ()
     * @param map                                   ()
     * @param simExit                               ()
     * @param simDim                                ()
     * @param settings_print                        ()
    */

/*
   _____ _                 _       _   _             
  / ____(_)               | |     | | (_)            
 | (___  _ _ __ ___  _   _| | __ _| |_ _  ___  _ __  
  \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \ 
  ____) | | | | | | | |_| | | (_| | |_| | (_) | | | |
 |_____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|
                                                     
*/

void progressBar(
    uint progress, 
    uint total, 
    uint width,
    uint settings_print
);

void shuffleIndex(
    uint ** PopulationIndex,        // (*) Because change
    uint simDimP,
    uint settings_print
);


// uint  lauchModel(
//     uint *** populationPosition, // (*) Because change
//     uint *** map,                // (*) Because change
//     uint * simPIn,               // (*) Because change
//     uint ** cost,
//     uint * simExit,   
//     uint2 simDim,
//     uint settings_print
// );

/*
  ______             
 |  ____|            
 | |__ _ __ ___  ___ 
 |  __| '__/ _ \/ _ \
 | |  | | |  __/  __/
 |_|  |_|  \___|\___|
                     
*/

void freeTab ( uint2 ** populationPosition, uint settings_print);

/*
  _    _ _   _ _     
 | |  | | | (_) |    
 | |  | | |_ _| |___ 
 | |  | | __| | / __|
 | |__| | |_| | \__ \
  \____/ \__|_|_|___/
                     
*/
void printMap(int * map, uint2 simDim, uint settings_print);
uint xPosof(uint value, uint dimX, uint dimY);
uint yPosof(uint value, uint dimX, uint dimY);
uint valueOfxy(uint xPos, uint yPos, uint dimX, uint dimY);
void printPopulationPosition(uint2 * population, uint simDimP);