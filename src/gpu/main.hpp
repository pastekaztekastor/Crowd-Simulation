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

// Enum
enum _Element { EMPTY, HUMAN, WALL, EXIT };

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
        int * simDimX,                  // (*) Because change
        int * simDimY,                  // (*) Because change
        int * simDimP,                  // (*) Because change
        int * simPIn,                   // (*) Because change
        int * simDimG,                  // (*) Because change
        int * settings_print,           // (*) Because change
        int * settings_debugMap,        // (*) Because change
        int * settings_model,           // (*) Because change
        int * settings_exportType,      // (*) Because change
        int * settings_exportFormat,    // (*) Because change
        int * settings_finishCondition, // (*) Because change
        std::string * settings_dir,     // (*) Because change
        std::string * settings_dirName, // (*) Because change
        std::string * settings_fileName // (*) Because change
);
    /**
     * @brief 
     * 
     * @param argc                      ()
     * @param argv                      ()
     * @param simDimX                   ()
     * @param simDimY                   ()
     * @param simDimP                   ()
     * @param simPIn                    ()
     * @param simDimG                   ()
     * @param settings_print            ()
     * @param settings_debugMap         ()
     * @param settings_model            ()
     * @param settings_exportType       ()
     * @param settings_exportFormat     ()
     * @param settings_finishCondition  ()
     * @param settings_dir              ()
     * @param settings_dirName          ()
     * @param settings_fileName         ()
    */

void initPopulationPositionMap(
    int *** populationPosition,         // (*) Because change
    enum _Element *** map,              // (*) Because change
    int * simExit,
    int simDimX,
    int simDimY,
    int simDimP,
    int settings_print
);
    /**
     * @brief 
     * 
     * @param populationPosition        ()
     * @param map                       ()
     * @param simExit                   ()
     * @param simDimX                   ()
     * @param simDimY                   ()
     * @param simDimP                   ()
     * @param settings_print            ()
    */

void initCost(
    int *** cost,                       // (*) Because change
    enum _Element ** map,
    int * simExit,   
    int simDimX,
    int simDimY,
    int settings_print
);
    /**
     * @brief 
     * 
     * @param cost                      ()
     * @param map                       ()
     * @param simExit                   ()
     * @param simDimX                   ()
     * @param simDimY                   ()
     * @param settings_print            ()
    */

void initSimExit(
    int ** simExit,                     // (*) Because change
    int simDimX,
    int simDimY,
    int settings_print
);
    /**
     * @brief 
     * 
     * @param simExit                   ()
     * @param simDimX                   ()
     * @param simDimY                   ()
     * @param settings_print            ()
    */

void initPopulationIndex(
    int ** populationIndex,             // (*) Because change
    int simDimP,
    int settings_print
);
    /**
     * @brief 
     * 
     * @param populationIndex           ()
     * @param simDimP                   ()
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
    int ** simExit,                     // (*) Because change
    int posX,
    int posY,
    int settings_print
);
    /**
     * @brief 
     * 
     * @param simExit                   ()
     * @param  posX                     ()
     * @param  posY                     ()
     * @param settings_print            ()
    */

void setPopulationPositionMap(
    int *** populationPosition,         // (*) Because change
    enum _Element *** map,              // (*) Because change
    int * simExit,
    int simDimX,
    int simDimY,
    int settings_print
);
    /**
     * @brief 
     * 
     * @param populationPosition        ()
     * @param map                       ()
     * @param simExit                   ()
     * @param simDimX                   ()
     * @param simDimY                   ()
     * @param settings_print            ()
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
    int progress, 
    int total, 
    int width,
    int settings_print
);

void shuffleIndex(
    int ** PopulationIndex,
    int simDimP,
    int settings_print
);


// int  lauchModel(
//     int *** populationPosition, // (*) Because change
//     enum _Element *** map,      // (*) Because change
//     int * simPIn,               // (*) Because change
//     int ** cost,
//     int * simExit,   
//     int simDimX,
//     int simDimY,
//     int settings_print
// );

/*
  ______             
 |  ____|            
 | |__ _ __ ___  ___ 
 |  __| '__/ _ \/ _ \
 | |  | | |  __/  __/
 |_|  |_|  \___|\___|
                     
*/

void freePopulationPosition (
    int *** populationPosition,         // (*) Because change
    int simDimP,
    int settings_print
);
void freePopulationIndex (
    int ** populationIndex,             // (*) Because change
    int settings_print
);
void freeCost (
    int *** cost,                       // (*) Because change
    int simDimY,
    int settings_print
);
void freeMap (
    enum _Element *** map,              // (*) Because change
    int simDimY,
    int settings_print
);