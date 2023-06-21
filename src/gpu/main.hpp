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
        unsigned int * simDimX,                     // (*) Because change
        unsigned int * simDimY,                     // (*) Because change
        unsigned int * simDimP,                     // (*) Because change
        unsigned int * simPIn,                      // (*) Because change
        unsigned int * simDimG,                     // (*) Because change
        unsigned int * settings_print,              // (*) Because change
        unsigned int * settings_debugMap,           // (*) Because change
        unsigned int * settings_model,              // (*) Because change
        unsigned int * settings_exportType,         // (*) Because change
        unsigned int * settings_exportFormat,       // (*) Because change
        unsigned int * settings_finishCondition,    // (*) Because change
        std::string * settings_dir,                 // (*) Because change
        std::string * settings_dirName,             // (*) Because change
        std::string * settings_fileName             // (*) Because change
);
    /**
     * @brief 
     * 
     * @param argc                                  ()
     * @param argv                                  ()
     * @param simDimX                               ()
     * @param simDimY                               ()
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
    unsigned int *** populationPosition,            // (*) Because change
    unsigned int *** map,                           // (*) Because change
    unsigned int * simExit,
    unsigned int simDimX,
    unsigned int simDimY,
    unsigned int simDimP,
    unsigned int settings_print
);
    /**
     * @brief 
     * 
     * @param populationPosition                    ()
     * @param map                                   ()
     * @param simExit                               ()
     * @param simDimX                               ()
     * @param simDimY                               ()
     * @param simDimP                               ()
     * @param settings_print                        ()
    */

void initCost(
    unsigned int *** cost,                          // (*) Because change
    unsigned int ** map,
    unsigned int * simExit,   
    unsigned int simDimX,
    unsigned int simDimY,
    unsigned int settings_print
);
    /**
     * @brief 
     * 
     * @param cost                                  ()
     * @param map                                   ()
     * @param simExit                               ()
     * @param simDimX                               ()
     * @param simDimY                               ()
     * @param settings_print                        ()
    */

void initSimExit(
    unsigned int ** simExit,                        // (*) Because change
    unsigned int simDimX,
    unsigned int simDimY,
    unsigned int settings_print
);
    /**
     * @brief 
     * 
     * @param simExit                               ()
     * @param simDimX                               ()
     * @param simDimY                               ()
     * @param settings_print                        ()
    */

void initPopulationIndex(
    unsigned int ** populationIndex,                // (*) Because change
    unsigned int simDimP,
    unsigned int settings_print
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
    unsigned int ** simExit,                        // (*) Because change
    unsigned int posX,
    unsigned int posY,
    unsigned int settings_print
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
    unsigned int *** populationPosition,            // (*) Because change
    unsigned int *** map,                           // (*) Because change
    unsigned int * simExit,
    unsigned int simDimX,
    unsigned int simDimY,
    unsigned int settings_print
);
    /**
     * @brief 
     * 
     * @param populationPosition                    ()
     * @param map                                   ()
     * @param simExit                               ()
     * @param simDimX                               ()
     * @param simDimY                               ()
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
    unsigned int progress, 
    unsigned int total, 
    unsigned int width,
    unsigned int settings_print
);

void shuffleIndex(
    unsigned int ** PopulationIndex,
    unsigned int simDimP,
    unsigned int settings_print
);


// unsigned int  lauchModel(
//     unsigned int *** populationPosition, // (*) Because change
//     unsigned int *** map,                // (*) Because change
//     unsigned int * simPIn,               // (*) Because change
//     unsigned int ** cost,
//     unsigned int * simExit,   
//     unsigned int simDimX,
//     unsigned int simDimY,
//     unsigned int settings_print
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
    unsigned int *** populationPosition,            // (*) Because change
    unsigned int simDimP,
    unsigned int settings_print
);
void freePopulationIndex (
    unsigned int ** populationIndex,                // (*) Because change
    unsigned int settings_print
);
void freeCost (
    unsigned int *** cost,                          // (*) Because change
    unsigned int simDimY,
    unsigned int settings_print
);
void freeMap (
    unsigned int *** map,                           // (*) Because change
    unsigned int simDimY,
    unsigned int settings_print
);

/*
  _    _ _   _ _     
 | |  | | | (_) |    
 | |  | | |_ _| |___ 
 | |  | | __| | / __|
 | |__| | |_| | \__ \
  \____/ \__|_|_|___/
                     
*/
void printMap(unsigned int ** map, unsigned int simDimX, unsigned int simDimY, unsigned int settings_print);