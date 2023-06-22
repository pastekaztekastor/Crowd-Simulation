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

// Struct
typedef struct {
    uint2 * populationPosition          ; // Position table of all the individuals in the simulation [[x,y],[x,y],...]
    uint  * cost                        ; //
    int   * map                         ; // 2D map composed of the uint 0: empty, 1 humain, 2 wall, 3 exit on 1 line
    uint2   exit                        ; // [x,y] coordinate of simulation output
    uint  * populationIndex             ; // List of individual indexes so as not to disturb the order of the individuals
    uint2   dimension                   ; // Simulation x-dimension
    uint    nbIndividual                ; // Number of individuals who must evolve during the simulation
    uint    pInSim                      ; //
    uint    isFinish                    ; //
} simParam;

typedef struct {
    uint            print               ; // For display the debug [lvl to print]
    uint            debugMap            ; // For display map
    uint            model               ; //
    uint            exportType          ; //
    uint            exportFormat        ; //
    uint            finishCondition     ; //
    std::string     dir                 ; // For chose the directory to exporte bin files
    std::string     dirName             ; //
    std::string     fileName            ; //
} settings;

typedef struct {
    uint **     populationPosition      ;
    uint **     map                     ;
    uint *      simPIn                  ;
    uint        nb_threads              ;
    dim3        blocks                  ;
    dim3        threads                 ;
} kernelParam;

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
void initSimSettings            ( int argc, char const *argv[], simParam * _simParam, settings * _settings);
void initKernelParam            (kernelParam * _kernelParam, simParam _simParam, settings _settings);
void initPopulationPositionMap  (simParam * _simParam, settings _settings);
/*
   _____      _   _            
  / ____|    | | | |           
 | (___   ___| |_| |_ ___ _ __ 
  \___ \ / _ \ __| __/ _ \ '__|
  ____) |  __/ |_| ||  __/ |   
 |_____/ \___|\__|\__\___|_|   
                               
*/

void setSimExit                 (simParam * _simParam, settings _settings);
void setPopulationPositionMap   (simParam * _simParam, settings _settings);

/*
   _____ _                 _       _   _             
  / ____(_)               | |     | | (_)            
 | (___  _ _ __ ___  _   _| | __ _| |_ _  ___  _ __  
  \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \ 
  ____) | | | | | | | |_| | | (_| | |_| | (_) | | | |
 |_____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|
                                                     
*/
void progressBar(uint progress, uint total, uint width, settings _settings);
void shuffleIndex(simParam * _simParam, settings _settings);

/*
  ______             
 |  ____|            
 | |__ _ __ ___  ___ 
 |  __| '__/ _ \/ _ \
 | |  | | |  __/  __/
 |_|  |_|  \___|\___|
                     
*/
void freeSimParam (simParam * _simParam, settings _settings);

/*
  _    _ _   _ _     
 | |  | | | (_) |    
 | |  | | |_ _| |___ 
 | |  | | __| | / __|
 | |__| | |_| | \__ \
  \____/ \__|_|_|___/
                     
*/
void printMap               (simParam _simParam, settings _settings);
void printPopulationPosition(simParam _simParam, settings _settings);

uint xPosof(uint value, uint dimX, uint dimY);
uint yPosof(uint value, uint dimX, uint dimY);
uint valueOfxy(uint xPos, uint yPos, uint dimX, uint dimY);