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
#include <hdf5.h>
#include <opencv2/opencv.hpp>

using namespace std;

// Define
#define __MAP_EMPTY__         -1
#define __MAP_EXIT__          -2
#define __MAP_WALL__          -3
#define __MAP_HUMAN_QUITE__   make_int2(-1,-1)

#define __EXPORT_TYPE_GIF__   0    
#define __EXPORT_TYPE_VALUE__ 1    
#define __EXPORT_TYPE_ALL__   2 

#define __DEBUG_PRINT_NONE__  0 
#define __DEBUG_PRINT_STEP__  1 
#define __DEBUG_PRINT_ALL__   2 
#define __DEBUG_PRINT_DEBUG__ 3 

#define __MAX_X_DIM_JPEG__    1920
#define __MAX_Y_DIM_JPEG__    1080

#define __COLOR_WHITE__       cv::Scalar(255, 255, 255)
#define __COLOR_RED__         cv::Scalar(255, 0, 0)
#define __COLOR_GREEN__       cv::Scalar(0, 255, 0)
#define __COLOR_BLUE__        cv::Scalar(0, 0, 255)
#define __COLOR_BLACK__       cv::Scalar(0, 0, 0)

#define __GIF_FPS__           50


// Struct
typedef struct {
    float3* populationPosition          ; // Position table of all the individuals in the simulation [[x,y,att],[x,y,att],...]
    uint  * cost                        ; //
    int   * map                         ; // 2D map composed of the uint 0: empty, 1 humain, 2 wall, 3 exit on 1 line
    uint2   exit                        ; // [x,y] coordinate of simulation output
    uint  * populationIndex             ; // List of individual indexes so as not to disturb the order of the individuals
    uint2   dimension                   ; // Simulation x-dimension
    uint    nbIndividual                ; // Number of individuals who must evolve during the simulation
    uint    pInSim                      ; //
    uint    isFinish                    ; //
    uint    nbFrame                     ; // 
} simParam;

typedef struct {
    uint            print               ; // For display the debug [lvl to print]
    uint            debugMap            ; // For display map
    uint            model               ; //
    uint            exportDataType      ; //
    uint            exportDataFormat    ; //
    string          dir                 ; // For chose the directory to exportDatae bin files
    string          dirName             ; //
    string          fileName            ; //
} settings;

typedef struct {
    int2  *     populationPosition      ;
    int   *     map                     ;
    uint  *     pInSim                  ;
    uint        nb_threads              ;
    dim3        blocks                  ;
    dim3        threads                 ;
} kernelParam;

typedef struct {
    vector<cv::Mat>   gifFrames          ;
    string            gifOutputFilename  ;
    int               gifSizeFactor      ;
    string            gifPath            ; 
    int               gifRatioFrame      ;
    
} exportData;


/*
  _____       _ _ 
 |_   _|     (_) |  
   | |  _ __  _| |_ 
   | | | '_ \| | __|
  _| |_| | | | | |_ 
 |_____|_| |_|_|\__|

*/
void initSimSettings            ( int argc, char const *argv[], simParam * _simParam, settings * _settings);
void initPopulationPositionMap  (simParam * _simParam, settings _settings);
void initExportData                 (simParam _simParam, exportData * _exportData, settings _settings);

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
void progressBar                (uint progress, uint total, uint width, uint iteration);
void shuffleIndex               (simParam * _simParam, settings _settings); // NOT USED BUT CODED

void exportDataFrame            (simParam _simParam, exportData * _exportData, settings _settings);
void saveExportData             (simParam _simParam, exportData _exportData, settings _settings);

/*
  ______             
 |  ____|            
 | |__ _ __ ___  ___ 
 |  __| '__/ _ \/ _ \
 | |  | | |  __/  __/
 |_|  |_|  \___|\___|
                     
*/
void freeSimParam                 (simParam * _simParam, settings _settings);

/*
  _    _ _   _ _     
 | |  | | | (_) |    
 | |  | | |_ _| |___ 
 | |  | | __| | / __|
 | |__| | |_| | \__ \
  \____/ \__|_|_|___/
                     
*/
void printMap                     (simParam _simParam, settings _settings);
void printPopulationPosition      (simParam _simParam, settings _settings);

uint xPosof                       (uint value, uint dimX, uint dimY);
uint yPosof                       (uint value, uint dimX, uint dimY);
uint valueOfxy                    (uint xPos, uint yPos, uint dimX, uint dimY);