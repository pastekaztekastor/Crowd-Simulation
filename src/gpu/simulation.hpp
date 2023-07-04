/*******************************************************************************
* File Name: simulation.hpp var @VERSION@
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
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <climits>

using namespace std;

// Define
#define __MAP_EMPTY__         -1
#define __MAP_EXIT__          -2
#define __MAP_WALL__          -3
#define __MAP_HUMAN_QUITE__   make_float3(-1.f,-1.f,0.f)

#define __EXPORT_TYPE_VIDEO__   0    
#define __EXPORT_TYPE_VALUE__ 1    
#define __EXPORT_TYPE_ALL__   2 

#define __DEBUG_PRINT_NONE__  0 
#define __DEBUG_PRINT_STEP__  1 
#define __DEBUG_PRINT_ALL__   2 
#define __DEBUG_PRINT_DEBUG__ 3 

#define __MAX_X_DIM_JPEG__    1920
#define __MAX_Y_DIM_JPEG__    1080

#define __COLOR_WHITE__       cv::Scalar(255, 255, 255)
#define __COLOR_BLUE__        cv::Scalar(255, 0, 0)
#define __COLOR_GREEN__       cv::Scalar(0, 255, 0)
#define __COLOR_RED__         cv::Scalar(0, 0, 255)
#define __COLOR_BLACK__       cv::Scalar(0, 0, 0)

#define __VIDEO_FPS__         25

#define __SIM_MAX_WAITING__   100


// Struct
typedef struct {
    float3* populationPosition          ; // Position table of all the individuals in the simulation [[x,y,att],[x,y,att],...]
    uint2*  wallPosition                ; // Position table of all the individuals in the simulation [[x,y,att],[x,y,att],...]
    uint  * cost                        ; //
    int   * map                         ; // 2D map composed of the uint 0: empty, 1 humain, 2 wall, 3 exit on 1 line
    uint2   exit                        ; // [x,y] coordinate of simulation output
    uint  * populationIndex             ; // List of individual indexes so as not to disturb the order of the individuals
    uint2   dimension                   ; // Simulation x-dimension
    uint    nbIndividual                ; // Number of individuals who must evolve during the simulation
    uint    nbWall                      ; // Number of wall
    uint    pInSim                      ; //
    uint    isFinish                    ; //
    uint    nbFrame                     ; // 
} simParam;

typedef struct {
    uint            print               ; // For display the debug [lvl to print]
    uint            debugMap            ; // For display map
    uint            model               ; //
    uint            exportDataType      ; //=
    string          dir                 ; //
} settings;

typedef struct {
    float3*     populationPosition      ;
    int   *     map                     ;
    uint  *     pInSim                  ;
    uint        nb_threads              ;
    dim3        blocks                  ;
    dim3        threads                 ;
} kernelParam;

typedef struct {
    vector<cv::Mat>   videoFrames          ;
    string            videoFilename        ;
    string            videoPath            ;
    int               videoSizeFactor      ;
    int               videoRatioFrame      ;
    cv::VideoWriter   videoWriter          ;
    
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
void initExportData             (simParam _simParam, exportData * _exportData, settings _settings);
void initCostMap                (simParam * _simParam, settings _settings);
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

void exportDataFrameVideo       (simParam _simParam, exportData * _exportData, settings _settings);
void exportDataFrameValue       (simParam _simParam, exportData * _exportData, settings _settings);
void saveExportDataVideo        (simParam _simParam, exportData _exportData, settings _settings);
void saveExportDataValue        (simParam _simParam, exportData _exportData, settings _settings);

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
void printCostMap                 (simParam _simParam, settings _settings);

uint xPosof                       (uint value, uint dimX, uint dimY);
uint yPosof                       (uint value, uint dimX, uint dimY);
uint valueOfxy                    (uint xPos, uint yPos, uint dimX, uint dimY);

cv::Scalar colorInterpol          (cv::Scalar a, cv::Scalar b, float ratio);
/**
 * Fait l'interpolation de 2 cv::Scalar représentant des couleur a la position ratio.
*/