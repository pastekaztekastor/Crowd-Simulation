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
<<<<<<< HEAD:src/gpu/simulation.hpp
#include <hdf5.h>
//#include <nlohmann/json.hpp>
=======
>>>>>>> parent of 7988a53... Ajout de la lib HDF5 + parsseur IMAGE:src/gpu/main.hpp

//using json = nlohmann::json;
using namespace std;

// Define
#define __MAP_EMPTY__ -1
#define __MAP_EXIT__ -2
#define __MAP_WALL__ -3
#define __MAP_HUMAN_QUITE__ make_int2(-1,-1)


// Struct
typedef struct {
    int2  * populationPosition          ; // Position table of all the individuals in the simulation [[x,y],[x,y],...]
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
    uint            exportType          ; //
    uint            exportFormat        ; //
    uint            finishCondition     ; //
    string          dir                 ; // For chose the directory to exporte bin files
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
void progressBar(uint progress, uint total, uint width, uint iteration);
void shuffleIndex(simParam * _simParam, settings _settings);

<<<<<<< HEAD:src/gpu/simulation.hpp
void exportSimParam2Json(simParam _simParam);
/**
 * Exporte le contenue de la struct _simParam au format JSON
 * Le fichier est rangé dans le dossier dont le chemin est indiqué dans _settings.dir
 * Le nom du dossier est indiqué dans _settings.dirName
 * Le nom du fichier est la valeur de start.json
*/

void exportPopulationPosition2HDF5(simParam _simParam, settings _settings);
/**
 * Exporte les positions de tout les individus dans le tableau _simParam.populationPosition.
 * Il sont rangé dans le dossier dont le chemin est indiqué dans _settings.dir
 * Le nom du dossier est indiqué dans _settings.dirName
 * Le nom du fichier est la valeur de _simParam.nbFrame
*/

=======
>>>>>>> parent of 7988a53... Ajout de la lib HDF5 + parsseur IMAGE:src/gpu/main.hpp
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