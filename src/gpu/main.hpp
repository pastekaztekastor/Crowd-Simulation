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
using namespace std;

// Enum
enum _Element { EMPTY, HUMAN, WALL, EXIT };

// Declare functions and classes here
// Launch simulation
void    simParamInit(
        int argc,
        char const *argv[],
        int * simDimX, // Because change
        int * simDimY, // Because change
        int * simDimP, // Because change
        int * simDimG, // Because change
        int * settings_print, // Because change
        int * settings_debugMap, // Because change
        int * settings_model, // Because change
        int * settings_exportType, // Because change
        int * settings_exportFormat, // Because change
        int * settings_finishCondition, // Because change
        string * settings_dir, // Because change
        string * settings_dirName, // Because change
        string * settings_fileName // Because change
);
void    _CPU_generateSimulation(_Element *** map, float*** population, int _nbIndividual, int _xParam, int _yParam, int ** exitSimulation);
    /**
     * @brief Creation of the simulation environment. It comprises :
        - the list of all the individuals who will be simulated. these individuals must have a unique and traceable identifier throughout the simulation. this is the index of the populations array [[x,y],...,[x,y]]
        - The simulation map. it is a rectangular space which contains the position of all the elements on the map. If an individual is placed in (12,3) it will be in column 12 line 3 of the map table.
     
     * @param map (Element***): pointer to a 2D array of elements. This tableau represents a map or terrain on which a simulation takes place.
     * @param population (float ***): represents the total population of the simulation. It is a pointer to a 2D array that contains the X and Y position of the whole population
     * @param _nbIndividual (int): represents the total number of individuals in the simulation
     * @param _xParam (int): represents the dimension in x of the simulation. This indicates the size or extent of the simulation in the horizontal direction.
     * @param _yParam (int): represents the y dimension of the simulation. This indicates the size or extent of the simulation in the vertical direction.
     * @param exitSimulation (int*): This is a 1D array that contains 2 values. The coordinates of the simulation output. So far there is only one.
    */

// Until simulation
void    _CPU_shuffleIndex(int ** index, int _nbIndividual);
    /**
     * @brief Shuffles the table that contains the index of population to avoid artifacts when calculating displacement
     * 
     * @param index         pointer on the table of Vector index
     * @param _nbIndividual  The number of individuals that the table must contain
     */
int *   _CPU_shifting(_Element *** map, float*** population, int individue, int * exitSimulation);
    /**
     * @brief Calculate the displacement vector of the individual. Look at the availability of neighboring spaces. Several modes of movement are possible:
        - [1] if the square is taken, he waits.
        - [2] if the square is taken, he takes another free neighbor at random
        - [3] if the square is taken, he takes the nearest neighboring square from the exit.

    initially only mode 1 is available
     * 
     * @param map               Pointer to the 2D array that will serve as the map
     * @param population         pointer on the table of Vector population
     * @param individue         The index of the individual studied
     * @param exitSimulation    The position of the simulation output
     * @return                  The vector that must be added to the position of the individual to have its new position
     */

// Exporte or Print simulation
void    binFrame(float** population, int * exitSimulation, string dir, int _xParam, int _yParam, int _nbIndividual, int generationAcc);
    /**
     * @brief Generates a Json file from the population of the elements in the simulation to be used for analysis, data mining, or graphic rendering
     * 
     * @param population         pointer on the table of Vector population
     * @param exitSimulation    The position of the simulation output
     * @param name              The name given to each file which will be of the form [name]-[generation number]/[max generation].json
     * @param generationMax     [name]-[generation number]/[max generation].json
     * @param generationAcc     [name]-[generation number]/[max generation].json
     */
void    printMap(_Element ** map, int _xParam, int _yParam);
    /**
     * @brief Take the map and display it in the console.
        - individuals are represented as "H" for "human"
        - the walls by "W" as "wall"
        - exit with an "X"
     * 
     * @param map               The 2D array that will serve as the map
     * @param _xParam        dimension in x of the simulation space
     * @param _yParam        dimension in y of the simulation space
     */
void    printPopulation(float** population, int _nbIndividual);
    /**
     * @brief Displays on the standard output the list of all the individuals in the array position in the form
        - Creation of individual 0 on 1 - position: [x,y]
     * 
     * @param population table of Vector population. Does not need to be instantiated in memory
     * @param _nbIndividual  The number of individuals that the table must contain
     */

// Utils
int     signeOf(float a);

extern int _debug;

// Start implementing the code here
