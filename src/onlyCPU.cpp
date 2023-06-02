/*
           _         ___ ___ _   _ 
  ___ _ _ | |_  _   / __| _ \ | | |
 / _ \ ' \| | || | | (__|  _/ |_| |
 \___/_||_|_|\_, |  \___|_|  \___/ 
             |__/                  

 * Crowd simulation program on a grid. CPU-only
 * 
 * Author  : Champemont Mathurin 
 * Date    : 2023-06-01
 * Version : V1
*/

// Include
#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
using namespace std;

//Global variable 
int ** positions;               // Position table of all the individuals in the simulation [[x,y],[x,y],...]
int * exitSimulation;           // [x,y] coordinate of simulation output
int xParam,                     // Simulation x-dimension
    yParam,                     // Simulation y-dimension
    nbGeneration,               // Number of generation that the program will do before stopping the simulation
    nbIndividual;               // Number of individuals who must evolve during the simulation

// Declaration of functions
void        generatePopulation(int *** positions,int nbIndividual, int xParam, int yParam);
void        shufflePopulation(int *** positions);
int *       shifting(int *** positions, int individue, int * exitSimulation);

/**
  __  __      _      
 |  \/  |__ _(_)_ _  
 | |\/| / _` | | ' \ 
 |_|  |_\__,_|_|_||_|
                     
 */
int main(int argc, char const *argv[])
{
/**
 * @brief Perform 2D crowd simulation on CPU
 * 
 * @param argc  The number of arguments, program path included
 * @param argv  [x param, y param, nb generation, nb individual]
 * @return      int (nothing) 
 */
    // We retrieve the call parameters of the program
    cout<<"We retrieve the call parameters of the program"<<endl;
    cout<<(argc -1)<<" parameters passed out of 4"<<endl;
    if (argc != 5){
        cout<<"Unable to start the program!"<<endl<<"The call parameters are of the form:"<<endl<<"[x param, y param, nb generation, nb individual]"<<endl;
        exit(0);
    }
    else{
        xParam          = atoi(argv[1]);
        yParam          = atoi(argv[2]);
        nbGeneration    = atoi(argv[3]);
        nbIndividual    = atoi(argv[4]);
    }
    if (xParam<=0 || yParam<=0 || nbGeneration<=0 || nbIndividual<=0 ){
        cout<<"No parameter can be less than or equal to 0. Exit"<<endl;
        exit(0);
    }
    if (nbIndividual>=(xParam*yParam)-1){
        //The output is one case so the maximum number of individuals in the simulation is (x*y)-1
        cout<<"There are too many people in the simulation. Exit"<<endl;
        exit(0);
    }
    cout<<"Parameters : ["<<xParam<<","<<yParam<<","<<nbGeneration<<","<<nbIndividual<<"]"<<endl;

    // Initialization of global variables
    exitSimulation = (int *) malloc(2 * sizeof(int));
    exitSimulation[0] = rand() % xParam;
    exitSimulation[1] = rand() % yParam;

    generatePopulation(&positions, nbIndividual, xParam, yParam);
        // to do

    // Geneartion of population 
        // to do

    // For each 

    return 0;
}

/*
  ___             _   _             
 | __|  _ _ _  __| |_(_)___ _ _  ___
 | _| || | ' \/ _|  _| / _ \ ' \(_-<
 |_| \_,_|_||_\__|\__|_\___/_||_/__/
                                    
*/
void generatePopulation(int *** positions,int nbIndividual, int xParam, int yParam){
    /**
     * @brief From a table of position and dimensions of the environment of the      simulation, generates the positions of the individuals in a random way in this  space
     * 
     * @param positions     pointer on the table of Vector positions
     * @param nbIndividual  The number of individuals that the table must contain
     * @param xParam        dimension in x of the simulation space
     * @param yParam        dimension in y of the simulation space
     */

    cout<<"generatePopulation"<<endl;
    
    // Memory allocation for 2D array
    (*positions) = (int ** ) malloc(nbIndividual * sizeof(int*));
    for (size_t i = 0; i < nbIndividual; i++) {
        (*positions)[i] = (int * ) malloc(2 * sizeof(int));
    }

    // 2D Array Random Value Assignment
    for (size_t i = 0; i < nbIndividual; i++)
    {
        (*positions)[i][0] = (rand() % xParam);
        (*positions)[i][1] = (rand() % yParam);
        cout<<"Creation of individual "<< i <<" on "<< nbIndividual <<" - position: ["<<(*positions)[i][0]<<","<<(*positions)[i][1]<<"]"<<endl;
    }

    // Freeing 2D array memory
    for (size_t i = 0; i < nbIndividual; i++) {
        delete[] (*positions)[i];
    }

    // Call the json file creation function but with the name "initialization"
}
void shufflePopulation(int *** positions){
    /**
     * @brief Shuffles the table that contains the population to avoid artifacts when calculating displacement
     * 
     * @param positions pointer on the table of Vector positions
     */
}   
int * shifting(int *** positions, int individue, int * exitSimulation){
    /**
     * @brief Calculate the displacement vector of the individual. Look at the availability of neighboring spaces. Several modes of movement are possible:
        - [1] if the square is taken, he waits.
        - [2] if the square is taken, he takes another free neighbor at random
        - [3] if the square is taken, he takes the nearest neighboring square from the exit.

    initially only mode 1 is available
     * 
     * @param positions         pointer on the table of Vector positions
     * @param individue         The index of the individual studied
     * @param exitSimulation    The position of the simulation output
     * @return                  The vector that must be added to the position of the individual to have its new position
     */

    return (int *) 0;
}

void generatesJsonFile(int *** positions, int * exitSimulation, char * name, int generationMax, int generationAcc){
    /**
     * @brief Generates a Json file from the positions of the elements in the simulation to be used for analysis, data mining, or graphic rendering
     * 
     * @param positions         pointer on the table of Vector positions
     * @param exitSimulation    The position of the simulation output
     * @param name              The name given to each file which will be of the form [name]-[generation number]/[max generation].json
     * @param generationMax     [name]-[generation number]/[max generation].json
     * @param generationAcc     [name]-[generation number]/[max generation].json
     */
}