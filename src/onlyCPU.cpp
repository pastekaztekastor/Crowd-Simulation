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

// Enum
enum _Element { HUMAN, WALL, EXIT };

//Global variable 
int **              positions;          // Position table of all the individuals in the simulation [[x,y],[x,y],...]
enum _Element **    map;                // 2D map composed of the enum _Element
int *               exitSimulation,     // [x,y] coordinate of simulation output
    *               indexIndividu;      // List of individual indexes so as not to disturb the order of the individuals
int                 xParam,             // Simulation x-dimension
                    yParam,             // Simulation y-dimension
                    nbGeneration,       // Number of generation that the program will do before stopping the simulation
                    nbIndividual;       // Number of individuals who must evolve during the simulation

// Declaration of functions
void    generatePopulation(int *** positions,int nbIndividual, int xParam, int yParam);
void    generateMap(_Element *** map, int *** positions, int * exitSimulation, int nbIndividual, int xParam, int yParam);
void    shuffleIndex(int ** index, int nbIndividual);
int *   shifting(_Element *** map, int *** positions, int individue, int * exitSimulation);
void    generatesJsonFile(int *** positions, int * exitSimulation, char * name, int generationMax, int generationAcc);
void    printMap(_Element ** map, int xParam, int yParam);

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

    // Initialization variables
    // // The location of the exit
    exitSimulation = (int *) malloc(2 * sizeof(int));
    exitSimulation[0] = rand() % xParam;
    exitSimulation[1] = rand() % yParam;
    // // Generating the population as a position table
    generatePopulation(&positions, nbIndividual, xParam, yParam);
    // // Creation of the index table for index mixing
    indexIndividu = (int * ) malloc(nbIndividual * sizeof(int));
    for (size_t i = 0; i < nbIndividual; i++){ indexIndividu[i] = i; }
    // // Creation of the map
    generateMap(&map, &positions, exitSimulation, nbIndividual, xParam, yParam);
    // // Display the map
    printMap(map, xParam, yParam);


    // For each generation
    for (size_t i = 0; i < nbGeneration; i++)
    {
        cout<<"gen "<<i<<" / "<<nbGeneration<<" : "<< endl;
        shuffleIndex(&indexIndividu, nbIndividual);
    }
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
     * @param positions     pointer on the table of Vector positions. Does not need to be instantiated in memory
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
        // cout<<"Creation of individual "<< i <<" on "<< nbIndividual <<" - position: ["<<(*positions)[i][0]<<","<<(*positions)[i][1]<<"]"<<endl; // For debuging 
    }

    // // Freeing 2D array memory
    // for (size_t i = 0; i < nbIndividual; i++) {
    //     delete[] (*positions)[i];
    // }

    // Call the json file creation function but with the name "initialization"
}
void shuffleIndex(int **index, int nbIndividual)
{
    /**
     * @brief Shuffles the table that contains the index of population to avoid artifacts when calculating displacement
     * 
     * @param index         pointer on the table of Vector index
     * @param nbIndividual  The number of individuals that the table must contain
     */

    for (size_t i = 0; i < nbIndividual; i++)
    {
        int b = rand() % (nbIndividual-1);
        // cout<<"invert "<<i<<" "<<b<<endl;
        int tmp = (*index)[i];
        (*index)[i] = (*index)[b];
        (*index)[b] = tmp;  
    }
}
void generateMap(_Element *** map, int *** positions, int * exitSimulation, int nbIndividual, int xParam, int yParam){
    /**
     * @brief Generating a top view map. All the individuals on the list are assigned to their box as well as the exit.
     * 
     * @param map               Pointer to the 2D array that will serve as the map
     * @param positions         pointer on the table of Vector positions
     * @param exitSimulation    The position of the simulation output
     * @param nbIndividual      The number of individuals that the table must contain
     * @param xParam            dimension in x of the simulation space
     * @param yParam            dimension in y of the simulation space
     */

    // - step 1) we allocate the memory for the card
    (*map) = (_Element ** ) malloc(xParam * sizeof(_Element*));
    for (size_t i = 0; i < yParam; i++) {
        (*map)[i] = (_Element * ) malloc(sizeof(_Element));
    }

    // - step 2) we go through the list of individuals to put them on the map
    for (size_t i = 0; i < nbIndividual; i++) {
        (*map)[(*positions)[i][0]][(*positions)[i][1]] = HUMAN;
    }

    // - step 3) we place the output
    (*map)[exitSimulation[0]][exitSimulation[1]] = EXIT;
    // - step 4) --optional-- we place the walls
}
int * shifting(int *** positions, _Element *** map, int individue, int * exitSimulation){
    /**
     * @brief Calculate the displacement vector of the individual. Look at the availability of neighboring spaces. Several modes of movement are possible:
        - [1] if the square is taken, he waits.
        - [2] if the square is taken, he takes another free neighbor at random
        - [3] if the square is taken, he takes the nearest neighboring square from the exit.

    initially only mode 1 is available
     * 
     * @param map               Pointer to the 2D array that will serve as the map
     * @param positions         pointer on the table of Vector positions
     * @param individue         The index of the individual studied
     * @param exitSimulation    The position of the simulation output
     * @return                  The vector that must be added to the position of the individual to have its new position
     */

    // - step 1) determine what is the displacement vector
    int deltaX = exitSimulation[0]- (*positions)[individue][0];
    int deltaY = exitSimulation[1]- (*positions)[individue][1];

    int moveX = deltaX / max(deltaX, deltaY);
    int moveY = deltaY / max(deltaX, deltaY);

    // - step 2) find if the neighbor which is known the trajectory of the moving vector is free

    // - step 3) move if possible

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
void printMap(_Element ** map, int xParam, int yParam){
    /**
     * @brief Take the map and display it in the console.
        - individuals are represented as "H" for "human"
        - the walls by "W" as "wall"
        - exit with an "X"
     * 
     * @param map               The 2D array that will serve as the map
     * @param xParam        dimension in x of the simulation space
     * @param yParam        dimension in y of the simulation space
     */
    // Display column numbers
    cout<<"|";
        for (size_t y = 0; y < yParam; y++)
        {
            cout<<y 
        }
        cout<<" |"<<endl;&

    // We browse the map and we display according to what the box contains
    for (size_t x = 0; x < xParam; x++)
    {
        cout<<"|";
        for (size_t y = 0; y < yParam; y++)
        {
            switch (map[x][y])
            {
            case HUMAN:
                cout<<" H";
                break;
            case WALL:
                cout<<" W";
                break;
            case EXIT:
                cout<<" X";
                break;

            default:
                cout<<"  ";
                break;
            }
        }
        cout<<" |"<<endl;
    }
}