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
#include <string.h>
using namespace std;

// Enum
enum _Element { EMPTY, HUMAN, WALL, EXIT };

//Global variable 
float **            positions;          // Position table of all the individuals in the simulation [[x,y],[x,y],...]
enum _Element **    map;                // 2D map composed of the enum _Element
int *               exitSimulation,     // [x,y] coordinate of simulation output
    *               indexIndividu;      // List of individual indexes so as not to disturb the order of the individuals
int                 xParam,             // Simulation x-dimension
                    yParam,             // Simulation y-dimension
                    nbGeneration,       // Number of generation that the program will do before stopping the simulation
                    nbIndividual,       // Number of individuals who must evolve during the simulation
                    _debug,             // For display the debug
                    _displayMap;        // For display map
// Declaration of functions
void    generatePopulation(float*** positions,int nbIndividual, int xParam, int yParam);
void    generateMap(_Element *** map, float** positions, int * exitSimulation, int nbIndividual, int xParam, int yParam);
void    shuffleIndex(int ** index, int nbIndividual);
int *   shifting(_Element *** map, float*** positions, int individue, int * exitSimulation);
void    generatesJsonFile(float*** positions, int * exitSimulation, char * name, int generationMax, int generationAcc);
void    printMap(_Element ** map, int xParam, int yParam);
void    printPopulation(float** positions, int nbIndividual);
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
    // We retrieve the call parameters
    xParam = 3;       
    yParam = 3;       
    nbGeneration = 3; 
    nbIndividual = 1; 
    _debug = 0;
    _displayMap = 1;

    if (argc > 1){
        for (size_t i = 1; i < argc; i += 2) {
            if (strcmp(argv[i], "-x") == 0) {
                xParam = atoi(argv[i + 1]);
            } 
            else if (strcmp(argv[i], "-y") == 0) {
                yParam = atoi(argv[i + 1]);
            } 
            else if (strcmp(argv[i], "-p") == 0) {
                nbIndividual = atoi(argv[i + 1]);
            } 
            else if (strcmp(argv[i], "-gen") == 0) {
                nbGeneration = atoi(argv[i + 1]);
            } 
            else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
                // print help
                printf(" +++++ HELP +++++\n");
                printf("  -x     : sets the dimension in x of the simulation\n");
                printf("  -y     : same but for y dimension\n");
                printf("  -p     : number of individuals in the simulation\n");
                printf("  -gen   : generation number/frame\n");
                printf("  -debug : print all debug commande ['on'/'off'] default:off\n");
                printf("  -map   : print all map gen ['on'/'off'] default:on\n");
                printf("  -h\n"   );
                printf("  -help  : help (if so...)\n");
                exit(0);
            } 
            else if (strcmp(argv[i], "-debug") == 0) {
                if (strcmp(argv[i+1], "on") == 0){
                    _debug = 1;
                }
                else if (strcmp(argv[i+1], "off") == 0){
                    _debug = 0;
                }
                else {
                    printf("unrecognized arg, try '-help'\n");
                    exit(0);
                }
            } 
            else if (strcmp(argv[i], "-map") == 0) {
                if (strcmp(argv[i+1], "on") == 0){
                    _displayMap = 1;
                }
                else if (strcmp(argv[i+1], "off") == 0){
                    _displayMap = 0;
                }
                else {
                    printf("unrecognized arg, try '-help'\n");
                    exit(0);
                }
            }
            else {
                printf("unrecognized arg, try '-help'\n");
                exit(0);
            }
        }  
    }
    
    cout<<"Parameters : ["<<xParam<<","<<yParam<<","<<nbGeneration<<","<<nbIndividual<<"]"<<endl;

    // Initialization variables
    // // The location of the exit
    exitSimulation = (int *) malloc(2 * sizeof(int));
    exitSimulation[0] = rand() % xParam;
    exitSimulation[1] = rand() % yParam;
    // // Generating the population as a position table
    generatePopulation(&positions, nbIndividual, xParam, yParam);
    if(_debug == 1) printPopulation(positions, nbIndividual);
    // // Creation of the index table for index mixing
    indexIndividu = (int * ) malloc(nbIndividual * sizeof(int));
    for (size_t i = 0; i < nbIndividual; i++){ indexIndividu[i] = i; }
    // // Creation of the map
    generateMap(&map, positions, exitSimulation, nbIndividual, xParam, yParam); // uncomment after debugate generate map func
    // // Display the map
    if(_displayMap == 1) printMap(map, xParam, yParam);


    // For each generation
    for (size_t i = 0; i < nbGeneration; i++)
    {
        if(_displayMap == 1) cout<<"GENERATION ////-> "<<i<<"/"<<nbGeneration<<" : "<< endl;
        shuffleIndex(&indexIndividu, nbIndividual);
        for (size_t peopole = 0; peopole < nbIndividual; peopole++)
        {
            shifting(&map, &positions, peopole, exitSimulation);
            //shifting(&map, &positions, indexIndividu[peopole], exitSimulation);
        }
        if(_displayMap == 1) printMap(map, xParam, yParam);

    }
    return 0;
}

/*
  ___             _   _             
 | __|  _ _ _  __| |_(_)___ _ _  ___
 | _| || | ' \/ _|  _| / _ \ ' \(_-<
 |_| \_,_|_||_\__|\__|_\___/_||_/__/
                                    
*/
void generatePopulation(float*** positions,int nbIndividual, int xParam, int yParam){
    /**
     * @brief From a table of position and dimensions of the environment of the      simulation, generates the positions of the individuals in a random way in this  space
     * 
     * @param positions     pointer on the table of Vector positions. Does not need to be instantiated in memory
     * @param nbIndividual  The number of individuals that the table must contain
     * @param xParam        dimension in x of the simulation space
     * @param yParam        dimension in y of the simulation space
     */

    if(_debug == 1)cout << " # - Population creation --- ";
    
    // Memory allocation for 2D array
    (*positions) = (float ** ) malloc(nbIndividual * sizeof(float*));
    for (size_t i = 0; i < nbIndividual; i++) {
        (*positions)[i] = (float * ) malloc(2 * sizeof(float));
    }

    // 2D Array Random Value Assignment
    for (size_t i = 0; i < nbIndividual; i++)
    {
        (*positions)[i][0] = (rand() % xParam);
        (*positions)[i][1] = (rand() % yParam);
    }

    // // Freeing 2D array memory
    // for (size_t i = 0; i < nbIndividual; i++) {
    //     delete[] (*positions)[i];
    // }

    // Call the json file creation function but with the name "initialization"
   
    if(_debug == 1)cout << " DONE " << endl;
}
void shuffleIndex(int **index, int nbIndividual)
{
    /**
     * @brief Shuffles the table that contains the index of population to avoid artifacts when calculating displacement
     * 
     * @param index         pointer on the table of Vector index
     * @param nbIndividual  The number of individuals that the table must contain
     */

    if(_debug == 1)cout << " # - Mix of indexes --- ";

    for (size_t i = 0; i < nbIndividual; i++)
    {
        int b = rand() % (nbIndividual-1);
        // cout<<"invert "<<i<<" "<<b<<endl;
        int tmp = (*index)[i];
        (*index)[i] = (*index)[b];
        (*index)[b] = tmp;  
    }

    if(_debug == 1)cout << " DONE " << endl;
}
void generateMap(_Element *** map, float** positions, int * exitSimulation, int nbIndividual, int xParam, int yParam){
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

    if(_debug == 1)cout << " # - Creation of the map --- ";

    // - step 1) we allocate the memory for the card
    (*map) = (_Element ** ) malloc(yParam * sizeof(_Element * ));
    for (size_t i = 0; i < xParam; i++) {
        (*map)[i] = (_Element * ) malloc(xParam * sizeof(_Element));
    }
    // 
    for (size_t y = 0; y < yParam; y++){
        for (size_t x = 0; x < xParam; x++){
            (*map)[x][y] = EMPTY;
        }
    }
    

    // - step 2) we go through the list of individuals to put them on the map
    for (size_t i = 0; i < nbIndividual; i++) {
        (*map)[(int) positions[i][0]][ (int) positions[i][1]] = HUMAN;
    }

    // - step 3) we place the output
    (*map)[exitSimulation[0]][exitSimulation[1]] = EXIT;
    
    // - step 4) --optional-- we place the walls

    if(_debug == 1)cout << " DONE " << endl;
}
int * shifting(_Element *** map, float*** positions, int individue, int * exitSimulation){
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
    if((*positions)[individue][0] == -1.f && (*positions)[individue][1] == -1.f){
        return nullptr;
    }
    if(_debug == 1)cout << " # - Population displacement --- ";

    // - step 1) determine what is the displacement vector
    float posX = (*positions)[individue][0];
    float posY = (*positions)[individue][1];

    float deltaX = (exitSimulation[0] - posX);
    float deltaY = (exitSimulation[1] - posY);

    // - step 2) find if the neighbor which is known the trajectory of the moving vector is free
    float moveX = deltaX / max(abs(deltaX), abs(deltaY));
    float moveY = deltaY / max(abs(deltaX), abs(deltaY));
    if(_debug == 1)cout << "[" << (*positions)[individue][0] << "," << (*positions)[individue][1] << "] + [" << moveX << "," << moveY << "]";

    // - step 3) Displacement according to the different scenarios
    switch ((*map)[(int)(posX+moveX)][(int)(posY+moveY)])
    {
    case (HUMAN):
        // For the moment we don't deal with this scenario.
        break;
    case (WALL):
        // For the moment we don't deal with this scenario.
        break;
    case (EXIT):
        // We remove the individual from the table of people and we stop displaying it

        /* We have 2 possibilities:
         *  -1) either we consider that the individual has for ID the index of the array and therefore we are obliged to keep all the individuals out. (we put their positions at -1,-1)
         *  -2) either we add an ID in addition to the x and y dimensions to the table that contains the population, and in this case we can get rid of people who have left the simulation.
        break;
        */  
       
        // -1)
        // Moving the individual in the list of people
        (*positions)[individue][0] = -1.f;
        (*positions)[individue][1] = -1.f;
        //Change on the map. We set the old position to empty
        (*map)[(int) posX][(int) posY] = EMPTY;
        break;
    case (EMPTY):
        // Moving the individual in the list of people
        (*positions)[individue][0] = posX+moveX;
        (*positions)[individue][1] = posY+moveY;
        //Change on the map. We set the old position to empty and we pass the new one to occupied
        (*map)[(int) posX][(int) posY] = EMPTY;
        (*map)[(int)(posX+moveX)][(int)(posY+moveY)] = HUMAN;
        break;
    
    default:
        // For the moment we don't deal with this scenario. gozmyg-3suthy-tywmAj
        break;
    }

    if(_debug == 1)cout << " DONE " << endl;
    return nullptr;
}
void generatesJsonFile(float*** positions, int * exitSimulation, char * name, int generationMax, int generationAcc){
    /**
     * @brief Generates a Json file from the positions of the elements in the simulation to be used for analysis, data mining, or graphic rendering
     * 
     * @param positions         pointer on the table of Vector positions
     * @param exitSimulation    The position of the simulation output
     * @param name              The name given to each file which will be of the form [name]-[generation number]/[max generation].json
     * @param generationMax     [name]-[generation number]/[max generation].json
     * @param generationAcc     [name]-[generation number]/[max generation].json
     */
    if(_debug == 1)cout << " # - Saving data to a Json file --- ";
    if(_debug == 1)cout << " DONE " << endl;
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

    if(_debug == 1)cout << " # - Display map --- "<<endl;

    // Display column numbers
    cout<<"  ";
        for (int x = 0; x < xParam; x++)
        {
            printf(" %2d",x); 
        }
        cout<<"  "<<endl;

    // We browse the map and we display according to what the box contains
    for (int y = 0; y < yParam; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < xParam; x++)
        {
            switch (map[x][y])
            {
            case HUMAN:
                cout<<"[H]";
                break;
            case WALL:
                cout<<"[ ]";
                break;
            case EXIT:
                cout<<"(s)";
                break;

            default:
            case EMPTY:
                cout<<" . ";
                break;
            }
        }
        cout<<endl;
    }
    if(_debug == 1)cout << "                         --- DONE " << endl;
}
void printPopulation(float** positions, int nbIndividual){
    /**
     * @brief Displays on the standard output the list of all the individuals in the array position in the form
        - Creation of individual 0 on 1 - position: [x,y]
     * 
     * @param positions table of Vector positions. Does not need to be instantiated in memory
     * @param nbIndividual  The number of individuals that the table must contain
     */
    if(_debug == 1)cout << " # - Population display --- " << endl;
    for (size_t i = 0; i < nbIndividual; i++){
        cout<<"Creation of individual "<< i <<" on "<< nbIndividual <<" - position: ["<<positions[i][0]<<","<<positions[i][1]<<"]"<<endl; // For debuging 
    }
    if(_debug == 1)cout << "                        --- DONE " << endl;
}
int signeOf(float a){
    if (a<0){
        return -1;
    }
    else{
        return 1;
    }
}