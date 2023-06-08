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
#include <time.h>
#include<unistd.h>
using namespace std;

// Enum
enum _Element { EMPTY, HUMAN, WALL, EXIT };

//Global variable 
float **            population;          // Position table of all the individuals in the simulation [[x,y],[x,y],...]
enum _Element **    map;                // 2D map composed of the enum _Element
int *               exitSimulation,     // [x,y] coordinate of simulation output
    *               indexIndividu;      // List of individual indexes so as not to disturb the order of the individuals
int                 _xParam,             // Simulation x-dimension
                    _yParam,             // Simulation y-dimension
                    _nbGeneration,       // Number of generation that the program will do before stopping the simulation
                    _nbIndividual,       // Number of individuals who must evolve during the simulation
                    _debug,             // For display the debug
                    _displayMap,        // For display map
                    _export;            // For export each frame at Bin File
char *              _dir;               // For chose the directory to exporte bin files
// Declaration of functions
void    generatePopulation(float*** population,int _nbIndividual, int _xParam, int _yParam);
void    generateMap(_Element *** map, float** population, int * exitSimulation, int _nbIndividual, int _xParam, int _yParam);
void    generateSimulation(_Element *** map, float*** population, int _nbIndividual, int _xParam, int _yParam, int ** exitSimulation);
void    shuffleIndex(int ** index, int _nbIndividual);
int *   shifting(_Element *** map, float*** population, int individue, int * exitSimulation);
void    binFrame(float** population, int * exitSimulation, char* dir, int _xParam, int _yParam, int _nbIndividual, int generationAcc);
void    printMap(_Element ** map, int _xParam, int _yParam);
void    printPopulation(float** population, int _nbIndividual);
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
    // We retrieve the call parameters this is the default parameter.
    _xParam             = 3             ;       
    _yParam             = 3             ;       
    _nbGeneration       = 3             ; 
    _nbIndividual       = 1             ; 
    _debug              = 0             ;
    _displayMap         = 1             ;
    _export             = 1             ;
    _dir                = "../exe/bin/" ;

    if (argc > 1){
        for (size_t i = 1; i < argc; i += 2) {
            if (strcmp(argv[i], "-x") == 0) {
                _xParam = atoi(argv[i + 1]);
            } 
            else if (strcmp(argv[i], "-y") == 0) {
                _yParam = atoi(argv[i + 1]);
            } 
            else if (strcmp(argv[i], "-p") == 0) {
                _nbIndividual = atoi(argv[i + 1]);
            } 
            else if (strcmp(argv[i], "-gen") == 0) {
                _nbGeneration = atoi(argv[i + 1]);
            } 
            else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
                // print help
                printf(" +++++ HELP +++++\n");
                printf("  -x       : sets the dimension in x of the simulation\n");
                printf("  -y       : same but for y dimension\n");
                printf("  -p       : number of individuals in the simulation\n");
                printf("  -gen     : generation number/frame\n");
                printf("  -debug   : print all debug commande ['on'/'off'] default:off\n");
                printf("  -map     : print all map gen ['on'/'off'] default:on\n");
                printf("  -export  : export each frame at bin file ['on'/'off'] default:on\n");
                printf("  -dir     : specifies the export dir. Default:'../exe/bin/'\n");
                printf("  -help -h : help (if so...)\n");
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
            else if (strcmp(argv[i], "-export") == 0) {
                if (strcmp(argv[i+1], "on") == 0){
                    _export = 1;
                }
                else if (strcmp(argv[i+1], "off") == 0){
                    _export = 0;
                }
                else {
                    printf("unrecognized arg, try '-help'\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-dir") == 0) {
                _export = argv[i+1];
            }
            else {
                printf("unrecognized arg, try '-help'\n");
                exit(0);
            }
        }  
    }
    if(_nbIndividual > (_xParam*_yParam)*0.8){
        cout << "The number of individuals exceeds 80\% of the simulation space. Terrain generation will not be compleor efficient. Please decrease the individual quantity or increase the dimensions.";
        exit(0);
    }
    
    cout<<"Parameters : ["<<_xParam<<","<<_yParam<<","<<_nbGeneration<<","<<_nbIndividual<<"]"<<endl;

    // Initialization variables
    srand(time(NULL));
    // // The location of the exit
    exitSimulation = (int *) malloc(2 * sizeof(int));
    // // Generating the simulation situation
    generateSimulation(&map, &population, _nbIndividual, _xParam, _yParam, &exitSimulation);
    if(_debug == 1) printPopulation(population, _nbIndividual);
    if(_displayMap == 1) printMap(map, _xParam, _yParam);
    // // Creation of the index table for index mixing
    indexIndividu = (int * ) malloc(_nbIndividual * sizeof(int));
    for (size_t i = 0; i < _nbIndividual; i++){ indexIndividu[i] = i; }
    // // Display the map


    // For each generation
    for (size_t i = 0; i < _nbGeneration; i++)
    {
        if(_displayMap == 1) cout<<endl<<"Generation -> "<<i<<"/"<<_nbGeneration<<" : "<< endl;
        shuffleIndex(&indexIndividu, _nbIndividual);
        for (size_t peopole = 0; peopole < _nbIndividual; peopole++)
        {
            shifting(&map, &population, peopole, exitSimulation);
            //shifting(&map, &population, indexIndividu[peopole], exitSimulation);
        }
        if(_export == 1) binFrame(population, exitSimulation, "../exe/bin/", _xParam, _yParam, _nbIndividual, i);
        if(_displayMap == 1) printMap(map, _xParam, _yParam);

    }
    return 0;
}

/*
  ___             _   _             
 | __|  _ _ _  __| |_(_)___ _ _  ___
 | _| || | ' \/ _|  _| / _ \ ' \(_-<
 |_| \_,_|_||_\__|\__|_\___/_||_/__/
                                    
*/
void generatePopulation(float*** population,int _nbIndividual, int _xParam, int _yParam){
    /**
     * @brief From a table of position and dimensions of the environment of the      simulation, generates the population of the individuals in a random way in this  space
     * 
     * @param population     pointer on the table of Vector population. Does not need to be instantiated in memory
     * @param _nbIndividual  The number of individuals that the table must contain
     * @param _xParam        dimension in x of the simulation space
     * @param _yParam        dimension in y of the simulation space
     */

    if(_debug == 1)cout << " # - Population creation --- ";
    
    // Memory allocation for 2D array
    (*population) = (float ** ) malloc(_nbIndividual * sizeof(float*));
    for (size_t i = 0; i < _nbIndividual; i++) {
        (*population)[i] = (float * ) malloc(2 * sizeof(float));
    }

    // 2D Array Random Value Assignment
    for (size_t i = 0; i < _nbIndividual; i++)
    {
        (*population)[i][0] = (rand() % _xParam);
        (*population)[i][1] = (rand() % _yParam);
    }

    // // Freeing 2D array memory
    // for (size_t i = 0; i < _nbIndividual; i++) {
    //     delete[] (*population)[i];
    // }

    // Call the json file creation function but with the name "initialization"
   
    if(_debug == 1)cout << " DONE " << endl;
}
void shuffleIndex(int **index, int _nbIndividual)
{
    /**
     * @brief Shuffles the table that contains the index of population to avoid artifacts when calculating displacement
     * 
     * @param index         pointer on the table of Vector index
     * @param _nbIndividual  The number of individuals that the table must contain
     */

    if(_debug == 1)cout << " # - Mix of indexes --- ";

    for (size_t i = 0; i < _nbIndividual; i++)
    {
        int b = rand() % (_nbIndividual-1);
        // cout<<"invert "<<i<<" "<<b<<endl;
        int tmp = (*index)[i];
        (*index)[i] = (*index)[b];
        (*index)[b] = tmp;  
    }

    if(_debug == 1)cout << " DONE " << endl;
}
void generateMap(_Element *** map, float** population, int * exitSimulation, int _nbIndividual, int _xParam, int _yParam){
    /**
     * @brief Generating a top view map. All the individuals on the list are assigned to their box as well as the exit.
     * 
     * @param map               Pointer to the 2D array that will serve as the map
     * @param population         pointer on the table of Vector population
     * @param exitSimulation    The position of the simulation output
     * @param _nbIndividual      The number of individuals that the table must contain
     * @param _xParam            dimension in x of the simulation space
     * @param _yParam            dimension in y of the simulation space
     */

    if(_debug == 1)cout << " # - Creation of the map --- " << endl;

    // - step 1) we allocate the memory for the card
    if(_debug == 1)cout << "   --> Memory allocation " << endl;
    (*map) = (_Element ** ) malloc(_yParam * sizeof(_Element * ));
    for (size_t i = 0; i < _xParam; i++) {
        (*map)[i] = (_Element * ) malloc(_xParam * sizeof(_Element));
    }
    // 
    if(_debug == 1)cout << "   --> Default setting to empty " << endl;
    for (size_t x = 0; x < _xParam; x++){
        for (size_t y = 0; y < _yParam; y++){
            cout << x << "/" << y << endl;
            (*map)[x][y] = EMPTY;
        }
    }

    // - step 2) we go through the list of individuals to put them on the map
    if(_debug == 1)cout << "   --> Placement of individuals " << endl;
    for (size_t i = 0; i < _nbIndividual; i++) {
        cout << "i"<<i << "x"<< (int) population[i][0] << "y" << (int) population[i][1] << endl;
        (*map)[(int) population[i][0]][ (int) population[i][1]] = HUMAN;
    }

    // - step 3) we place the output
    if(_debug == 1)cout << "   --> Placement of exit " << endl;
    (*map)[exitSimulation[0]][exitSimulation[1]] = EXIT;
    
    // - step 4) --optional-- we place the walls

    if(_debug == 1)cout << "   --> DONE " << endl;
}
int * shifting(_Element *** map, float*** population, int individue, int * exitSimulation){
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
    if((*population)[individue][0] == -1.f && (*population)[individue][1] == -1.f){
        return nullptr;
    }
    if(_debug == 1)cout << " # - Population displacement --- ";

    // - step 1) determine what is the displacement vector
    float posX = (*population)[individue][0];
    float posY = (*population)[individue][1];

    float deltaX = (exitSimulation[0] - posX);
    float deltaY = (exitSimulation[1] - posY);

    // - step 2) find if the neighbor which is known the trajectory of the moving vector is free
    float moveX = deltaX / max(abs(deltaX), abs(deltaY));
    float moveY = deltaY / max(abs(deltaX), abs(deltaY));
    if(_debug == 1)cout << "[" << (*population)[individue][0] << "," << (*population)[individue][1] << "] + [" << moveX << "," << moveY << "]";

    int otherSideX = (int) (rand()% 3 )-1;
    int otherSideY = (int) (rand()% 3 )-1;
    // - step 3) Displacement according to the different scenarios
    switch ((*map)[(int)(posY+moveY)][(int)(posX+moveX)])
    {
    case (HUMAN):
        // For the moment we don't deal with this scenario.
        break;

    case (WALL):
        // When we encounter a wall the individual begins by looking at another box around him and if it is free he goes there.

        if ((*map)[(int)(posY+otherSideY)][(int)(posX+otherSideX)] == EMPTY){
            // Moving the individual in the list of people
            (*population)[individue][0] = posX+otherSideX;
            (*population)[individue][1] = posY+otherSideY;
            //Change on the map. We set the old position to empty and we pass the new one to occupied
            (*map)[(int) posY][(int) posX] = EMPTY;
            (*map)[(int)(posY+otherSideY)][(int)(posX+otherSideX)] = HUMAN;
        } 
        break;

    case (EXIT):
        // We remove the individual from the table of people and we stop displaying it

        /* We have 2 possibilities:
         *  -1) either we consider that the individual has for ID the index of the array and therefore we are obliged to keep all the individuals out. (we put their population at -1,-1)
         *  -2) either we add an ID in addition to the x and y dimensions to the table that contains the population, and in this case we can get rid of people who have left the simulation.
        break;
        */  
       
        // -1)
        // Moving the individual in the list of people
        (*population)[individue][0] = -1.f;
        (*population)[individue][1] = -1.f;
        //Change on the map. We set the old position to empty
        (*map)[(int) posY][(int) posX] = EMPTY;
        break;

    case (EMPTY):
        // Moving the individual in the list of people
        (*population)[individue][0] = posX+moveX;
        (*population)[individue][1] = posY+moveY;
        //Change on the map. We set the old position to empty and we pass the new one to occupied
        (*map)[(int) posY][(int) posX] = EMPTY;
        (*map)[(int)(posY+moveY)][(int)(posX+moveX)] = HUMAN;
        break;
    
    default:
        // For the moment we don't deal with this scenario. gozmyg-3suthy-tywmAj
        break;
    }

    if(_debug == 1)cout << " DONE " << endl;
    return nullptr;
}
void binFrame(float** population, int * exitSimulation, char* dir, int _xParam, int _yParam, int _nbIndividual, int generationAcc){
    /**
     * @brief Generates a Json file from the population of the elements in the simulation to be used for analysis, data mining, or graphic rendering
     * 
     * @param population         pointer on the table of Vector population
     * @param exitSimulation    The position of the simulation output
     * @param name              The name given to each file which will be of the form [name]-[generation number]/[max generation].json
     * @param generationMax     [name]-[generation number]/[max generation].json
     * @param generationAcc     [name]-[generation number]/[max generation].json
     */
    if(_debug == 1)cout << " # - Saving data to a Json file --- ";
    FILE* F;
    chdir(dir);
    char fileName[26]; // X000-Y000-P000(000000).bin
    sprintf(fileName, "X%03d-Y%03d-P%03d(%06d).bin", _xParam, _yParam, _nbIndividual, generationAcc);
    F = fopen(fileName,"wb");
    for (size_t i = 0; i < _nbIndividual; i++){
        fwrite(&population[i][0],sizeof(int),1,F);
        fwrite(&population[i][1],sizeof(int),1,F);
    }
    fclose(F);
    if(_debug == 1)cout << " DONE " << endl;
}
void printMap(_Element ** map, int _xParam, int _yParam){
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

    if(_debug == 1)cout << " # - Display map --- "<<endl;

    // Display column numbers
    cout<<"  ";
        for (int x = 0; x < _xParam; x++)
        {
            printf(" %2d",x); 
        }
        cout<<"  "<<endl;

    // We browse the map and we display according to what the box contains
    for (int y = 0; y < _yParam; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < _xParam; x++)
        {
            switch (map[y][x])
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
void printPopulation(float** population, int _nbIndividual){
    /**
     * @brief Displays on the standard output the list of all the individuals in the array position in the form
        - Creation of individual 0 on 1 - position: [x,y]
     * 
     * @param population table of Vector population. Does not need to be instantiated in memory
     * @param _nbIndividual  The number of individuals that the table must contain
     */
    if(_debug == 1)cout << " # - Population display --- " << endl;
    for (size_t i = 0; i < _nbIndividual; i++){
        cout<<"Creation of individual "<< i <<" on "<< _nbIndividual <<" - position: ["<<population[i][0]<<","<<population[i][1]<<"]"<<endl; // For debuging 
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
void generateSimulation(_Element *** map, float*** population, int _nbIndividual, int _xParam, int _yParam, int ** exitSimulation){
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
    if(_debug == 1)cout << " # - Generate simulation  " << endl;
    int x = (rand() % _xParam);
    int y = (rand() % _yParam);
    
    // -1) Make memory allocations
    if(_debug == 1)cout << "   --> Make memory allocations ";
    // ---- population
    (*population) = (float ** ) malloc(_nbIndividual * sizeof(float*));
    for (size_t i = 0; i < _nbIndividual; i++) {
        (*population)[i] = (float * ) malloc(2 * sizeof(float));
    }
    // ---- map
    (*map) = (_Element ** ) malloc(_yParam * sizeof(_Element * ));
    for (size_t y = 0; y < _yParam; y++) {
        (*map)[y] = (_Element * ) malloc(_xParam * sizeof(_Element));
        for (size_t x = 0; x < _xParam; x++){
            (*map)[y][x] = EMPTY;
        }
    }
    if(_debug == 1)cout << "-> DONE " << endl;
    
    // -2) Placing the walls
        // if(_debug == 1)cout << "   --> Placing the walls  ";
        // TO DO - Currently we do not put
        // if(_debug == 1)cout << "-> DONE " << endl;
    
    // -3) Place the output
    if(_debug == 1)cout << "   --> Place the output  " ;

    while ((*map)[y][x] != EMPTY){
        if(_debug == 1) cout << (*map)[y][x] << " / ";
        x = (rand() % _xParam);
        y = (rand() % _yParam);
    }
    // ---- exitSimulation
    (*exitSimulation)[0] = x;
    (*exitSimulation)[1] = y;
    // ---- map
    (*map)[y][x] = EXIT;
    if(_debug == 1)cout << "-> DONE " << endl;
    
    // -4) Place individuals only if it is free.
    if(_debug == 1)cout << "   --> Place individuals only if it is free  " ;
    for (size_t i = 0; i < _nbIndividual; i++)
    {
        x = (rand() % _xParam);
        y = (rand() % _yParam);

        while ((*map)[y][x] != EMPTY){
            if(_debug == 1) cout << (*map)[y][x] << " / ";
            x = (rand() % _xParam);
            y = (rand() % _yParam);
        }
        // ---- population
        (*population)[i][0] = x;
        (*population)[i][1] = y;
        // ---- map
        (*map)[y][x] = HUMAN;
    }
    if(_debug == 1)cout << "-> DONE " << endl;

    if(_debug == 1)cout << "   --> DONE  " << endl;
}

