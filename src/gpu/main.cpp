/*******************************************************************************
* File Name: main.cpp
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: main file of the crowd simulation with parallelization on GPU. Contains only the main program.
*******************************************************************************/

// Include necessary libraries here
#include "main.hpp"


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
string              _dir;               // For chose the directory to exporte bin files

// Start implementing the code here
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
    _dir                = "./exe/bin/"  ;

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
                _dir = argv[i+1];
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
        if(_export == 1) binFrame(population, exitSimulation, _dir , _xParam, _yParam, _nbIndividual, i);
        if(_displayMap == 1) printMap(map, _xParam, _yParam);

    }
    return 0;
}
