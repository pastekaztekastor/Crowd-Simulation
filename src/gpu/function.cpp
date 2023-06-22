/*******************************************************************************
* File Name: function.cpp 
* Author: Mathurin Champemont
* Created Date: 2023-06-14
* Last Modified: 2023-06-14
* Description: This file contains all the functions
*******************************************************************************/

// Include necessary libraries here
#include "main.hpp"

// Declare functions and classes here

/*
  _____       _ _ 
 |_   _|     (_) |  
   | |  _ __  _| |_ 
   | | | '_ \| | __|
  _| |_| | | | | |_ 
 |_____|_| |_|_|\__|

*/
void initSimParam( int argc, char const *argv[], uint2* simDim, uint * simDimP, uint * simPIn, uint * simDimG, uint * settings_print, uint * settings_debugMap, uint * settings_model, uint * settings_exportType, uint * settings_exportFormat, uint * settings_finishCondition,  std::string * settings_dir, std::string * settings_dirName, std::string * settings_fileName){
    if (argc > 1){
        for (size_t i = 1; i < argc; i += 2) {
            if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-help") == 0) {
                // print help
                printf(" +++++ HELP GPU VERSION +++++\n");

                printf("  -x        : sets the dimension in x of the simulation\n");
                printf("  -y        : same but for y dimension\n");
                printf("  -p        : number of individuals in the simulation\n");
                printf("  -g        : generation number/frame\n");
                printf("  -debug    : settings_print           param [val] default:'normal'\n");
                printf("              - 'off' : print nothing\n");
                printf("              - 'time' : print only time execution\n");
                printf("              - 'normal' : print time execution and simlulation progression\n");
                printf("              - 'all' : print all\n");
                printf("  -debugMap : settings_debugMap        param ['on'/'off'] default:'off'\n");
                printf("  -model    : settings_model           param [num] default:0\n");
                printf("              - 0 : Acctuel\n");
                printf("              - 1 : Rng\n");
                printf("              - 2 : Sage ignorant\n");
                printf("              - 3 : Impatient ingorant\n");
                printf("              - 4 : Forcée\n");
                printf("              - 5 : Cone de vision\n");
                printf("              - 6 : Meilleur cout\n");
                printf("              - 7 : Meilleur cout & déplacement forcé\n");
                printf("  -exptT    : settings_exportType      param ['type'] default:'txt'\n");
                printf("              - 'txt', 'jpeg', 'bin', ...\n");
                printf("  -exptF    : settings_exportFormat    param ['type'] default:'m'\n");
                printf("              - 'm' : export map\n");
                printf("              - 'p' : export position [ONLY 'txt' OR 'bin' exptT]\n");
                printf("              - 'c' : export congestion\n");
                printf("  -fCon     : settings_finishCondition param ['fix'/'empty'] default:empty\n");
                printf("  -dir      : settings_dir             param ['dir'] default:'bin/'\n");
                printf("              - Chose a custom directory path to export\n");
                printf("  -dirName  : settings_dirName         param ['name'] default:'X-Y-P'\n");
                printf("              - Chose a custom directory name to export\n");
                printf("  -fileName : settings_fileName        param ['std::string'] default:''\n");
                printf("              - Chose a custom file name to export\n");
                printf("  -help -h : help (if so...)\n");
                exit(0);
            } 
            else if (strcmp(argv[i], "-x") == 0) {
                simDim->x = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-y") == 0) {
                simDim->y = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-p") == 0) {
                (* simDimP) = atoi(argv[i + 1]);
                (* simPIn) = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-g") == 0) {
                (* simDimG) = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-debug") == 0) {
                if (strcmp(argv[i+1], "off") == 0) {
                    (* settings_print) = 0;
                }
                else if (strcmp(argv[i+1], "time") == 0) {
                    (* settings_print) = 10;
                }
                else if (strcmp(argv[i+1], "normal") == 0) {
                    (* settings_print) = 20;
                }
                else if (strcmp(argv[i+1], "all") == 0) {
                    (* settings_print) = 30;
                }
                else {
                    printf("Unrecognized argument for -debug param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-debugMap") == 0) {
                if (strcmp(argv[i+1], "off") == 0) {
                    (* settings_debugMap) = 0;
                }
                else if (strcmp(argv[i+1], "on") == 0) {
                    (* settings_debugMap) = 1;
                }
                else {
                    printf("Unrecognized argument for -debugMap param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-model") == 0) {
                (* settings_model) = atoi(argv[i + 1]);
            }
            else if (strcmp(argv[i], "-exptT") == 0) {
                if (strcmp(argv[i+1], "txt") == 0) {
                    (* settings_exportType) = 0;
                }
                else if (strcmp(argv[i+1], "bin") == 0) {
                    (* settings_exportType) = 1;
                }
                else if (strcmp(argv[i+1], "jpeg") == 0) {
                    (* settings_exportType) = 2;
                }
                else {
                    printf("Unrecognized argument for -exptT param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-exptF") == 0) {
                if (strcmp(argv[i+1], "m") == 0) {
                    (* settings_exportFormat) = 0;
                }
                else if (strcmp(argv[i+1], "p") == 0) {
                    (* settings_exportFormat) = 1;
                }
                else if (strcmp(argv[i+1], "c") == 0) {
                    (* settings_exportFormat) = 2;
                }
                else {
                    printf("Unrecognized argument for -exptF param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-fCon") == 0) {
                if (strcmp(argv[i+1], "fix") == 0) {
                    (* settings_finishCondition) = 0;
                }
                else if (strcmp(argv[i+1], "empty") == 0) {
                    (* settings_finishCondition) = 1;
                }
                else {
                    printf("Unrecognized argument for -fCon param\n");
                    exit(0);
                }
            }
            else if (strcmp(argv[i], "-dir") == 0) {
                (* settings_dir) = argv[i+1];
            }
            else if (strcmp(argv[i], "-dirName") == 0) {
                (* settings_dirName) = argv[i+1];
            }
            else if (strcmp(argv[i], "-fileName") == 0) {
                (* settings_fileName) = argv[i+1];
            }
            else{
                printf("Unrecognized argument, try -h or -help\n");
                exit(0);
            }
        }  
    }
    if((* simDimP) > (simDim->x*simDim->y)*0.8){
        std::cout << "The number of individuals exceeds 80\% of the simulation space. Terrain generation will not be compleor efficient. Please decrease the individual quantity or increase the dimensions.";
        exit(0);
    }
    
    if((*settings_print) > 1) std::cout << " *** WELCOME TO CROWD SIMULATION ON CUDA *** " << std::endl 
        << "\t # simDimX = " << simDim->x <<std::endl
        << "\t # simDimY = " << simDim->y <<std::endl
        << "\t # simDimG = " << (*simDimG) <<std::endl
        << "\t # simDimP = " << (*simDimP) <<std::endl
        << "\t # settings_print = " << (*settings_print) <<std::endl
        << "\t # settings_debugMap = " << (*settings_debugMap) <<std::endl
        << "\t # settings_model = " << (*settings_model) <<std::endl
        << "\t # settings_exportType = " << (*settings_exportType) <<std::endl
        << "\t # settings_exportFormat = " << (*settings_exportFormat) <<std::endl
        << "\t # settings_finishCondition = " << (*settings_finishCondition) <<std::endl
        << "\t # settings_dir = " << (*settings_dir) <<std::endl
        << "\t # settings_dirName = " << (*settings_dirName) <<std::endl
        << "\t # settings_fileName = " << (*settings_fileName) <<std::endl <<std::endl;
}
void initPopulationPositionMap( uint2 ** populationPosition, int ** map, uint2 simExit, uint2 simDim, uint simDimP, uint settings_print){
    if(settings_print >2) std::cout << "\t# initPopulationPositionMap  " << std::endl;
    uint2 coord = make_uint2((rand() % simDim.x),(rand() % simDim.y));
    
    // -1) Make memory allocations
    if(settings_print >2) std::cout << "\t -> Make memory allocations ";

    // ---- population
    (*populationPosition) = ( uint2 * ) calloc(simDimP, sizeof( uint2 ));

    // ---- map
    (*map) = ( int * ) calloc(simDim.x * simDim.y , sizeof( int ));
    for (size_t i = 0; i < simDim.x * simDim.y; i++){
        (*map)[i]=-1;
    }
    

    if(settings_print >2) std::cout << "\tOK " << std::endl ;
    
    // -2) Placing the walls
        // if(settings_print >2)std::cout << "   --> Placing the walls  ";
        // TO DO - Currently we do not put
        // if(settings_print >2)std::cout << "-> DONE " << std::endl;

    // -3) Placing exit
    if(settings_print >2) std::cout << "\t -> Placing element"<< std::endl;
    if(settings_print >2) std::cout << "\t --> Wall " ;
    (*map)[valueOfxy(simExit.x,simExit.y,simDim.x, simDim.y)] = -2;
    if(settings_print >2) std::cout << "\tOK " << std::endl ;
    
    // -4) Place individuals only if it is free.
    if(settings_print >2) std::cout << "\t --> People " ;
    for (size_t i = 0; i < simDimP; i++){
        coord = make_uint2((rand() % simDim.x),(rand() % simDim.y));

        while ((*map)[valueOfxy(coord.x,coord.y,simDim.x,simDim.y)] != -1){
            coord = make_uint2((rand() % simDim.x),(rand() % simDim.y));
        }
        // ---- population
        (*populationPosition)[i] = make_uint2(coord.x, coord.y) ;
        // ---- map
        (*map)[valueOfxy(coord.x,coord.y,simDim.x,simDim.y)] = i;
    }
    printPopulationPosition((*populationPosition), simDimP);

    if(settings_print >2)std::cout << "\tOK " << std::endl ;
}
void initPopulationIndex( uint ** populationIndex, uint simDimP, uint settings_print){
    if(settings_print >2)std::cout << "\t# initPopulationIndex  " ;
    (* populationIndex) = ( uint * ) calloc(simDimP * 2, sizeof( uint));
    if(settings_print >2)std::cout << "\tOK " << std::endl;
}
void initSimExit( uint2 * simExit, uint2 simDim, uint settings_print ){
    if(settings_print >2 ) std::cout << "\t# initSimExit  " ;
    (*simExit) = make_uint2((rand() % simDim.x),(rand() % simDim.y));
    if(settings_print >2) std::cout << "\tOK "  << std::endl;
}
void initCost( uint ** cost, int * map, uint2 simExit, uint2 simDim, uint settings_print){
    // TO DO
}

/*
   _____      _   _            
  / ____|    | | | |           
 | (___   ___| |_| |_ ___ _ __ 
  \___ \ / _ \ __| __/ _ \ '__|
  ____) |  __/ |_| ||  __/ |   
 |_____/ \___|\__|\__\___|_|   
                               
*/
void setSimExit( uint ** simExit, uint posX, uint posY, uint settings_print){
    // TO DO
}
void setPopulationPositionMap( uint *** populationPosition, uint *** map, uint * simExit, uint2 simDim, uint settings_print){
    // TO DO
}

/*
   _____ _                 _       _   _             
  / ____(_)               | |     | | (_)            
 | (___  _ _ __ ___  _   _| | __ _| |_ _  ___  _ __  
  \___ \| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \ 
  ____) | | | | | | | |_| | | (_| | |_| | (_) | | | |
 |_____/|_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|
                                                     
*/
void progressBar( uint progress, uint total, uint width, uint settings_print) {
    // From ChatGPT
    if(settings_print >2)std::cout << "\t# progressBar  " ;
    float percentage = (float)progress / total;
   uint filledWidth = ( uint)(percentage * width);
    printf("[");
    for ( uint i = 0; i < width; i++) {
        if (i < filledWidth) {
            printf("=");
        } else {
            printf(" ");
        }
    }
    printf("] %.2f%%\r", percentage * 100);
    fflush(stdout);
    if(settings_print >2)std::cout << "\tOK " << std::endl;
}
void shuffleIndex( uint ** PopulationIndex, uint simDimP, uint settings_print){
    if(settings_print >2)std::cout << "\t# shuffleIndex  " ;
    for ( uint i = simDimP - 1; i > 0; i--) {
       uint j = rand() % (i + 1);
       uint *temp = PopulationIndex[i];
        PopulationIndex[i] = PopulationIndex[j];
        PopulationIndex[j] = temp;
    }
    if(settings_print >2)std::cout << "\tOK " << std::endl;
}

/*
  ______             
 |  ____|            
 | |__ _ __ ___  ___ 
 |  __| '__/ _ \/ _ \
 | |  | | |  __/  __/
 |_|  |_|  \___|\___|
                     
*/
void freeTab ( uint ** populationPosition, uint settings_print){
    if(settings_print >2)std::cout << "\t# freeTab  " ;
    free(*populationPosition);
    *populationPosition = NULL;
    if(settings_print >2)std::cout << "\tOK " << std::endl;
}

/*
  _    _ _   _ _     
 | |  | | | (_) |    
 | |  | | |_ _| |___ 
 | |  | | __| | / __|
 | |__| | |_| | \__ \
  \____/ \__|_|_|___/
                     
*/
void printMap(int * map, uint2 simDim, uint settings_print){
    if(settings_print > 2)std::cout << " # - Display map --- "<<std::endl;

    // Display column numbers
    std::cout<<"  ";
        for (int x = 0; x < simDim.x; x++)
        {
            printf(" %2d",x); 
        }
        std::cout<<"  "<<std::endl;

    // We browse the map and we display according to what the box contains
    for (int y = 0; y < simDim.y; y++)
    {
        printf("%2d ",y); 
        for (int x = 0; x < simDim.x; x++)
        {
            switch (map[valueOfxy(x,y,simDim.x,simDim.y)])
            {
            case -3:
                std::cout<<"///";
                break;
            case -2:
                std::cout<<"(s)";
                break;
            case -1:
                std::cout<<" . ";
                break;
            default:
                std::cout<<"[H]";
                break;
            }
        }
        std::cout<<std::endl;
    }
    if(settings_print < 2)std::cout << "                         --- DONE " << std::endl;
}
uint xPosof(uint value, uint dimX, uint dimY) {
    return value % dimX;
}
uint yPosof(uint value, uint dimX, uint dimY) {
    return value / dimX;
}
uint valueOfxy(uint xPos, uint yPos, uint dimX, uint dimY) {
    return yPos * dimX + xPos;
}
void printPopulationPosition(uint2 * population, uint simDimP){
    std::cout << std::endl << "\t# printPopulationPosition"<< std::endl << "\t   Id \t x \t y " <<std::endl;
    for (size_t i = 0; i < simDimP; i++)
    {
        std::cout << "\t  - " << i << " )\t" << population[i].x << "\t" << population[i].y <<std::endl;
    }
    std::cout << "\t OK" <<std::endl; 
}

